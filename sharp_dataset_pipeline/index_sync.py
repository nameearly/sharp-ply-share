import json
import os
import shutil
import threading
import time

from . import hf_utils


class IndexSync:
    def __init__(
        self,
        repo_id: str,
        *,
        repo_type: str,
        repo_path: str,
        save_dir: str,
        hf_upload: bool,
        hf_index_flush_every: int,
        hf_index_flush_secs: float,
        hf_index_refresh_secs: float = 300.0,
        debug_fn,
    ):
        self.repo_id = str(repo_id or "").strip()
        self.repo_type = str(repo_type or "dataset").strip().lower()
        self.repo_path = str(repo_path or "").strip().lstrip("/")
        self.save_dir = os.path.abspath(str(save_dir or os.getcwd()))
        self.local_path = os.path.join(
            self.save_dir,
            os.path.basename(self.repo_path) if self.repo_path else "train.jsonl",
        )

        self.hf_upload = bool(hf_upload)
        self.hf_index_flush_every = int(hf_index_flush_every)
        self.hf_index_flush_secs = float(hf_index_flush_secs)
        self.hf_index_refresh_secs = float(hf_index_refresh_secs)
        self.debug_fn = debug_fn

        self.lock = threading.Lock()
        self.indexed: set[str] = set()
        self.pending = 0
        self.last_flush_ts = 0.0
        self.last_refresh_ts = 0.0
        self._refresh_inflight = False
        self._last_remote_local = ""

        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except Exception:
            pass

        self._init_from_remote()
        changed = self._sanitize_local_index()
        self._load_indexed_ids()
        try:
            if changed:
                self.pending = max(int(self.pending), 1)
                self.maybe_flush(True)
        except Exception:
            pass

    def _print(self, msg: str):
        try:
            if self.debug_fn:
                self.debug_fn(msg)
        except Exception:
            return

    def _normalize_list_str(self, v):
        if v is None:
            return []
        if isinstance(v, str):
            s = v.strip()
            return [s] if s else []
        if isinstance(v, list):
            out = []
            for item in v:
                if item is None:
                    continue
                if isinstance(item, str):
                    s = item.strip()
                elif isinstance(item, dict):
                    s = str(item.get("title") or item.get("name") or item.get("slug") or "").strip()
                else:
                    s = str(item).strip()
                if s:
                    out.append(s)
            return out
        s = str(v).strip()
        return [s] if s else []

    def _normalize_row(self, row: dict):
        if not isinstance(row, dict):
            return None
        pid = str(row.get("image_id") or "").strip()
        if not pid:
            return None
        out = dict(row)
        out["image_id"] = pid

        for k in (
            "image_url",
            "ply_url",
            "spz_url",
            "gsplat_url",
            "gsplat_share_id",
            "gsplat_order_id",
            "gsplat_model_file_url",
            "unsplash_id",
            "unsplash_url",
            "created_at",
            "user_username",
            "user_name",
        ):
            try:
                v = out.get(k)
                out[k] = "" if v is None else str(v)
            except Exception:
                out[k] = ""

        tags = self._normalize_list_str(out.get("tags"))
        topics = self._normalize_list_str(out.get("topics"))
        tags_text = " ".join(tags)
        topics_text = " ".join(topics)
        out["tags"] = tags_text
        out["topics"] = topics_text
        out["tags_text"] = tags_text
        out["topics_text"] = topics_text

        for k in ("description", "alt_description"):
            try:
                if out.get(k) is None:
                    out[k] = ""
                    continue
                if isinstance(out.get(k), (dict, list)):
                    out[k] = json.dumps(out.get(k), ensure_ascii=False)
                else:
                    out[k] = str(out.get(k))
            except Exception:
                try:
                    out[k] = str(out.get(k))
                except Exception:
                    out[k] = ""

        return out

    def _sanitize_local_index(self) -> bool:
        try:
            if not os.path.exists(self.local_path):
                return False
            tmp_path = self.local_path + ".tmp"
            seen = set()
            changed = False
            with open(self.local_path, "r", encoding="utf-8") as rf, open(tmp_path, "w", encoding="utf-8") as wf:
                for line in rf:
                    s = line.strip()
                    if not s:
                        changed = True
                        continue
                    try:
                        obj = json.loads(s)
                    except Exception:
                        changed = True
                        continue
                    norm = self._normalize_row(obj)
                    if not norm:
                        changed = True
                        continue
                    pid = str(norm.get("image_id") or "")
                    if not pid or pid in seen:
                        changed = True
                        continue
                    seen.add(pid)
                    wf.write(json.dumps(norm, ensure_ascii=False) + "\n")
                    if norm is not obj:
                        changed = True

            if changed:
                try:
                    os.replace(tmp_path, self.local_path)
                except Exception:
                    try:
                        shutil.copyfile(tmp_path, self.local_path)
                    except Exception:
                        pass
            else:
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            return bool(changed)
        except Exception:
            return False

    def _init_from_remote(self):
        if (not self.repo_path) or (not self.repo_id):
            return
        try:
            from huggingface_hub import hf_hub_download

            remote_local = hf_hub_download(repo_id=self.repo_id, repo_type=self.repo_type, filename=self.repo_path)
            try:
                self._last_remote_local = str(remote_local or "")
            except Exception:
                self._last_remote_local = ""
            try:
                self.last_refresh_ts = time.time()
            except Exception:
                self.last_refresh_ts = 0.0
            try:
                shutil.copyfile(remote_local, self.local_path)
            except Exception:
                try:
                    with open(remote_local, "rb") as rf, open(self.local_path, "wb") as wf:
                        wf.write(rf.read())
                except Exception:
                    return
        except Exception:
            return

    def _iter_ids_from_jsonl(self, path: str):
        try:
            if not path or (not os.path.exists(path)):
                return
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if not s:
                        continue
                    try:
                        obj = json.loads(s)
                        pid = str(obj.get("image_id") or "").strip()
                        if pid:
                            yield pid
                    except Exception:
                        continue
        except Exception:
            return

    def _load_indexed_ids(self):
        try:
            if not os.path.exists(self.local_path):
                return
            for pid in self._iter_ids_from_jsonl(self.local_path):
                self.indexed.add(pid)
        except Exception:
            return

    def maybe_refresh(self, force: bool = False) -> bool:
        if (not self.repo_path) or (not self.repo_id):
            return False

        now = time.time()
        try:
            with self.lock:
                if self._refresh_inflight:
                    return False
                if (not force) and (now - float(self.last_refresh_ts)) < float(self.hf_index_refresh_secs):
                    return False
                self._refresh_inflight = True
        except Exception:
            return False

        remote_local = ""
        changed = False
        try:
            from huggingface_hub import hf_hub_download

            remote_local = hf_hub_download(repo_id=self.repo_id, repo_type=self.repo_type, filename=self.repo_path)
            remote_local = str(remote_local or "")

            try:
                with self.lock:
                    if remote_local and (remote_local == str(self._last_remote_local or "")):
                        return False
            except Exception:
                pass

            new_ids = set(self._iter_ids_from_jsonl(remote_local))
            if new_ids:
                with self.lock:
                    before = len(self.indexed)
                    self.indexed |= new_ids
                    changed = len(self.indexed) > before
        except Exception:
            return False
        finally:
            try:
                with self.lock:
                    self.last_refresh_ts = now
                    if remote_local:
                        self._last_remote_local = remote_local
                    self._refresh_inflight = False
            except Exception:
                pass

        return bool(changed)

    def add_row(self, row: dict):
        norm = self._normalize_row(row)
        if not norm:
            return
        pid = str(norm.get("image_id") or "")
        with self.lock:
            if pid in self.indexed:
                return
            try:
                with open(self.local_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(norm, ensure_ascii=False) + "\n")
                self.indexed.add(pid)
                self.pending += 1
            except Exception:
                return
        self.maybe_flush(False)

    def maybe_flush(self, force: bool):
        if (not self.hf_upload) or (not self.repo_id) or (not self.repo_path):
            return
        try:
            with self.lock:
                now = time.time()
                if int(self.pending) <= 0:
                    return
                if not force:
                    if self.pending < int(self.hf_index_flush_every) and (now - float(self.last_flush_ts)) < float(
                        self.hf_index_flush_secs
                    ):
                        return

                if not os.path.isfile(self.local_path):
                    return

                from huggingface_hub import CommitOperationAdd, HfApi

                api = HfApi()
                self._sanitize_local_index()
                ops = [CommitOperationAdd(path_in_repo=self.repo_path, path_or_fileobj=self.local_path)]
                try:
                    api.create_commit(
                        repo_id=self.repo_id,
                        repo_type=self.repo_type,
                        operations=ops,
                        commit_message="index update",
                    )
                except Exception as e:
                    if not hf_utils.should_retry_with_pr(e):
                        raise
                    api.create_commit(
                        repo_id=self.repo_id,
                        repo_type=self.repo_type,
                        operations=ops,
                        commit_message="index update",
                        create_pr=True,
                    )

                self.pending = 0
                self.last_flush_ts = now
        except Exception as e:
            self._print(f"HF index 上传失败（可忽略） | err={str(e)}")
            return
