import json
import os
import shutil
import threading
import time
from urllib.parse import urlparse

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

        try:
            self.compact = str(os.getenv("HF_INDEX_COMPACT", "0") or "0").strip().lower() in ("1", "true", "yes", "y")
        except Exception:
            self.compact = False
        try:
            default_drop = "1" if bool(self.compact) else "0"
            self.compact_drop_empty = str(os.getenv("HF_INDEX_COMPACT_DROP_EMPTY", default_drop) or default_drop).strip().lower() in (
                "1",
                "true",
                "yes",
                "y",
            )
        except Exception:
            self.compact_drop_empty = bool(self.compact)

        try:
            default_mode = "path" if bool(self.compact) else "url"
            self.asset_mode = str(os.getenv("HF_INDEX_ASSET_MODE", default_mode) or default_mode).strip().lower()
        except Exception:
            self.asset_mode = "path" if bool(self.compact) else "url"
        if str(self.asset_mode) not in ("url", "path", "both", "none"):
            self.asset_mode = "url"

        try:
            default_text = "full"
            self.text_mode = str(os.getenv("HF_INDEX_TEXT_MODE", default_text) or default_text).strip().lower()
        except Exception:
            self.text_mode = "full"
        if str(self.text_mode) not in ("full", "minimal", "none"):
            self.text_mode = "full"

        try:
            default_drop_urls = "1" if bool(self.compact) else "0"
            self.drop_derivable_urls = str(os.getenv("HF_INDEX_DROP_DERIVABLE_URLS", default_drop_urls) or default_drop_urls).strip().lower() in (
                "1",
                "true",
                "yes",
                "y",
            )
        except Exception:
            self.drop_derivable_urls = bool(self.compact)

        try:
            default_drop_user_name = "1" if bool(self.compact) else "0"
            self.drop_user_name = str(os.getenv("HF_INDEX_DROP_USER_NAME", default_drop_user_name) or default_drop_user_name).strip().lower() in (
                "1",
                "true",
                "yes",
                "y",
            )
        except Exception:
            self.drop_user_name = bool(self.compact)

        try:
            default_drop_unsplash_id = "1" if bool(self.compact) else "0"
            self.drop_unsplash_id = str(
                os.getenv("HF_INDEX_DROP_UNSPLASH_ID", default_drop_unsplash_id) or default_drop_unsplash_id
            ).strip().lower() in (
                "1",
                "true",
                "yes",
                "y",
            )
        except Exception:
            self.drop_unsplash_id = bool(self.compact)

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

        def _try_to_repo_path(v: str) -> str:
            try:
                s = str(v or "").strip()
                if not s:
                    return ""
                if s.startswith("/") and ("/resolve/" not in s):
                    return s.lstrip("/")
                if not (s.startswith("http://") or s.startswith("https://")):
                    return s.lstrip("/")
                p = urlparse(s)
                parts = [x for x in str(p.path or "").strip("/").split("/") if x]
                if len(parts) >= 6 and parts[0] in ("datasets", "models") and parts[3] == "resolve":
                    return "/".join(parts[5:])
                return s.lstrip("/")
            except Exception:
                try:
                    return str(v or "").strip().lstrip("/")
                except Exception:
                    return ""

        for k in ("image_url", "ply_url", "spz_url"):
            try:
                v = out.get(k)
                out[k] = "" if v is None else str(v)
            except Exception:
                out[k] = ""

        if bool(getattr(self, "drop_derivable_urls", False)):
            try:
                out.pop("image_url", None)
                out.pop("ply_url", None)
                out.pop("spz_url", None)
            except Exception:
                pass

        if (not bool(getattr(self, "drop_derivable_urls", False))) and str(getattr(self, "asset_mode", "url")) in ("path", "both"):
            try:
                imgp = _try_to_repo_path(out.get("image_url"))
                plyp = _try_to_repo_path(out.get("ply_url"))
                spzp = _try_to_repo_path(out.get("spz_url"))
                if imgp:
                    out["image_path"] = imgp
                if plyp:
                    out["ply_path"] = plyp
                if spzp:
                    out["spz_path"] = spzp
                if str(getattr(self, "asset_mode", "url")) == "path":
                    out.pop("image_url", None)
                    out.pop("ply_url", None)
                    out.pop("spz_url", None)
            except Exception:
                pass
        if bool(getattr(self, "drop_derivable_urls", False)) or str(getattr(self, "asset_mode", "url")) == "none":
            try:
                out.pop("image_path", None)
                out.pop("ply_path", None)
                out.pop("spz_path", None)
            except Exception:
                pass

        for k in (
            "gsplat_url",
            "gsplat_share_id",
            "gsplat_order_id",
            "gsplat_model_file_url",
            "unsplash_url",
            "created_at",
            "user_username",
            "user_name",
        ):
            try:
                if (not bool(getattr(self, "compact", False))) or (k in out):
                    v = out.get(k)
                    out[k] = "" if v is None else str(v)
            except Exception:
                if (not bool(getattr(self, "compact", False))) or (k in out):
                    out[k] = ""

        try:
            s = str(out.get("gsplat_model_file_url") or "").strip()
            if s:
                # Accept formats like:
                # - /share/file/<token>.ply
                # - share/file/<token>.ply
                # - <token>.ply
                # - <token>
                s2 = s
                try:
                    s2 = s2.split("?", 1)[0]
                    s2 = s2.split("#", 1)[0]
                except Exception:
                    s2 = s2
                s2 = s2.strip().lstrip("/")
                if "/share/file/" in "/" + s2:
                    try:
                        s2 = ("/" + s2).split("/share/file/", 1)[1]
                    except Exception:
                        s2 = s2
                # keep only last path segment
                try:
                    if "/" in s2:
                        s2 = s2.rsplit("/", 1)[-1]
                except Exception:
                    s2 = s2
                if s2.endswith(".ply"):
                    s2 = s2[: -len(".ply")]
                out["gsplat_model_file_url"] = str(s2).strip()
        except Exception:
            pass

        if bool(getattr(self, "drop_derivable_urls", False)):
            try:
                out.pop("gsplat_url", None)
                out.pop("unsplash_url", None)
            except Exception:
                pass

        if bool(getattr(self, "drop_user_name", False)):
            try:
                out.pop("user_name", None)
            except Exception:
                pass

        if bool(getattr(self, "drop_unsplash_id", False)):
            try:
                out.pop("unsplash_id", None)
            except Exception:
                pass

        def _tokenize(v):
            try:
                if v is None:
                    return []
                if isinstance(v, list):
                    items = []
                    for it in v:
                        if it is None:
                            continue
                        items.append(str(it))
                    s = " ".join(items)
                else:
                    s = str(v)
                s = s.replace("\u3001", ",")
                parts = []
                cur = ""
                for ch in s:
                    if ch.isspace() or ch == ",":
                        if cur:
                            parts.append(cur)
                            cur = ""
                        continue
                    cur += ch
                if cur:
                    parts.append(cur)
                outp = []
                seen = set()
                for p in parts:
                    t = str(p or "").strip()
                    if not t:
                        continue
                    key = t.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    outp.append(t)
                return outp
            except Exception:
                return []

        tags_tokens = _tokenize(out.get("tags")) + _tokenize(out.get("tags_text"))
        topics_tokens = _tokenize(out.get("topics")) + _tokenize(out.get("topics_text"))

        # Dedupe again after concatenation
        def _dedupe(tokens):
            res = []
            seen = set()
            for t in tokens:
                tt = str(t or "").strip()
                if not tt:
                    continue
                key = tt.lower()
                if key in seen:
                    continue
                seen.add(key)
                res.append(tt)
            return res

        tags_tokens = _dedupe(tags_tokens)
        topics_tokens = _dedupe(topics_tokens)

        tags_text = " ".join(tags_tokens)
        topics_text = " ".join(topics_tokens)
        out["tags"] = tags_text
        out["topics"] = topics_text
        out["tags_text"] = tags_text
        out["topics_text"] = topics_text

        try:
            tm = str(getattr(self, "text_mode", "full") or "full").strip().lower()
        except Exception:
            tm = "full"
        if tm == "none":
            try:
                out.pop("tags", None)
                out.pop("topics", None)
                out.pop("tags_text", None)
                out.pop("topics_text", None)
            except Exception:
                pass
        elif tm == "minimal":
            try:
                if "topics" in out and str(out.get("topics") or "") == "":
                    out.pop("topics", None)
            except Exception:
                pass

        for k in ("description", "alt_description"):
            try:
                if tm != "none":
                    if out.get(k) is None:
                        out[k] = ""
                        continue
                    if isinstance(out.get(k), (dict, list)):
                        out[k] = json.dumps(out.get(k), ensure_ascii=False)
                    else:
                        out[k] = str(out.get(k))
            except Exception:
                try:
                    if tm != "none":
                        out[k] = str(out.get(k))
                except Exception:
                    if tm != "none":
                        out[k] = ""

        if tm == "none":
            try:
                out.pop("description", None)
                out.pop("alt_description", None)
            except Exception:
                pass
        elif tm == "minimal":
            try:
                if "description" in out and str(out.get("description") or "") == "":
                    out.pop("description", None)
                if "alt_description" in out and str(out.get("alt_description") or "") == "":
                    out.pop("alt_description", None)
            except Exception:
                pass

        if bool(getattr(self, "compact", False)) and bool(getattr(self, "compact_drop_empty", False)):
            for k in (
                "gsplat_url",
                "gsplat_share_id",
                "gsplat_order_id",
                "gsplat_model_file_url",
            ):
                try:
                    if k in out and str(out.get(k) or "") == "":
                        out.pop(k, None)
                except Exception:
                    continue

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
