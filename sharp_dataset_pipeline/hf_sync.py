import io
import os
import re
import json
import threading
import time
import uuid


_HF_UPLOAD = False
_HF_REPO_TYPE = "model"
_HF_LOCKS_DIR = "locks"
_HF_DONE_DIR = "done"
_RANGE_LOCKS_DIR = "ranges/locks"
_RANGE_DONE_DIR = "ranges/done"
_RANGE_PROGRESS_DIR = "ranges/progress"
_RANGE_ABANDONED_DIR = "ranges/abandoned"
_HF_LOCK_STALE_SECS = 21600.0
_RANGE_LOCK_STALE_SECS = 21600.0
_debug = None

_hf_exists_cache_lock = threading.Lock()
_hf_exists_cache = {}


def _hf_hub_download_quiet(*, repo_id: str, filename: str):
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo_id=repo_id, repo_type=_HF_REPO_TYPE, filename=filename, disable_tqdm=True)
    except TypeError:
        pass
    try:
        return hf_hub_download(repo_id=repo_id, repo_type=_HF_REPO_TYPE, filename=filename, tqdm_class=None)
    except TypeError:
        pass
    return hf_hub_download(repo_id=repo_id, repo_type=_HF_REPO_TYPE, filename=filename)


def configure_hf_sync(
    *,
    hf_upload: bool,
    repo_type: str,
    hf_locks_dir: str,
    hf_done_dir: str,
    range_locks_dir: str,
    range_done_dir: str,
    range_progress_dir: str = "ranges/progress",
    range_abandoned_dir: str = "ranges/abandoned",
    hf_lock_stale_secs: float,
    range_lock_stale_secs: float,
    debug_fn=None,
):
    global _HF_UPLOAD, _HF_REPO_TYPE, _HF_LOCKS_DIR, _HF_DONE_DIR
    global _RANGE_LOCKS_DIR, _RANGE_DONE_DIR, _RANGE_PROGRESS_DIR, _RANGE_ABANDONED_DIR
    global _HF_LOCK_STALE_SECS, _RANGE_LOCK_STALE_SECS, _debug
    _HF_UPLOAD = bool(hf_upload)
    _HF_REPO_TYPE = str(repo_type)
    _HF_LOCKS_DIR = str(hf_locks_dir).strip().strip('/')
    _HF_DONE_DIR = str(hf_done_dir).strip().strip('/')
    _RANGE_LOCKS_DIR = str(range_locks_dir).strip().strip('/')
    _RANGE_DONE_DIR = str(range_done_dir).strip().strip('/')
    _RANGE_PROGRESS_DIR = str(range_progress_dir).strip().strip('/')
    _RANGE_ABANDONED_DIR = str(range_abandoned_dir).strip().strip('/')
    try:
        _HF_LOCK_STALE_SECS = float(hf_lock_stale_secs)
    except Exception:
        _HF_LOCK_STALE_SECS = 21600.0
    try:
        _RANGE_LOCK_STALE_SECS = float(range_lock_stale_secs)
    except Exception:
        _RANGE_LOCK_STALE_SECS = 21600.0
    _debug = debug_fn


def _d(msg: str) -> None:
    if _debug is None:
        return
    try:
        _debug(msg)
    except Exception:
        pass


def hf_locks_repo_path(image_id: str) -> str:
    base = str(_HF_LOCKS_DIR).strip().strip('/')
    if base:
        return f"{base}/{image_id}"
    return str(image_id)


def hf_done_repo_path(image_id: str) -> str:
    base = str(_HF_DONE_DIR).strip().strip('/')
    if base:
        return f"{base}/{image_id}"
    return str(image_id)


def _hf_range_lock_repo_path(range_start: int, range_end: int) -> str:
    base = str(_RANGE_LOCKS_DIR).strip().strip('/')
    name = f"{int(range_start)}-{int(range_end)}"
    if base:
        return f"{base}/{name}"
    return name


def _hf_range_progress_repo_path(range_start: int, range_end: int) -> str:
    base = str(_RANGE_PROGRESS_DIR).strip().strip('/')
    name = f"{int(range_start)}-{int(range_end)}.json"
    if base:
        return f"{base}/{name}"
    return name


def _hf_range_abandoned_repo_path(range_start: int, range_end: int) -> str:
    base = str(_RANGE_ABANDONED_DIR).strip().strip('/')
    name = f"{int(range_start)}-{int(range_end)}.json"
    if base:
        return f"{base}/{name}"
    return name


def _hf_range_done_repo_path(range_start: int, range_end: int) -> str:
    base = str(_RANGE_DONE_DIR).strip().strip('/')
    name = f"{int(range_start)}-{int(range_end)}"
    if base:
        return f"{base}/{name}"
    return name


def hf_file_exists_cached(repo_id: str, path_in_repo: str, ttl_s: float = 120.0) -> bool:
    if (not _HF_UPLOAD) or (not repo_id) or (not path_in_repo):
        return False
    key = (str(repo_id), str(path_in_repo))
    now = time.time()
    try:
        with _hf_exists_cache_lock:
            hit = _hf_exists_cache.get(key)
        if hit is not None:
            ok, ts = hit
            if (now - float(ts)) <= float(ttl_s):
                return bool(ok)
    except Exception:
        pass
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        ok = bool(api.file_exists(repo_id=repo_id, repo_type=_HF_REPO_TYPE, filename=key[1]))
    except Exception:
        ok = False
    try:
        with _hf_exists_cache_lock:
            _hf_exists_cache[key] = (bool(ok), float(now))
    except Exception:
        pass
    return bool(ok)


def _hf_try_read_json(repo_id: str, repo_path: str):
    if (not _HF_UPLOAD) or (not repo_id) or (not repo_path):
        return None
    try:
        local = _hf_hub_download_quiet(repo_id=repo_id, filename=repo_path)
        with open(local, 'r', encoding='utf-8') as f:
            return json.loads(f.read() or "{}")
    except Exception:
        return None


def _hf_try_write_json(repo_id: str, repo_path: str, payload_obj: dict, commit_message: str) -> bool:
    if (not _HF_UPLOAD) or (not repo_id) or (not repo_path):
        return False
    try:
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi()
        blob = (json.dumps(payload_obj, ensure_ascii=False) + "\n").encode('utf-8')
        ops = [CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=io.BytesIO(blob))]
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type=_HF_REPO_TYPE,
                operations=ops,
                commit_message=str(commit_message or 'range meta update'),
            )
        except Exception as e:
            if not _hf_should_retry_with_pr(e):
                raise
            api.create_commit(
                repo_id=repo_id,
                repo_type=_HF_REPO_TYPE,
                operations=ops,
                commit_message=str(commit_message or 'range meta update'),
                create_pr=True,
            )
        return True
    except Exception as e:
        _d(f"HF range meta 写入失败（可忽略） | err={str(e)}")
        return False


def _parse_range_name(name: str):
    try:
        s = str(name or '').strip()
        m = re.match(r"^(\d+)-(\d+)$", s)
        if not m:
            return None
        a = int(m.group(1))
        b = int(m.group(2))
        if a < 0 or b < a:
            return None
        return (a, b)
    except Exception:
        return None


def _hf_try_list_dir_ids(repo_id: str, prefix_dir: str) -> set:
    if (not _HF_UPLOAD) or (not repo_id) or (not prefix_dir):
        return set()
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id, repo_type=_HF_REPO_TYPE)
        prefix = str(prefix_dir).strip().strip('/') + '/'
        out = set()
        for fp in files or []:
            s = str(fp)
            if not s.startswith(prefix):
                continue
            image_id = s[len(prefix) :].strip().strip('/')
            if image_id:
                out.add(image_id)
        return out
    except Exception as e:
        _d(f"HF 列目录失败（可忽略） | err={str(e)}")
        return set()


def _hf_try_list_dir_ranges(repo_id: str, prefix_dir: str) -> set:
    if (not _HF_UPLOAD) or (not repo_id) or (not prefix_dir):
        return set()
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        files = api.list_repo_files(repo_id=repo_id, repo_type=_HF_REPO_TYPE)
        prefix = str(prefix_dir).strip().strip('/') + '/'
        out = set()
        for fp in files or []:
            s = str(fp)
            if not s.startswith(prefix):
                continue
            name = s[len(prefix) :].strip().strip('/')
            parsed = _parse_range_name(name)
            if parsed:
                out.add(parsed)
        return out
    except Exception as e:
        _d(f"HF 列目录失败（可忽略） | err={str(e)}")
        return set()


def _hf_should_retry_with_pr(err: Exception) -> bool:
    try:
        s = str(err)
        return ("create_pr=1" in s) or ("create_pr" in s and "Pull Request" in s)
    except Exception:
        return False


def _hf_try_get_lock_info_status(repo_id: str, image_id: str):
    if (not _HF_UPLOAD) or (not repo_id) or (not image_id):
        return (None, False)
    try:
        lock_path = hf_locks_repo_path(image_id)
        local = _hf_hub_download_quiet(repo_id=repo_id, filename=lock_path)
        ts = None
        owner = None
        with open(local, 'r', encoding='utf-8') as f:
            lines = [x.strip() for x in f.read().splitlines()]
        if len(lines) >= 1 and lines[0]:
            try:
                ts = float(lines[0])
            except Exception:
                ts = None
        if len(lines) >= 2 and lines[1]:
            owner = lines[1]
        extra = None
        if len(lines) >= 3 and lines[2]:
            extra = lines[2]
        return ({"ts": ts, "owner": owner, "extra": extra}, False)
    except Exception as e:
        try:
            s = str(e)
            if "404" in s or "EntryNotFound" in s or "Not Found" in s:
                return (None, False)
        except Exception:
            pass
        return (None, True)


def _hf_try_write_lock(repo_id: str, image_id: str, owner: str, ts: float, extra: str | None = None) -> bool:
    if (not _HF_UPLOAD) or (not repo_id) or (not image_id):
        return False
    try:
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi()
        lock_path = hf_locks_repo_path(image_id)
        extra_s = str(extra).strip() if extra is not None else ""
        payload = f"{float(ts)}\n{str(owner or '')}\n{extra_s}\n".encode('utf-8')
        ops = [CommitOperationAdd(path_in_repo=lock_path, path_or_fileobj=io.BytesIO(payload))]
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type=_HF_REPO_TYPE,
                operations=ops,
                commit_message=f"lock {image_id}",
            )
        except Exception as e:
            if not _hf_should_retry_with_pr(e):
                raise
            api.create_commit(
                repo_id=repo_id,
                repo_type=_HF_REPO_TYPE,
                operations=ops,
                commit_message=f"lock {image_id}",
                create_pr=True,
            )
        return True
    except Exception as e:
        _d(f"HF lock 写入失败（可忽略） | err={str(e)}")
        return False


def _hf_try_write_done(repo_id: str, image_id: str) -> bool:
    if (not _HF_UPLOAD) or (not repo_id) or (not image_id):
        return False
    try:
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi()
        done_path = hf_done_repo_path(image_id)
        ops = [CommitOperationAdd(path_in_repo=done_path, path_or_fileobj=io.BytesIO(b""))]
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type=_HF_REPO_TYPE,
                operations=ops,
                commit_message=f"done {image_id}",
            )
        except Exception as e:
            if not _hf_should_retry_with_pr(e):
                raise
            api.create_commit(
                repo_id=repo_id,
                repo_type=_HF_REPO_TYPE,
                operations=ops,
                commit_message=f"done {image_id}",
                create_pr=True,
            )
        return True
    except Exception as e:
        _d(f"HF done 写入失败（可忽略） | err={str(e)}")
        return False


def _hf_try_get_range_lock_info(repo_id: str, range_start: int, range_end: int):
    if (not _HF_UPLOAD) or (not repo_id):
        return None
    try:
        lock_path = _hf_range_lock_repo_path(range_start, range_end)
        local = _hf_hub_download_quiet(repo_id=repo_id, filename=lock_path)
        ts = None
        owner = None
        with open(local, 'r', encoding='utf-8') as f:
            lines = [x.strip() for x in f.read().splitlines()]
        if len(lines) >= 1 and lines[0]:
            try:
                ts = float(lines[0])
            except Exception:
                ts = None
        if len(lines) >= 2 and lines[1]:
            owner = lines[1]
        return {"ts": ts, "owner": owner}
    except Exception:
        return None


def _hf_try_write_range_lock(repo_id: str, range_start: int, range_end: int, owner: str, ts: float, extra: str | None = None) -> bool:
    if (not _HF_UPLOAD) or (not repo_id):
        return False
    try:
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi()
        lock_path = _hf_range_lock_repo_path(range_start, range_end)
        extra_s = str(extra).strip() if extra is not None else ""
        payload = f"{float(ts)}\n{str(owner or '')}\n{extra_s}\n".encode('utf-8')
        ops = [CommitOperationAdd(path_in_repo=lock_path, path_or_fileobj=io.BytesIO(payload))]
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type=_HF_REPO_TYPE,
                operations=ops,
                commit_message=f"range lock {range_start}-{range_end}",
            )
        except Exception as e:
            if not _hf_should_retry_with_pr(e):
                raise
            api.create_commit(
                repo_id=repo_id,
                repo_type=_HF_REPO_TYPE,
                operations=ops,
                commit_message=f"range lock {range_start}-{range_end}",
                create_pr=True,
            )
        return True
    except Exception as e:
        _d(f"HF range lock 写入失败（可忽略） | err={str(e)}")
        return False


def _hf_try_write_range_done(repo_id: str, range_start: int, range_end: int) -> bool:
    if (not _HF_UPLOAD) or (not repo_id):
        return False
    try:
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi()
        done_path = _hf_range_done_repo_path(range_start, range_end)
        ops = [CommitOperationAdd(path_in_repo=done_path, path_or_fileobj=io.BytesIO(b""))]
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type=_HF_REPO_TYPE,
                operations=ops,
                commit_message=f"range done {range_start}-{range_end}",
            )
        except Exception as e:
            if not _hf_should_retry_with_pr(e):
                raise
            api.create_commit(
                repo_id=repo_id,
                repo_type=_HF_REPO_TYPE,
                operations=ops,
                commit_message=f"range done {range_start}-{range_end}",
                create_pr=True,
            )
        return True
    except Exception as e:
        _d(f"HF range done 写入失败（可忽略） | err={str(e)}")
        return False


class LockDoneSync:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.instance_id = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.done = _hf_try_list_dir_ids(repo_id, _HF_DONE_DIR)
        self._recent = {}

    def is_done(self, image_id: str) -> bool:
        with self.lock:
            return image_id in (self.done or set())

    def try_lock_status(self, image_id: str, extra: str | None = None):
        if not image_id:
            return ("error", None)
        if self.is_done(image_id):
            return ("done", None)

        try:
            now = time.time()
            with self.lock:
                rec = self._recent.get(str(image_id))
            if rec is not None:
                st, until = rec
                if (until is not None) and float(until) > float(now):
                    return (st, float(until))
        except Exception:
            pass

        info, info_err = _hf_try_get_lock_info_status(self.repo_id, image_id)
        if info_err:
            ra = time.time() + 30.0
            try:
                with self.lock:
                    self._recent[str(image_id)] = ("error", float(ra))
            except Exception:
                pass
            return ("error", ra)

        if info is not None:
            ts = info.get("ts")
            if ts is not None:
                try:
                    tsf = float(ts)
                    age = time.time() - tsf
                    if age < float(_HF_LOCK_STALE_SECS):
                        ra = tsf + float(_HF_LOCK_STALE_SECS)
                        try:
                            with self.lock:
                                self._recent[str(image_id)] = ("locked_by_other", float(ra))
                        except Exception:
                            pass
                        return ("locked_by_other", ra)
                except Exception:
                    ra = time.time() + 30.0
                    try:
                        with self.lock:
                            self._recent[str(image_id)] = ("error", float(ra))
                    except Exception:
                        pass
                    return ("error", ra)

        ok = _hf_try_write_lock(self.repo_id, image_id, self.instance_id, time.time(), extra=extra)
        if ok:
            ra = time.time() + float(_HF_LOCK_STALE_SECS)
            try:
                with self.lock:
                    self._recent[str(image_id)] = ("acquired", float(ra))
            except Exception:
                pass
            return ("acquired", ra)
        ra = time.time() + 30.0
        try:
            with self.lock:
                self._recent[str(image_id)] = ("error", float(ra))
        except Exception:
            pass
        return ("error", ra)

    def try_lock(self, image_id: str, extra: str | None = None) -> bool:
        st, _ = self.try_lock_status(image_id, extra=extra)
        return st == "acquired"

    def mark_done(self, image_id: str) -> bool:
        if not image_id:
            return False
        ok = _hf_try_write_done(self.repo_id, image_id)
        if ok:
            with self.lock:
                if self.done is None:
                    self.done = set()
                self.done.add(image_id)
        return bool(ok)


class RangeLockSync:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.instance_id = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.done_ranges = _hf_try_list_dir_ranges(repo_id, _RANGE_DONE_DIR) if _RANGE_DONE_DIR else set()
        self.done_prefix = self._compute_done_prefix(self.done_ranges)
        self._last_heartbeat_ts = {}
        self._last_progress_ts = {}
        self._last_abandoned_ts = {}
        self.heartbeat_secs = 600.0
        self.progress_secs = 300.0
        self.abandoned_secs = 60.0

    def _compute_done_prefix(self, ranges: set) -> int:
        try:
            expected = 0
            if not ranges:
                return 0
            sorted_ranges = sorted(list(ranges), key=lambda x: (int(x[0]), int(x[1])))
            for a, b in sorted_ranges:
                if int(a) != int(expected):
                    break
                expected = int(b) + 1
            return int(expected)
        except Exception:
            return 0

    def refresh_done_prefix(self) -> int:
        try:
            done_ranges = _hf_try_list_dir_ranges(self.repo_id, _RANGE_DONE_DIR) if _RANGE_DONE_DIR else set()
            with self.lock:
                self.done_ranges = done_ranges
                self.done_prefix = self._compute_done_prefix(done_ranges)
                return int(self.done_prefix)
        except Exception:
            return int(self.done_prefix or 0)

    def try_lock_range(self, range_start: int, range_end: int) -> bool:
        if (not self.repo_id) or range_start < 0 or range_end < range_start:
            return False
        try:
            if _RANGE_DONE_DIR:
                with self.lock:
                    if (int(range_start), int(range_end)) in (self.done_ranges or set()):
                        return False
        except Exception:
            pass

        info = _hf_try_get_range_lock_info(self.repo_id, range_start, range_end)
        if info is not None:
            ts = info.get('ts')
            if ts is not None:
                try:
                    age = time.time() - float(ts)
                    if age < float(_RANGE_LOCK_STALE_SECS):
                        return False
                except Exception:
                    return False
        ok = _hf_try_write_range_lock(self.repo_id, range_start, range_end, self.instance_id, time.time())
        return bool(ok)

    def read_progress(self, range_start: int, range_end: int):
        try:
            repo_path = _hf_range_progress_repo_path(range_start, range_end)
            obj = _hf_try_read_json(self.repo_id, repo_path)
            if isinstance(obj, dict) and obj:
                return obj
            return None
        except Exception:
            return None

    def write_progress(self, range_start: int, range_end: int, progress_obj: dict) -> bool:
        if not isinstance(progress_obj, dict):
            return False
        now = time.time()
        key = (int(range_start), int(range_end))
        try:
            with self.lock:
                last = float(self._last_progress_ts.get(key, 0.0) or 0.0)
                if (now - last) < float(self.progress_secs):
                    return True
                self._last_progress_ts[key] = float(now)
        except Exception:
            pass
        payload = dict(progress_obj)
        payload['_updated_at'] = float(now)
        payload['_owner'] = str(self.instance_id)
        repo_path = _hf_range_progress_repo_path(range_start, range_end)
        return bool(_hf_try_write_json(self.repo_id, repo_path, payload, f"range progress {range_start}-{range_end}"))

    def heartbeat(self, range_start: int, range_end: int, progress_obj: dict | None = None) -> bool:
        now = time.time()
        key = (int(range_start), int(range_end))
        try:
            with self.lock:
                last = float(self._last_heartbeat_ts.get(key, 0.0) or 0.0)
                if (now - last) < float(self.heartbeat_secs):
                    if progress_obj is not None:
                        self.write_progress(range_start, range_end, progress_obj)
                    return True
                self._last_heartbeat_ts[key] = float(now)
        except Exception:
            pass

        info = _hf_try_get_range_lock_info(self.repo_id, range_start, range_end)
        if not info:
            return False
        if str(info.get('owner') or '') != str(self.instance_id):
            return False

        ok = _hf_try_write_range_lock(self.repo_id, range_start, range_end, self.instance_id, now)
        if progress_obj is not None:
            self.write_progress(range_start, range_end, progress_obj)
        return bool(ok)

    def mark_abandoned_range(self, range_start: int, range_end: int, reason: str) -> bool:
        now = time.time()
        key = (int(range_start), int(range_end))
        try:
            with self.lock:
                last = float(self._last_abandoned_ts.get(key, 0.0) or 0.0)
                if (now - last) < float(self.abandoned_secs):
                    return True
                self._last_abandoned_ts[key] = float(now)
        except Exception:
            pass
        payload = {
            'range_start': int(range_start),
            'range_end': int(range_end),
            'reason': str(reason or ''),
            'ts': float(now),
            'owner': str(self.instance_id),
        }
        repo_path = _hf_range_abandoned_repo_path(range_start, range_end)
        return bool(_hf_try_write_json(self.repo_id, repo_path, payload, f"range abandoned {range_start}-{range_end}"))

    def mark_done_range(self, range_start: int, range_end: int) -> bool:
        ok = _hf_try_write_range_done(self.repo_id, range_start, range_end)
        if ok:
            with self.lock:
                if self.done_ranges is None:
                    self.done_ranges = set()
                self.done_ranges.add((int(range_start), int(range_end)))
                self.done_prefix = self._compute_done_prefix(self.done_ranges)
        return bool(ok)
