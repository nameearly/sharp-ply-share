import io
import os
import re
import json
import socket
import sqlite3
import threading
import time
import uuid


_HF_UPLOAD = False
_HF_REPO_TYPE = "model"
_HF_LOCKS_DIR = "locks"
_HF_DONE_DIR = "done"
_HF_COWORKERS_DIR = "coworkers"
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
    global _HF_UPLOAD, _HF_REPO_TYPE, _HF_LOCKS_DIR, _HF_DONE_DIR, _HF_COWORKERS_DIR
    global _RANGE_LOCKS_DIR, _RANGE_DONE_DIR, _RANGE_PROGRESS_DIR, _RANGE_ABANDONED_DIR
    global _HF_LOCK_STALE_SECS, _RANGE_LOCK_STALE_SECS, _debug
    _HF_UPLOAD = bool(hf_upload)
    _HF_REPO_TYPE = str(repo_type)
    _HF_LOCKS_DIR = str(hf_locks_dir).strip().strip('/')
    _HF_DONE_DIR = str(hf_done_dir).strip().strip('/')
    try:
        v = str(os.getenv("HF_COWORKERS_DIR", "") or "").strip().strip('/')
        if v:
            _HF_COWORKERS_DIR = v
    except Exception:
        pass
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


def _hf_range_done_prefix_repo_path() -> str:
    base = str(_RANGE_PROGRESS_DIR).strip().strip('/')
    name = "done_prefix.json"
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


def _hf_coworker_events_prefix() -> str:
    base = str(_HF_COWORKERS_DIR).strip().strip('/')
    if base:
        return f"{base}/events"
    return "events"


def _hf_coworker_active_repo_path() -> str:
    base = str(_HF_COWORKERS_DIR).strip().strip('/')
    if base:
        return f"{base}/active.json"
    return "active.json"


def _coworker_owner_default() -> str:
    try:
        v = str(os.getenv("HF_COWORKER_OWNER", "") or "").strip()
        if v:
            return v
    except Exception:
        pass
    for k in ("USERNAME", "USER", "LOGNAME"):
        try:
            v = str(os.getenv(k, "") or "").strip()
            if v:
                return v
        except Exception:
            pass
    try:
        h = str(socket.gethostname() or "").strip()
        if h:
            return h
    except Exception:
        pass
    return "unknown"


def _coworker_ttl_s_default() -> float:
    try:
        v = float(os.getenv("HF_COWORKER_TTL_SECS", "1800") or "1800")
        if v > 0:
            return float(v)
    except Exception:
        pass
    return 1800.0


def _coworker_heartbeat_s_default(ttl_s: float) -> float:
    try:
        v = float(os.getenv("HF_COWORKER_HEARTBEAT_SECS", "") or "")
        if v > 0:
            return float(v)
    except Exception:
        pass
    try:
        return max(60.0, min(float(ttl_s) * 0.5, 900.0))
    except Exception:
        return 900.0


def _hf_coworker_event_repo_path(owner: str, session_id: str, ts: float, kind: str) -> str:
    base = _hf_coworker_events_prefix().strip().strip('/')
    owner_s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(owner or "unknown").strip())
    sess_s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(session_id or "").strip())
    if not sess_s:
        sess_s = uuid.uuid4().hex
    ts_i = int(float(ts) * 1000.0)
    kind_s = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(kind or "event").strip())
    fn = f"{ts_i}_{kind_s}.json"
    if base:
        return f"{base}/{owner_s}/{sess_s}/{fn}"
    return f"{owner_s}/{sess_s}/{fn}"


def _hf_try_write_coworker_event_pr(
    repo_id: str,
    *,
    owner: str,
    session_id: str,
    kind: str,
    ts: float | None = None,
    extra: dict | None = None,
) -> bool:
    if (not _HF_UPLOAD) or (not repo_id):
        return False
    try:
        from huggingface_hub import CommitOperationAdd, HfApi

        api = HfApi()
        now = time.time() if ts is None else float(ts)
        payload = {
            "v": 1,
            "kind": str(kind),
            "owner": str(owner),
            "session": str(session_id),
            "ts": float(now),
        }
        try:
            if isinstance(extra, dict):
                payload.update(extra)
        except Exception:
            pass

        path = _hf_coworker_event_repo_path(str(owner), str(session_id), float(now), str(kind))
        buf = io.BytesIO(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        ops = [CommitOperationAdd(path_in_repo=path, path_or_fileobj=buf)]
        _hf_create_commit_retry(
            api,
            repo_id=str(repo_id),
            operations=ops,
            commit_message=f"coworker {str(kind)} {str(owner)}",
            create_pr=True,
        )
        return True
    except Exception as e:
        _d(f"HF coworker event PR failed（可忽略） | kind={str(kind)} owner={str(owner)} | err={type(e).__name__}: {str(e)}")
        return False


def _hf_try_read_coworker_active(repo_id: str):
    try:
        return _hf_try_read_json(str(repo_id), _hf_coworker_active_repo_path())
    except Exception:
        return None


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
        self.done = set()
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

    def start(self) -> None:
        return

    def close(self) -> None:
        return


class LocalLockDoneSync:
    def __init__(self, save_dir: str, *, lock_stale_secs: float = 21600.0):
        self.save_dir = os.path.abspath(str(save_dir or os.getcwd()))
        self.lock_stale_secs = float(lock_stale_secs)
        self.instance_id = uuid.uuid4().hex
        self.lock = threading.Lock()
        self._recent = {}

        self.db_path = os.path.join(self.save_dir, "local_lock_done.sqlite3")
        self._conn = None
        try:
            os.makedirs(self.save_dir, exist_ok=True)
        except Exception:
            pass
        self._ensure_db()

    def _ensure_db(self) -> None:
        with self.lock:
            if self._conn is None:
                self._conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
                try:
                    self._conn.execute("PRAGMA journal_mode=WAL")
                except Exception:
                    pass
                try:
                    self._conn.execute("PRAGMA synchronous=NORMAL")
                except Exception:
                    pass
            c = self._conn
            c.execute(
                "CREATE TABLE IF NOT EXISTS done (id TEXT PRIMARY KEY, ts REAL)"
            )
            c.execute(
                "CREATE TABLE IF NOT EXISTS locks (id TEXT PRIMARY KEY, ts REAL, owner TEXT, extra TEXT)"
            )
            c.commit()

    def _q1(self, sql: str, args: tuple):
        self._ensure_db()
        with self.lock:
            cur = self._conn.execute(sql, args)
            return cur.fetchone()

    def _exec(self, sql: str, args: tuple = ()) -> None:
        self._ensure_db()
        with self.lock:
            self._conn.execute(sql, args)
            self._conn.commit()

    def iter_done_ids(self):
        self._ensure_db()
        with self.lock:
            cur = self._conn.execute("SELECT id FROM done")
        while True:
            with self.lock:
                rows = cur.fetchmany(1024)
            if not rows:
                break
            for (rid,) in rows:
                if rid:
                    yield str(rid)

    def iter_locks(self):
        self._ensure_db()
        with self.lock:
            cur = self._conn.execute("SELECT id, ts, owner, extra FROM locks")
        while True:
            with self.lock:
                rows = cur.fetchmany(1024)
            if not rows:
                break
            for rid, ts, owner, extra in rows:
                if rid:
                    yield {
                        "id": str(rid),
                        "ts": float(ts) if ts is not None else None,
                        "owner": str(owner) if owner is not None else "",
                        "extra": str(extra) if extra is not None else "",
                    }

    def is_done(self, image_id: str) -> bool:
        if not image_id:
            return False
        try:
            row = self._q1("SELECT 1 FROM done WHERE id=?", (str(image_id),))
            return row is not None
        except Exception:
            return False

    def start(self) -> None:
        return

    def close(self) -> None:
        return

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

        now = time.time()

        try:
            row = self._q1("SELECT ts, owner FROM locks WHERE id=?", (str(image_id),))
        except Exception:
            row = None

        if row is not None:
            try:
                tsf = float(row[0]) if row[0] is not None else None
            except Exception:
                tsf = None
            if tsf is not None:
                age = float(now) - float(tsf)
                if age < float(self.lock_stale_secs):
                    ra = float(tsf) + float(self.lock_stale_secs)
                    try:
                        with self.lock:
                            self._recent[str(image_id)] = ("locked_by_other", float(ra))
                    except Exception:
                        pass
                    return ("locked_by_other", float(ra))

            try:
                self._exec("DELETE FROM locks WHERE id=?", (str(image_id),))
            except Exception:
                pass

        try:
            self._ensure_db()
            with self.lock:
                try:
                    self._conn.execute(
                        "INSERT INTO locks(id, ts, owner, extra) VALUES(?,?,?,?)",
                        (str(image_id), float(now), str(self.instance_id), str(extra or "")),
                    )
                    self._conn.commit()
                    ra = float(now) + float(self.lock_stale_secs)
                    try:
                        self._recent[str(image_id)] = ("acquired", float(ra))
                        if len(self._recent) > 200000:
                            self._recent = {}
                    except Exception:
                        pass
                    return ("acquired", float(ra))
                except sqlite3.IntegrityError:
                    pass
        except Exception:
            ra = float(now) + 30.0
            try:
                with self.lock:
                    self._recent[str(image_id)] = ("error", float(ra))
            except Exception:
                pass
            return ("error", float(ra))

        try:
            row2 = self._q1("SELECT ts FROM locks WHERE id=?", (str(image_id),))
            ts2 = float(row2[0]) if row2 is not None and row2[0] is not None else float(now)
            ra = float(ts2) + float(self.lock_stale_secs)
        except Exception:
            ra = float(now) + 30.0

        try:
            with self.lock:
                self._recent[str(image_id)] = ("locked_by_other", float(ra))
                if len(self._recent) > 200000:
                    self._recent = {}
        except Exception:
            pass
        return ("locked_by_other", float(ra))

    def try_lock(self, image_id: str, extra: str | None = None) -> bool:
        st, _ = self.try_lock_status(image_id, extra=extra)
        return st == "acquired"

    def mark_done(self, image_id: str) -> bool:
        if not image_id:
            return False
        try:
            now = time.time()
            self._exec(
                "INSERT OR REPLACE INTO done(id, ts) VALUES(?,?)",
                (str(image_id), float(now)),
            )
            try:
                self._exec("DELETE FROM locks WHERE id=?", (str(image_id),))
            except Exception:
                pass
            return True
        except Exception:
            return False

    def start(self) -> None:
        return

    def close(self) -> None:
        return


def _hf_is_rate_limited(err: Exception) -> tuple[bool, float | None]:
    try:
        s = str(err).lower()
    except Exception:
        s = ""
    if ("429" not in s) and ("too many requests" not in s):
        return (False, None)
    if "repository commits" in s or "128 per hour" in s:
        return (True, 3600.0)
    try:
        m = re.search(r"retry after\s+(\d+)\s+seconds", s)
        if m:
            return (True, float(int(m.group(1))))
    except Exception:
        pass
    return (True, 30.0)


def _hf_create_commit_retry(api, *, repo_id: str, operations, commit_message: str, create_pr: bool = False):
    last = None
    attempt = 0
    while attempt < 6:
        try:
            api.create_commit(
                repo_id=repo_id,
                repo_type=_HF_REPO_TYPE,
                operations=operations,
                commit_message=commit_message,
                create_pr=bool(create_pr),
            )
            return
        except Exception as e:
            last = e
            rl, wait_s = _hf_is_rate_limited(e)
            if rl:
                try:
                    time.sleep(max(1.0, float(wait_s or 30.0)) * (0.8 + 0.4 * (time.time() % 1.0)))
                except Exception:
                    time.sleep(5.0)
                continue
            if _hf_should_retry_with_pr(e) and (not create_pr):
                create_pr = True
                continue
            if ("412" not in str(e)) and ("Precondition" not in str(e)):
                raise
            if attempt >= 5:
                raise
            try:
                time.sleep(min(8.0, float(0.5 * (2**attempt))))
            except Exception:
                time.sleep(0.5)
            attempt += 1
    if last is not None:
        raise last


class AdaptiveLockDoneSync:
    def __init__(self, save_dir: str, *, repo_id: str, lock_stale_secs: float = 21600.0, check_interval_s: float = 300.0):
        self.repo_id = str(repo_id)
        self.lock_stale_secs = float(lock_stale_secs)
        self.check_interval_s = float(check_interval_s)

        self._lock = threading.Lock()
        self._mode = "local"
        self._last_check_ts = 0.0

        self.owner = _coworker_owner_default()
        self.session_id = uuid.uuid4().hex
        self.coworker_ttl_s = float(_coworker_ttl_s_default())
        self.coworker_heartbeat_s = float(_coworker_heartbeat_s_default(float(self.coworker_ttl_s)))
        self._presence_started = False
        self._last_presence_ts = 0.0
        self._presence_closed = False

        self.local = LocalLockDoneSync(save_dir, lock_stale_secs=float(lock_stale_secs))
        self.hf = LockDoneSync(self.repo_id)

    def start(self) -> None:
        try:
            with self._lock:
                if self._presence_started:
                    return
                self._presence_started = True
        except Exception:
            pass
        try:
            _hf_try_write_coworker_event_pr(
                self.repo_id,
                owner=str(self.owner),
                session_id=str(self.session_id),
                kind="start",
                extra={"ttl_s": float(self.coworker_ttl_s)},
            )
        except Exception:
            pass
        try:
            self._last_presence_ts = float(time.time())
        except Exception:
            self._last_presence_ts = 0.0

    def close(self) -> None:
        try:
            with self._lock:
                if self._presence_closed:
                    return
                self._presence_closed = True
        except Exception:
            pass
        try:
            _hf_try_write_coworker_event_pr(
                self.repo_id,
                owner=str(self.owner),
                session_id=str(self.session_id),
                kind="end",
                extra={"ttl_s": float(self.coworker_ttl_s)},
            )
        except Exception:
            pass

    def _maybe_presence_heartbeat(self) -> None:
        try:
            now = float(time.time())
        except Exception:
            return
        try:
            if (not self._presence_started) or self._presence_closed:
                return
        except Exception:
            pass
        try:
            if float(self.coworker_heartbeat_s) <= 0:
                return
        except Exception:
            return
        try:
            if (now - float(self._last_presence_ts)) < float(self.coworker_heartbeat_s):
                return
        except Exception:
            pass
        try:
            _hf_try_write_coworker_event_pr(
                self.repo_id,
                owner=str(self.owner),
                session_id=str(self.session_id),
                kind="heartbeat",
                extra={"ttl_s": float(self.coworker_ttl_s)},
            )
        except Exception:
            pass
        try:
            self._last_presence_ts = float(now)
        except Exception:
            pass

    def _other_active_coworkers_exist(self) -> bool:
        try:
            obj = _hf_try_read_coworker_active(self.repo_id)
        except Exception:
            obj = None
        if not isinstance(obj, dict):
            return False
        try:
            arr = obj.get("active")
            if arr is None:
                arr = obj.get("coworkers")
        except Exception:
            arr = None
        if not isinstance(arr, list):
            return False
        try:
            now = float(time.time())
        except Exception:
            now = None
        for rec in arr:
            if not isinstance(rec, dict):
                continue
            try:
                o = str(rec.get("owner") or rec.get("id") or "").strip()
            except Exception:
                o = ""
            if not o:
                continue
            if o == str(self.owner):
                continue
            exp = None
            try:
                exp = rec.get("expires_ts")
            except Exception:
                exp = None
            if exp is None:
                try:
                    last_ts = rec.get("last_ts")
                    if last_ts is not None:
                        exp = float(last_ts) + float(self.coworker_ttl_s)
                except Exception:
                    exp = None
            if exp is not None and now is not None:
                try:
                    if float(exp) < float(now):
                        continue
                except Exception:
                    pass
            return True
        return False

    def _maybe_promote(self) -> None:
        with self._lock:
            if self._mode != "local":
                return
            now = time.time()
            if (now - float(self._last_check_ts)) < float(self.check_interval_s):
                return
            self._last_check_ts = float(now)

        try:
            try:
                self._maybe_presence_heartbeat()
            except Exception:
                pass
            if not self._other_active_coworkers_exist():
                return

            try:
                _d(f"AdaptiveLockDoneSync promote local -> hf | reason=coworkers_active | owner={str(self.owner)}")
            except Exception:
                pass

            from huggingface_hub import HfApi

            api = HfApi()

            existing_done = set()
            existing_locks = set()
            try:
                base_done = str(_HF_DONE_DIR).strip().strip('/')
                if base_done:
                    for ent in api.list_repo_tree(
                        repo_id=self.repo_id,
                        repo_type=_HF_REPO_TYPE,
                        path_in_repo=base_done,
                        recursive=False,
                    ):
                        p = None
                        for attr in ("path", "path_in_repo", "rfilename"):
                            if hasattr(ent, attr):
                                try:
                                    p = getattr(ent, attr)
                                    break
                                except Exception:
                                    p = None
                        if p:
                            name = os.path.basename(str(p))
                            if name:
                                existing_done.add(str(name))
            except Exception:
                existing_done = set()

            try:
                base_locks = str(_HF_LOCKS_DIR).strip().strip('/')
                if base_locks:
                    for ent in api.list_repo_tree(
                        repo_id=self.repo_id,
                        repo_type=_HF_REPO_TYPE,
                        path_in_repo=base_locks,
                        recursive=False,
                    ):
                        p = None
                        for attr in ("path", "path_in_repo", "rfilename"):
                            if hasattr(ent, attr):
                                try:
                                    p = getattr(ent, attr)
                                    break
                                except Exception:
                                    p = None
                        if p:
                            name = os.path.basename(str(p))
                            if name:
                                existing_locks.add(str(name))
            except Exception:
                existing_locks = set()

            ops = []
            try:
                from huggingface_hub import CommitOperationAdd

                for rid in self.local.iter_done_ids():
                    if rid in existing_done:
                        continue
                    ops.append(CommitOperationAdd(path_in_repo=hf_done_repo_path(str(rid)), path_or_fileobj=io.BytesIO(b"")))
                    if len(ops) >= 64:
                        _hf_create_commit_retry(api, repo_id=self.repo_id, operations=list(ops), commit_message="export local done")
                        for op in ops:
                            try:
                                p = str(getattr(op, 'path_in_repo', '') or '')
                                name = os.path.basename(p)
                                if name:
                                    self.hf.done.add(name)
                            except Exception:
                                pass
                        ops = []

                if ops:
                    _hf_create_commit_retry(api, repo_id=self.repo_id, operations=list(ops), commit_message="export local done")
                    for op in ops:
                        try:
                            p = str(getattr(op, 'path_in_repo', '') or '')
                            name = os.path.basename(p)
                            if name:
                                self.hf.done.add(name)
                        except Exception:
                            pass
            except Exception:
                pass

            ops2 = []
            try:
                from huggingface_hub import CommitOperationAdd

                for rec in self.local.iter_locks():
                    rid = str(rec.get('id') or '')
                    if not rid:
                        continue
                    if rid in existing_locks:
                        continue
                    ts = rec.get('ts')
                    if ts is None:
                        ts = time.time()
                    payload = f"{float(ts)}\n{str(rec.get('owner') or '')}\n{str(rec.get('extra') or '')}\n".encode('utf-8')
                    ops2.append(CommitOperationAdd(path_in_repo=hf_locks_repo_path(rid), path_or_fileobj=io.BytesIO(payload)))
                    if len(ops2) >= 64:
                        _hf_create_commit_retry(api, repo_id=self.repo_id, operations=list(ops2), commit_message="export local locks")
                        ops2 = []
                if ops2:
                    _hf_create_commit_retry(api, repo_id=self.repo_id, operations=list(ops2), commit_message="export local locks")
            except Exception:
                pass

            with self._lock:
                self._mode = "hf"
        except Exception:
            return

    def is_done(self, image_id: str) -> bool:
        try:
            self._maybe_promote()
        except Exception:
            pass
        with self._lock:
            mode = self._mode
        if mode == "hf":
            return self.hf.is_done(image_id)
        return self.local.is_done(image_id)

    def try_lock_status(self, image_id: str, extra: str | None = None):
        try:
            self._maybe_promote()
        except Exception:
            pass
        with self._lock:
            mode = self._mode
        if mode == "hf":
            return self.hf.try_lock_status(image_id, extra=extra)
        return self.local.try_lock_status(image_id, extra=extra)

    def try_lock(self, image_id: str, extra: str | None = None) -> bool:
        st, _ = self.try_lock_status(image_id, extra=extra)
        return st == "acquired"

    def mark_done(self, image_id: str) -> bool:
        try:
            self._maybe_promote()
        except Exception:
            pass
        with self._lock:
            mode = self._mode
        if mode == "hf":
            ok = self.hf.mark_done(image_id)
            if ok:
                try:
                    self.local.mark_done(image_id)
                except Exception:
                    pass
            return bool(ok)
        return self.local.mark_done(image_id)


class RangeLockSync:
    def __init__(self, repo_id: str):
        self.repo_id = repo_id
        self.instance_id = uuid.uuid4().hex
        self.lock = threading.Lock()
        self.done_ranges = set()
        self.done_prefix = 0
        self.range_size = None
        try:
            obj = _hf_try_read_json(self.repo_id, _hf_range_done_prefix_repo_path())
            if isinstance(obj, dict):
                dp = obj.get('done_prefix')
                if dp is not None:
                    self.done_prefix = int(dp)
        except Exception:
            self.done_prefix = 0
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
            obj = _hf_try_read_json(self.repo_id, _hf_range_done_prefix_repo_path())
            if isinstance(obj, dict):
                dp = obj.get('done_prefix')
                if dp is not None:
                    with self.lock:
                        self.done_prefix = int(dp)
                        return int(self.done_prefix)
        except Exception:
            pass

        try:
            if not _RANGE_DONE_DIR:
                return int(self.done_prefix or 0)
            with self.lock:
                dp = int(self.done_prefix or 0)
                rs = self.range_size
            if rs is None:
                return int(dp)
            rs = int(rs)
            if rs <= 0:
                return int(dp)

            advanced = False
            for _ in range(0, 30):
                a = int(dp)
                b = int(a + rs - 1)
                if hf_file_exists_cached(self.repo_id, _hf_range_done_repo_path(a, b), ttl_s=60.0):
                    dp = int(b + 1)
                    advanced = True
                    continue
                break

            if advanced:
                payload = {"done_prefix": int(dp)}
                _hf_try_write_json(self.repo_id, _hf_range_done_prefix_repo_path(), payload, "range done prefix")
                with self.lock:
                    self.done_prefix = int(dp)
            return int(dp)
        except Exception:
            return int(self.done_prefix or 0)

    def try_lock_range(self, range_start: int, range_end: int) -> bool:
        if (not self.repo_id) or range_start < 0 or range_end < range_start:
            return False
        try:
            if _RANGE_DONE_DIR:
                with self.lock:
                    dp = int(self.done_prefix or 0)
                    if int(range_end) < int(dp):
                        return False
        except Exception:
            pass

        try:
            if _RANGE_DONE_DIR and hf_file_exists_cached(self.repo_id, _hf_range_done_repo_path(int(range_start), int(range_end)), ttl_s=60.0):
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
            try:
                with self.lock:
                    dp = int(self.done_prefix or 0)
                    rs = self.range_size
                if rs is not None:
                    rs = int(rs)
                if rs is not None and rs > 0:
                    if int(range_start) <= int(dp) <= int(range_end) + 1:
                        dp = int(max(int(dp), int(range_end) + 1))
                    for _ in range(0, 30):
                        a = int(dp)
                        b = int(a + int(rs) - 1)
                        if hf_file_exists_cached(self.repo_id, _hf_range_done_repo_path(a, b), ttl_s=10.0):
                            dp = int(b + 1)
                            continue
                        break
                    payload = {"done_prefix": int(dp)}
                    _hf_try_write_json(self.repo_id, _hf_range_done_prefix_repo_path(), payload, "range done prefix")
                    with self.lock:
                        self.done_prefix = int(dp)
            except Exception:
                pass
        return bool(ok)
