import hashlib
import io
import json
import os
import re
import signal
import subprocess
import threading
import time
from datetime import datetime
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import requests

from . import hf_sync
from . import hf_upload
from . import hf_utils
from . import unsplash

_env_str = hf_utils.env_str
_env_int = hf_utils.env_int
_env_flag = hf_utils.env_flag

def _is_precondition_failed(err: Exception) -> bool:
    try:
        s = str(err)
        return (" 412" in s) or ("412" in s and "Precondition" in s) or ("Precondition Failed" in s) or (
            "A commit has happened since" in s
        )
    except Exception:
        return False


def _print(msg: str):
    try:
        print(msg, flush=True)
    except Exception:
        pass


def _control_dir() -> str:
    try:
        cd = os.getenv("CONTROL_DIR")
        if cd is not None and str(cd).strip():
            return os.path.abspath(str(cd).strip())
    except Exception:
        pass
    try:
        sd = os.getenv("REQ_SAVE_DIR")
        if sd is not None and str(sd).strip():
            return os.path.abspath(str(sd).strip())
    except Exception:
        pass
    return os.path.abspath(os.getcwd())


def _control_path(name: str) -> str:
    return os.path.join(_control_dir(), str(name))


def _pause_file() -> str:
    return str(os.getenv("PAUSE_FILE", "PAUSE") or "PAUSE").strip() or "PAUSE"


def _stop_file() -> str:
    return str(os.getenv("STOP_FILE", "STOP") or "STOP").strip() or "STOP"


def pause_requested() -> bool:
    try:
        return os.path.exists(_control_path(_pause_file()))
    except Exception:
        return False


def stop_requested() -> bool:
    try:
        return os.path.exists(_control_path(_stop_file()))
    except Exception:
        return False


def wait_if_paused() -> None:
    while (not stop_requested()) and pause_requested():
        time.sleep(0.2)


def gate() -> bool:
    if stop_requested():
        return False
    wait_if_paused()
    if stop_requested():
        return False
    return True


def set_pause(paused: bool) -> bool:
    p = _control_path(_pause_file())
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    except Exception:
        pass
    try:
        if paused:
            with open(p, "a", encoding="utf-8"):
                pass
        else:
            if os.path.exists(p):
                os.remove(p)
        return bool(paused)
    except Exception:
        return bool(paused)


def touch_stop() -> None:
    p = _control_path(_stop_file())
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
    except Exception:
        pass
    try:
        with open(p, "a", encoding="utf-8"):
            pass
    except Exception:
        pass


def _parse_code_blocks(text: str) -> list[str]:
    s = str(text or "")
    out = []
    for m in re.finditer(r"```\s*sharp-request\s*\n([\s\S]*?)\n```", s, flags=re.IGNORECASE):
        out.append(m.group(1) or "")
    for m in re.finditer(r"'''\s*sharp-request\s*\n([\s\S]*?)\n'''", s, flags=re.IGNORECASE):
        out.append(m.group(1) or "")
    if out:
        return out
    for m in re.finditer(r"sharp-request\s*\n([\s\S]{0,800})", s, flags=re.IGNORECASE):
        out.append(m.group(1) or "")
    return out


def _extract_unsplash_id_from_text(text: str) -> str | None:
    try:
        s = str(text or "")
        m = re.search(r"https?://(?:www\.)?unsplash\.com/photos/([a-zA-Z0-9_-]{6,})\b", s, flags=re.IGNORECASE)
        if m:
            return str(m.group(1) or "").strip() or None
    except Exception:
        return None
    return None


def _extract_http_urls(text: str) -> list[str]:
    try:
        s = str(text or "")
        out = []
        for m in re.finditer(r"https?://[^\s\]\)\>\"']+", s, flags=re.IGNORECASE):
            u = str(m.group(0) or "").strip().strip(",.;")
            if u and u not in out:
                out.append(u)
        return out
    except Exception:
        return []


def _looks_like_unsplash_id(s: str) -> bool:
    try:
        v = str(s or "").strip()
        if not v:
            return False
        if len(v) < 6 or len(v) > 32:
            return False
        return bool(re.match(r"^[a-zA-Z0-9_-]+$", v))
    except Exception:
        return False


def _parse_want_tokens(block: str) -> list[str]:
    b = str(block or "")
    m = re.search(r"(?im)^\s*want\s*[:=]\s*(.+?)\s*$", b)
    if not m:
        return []
    raw = str(m.group(1) or "")
    raw = raw.strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    toks = [t.strip().strip("'\"") for t in re.split(r"[^a-zA-Z0-9_]+", raw) if t.strip()]
    out = []
    for t in toks:
        low = str(t).strip().lower()
        if low in ("jpg", "image", "jpeg"):
            low = "jpg"
        if low in ("ply", "spz", "share", "jpg") and low not in out:
            out.append(low)
    return out


def _normalize_url(url: str) -> str:
    try:
        u = str(url or "").strip()
        if not u:
            return ""
        p = urlparse(u)
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if not str(k).lower().startswith("utm_")]
        q = sorted(q, key=lambda kv: kv[0])
        p2 = p._replace(fragment="", query=urlencode(q, doseq=True))
        return urlunparse(p2)
    except Exception:
        return str(url or "").strip()


def _extract_requests(block: str, *, origin: dict) -> list[dict]:
    b = str(block or "")
    want = _parse_want_tokens(b)
    note = ""
    m_note = re.search(r"(?im)^\s*note\s*:\s*(.+?)\s*$", b)
    if m_note:
        note = str(m_note.group(1) or "").strip().strip('"\'')

    items = []

    m_manifest = re.search(r"(?im)^\s*(manifest_url|manifest|list_url)\s*:\s*(.+?)\s*$", b)
    if m_manifest:
        mu = str(m_manifest.group(2) or "").strip().strip('"\'')
        if mu:
            items.append({"src": "manifest", "manifest_url": mu})

    if re.search(r"(?im)^\s*(items|urls|links|ids)\s*:\s*$", b):
        for line in b.splitlines():
            low = line.strip().lower()
            if not low.startswith("-"):
                continue
            payload = line.strip()[1:].strip()
            if not payload:
                continue
            if payload.startswith("{") and payload.endswith("}"):
                try:
                    payload = payload[1:-1]
                except Exception:
                    pass
            mid = re.search(r"(unsplash_id|image_id|photo_id|id)\s*[:=]\s*['\"]?([a-zA-Z0-9_-]{6,})", payload)
            if mid:
                items.append({"src": "unsplash", "unsplash_id": str(mid.group(2))})
                continue
            murl = re.search(r"(url|link|href)\s*[:=]\s*['\"]?([^'\"\s]+)", payload)
            if murl:
                items.append({"src": "url", "url": str(murl.group(2))})
                continue
            if re.match(r"^https?://", payload, flags=re.IGNORECASE):
                mid2 = _extract_unsplash_id_from_text(payload)
                if mid2:
                    items.append({"src": "unsplash", "unsplash_id": mid2})
                else:
                    items.append({"src": "url", "url": payload})
                continue
            if _looks_like_unsplash_id(payload):
                items.append({"src": "unsplash", "unsplash_id": payload})
                continue

    mid = re.search(r"(?im)^\s*(unsplash_id|image_id|photo_id|id)\s*:\s*([a-zA-Z0-9_-]{6,})\s*$", b)
    if mid:
        items.append({"src": "unsplash", "unsplash_id": str(mid.group(2)).strip()})

    murl = re.search(r"(?im)^\s*(url|link|image_url|href)\s*:\s*(.+?)\s*$", b)
    if murl:
        u = str(murl.group(2) or "").strip().strip('"\'')
        if u:
            mid2 = _extract_unsplash_id_from_text(u)
            if mid2:
                items.append({"src": "unsplash", "unsplash_id": mid2})
            else:
                items.append({"src": "url", "url": u})

    if not items:
        try:
            urls = _extract_http_urls(b)
            for u in urls:
                mid2 = _extract_unsplash_id_from_text(u)
                if mid2:
                    items.append({"src": "unsplash", "unsplash_id": mid2})
                else:
                    items.append({"src": "url", "url": u})
        except Exception:
            pass

    out = []
    for it in items:
        if not isinstance(it, dict):
            continue
        r = {
            "origin": dict(origin or {}),
            "want": list(want or []),
            "note": note,
            **it,
        }
        out.append(r)
    return out


def _sha1(s: str) -> str:
    try:
        return hashlib.sha1(str(s).encode("utf-8", errors="ignore")).hexdigest()
    except Exception:
        return hashlib.sha1(b"").hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _download_url_to_file(url: str, out_path: str) -> bool:
    try:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    except Exception:
        pass
    tries = 0
    while True:
        tries += 1
        try:
            r = requests.get(url, timeout=30, stream=True, headers={"User-Agent": "sharp-ply-share"})
            if r.status_code != 200:
                if tries >= 3:
                    return False
                time.sleep(1.0)
                continue
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    f.write(chunk)
            return os.path.isfile(out_path) and os.path.getsize(out_path) > 0
        except Exception:
            if tries >= 3:
                return False
            time.sleep(1.0)


def _run_sharp_predict(jpg_path: str, gaussians_dir: str) -> str | None:
    ml_sharp_dir = _env_str("ML_SHARP_DIR", "").strip()
    conda_env = _env_str("CONDA_ENV_NAME", "sharp").strip() or "sharp"
    device = _env_str("SHARP_DEVICE", "default").strip() or "default"
    verbose = _env_flag("SHARP_VERBOSE", False)

    def _is_cuda_device(dev: str) -> bool:
        try:
            s = str(dev or "").strip().lower()
            if not s or s == "default":
                return True
            if "cuda" in s or s in ("gpu", "nvidia"):
                return True
            return False
        except Exception:
            return True

    def _predict_timeout_s() -> float | None:
        try:
            raw = str(os.getenv("SHARP_PREDICT_TIMEOUT_SECS", "") or "").strip()
            if raw:
                v = float(raw)
                return None if v <= 0 else float(v)
        except Exception:
            pass
        try:
            if _is_cuda_device(device):
                v = float(str(os.getenv("SHARP_PREDICT_TIMEOUT_SECS_CUDA", "1200") or "1200").strip())
            else:
                v = float(str(os.getenv("SHARP_PREDICT_TIMEOUT_SECS_CPU", "3600") or "3600").strip())
            return None if v <= 0 else float(v)
        except Exception:
            return 1200.0

    if not ml_sharp_dir:
        return None

    try:
        os.makedirs(gaussians_dir, exist_ok=True)
    except Exception:
        pass

    extra = ["-v"] if verbose else []
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "sharp",
        "predict",
        "-i",
        jpg_path,
        "-o",
        gaussians_dir,
        "--device",
        device,
        *extra,
    ]

    popen_kw = {}
    try:
        if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    except Exception:
        popen_kw = {}

    try:
        to_s = _predict_timeout_s()
        p = subprocess.Popen(cmd, cwd=ml_sharp_dir, **popen_kw)
        try:
            p.wait(timeout=to_s)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "nt":
                    import signal

                    if hasattr(signal, "CTRL_BREAK_EVENT"):
                        p.send_signal(signal.CTRL_BREAK_EVENT)
                        try:
                            p.wait(timeout=5)
                        except Exception:
                            pass
            except Exception:
                pass
            try:
                p.kill()
            except Exception:
                pass
            try:
                p.wait(timeout=5)
            except Exception:
                pass
            return None
        if int(p.returncode or 0) != 0:
            return None
    except Exception:
        return None

    base = os.path.splitext(os.path.basename(jpg_path))[0]
    cand = os.path.join(gaussians_dir, base + ".ply")
    if os.path.isfile(cand) and os.path.getsize(cand) > 0:
        return cand

    newest = None
    newest_ts = None
    try:
        for fn in os.listdir(gaussians_dir):
            if not str(fn).lower().endswith(".ply"):
                continue
            if ".vertexonly.binary" in str(fn).lower():
                continue
            p = os.path.join(gaussians_dir, fn)
            if not os.path.isfile(p):
                continue
            ts = os.path.getmtime(p)
            if newest_ts is None or ts > newest_ts:
                newest_ts = ts
                newest = p
    except Exception:
        newest = None

    if newest and os.path.isfile(newest) and os.path.getsize(newest) > 0:
        return newest
    return None


def _create_commit_retry(api, *, repo_id: str, repo_type: str, operations, commit_message: str, debug_fn=None):
    last_err = None
    for attempt in range(0, 6):
        try:
            api.create_commit(repo_id=repo_id, repo_type=repo_type, operations=operations, commit_message=commit_message)
            return
        except Exception as e:
            last_err = e
            if not _is_precondition_failed(e):
                raise
            if attempt >= 5:
                raise
            try:
                wait_s = min(8.0, float(0.5 * (2**attempt)))
                wait_s = float(wait_s) * (0.5 + (hash(str(e)) % 1000) / 1000.0)
                if debug_fn:
                    debug_fn(f"HF commit 冲突 412（可重试） | wait={wait_s:.2f}s | attempt={attempt + 1}/6")
                time.sleep(wait_s)
            except Exception:
                time.sleep(0.5)
    if last_err is not None:
        raise last_err


def _hf_write_ops(api, *, repo_id: str, repo_type: str, operations, commit_message: str, dry_run: bool, debug_fn=None):
    if dry_run:
        try:
            if debug_fn:
                debug_fn(f"DRY_RUN: skip commit | msg={commit_message} | ops={len(list(operations or []))}")
        except Exception:
            pass
        return
    _create_commit_retry(api, repo_id=repo_id, repo_type=repo_type, operations=operations, commit_message=commit_message, debug_fn=debug_fn)


def _try_load_manifest(manifest_url: str) -> list[dict]:
    try:
        u = str(manifest_url or "").strip()
        if not u:
            return []
        r = requests.get(u, timeout=30)
        if r.status_code != 200:
            return []
        obj = r.json()
        items = obj.get("items") if isinstance(obj, dict) else None
        if not isinstance(items, list):
            return []
        out = []
        for it in items:
            if isinstance(it, dict):
                out.append(it)
        return out
    except Exception:
        return []


def _dedupe_key(item: dict) -> str:
    try:
        src = str(item.get("src") or "").strip().lower()
        want = item.get("want") if isinstance(item.get("want"), list) else []
        want_s = ",".join(sorted([str(x).strip().lower() for x in want if str(x).strip()]))
        if src == "unsplash":
            pid = str(item.get("unsplash_id") or "").strip()
            return f"unsplash:{pid}|want:{want_s}"
        if src == "url":
            u = _normalize_url(str(item.get("url") or "").strip())
            return f"url:{u}|want:{want_s}"
        if src == "manifest":
            u = _normalize_url(str(item.get("manifest_url") or "").strip())
            return f"manifest:{u}|want:{want_s}"
        return f"unknown:{_sha1(json.dumps(item, ensure_ascii=False))}|want:{want_s}"
    except Exception:
        return f"unknown:{_sha1(str(item))}"


def _hf_download_text(*, repo_id: str, repo_type: str, filename: str) -> str | None:
    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=filename)
        local = str(local or "")
        if not local or (not os.path.isfile(local)):
            return None
        return open(local, "r", encoding="utf-8").read()
    except Exception:
        return None


def _hf_download_json(*, repo_id: str, repo_type: str, filename: str) -> dict | None:
    try:
        s = _hf_download_text(repo_id=repo_id, repo_type=repo_type, filename=filename)
        if not s:
            return None
        return json.loads(s)
    except Exception:
        return None


def _list_req_files_in_dir(api, *, repo_id: str, repo_type: str, dir_path: str, ext: str) -> list[str]:
    base = str(dir_path).strip().strip("/")
    out = []
    if not base:
        return []
    try:
        for ent in api.list_repo_tree(repo_id=repo_id, repo_type=repo_type, path_in_repo=base, recursive=True):
            p = None
            for attr in ("path", "path_in_repo", "rfilename"):
                if hasattr(ent, attr):
                    try:
                        p = getattr(ent, attr)
                        break
                    except Exception:
                        p = None
            if not p:
                continue
            p = str(p)
            if ext and (not p.lower().endswith(str(ext).lower())):
                continue
            out.append(p)
        if out:
            return sorted(out)
    except Exception:
        pass

    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        prefix = base + "/"
        for f in files or []:
            p = str(f or "")
            if not p.startswith(prefix):
                continue
            if ext and (not p.lower().endswith(str(ext).lower())):
                continue
            out.append(p)
        return sorted(list(set(out)))
    except Exception:
        return []


def _list_done_ids(api, *, repo_id: str, repo_type: str, done_dir: str) -> set[str]:
    out = set()
    base = str(done_dir).strip().strip("/")
    if not base:
        return out
    try:
        for ent in api.list_repo_tree(repo_id=repo_id, repo_type=repo_type, path_in_repo=base, recursive=False):
            p = None
            for attr in ("path", "path_in_repo", "rfilename"):
                if hasattr(ent, attr):
                    try:
                        p = getattr(ent, attr)
                        break
                    except Exception:
                        p = None
            if not p:
                continue
            name = os.path.basename(str(p))
            if name:
                out.add(str(name))
    except Exception:
        pass
    return out


def run_once():
    from huggingface_hub import HfApi
    from huggingface_hub import CommitOperationAdd

    wait_if_paused()
    if stop_requested():
        return {"mode": "", "ingested": 0, "processed": 0}

    repo_id = _env_str("HF_REPO_ID", "eatmorefruit/sharp-ply-share").strip()
    repo_type = _env_str("HF_REPO_TYPE", "dataset").strip().lower() or "dataset"

    dry_run = _env_flag("REQ_DRY_RUN", False)

    mode = _env_str("REQ_MODE", "both").strip().lower() or "both"
    ingest_enabled = mode in ("both", "ingest")
    process_enabled = mode in ("both", "process")

    discussion_num = _env_int("REQ_DISCUSSION_NUM", 2)
    req_dir = _env_str("REQ_DIR", "requests").strip().strip("/") or "requests"
    inbox_dir = f"{req_dir}/inbox"
    status_dir = f"{req_dir}/status"
    seen_dir = f"{req_dir}/seen"

    write_index = _env_flag("REQ_WRITE_INDEX", True)

    hf_subdir_unsplash = _env_str("HF_SUBDIR", "unsplash").strip().strip("/") or "unsplash"
    hf_subdir_external = _env_str("REQ_EXTERNAL_SUBDIR", "external").strip().strip("/") or "external"

    save_dir = os.path.abspath(_env_str("REQ_SAVE_DIR", os.path.join(os.getcwd(), "runs", "requests_worker")))
    images_dir = os.path.join(save_dir, "images")
    gaussians_dir = os.path.join(save_dir, "gaussians")

    try:
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(gaussians_dir, exist_ok=True)
    except Exception:
        pass

    if not repo_id:
        raise RuntimeError("HF_REPO_ID 为空")

    api = HfApi()
    hf_utils.ensure_repo(repo_id, repo_type=repo_type, debug_fn=_print)

    hf_sync.configure_hf_sync(
        hf_upload=True,
        repo_type=repo_type,
        hf_locks_dir=f"{req_dir}/locks",
        hf_done_dir=f"{req_dir}/done",
        range_locks_dir=f"{req_dir}/ranges/locks",
        range_done_dir=f"{req_dir}/ranges/done",
        range_progress_dir=f"{req_dir}/ranges/progress",
        range_abandoned_dir=f"{req_dir}/ranges/abandoned",
        hf_lock_stale_secs=float(os.getenv("HF_LOCK_STALE_SECS", "21600") or "21600"),
        range_lock_stale_secs=float(os.getenv("RANGE_LOCK_STALE_SECS", "21600") or "21600"),
        debug_fn=_print,
    )
    coord = hf_sync.LockDoneSync(repo_id)

    index_sync_obj = None
    if bool(write_index) and (not dry_run):
        try:
            from . import index_sync as hf_index_sync

            index_sync_obj = hf_index_sync.IndexSync(
                repo_id,
                repo_type=repo_type,
                repo_path=_env_str("HF_INDEX_REPO_PATH", "data/train.jsonl").strip().lstrip("/"),
                save_dir=save_dir,
                hf_upload=True,
                hf_index_flush_every=_env_int("HF_INDEX_FLUSH_EVERY", 20),
                hf_index_flush_secs=float(_env_int("HF_INDEX_FLUSH_SECS", 30)),
                hf_index_refresh_secs=float(_env_int("HF_INDEX_REFRESH_SECS", 300)),
                debug_fn=_print,
            )
        except Exception:
            index_sync_obj = None

    unsplash_key_pool = None
    if not str(os.getenv("UNSPLASH_ACCESS_KEY", "") or "").strip():
        keys_path = _env_str(
            "UNSPLASH_ACCESS_KEY_JSON",
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "UNSPLASH_ACCESS_KEY.json"),
        )
        unsplash_key_pool = unsplash.load_unsplash_key_pool(
            keys_path,
            default_app_name=_env_str("UNSPLASH_APP_NAME", "sharp-ply-share"),
        )

    if str(os.getenv("UNSPLASH_ACCESS_KEY", "") or "").strip() or unsplash_key_pool:
        unsplash.configure_unsplash(
            access_key=(unsplash_key_pool if unsplash_key_pool else os.getenv("UNSPLASH_ACCESS_KEY", "")),
            app_name=_env_str("UNSPLASH_APP_NAME", "sharp-ply-share"),
            api_base=_env_str("UNSPLASH_API_BASE", "https://api.unsplash.com"),
            per_page=_env_int("PER_PAGE", 10),
            list_per_page=30,
            stop_on_rate_limit=_env_flag("STOP_ON_RATE_LIMIT", True),
            debug_fn=_print,
        )

    ingested = 0
    processed = 0

    if ingest_enabled:
        if not gate():
            return {"mode": mode, "ingested": int(ingested), "processed": int(processed)}
        d = api.get_discussion_details(repo_id=repo_id, repo_type=repo_type, discussion_num=int(discussion_num))
        events = list(d.events or [])
        for evt in events:
            if not gate():
                break
            try:
                if getattr(evt, "type", None) != "comment":
                    continue
                eid = str(getattr(evt, "id", "") or "").strip()
                content = str(getattr(evt, "content", "") or "")
                if not eid or ("sharp-request" not in content.lower() and "unsplash.com/photos/" not in content.lower()):
                    continue
                seen_path = f"{seen_dir}/{eid}"
                try:
                    if api.file_exists(repo_id=repo_id, repo_type=repo_type, filename=seen_path):
                        continue
                except Exception:
                    pass

                origin = {
                    "discussion_num": int(discussion_num),
                    "event_id": eid,
                    "author": str(getattr(evt, "author", "") or ""),
                    "created_at": str(getattr(evt, "created_at", "") or ""),
                }

                blocks = _parse_code_blocks(content)
                req_items = []
                for b in blocks:
                    req_items.extend(_extract_requests(b, origin=origin))

                try:
                    if not req_items:
                        mid = _extract_unsplash_id_from_text(content)
                        if mid:
                            req_items.append({"src": "unsplash", "unsplash_id": mid, "want": ["ply", "spz"], "note": "", "origin": dict(origin)})
                except Exception:
                    pass

                expanded = []
                for it in req_items:
                    if str(it.get("src") or "").strip().lower() == "manifest":
                        mu = str(it.get("manifest_url") or "").strip().strip('"\'')
                        for sub in _try_load_manifest(mu):
                            if not isinstance(sub, dict):
                                continue
                            sub_src = str(sub.get("src") or "").strip().lower()
                            if sub_src in ("unsplash", "url"):
                                expanded.append(
                                    {"src": sub_src, **sub, "want": it.get("want") or [], "note": it.get("note") or "", "origin": it.get("origin")}
                                )
                        continue
                    if str(it.get("src") or "").strip().lower() in ("unsplash", "url"):
                        expanded.append(it)

                ops = []
                for item in expanded:
                    src = str(item.get("src") or "").strip().lower()
                    want = item.get("want") if isinstance(item.get("want"), list) else []
                    want = [str(x).strip().lower() for x in want if str(x).strip()]
                    if not want:
                        want = ["ply", "spz"]

                    key = _dedupe_key({**item, "want": want})
                    req_id = "req_" + _sha1(key)[:12]
                    req_path = f"{inbox_dir}/{req_id}.req"
                    status_path = f"{status_dir}/{req_id}.json"

                    req_obj = {
                        "request_id": req_id,
                        "created_at": datetime.utcnow().isoformat() + "Z",
                        "src": src,
                        "want": want,
                        "note": str(item.get("note") or ""),
                        "origin": dict(origin),
                    }
                    if src == "unsplash":
                        req_obj["unsplash_id"] = str(item.get("unsplash_id") or "").strip()
                    elif src == "url":
                        req_obj["url"] = _normalize_url(str(item.get("url") or "").strip())
                    if not req_obj.get("unsplash_id") and not req_obj.get("url"):
                        continue

                    try:
                        if api.file_exists(repo_id=repo_id, repo_type=repo_type, filename=status_path):
                            continue
                    except Exception:
                        pass

                    blob_req = (json.dumps(req_obj, ensure_ascii=False) + "\n").encode("utf-8")
                    blob_st = (
                        json.dumps({"request_id": req_id, "status": "queued", "created_at": req_obj["created_at"]}, ensure_ascii=False)
                        + "\n"
                    ).encode("utf-8")
                    ops.append(CommitOperationAdd(path_in_repo=req_path, path_or_fileobj=io.BytesIO(blob_req)))
                    ops.append(CommitOperationAdd(path_in_repo=status_path, path_or_fileobj=io.BytesIO(blob_st)))

                ops.append(
                    CommitOperationAdd(path_in_repo=seen_path, path_or_fileobj=io.BytesIO((str(time.time()) + "\n").encode("utf-8")))
                )
                if ops:
                    _hf_write_ops(api, repo_id=repo_id, repo_type=repo_type, operations=ops, commit_message=f"ingest {eid}", dry_run=dry_run, debug_fn=_print)
                    ingested += 1
            except Exception as e:
                _print(f"ingest error (ignored) | err={str(e)}")

    if process_enabled:
        if not gate():
            return {"mode": mode, "ingested": int(ingested), "processed": int(processed)}
        max_per_run = max(1, min(_env_int("REQ_MAX_PER_RUN", 16), 256))
        done_ids = set()
        try:
            done_ids = _list_done_ids(api, repo_id=repo_id, repo_type=repo_type, done_dir=f"{req_dir}/done")
        except Exception:
            done_ids = set()

        inbox_paths = _list_req_files_in_dir(api, repo_id=repo_id, repo_type=repo_type, dir_path=inbox_dir, ext=".req")
        did = 0
        lines = []
        for rp in inbox_paths:
            if did >= int(max_per_run):
                break
            if not gate():
                break
            try:
                req_obj = _hf_download_json(repo_id=repo_id, repo_type=repo_type, filename=rp)
                if not isinstance(req_obj, dict):
                    continue
                req_id = str(req_obj.get("request_id") or "").strip() or os.path.splitext(os.path.basename(rp))[0]

                if req_id and req_id in done_ids:
                    continue

                status_path = f"{status_dir}/{req_id}.json"
                try:
                    st_obj = _hf_download_json(repo_id=repo_id, repo_type=repo_type, filename=status_path) or {}
                except Exception:
                    st_obj = {}
                if isinstance(st_obj, dict) and str(st_obj.get("status") or "").strip().lower() in ("done", "failed"):
                    continue

                st, _ = coord.try_lock_status(req_id, extra=str(req_obj.get("origin") or ""))
                if st != "acquired":
                    continue

                src = str(req_obj.get("src") or "").strip().lower()
                want = req_obj.get("want") if isinstance(req_obj.get("want"), list) else []
                want = [str(x).strip().lower() for x in want if str(x).strip()]
                if not want:
                    want = ["ply", "spz"]

                result = None
                err = None
                try:
                    if src == "unsplash":
                        if not gate():
                            raise RuntimeError("stopped")
                        pid = str(req_obj.get("unsplash_id") or "").strip()
                        if not pid:
                            raise RuntimeError("unsplash_id missing")
                        rel_dir = f"{hf_subdir_unsplash}/{pid}"
                        rel_ply = f"{rel_dir}/{pid}.ply"
                        rel_spz = f"{rel_dir}/{pid}.spz"
                        rel_jpg = f"{rel_dir}/{pid}.jpg"
                        try:
                            ply_exists = bool(api.file_exists(repo_id=repo_id, repo_type=repo_type, filename=rel_ply))
                        except Exception:
                            ply_exists = False
                        try:
                            spz_exists = bool(api.file_exists(repo_id=repo_id, repo_type=repo_type, filename=rel_spz))
                        except Exception:
                            spz_exists = False
                        if ply_exists and (("spz" not in want) or spz_exists):
                            result = {
                                "image_url": hf_utils.build_resolve_url(repo_id, rel_jpg, repo_type=repo_type),
                                "ply_url": hf_utils.build_resolve_url(repo_id, rel_ply, repo_type=repo_type),
                                "spz_url": hf_utils.build_resolve_url(repo_id, rel_spz, repo_type=repo_type) if spz_exists else None,
                                "duplicate": True,
                            }
                        else:
                            if not gate():
                                raise RuntimeError("stopped")
                            details = unsplash.fetch_photo_details(pid)
                            if not details:
                                raise RuntimeError("unsplash fetch_photo_details failed")
                            download_location = ((details.get("links") or {}).get("download_location"))
                            if not download_location:
                                raise RuntimeError("unsplash download_location missing")
                            jpg_local = os.path.join(images_dir, f"{pid}.jpg")
                            if (not os.path.isfile(jpg_local)) or os.path.getsize(jpg_local) <= 0:
                                if not gate():
                                    raise RuntimeError("stopped")
                                if not unsplash.download_image(download_location, jpg_local):
                                    raise RuntimeError("unsplash download failed")
                            if not gate():
                                raise RuntimeError("stopped")
                            ply_local = _run_sharp_predict(jpg_local, gaussians_dir)
                            if not ply_local:
                                raise RuntimeError("sharp predict failed")
                            spz_enabled = "spz" in want
                            gsplat_enabled = ("share" in want) or _env_flag("REQ_GSPLAT_DEFAULT", False)
                            if not gate():
                                raise RuntimeError("stopped")
                            result = hf_upload.upload_sample_pair(
                                repo_id,
                                pid,
                                jpg_local,
                                ply_local,
                                hf_subdir=hf_subdir_unsplash,
                                repo_type=repo_type,
                                gsplat_enabled=gsplat_enabled,
                                gsplat_base=_env_str("GSPLAT_BASE", "https://gsplat.org").strip().rstrip("/"),
                                gsplat_expiration_type=_env_str("GSPLAT_EXPIRATION_TYPE", "1week").strip(),
                                gsplat_filter_visibility=_env_int("GSPLAT_FILTER_VISIBILITY", 20000),
                                splat_transform_bin=_env_str("SPLAT_TRANSFORM_BIN", "splat-transform").strip(),
                                gsplat_use_small_ply=_env_flag("GSPLAT_USE_SMALL_PLY", False),
                                spz_enabled=spz_enabled,
                                spz_tool=_env_str("SPZ_TOOL", ""),
                                gsbox_bin=_env_str("GSBOX_BIN", "gsbox"),
                                gsbox_spz_quality=_env_int("GSBOX_SPZ_QUALITY", 5),
                                gsbox_spz_version=_env_int("GSBOX_SPZ_VERSION", 0),
                                gsconverter_bin=_env_str("GSCONVERTER_BIN", "3dgsconverter"),
                                gsconverter_compression_level=_env_int("GSCONVERTER_COMPRESSION_LEVEL", 6),
                                debug_fn=_print,
                            )

                            if index_sync_obj is not None:
                                try:
                                    tags = []
                                    for t in (details.get("tags") or []):
                                        if isinstance(t, dict):
                                            tt = (t.get("title") or "").strip()
                                            if tt:
                                                tags.append(tt)
                                    topics = []
                                    for t in (details.get("topics") or []):
                                        if isinstance(t, dict):
                                            tt = (t.get("title") or "").strip()
                                            if tt:
                                                topics.append(tt)
                                    alt_desc = details.get("alt_description")
                                    desc = details.get("description")
                                    user = details.get("user") if isinstance(details.get("user"), dict) else {}
                                    meta = {
                                        "tags": tags,
                                        "topics": topics,
                                        "tags_text": ",".join(tags),
                                        "topics_text": ",".join(topics),
                                        "alt_description": alt_desc,
                                        "description": desc,
                                        "unsplash_id": pid,
                                        "unsplash_url": (details.get("links") or {}).get("html"),
                                        "created_at": details.get("created_at"),
                                        "user_username": user.get("username") if isinstance(user, dict) else None,
                                        "user_name": user.get("name") if isinstance(user, dict) else None,
                                    }
                                    row = {
                                        "image": (result or {}).get("image_url"),
                                        "image_id": pid,
                                        "image_url": (result or {}).get("image_url"),
                                        "ply_url": (result or {}).get("ply_url"),
                                    }
                                    try:
                                        if isinstance(result, dict):
                                            for k, v in result.items():
                                                if k not in row:
                                                    row[k] = v
                                    except Exception:
                                        pass
                                    row.update(meta)
                                    index_sync_obj.add_row(row)
                                except Exception:
                                    pass
                    elif src == "url":
                        if not gate():
                            raise RuntimeError("stopped")
                        url = str(req_obj.get("url") or "").strip()
                        if not url:
                            raise RuntimeError("url missing")
                        if not gate():
                            raise RuntimeError("stopped")
                        tmp_path = _download_url_to_file(url, images_dir)
                        if not tmp_path:
                            raise RuntimeError("download_url failed")
                        sha = _sha256_file(tmp_path)
                        eid2 = str(sha)[:16]
                        jpg_local = os.path.join(images_dir, f"{eid2}.jpg")
                        rel_dir = f"{hf_subdir_external}/{eid2}"
                        rel_ply = f"{rel_dir}/{eid2}.ply"
                        rel_spz = f"{rel_dir}/{eid2}.spz"
                        rel_jpg = f"{rel_dir}/{eid2}.jpg"
                        try:
                            ply_exists = bool(api.file_exists(repo_id=repo_id, repo_type=repo_type, filename=rel_ply))
                        except Exception:
                            ply_exists = False
                        try:
                            spz_exists = bool(api.file_exists(repo_id=repo_id, repo_type=repo_type, filename=rel_spz))
                        except Exception:
                            spz_exists = False
                        if ply_exists and (("spz" not in want) or spz_exists):
                            result = {
                                "image_url": hf_utils.build_resolve_url(repo_id, rel_jpg, repo_type=repo_type),
                                "ply_url": hf_utils.build_resolve_url(repo_id, rel_ply, repo_type=repo_type),
                                "spz_url": hf_utils.build_resolve_url(repo_id, rel_spz, repo_type=repo_type) if spz_exists else None,
                                "sha256": sha,
                                "normalized_url": url,
                                "duplicate": True,
                            }
                        else:
                            if not gate():
                                raise RuntimeError("stopped")
                            if (not os.path.isfile(jpg_local)) or os.path.getsize(jpg_local) <= 0:
                                try:
                                    os.replace(tmp_path, jpg_local)
                                except Exception:
                                    try:
                                        import shutil

                                        shutil.copyfile(tmp_path, jpg_local)
                                    except Exception:
                                        pass
                            else:
                                try:
                                    if os.path.normcase(os.path.abspath(str(tmp_path))) != os.path.normcase(
                                        os.path.abspath(str(jpg_local))
                                    ) and os.path.isfile(tmp_path):
                                        os.remove(tmp_path)
                                except Exception:
                                    pass
                            if not gate():
                                raise RuntimeError("stopped")
                            ply_local = _run_sharp_predict(jpg_local, gaussians_dir)
                            if not ply_local:
                                raise RuntimeError("sharp predict failed")
                            spz_enabled = "spz" in want
                            gsplat_enabled = ("share" in want) or _env_flag("REQ_GSPLAT_DEFAULT", False)
                            if not gate():
                                raise RuntimeError("stopped")
                            result = hf_upload.upload_sample_pair(
                                repo_id,
                                eid2,
                                jpg_local,
                                ply_local,
                                hf_subdir=hf_subdir_external,
                                repo_type=repo_type,
                                gsplat_enabled=gsplat_enabled,
                                gsplat_base=_env_str("GSPLAT_BASE", "https://gsplat.org").strip().rstrip("/"),
                                gsplat_expiration_type=_env_str("GSPLAT_EXPIRATION_TYPE", "1week").strip(),
                                gsplat_filter_visibility=_env_int("GSPLAT_FILTER_VISIBILITY", 20000),
                                splat_transform_bin=_env_str("SPLAT_TRANSFORM_BIN", "splat-transform").strip(),
                                gsplat_use_small_ply=_env_flag("GSPLAT_USE_SMALL_PLY", False),
                                spz_enabled=spz_enabled,
                                spz_tool=_env_str("SPZ_TOOL", ""),
                                gsbox_bin=_env_str("GSBOX_BIN", "gsbox"),
                                gsbox_spz_quality=_env_int("GSBOX_SPZ_QUALITY", 5),
                                gsbox_spz_version=_env_int("GSBOX_SPZ_VERSION", 0),
                                gsconverter_bin=_env_str("GSCONVERTER_BIN", "3dgsconverter"),
                                gsconverter_compression_level=_env_int("GSCONVERTER_COMPRESSION_LEVEL", 6),
                                debug_fn=_print,
                            )
                            result = {**(result or {}), "sha256": sha, "normalized_url": url}
                except Exception as e:
                    err = str(e)

                status_obj = {
                    "request_id": req_id,
                    "updated_at": datetime.utcnow().isoformat() + "Z",
                    "status": "done" if result else "failed",
                    "result": result,
                    "error": err,
                }
                ops2 = [
                    CommitOperationAdd(
                        path_in_repo=status_path,
                        path_or_fileobj=io.BytesIO((json.dumps(status_obj, ensure_ascii=False) + "\n").encode("utf-8")),
                    )
                ]
                _hf_write_ops(
                    api,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    operations=ops2,
                    commit_message=f"status {status_obj['status']} {req_id}",
                    dry_run=dry_run,
                    debug_fn=_print,
                )
                try:
                    coord.mark_done(req_id)
                except Exception:
                    pass
                if result:
                    lines.append(f"- done `{req_id}` | jpg={result.get('image_url')} | ply={result.get('ply_url')} | spz={result.get('spz_url')}")
                else:
                    lines.append(f"- failed `{req_id}` | err={err}")
                did += 1
            except Exception as e:
                _print(f"process error (ignored) | err={str(e)}")
                continue

        if lines:
            msg = "Status update\n\n" + "\n".join(lines)
            try:
                if not dry_run:
                    api.comment_discussion(repo_id=repo_id, repo_type=repo_type, discussion_num=int(discussion_num), comment=msg)
            except Exception as e:
                _print(f"comment_discussion failed (ignored) | err={str(e)}")

        processed = int(did)

    _print(f"run_once done | mode={mode} | processed={processed}")
    try:
        _print(
            "REQ_RESULT "
            + json.dumps({"mode": mode, "ingested": int(ingested), "processed": int(processed)}, ensure_ascii=False)
        )
    except Exception:
        pass
    return {"mode": mode, "ingested": int(ingested), "processed": int(processed)}


def main():
    once = _env_flag("REQ_ONCE", True)
    poll = float(_env_int("REQ_POLL_SECS", 60))
    stop_flag = {"stop": False}

    def _request_stop(reason: str):
        try:
            touch_stop()
        except Exception:
            pass
        stop_flag["stop"] = True
        _print(f"REQ: STOP requested | reason={reason} | stop_file={_control_path(_stop_file())}")

    def _sigint_handler(signum, frame):
        try:
            _request_stop("SIGINT")
        except Exception:
            return

    try:
        signal.signal(signal.SIGINT, _sigint_handler)
    except Exception:
        pass

    def _key_loop():
        try:
            import msvcrt

            while (not stop_flag.get("stop")) and (not stop_requested()):
                try:
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        try:
                            c2 = str(ch or "")
                        except Exception:
                            c2 = ""
                        c2 = c2.lower()
                        if c2 == "p":
                            paused = bool(pause_requested())
                            set_pause((not paused))
                            _print(f"REQ: {'pause' if (not paused) else 'resume'} requested | pause_file={_control_path(_pause_file())}")
                        elif c2 == "q":
                            _request_stop("KEY_Q")
                            break
                except Exception:
                    time.sleep(0.1)
                    continue
                time.sleep(0.1)
        except Exception:
            return

    try:
        t = threading.Thread(target=_key_loop, daemon=True)
        t.start()
    except Exception:
        pass

    if once:
        if stop_requested() or stop_flag.get("stop"):
            return
        if pause_requested():
            return
        run_once()
        return

    while True:
        if stop_requested() or stop_flag.get("stop"):
            break
        if pause_requested():
            time.sleep(max(0.2, float(poll)))
            continue
        run_once()
        time.sleep(max(5.0, float(poll)))


if __name__ == "__main__":
    main()
