import hashlib
import io
import json
import os
import re
import subprocess
import time
from datetime import datetime
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

import requests

from . import hf_sync
from . import hf_upload
from . import hf_utils
from . import unsplash


def _env_str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        return str(default)
    return str(v)


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return int(default)
    try:
        return int(str(v).strip())
    except Exception:
        return int(default)


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "y")


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


def _load_unsplash_key_pool(json_path: str):
    try:
        p = str(json_path or "").strip()
        if not p:
            return []
        if not os.path.exists(p):
            return []
        raw = open(p, "r", encoding="utf-8").read()
    except Exception:
        return []

    def _normalize_items(obj):
        if obj is None:
            return []
        if isinstance(obj, dict):
            obj = [obj]
        if not isinstance(obj, list):
            return []
        out = []
        for it in obj:
            if not isinstance(it, dict):
                continue
            k = it.get("UNSPLASH_ACCESS_KEY") or it.get("unsplash_access_key") or it.get("access_key")
            k = str(k or "").strip()
            if not k:
                continue
            an = it.get("UNSPLASH_APP_NAME") or it.get("unsplash_app_name") or it.get("app_name")
            an = str(an or "").strip()
            out.append({"UNSPLASH_ACCESS_KEY": k, "UNSPLASH_APP_NAME": an})
        return out

    try:
        return _normalize_items(json.loads(raw))
    except Exception:
        pass

    try:
        s = str(raw or "").strip()
        if not s:
            return []
        if s.startswith("{") and s.endswith("}"):
            s = "[" + s[1:-1] + "]"
        s = re.sub(r"(?m)\b(UNSPLASH_APP_NAME|UNSPLASH_ACCESS_KEY)\s*:", r'"\\1":', s)
        s = re.sub(r",\s*([}\]])", r"\\1", s)
        obj = json.loads(s)
        return _normalize_items(obj)
    except Exception:
        pass

    try:
        s = str(raw or "")
        blocks = []
        try:
            blocks = re.findall(r"\{[^{}]*\}", s, flags=re.DOTALL)
        except Exception:
            blocks = []

        out = []
        for b in blocks or []:
            try:
                km = re.search(r"\bUNSPLASH_ACCESS_KEY\s*:\s*['\"]([^'\"]+)['\"]", b)
                if not km:
                    km = re.search(r"\bunsplash_access_key\s*:\s*['\"]([^'\"]+)['\"]", b)
                if not km:
                    km = re.search(r"\baccess_key\s*:\s*['\"]([^'\"]+)['\"]", b)
                if not km:
                    continue
                k = str(km.group(1) or "").strip()
                if not k:
                    continue

                am = re.search(r"\bUNSPLASH_APP_NAME\s*:\s*['\"]([^'\"]+)['\"]", b)
                if not am:
                    am = re.search(r"\bunsplash_app_name\s*:\s*['\"]([^'\"]+)['\"]", b)
                if not am:
                    am = re.search(r"\bapp_name\s*:\s*['\"]([^'\"]+)['\"]", b)
                an = str(am.group(1) if am else "").strip()

                out.append({"UNSPLASH_ACCESS_KEY": k, "UNSPLASH_APP_NAME": an})
            except Exception:
                continue

        out = _normalize_items(out)
        if out:
            return out
    except Exception:
        pass

    return []


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
        m = re.search(r"https?://(www\.)?unsplash\.com/photos/([a-zA-Z0-9_-]{6,})", s, flags=re.IGNORECASE)
        if m:
            return str(m.group(2) or "").strip() or None
    except Exception:
        return None
    return None


def _parse_want_tokens(block: str) -> list[str]:
    b = str(block or "")
    m = re.search(r"(?im)^\s*want\s*:\s*(.+?)\s*$", b)
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

    m_manifest = re.search(r"(?im)^\s*manifest_url\s*:\s*(.+?)\s*$", b)
    if m_manifest:
        mu = str(m_manifest.group(1) or "").strip().strip('"\'')
        if mu:
            items.append({"src": "manifest", "manifest_url": mu})

    if re.search(r"(?im)^\s*items\s*:\s*$", b):
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
            mid = re.search(r"unsplash_id\s*[:=]\s*['\"]?([a-zA-Z0-9_-]{6,})", payload)
            if mid:
                items.append({"src": "unsplash", "unsplash_id": str(mid.group(1))})
                continue
            murl = re.search(r"url\s*[:=]\s*['\"]?([^'\"\s]+)", payload)
            if murl:
                items.append({"src": "url", "url": str(murl.group(1))})
                continue
            if re.match(r"^https?://", payload, flags=re.IGNORECASE):
                items.append({"src": "url", "url": payload})
                continue

    mid = re.search(r"(?im)^\s*unsplash_id\s*:\s*([a-zA-Z0-9_-]{6,})\s*$", b)
    if mid:
        items.append({"src": "unsplash", "unsplash_id": str(mid.group(1)).strip()})

    murl = re.search(r"(?im)^\s*url\s*:\s*(.+?)\s*$", b)
    if murl:
        u = str(murl.group(1) or "").strip().strip('"\'')
        if u:
            mid2 = _extract_unsplash_id_from_text(u)
            if mid2:
                items.append({"src": "unsplash", "unsplash_id": mid2})
            else:
                items.append({"src": "url", "url": u})

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
        subprocess.run(cmd, cwd=ml_sharp_dir, check=True, **popen_kw)
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


def _list_inbox_req_paths(api, *, repo_id: str, repo_type: str, inbox_dir: str) -> list[str]:
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type)
        prefix = str(inbox_dir).strip().strip("/") + "/"
        out = []
        for f in files or []:
            p = str(f or "")
            if not p.startswith(prefix):
                continue
            if not p.lower().endswith(".req"):
                continue
            out.append(p)
        return sorted(out)
    except Exception:
        return []


def run_once():
    from huggingface_hub import HfApi
    from huggingface_hub import CommitOperationAdd

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
        unsplash_key_pool = _load_unsplash_key_pool(keys_path)

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

    processed = 0

    if ingest_enabled:
        d = api.get_discussion_details(repo_id=repo_id, repo_type=repo_type, discussion_num=int(discussion_num))
        events = list(d.events or [])
        for evt in events:
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
                    processed += 1
            except Exception as e:
                _print(f"ingest error (ignored) | err={str(e)}")

    if process_enabled:
        max_per_run = max(1, min(_env_int("REQ_MAX_PER_RUN", 16), 256))
        inbox_paths = _list_inbox_req_paths(api, repo_id=repo_id, repo_type=repo_type, inbox_dir=inbox_dir)
        did = 0
        lines = []
        for rp in inbox_paths:
            if did >= int(max_per_run):
                break
            try:
                req_obj = _hf_download_json(repo_id=repo_id, repo_type=repo_type, filename=rp)
                if not isinstance(req_obj, dict):
                    continue
                req_id = str(req_obj.get("request_id") or "").strip() or os.path.splitext(os.path.basename(rp))[0]
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
                            details = unsplash.fetch_photo_details(pid)
                            if not details:
                                raise RuntimeError("unsplash fetch_photo_details failed")
                            download_location = ((details.get("links") or {}).get("download_location"))
                            if not download_location:
                                raise RuntimeError("unsplash download_location missing")
                            jpg_local = os.path.join(images_dir, f"{pid}.jpg")
                            if (not os.path.isfile(jpg_local)) or os.path.getsize(jpg_local) <= 0:
                                if not unsplash.download_image(download_location, jpg_local):
                                    raise RuntimeError("unsplash download failed")
                            ply_local = _run_sharp_predict(jpg_local, gaussians_dir)
                            if not ply_local:
                                raise RuntimeError("sharp predict failed")
                            spz_enabled = "spz" in want
                            gsplat_enabled = ("share" in want) or _env_flag("REQ_GSPLAT_DEFAULT", False)
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
                        u = _normalize_url(str(req_obj.get("url") or "").strip())
                        if not u:
                            raise RuntimeError("url missing")
                        tmp_name = "ext_" + _sha1(u)[:12] + ".jpg"
                        tmp_path = os.path.join(images_dir, tmp_name)
                        if (not os.path.isfile(tmp_path)) or os.path.getsize(tmp_path) <= 0:
                            if not _download_url_to_file(u, tmp_path):
                                raise RuntimeError("download url failed")
                        sha = _sha256_file(tmp_path)
                        eid2 = str(sha)[:16]
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
                                "normalized_url": u,
                                "duplicate": True,
                            }
                        else:
                            jpg_local = os.path.join(images_dir, f"{eid2}.jpg")
                            if not os.path.isfile(jpg_local):
                                try:
                                    os.replace(tmp_path, jpg_local)
                                except Exception:
                                    try:
                                        import shutil

                                        shutil.copyfile(tmp_path, jpg_local)
                                    except Exception:
                                        pass
                            ply_local = _run_sharp_predict(jpg_local, gaussians_dir)
                            if not ply_local:
                                raise RuntimeError("sharp predict failed")
                            spz_enabled = "spz" in want
                            gsplat_enabled = ("share" in want) or _env_flag("REQ_GSPLAT_DEFAULT", False)
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
                            result = {**(result or {}), "sha256": sha, "normalized_url": u}
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

    _print(f"run_once done | mode={mode} | processed={processed}")


def main():
    once = _env_flag("REQ_ONCE", True)
    poll = float(_env_int("REQ_POLL_SECS", 60))
    if once:
        run_once()
        return
    while True:
        run_once()
        time.sleep(max(5.0, float(poll)))


if __name__ == "__main__":
    main()
