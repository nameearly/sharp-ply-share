import json
import os
import time

from . import hf_utils

_env_flag = hf_utils.env_flag
_env_str = hf_utils.env_str

def snapshot() -> dict:
    try:
        import psutil

        p = psutil.Process(os.getpid())
        rss = None
        try:
            rss = int(getattr(p.memory_info(), "rss", 0) or 0)
        except Exception:
            rss = None
        vm = None
        try:
            vm = psutil.virtual_memory()
        except Exception:
            vm = None

        out = {}
        if rss is not None:
            out["proc_rss_mb"] = round(float(rss) / (1024.0 * 1024.0), 3)
        if vm is not None:
            try:
                out["vmem_pct"] = float(getattr(vm, "percent", 0.0) or 0.0)
            except Exception:
                pass
        return out
    except Exception:
        return {}


def emit(event: str, *, debug_fn=None, **fields) -> None:
    if not _env_flag("PIPELINE_PROFILE", False):
        return

    rec = {
        "ts": float(time.time()),
        "event": str(event or ""),
    }
    try:
        rec.update({k: v for k, v in (fields or {}).items() if v is not None})
    except Exception:
        pass

    line = json.dumps(rec, ensure_ascii=False)
    msg = "METRIC " + line

    try:
        if debug_fn is not None:
            debug_fn(msg)
        else:
            print(msg, flush=True)
    except Exception:
        pass

    path = _env_str("PIPELINE_PROFILE_JSONL", "").strip()
    if not path:
        return
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    except Exception:
        pass
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
