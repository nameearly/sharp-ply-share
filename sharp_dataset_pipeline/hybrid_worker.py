import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

from . import requests_worker


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


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return float(default)
    try:
        return float(str(v).strip())
    except Exception:
        return float(default)


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "y")


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


def _with_env(overrides: dict, fn):
    old = {}
    try:
        for k, v in (overrides or {}).items():
            old[k] = os.environ.get(k)
            os.environ[k] = str(v)
        return fn()
    finally:
        for k in (overrides or {}).keys():
            if old.get(k) is None:
                try:
                    os.environ.pop(k, None)
                except Exception:
                    pass
            else:
                os.environ[k] = str(old.get(k))


def _run_requests_once(*, max_per_run: int | None = None) -> dict:
    ingest = _env_flag("HYBRID_REQ_INGEST", False)
    mode = "both" if ingest else "process"
    overrides = {
        "REQ_MODE": mode,
        "REQ_ONCE": "1",
        "REQ_DRY_RUN": "0",
        "REQ_WRITE_INDEX": os.getenv("REQ_WRITE_INDEX", "0"),
    }
    if max_per_run is not None:
        try:
            overrides["REQ_MAX_PER_RUN"] = str(max(0, int(max_per_run)))
        except Exception:
            pass
    return _with_env(overrides, requests_worker.run_once) or {"mode": mode, "ingested": 0, "processed": 0}


def _run_normal_once(*, max_images_override: int | None = None) -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script = os.path.join(root, "sharp_dataset_pipeline_main.py")
    if not os.path.isfile(script):
        _print(f"HYBRID: missing sharp_dataset_pipeline_main.py | path={script}")
        return 1

    batch_images = _env_int("HYBRID_NORMAL_BATCH_IMAGES", 3)
    batch_images = max(1, int(batch_images))

    if max_images_override is not None:
        try:
            normal_max_images = int(max_images_override)
        except Exception:
            normal_max_images = 3
    else:
        try:
            normal_max_images = (
                _env_int("HYBRID_NORMAL_MAX_IMAGES", 0)
                if os.getenv("HYBRID_NORMAL_MAX_IMAGES") is not None
                else _env_int("MAX_IMAGES", 3)
            )
        except Exception:
            normal_max_images = 3

        # In hybrid mode, keep the normal pipeline run short by default,
        # even if MAX_IMAGES is -1 or very large. If the user wants a
        # different behavior, they can set HYBRID_NORMAL_MAX_IMAGES explicitly.
        try:
            normal_max_images = int(batch_images)
        except Exception:
            normal_max_images = 3

    try:
        normal_max_candidates = (
            _env_int("HYBRID_NORMAL_MAX_CANDIDATES", 0)
            if os.getenv("HYBRID_NORMAL_MAX_CANDIDATES") is not None
            else _env_int("MAX_CANDIDATES", 50)
        )
    except Exception:
        normal_max_candidates = 50
    normal_source = _env_str("HYBRID_NORMAL_SOURCE", "").strip().lower()

    run_id = _env_str("RUN_ID", "")
    if not run_id:
        run_id = "hybrid_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    env = dict(os.environ)
    env["RUN_ID"] = run_id
    env["MAX_IMAGES"] = str(int(normal_max_images))
    env["MAX_CANDIDATES"] = str(int(normal_max_candidates))
    env["CONTROL_DIR"] = _control_dir()
    if normal_source in ("search", "list"):
        env["SOURCE"] = normal_source

    try:
        p = subprocess.Popen([sys.executable, script], cwd=root, env=env)
        while True:
            try:
                return int(p.wait(timeout=1.0))
            except subprocess.TimeoutExpired:
                continue
            except KeyboardInterrupt:
                continue
    except Exception as e:
        _print(f"HYBRID: normal pipeline failed | err={str(e)}")
        return 2


def run_loop():
    once = _env_flag("HYBRID_ONCE", False)
    sleep_s = max(1.0, float(_env_float("HYBRID_SLEEP_SECS", 10.0)))

    normal_enabled = _env_flag("HYBRID_NORMAL_ENABLED", True)
    normal_when_idle = _env_flag("HYBRID_NORMAL_WHEN_IDLE", True)

    # Global budget across requests+normal: if MAX_IMAGES env is unset => unlimited
    # If set and >=0 => stop after total_done reaches it.
    max_images_raw = os.getenv("MAX_IMAGES")
    global_limit = None
    if max_images_raw is not None:
        try:
            vv = int(str(max_images_raw).strip())
            if vv >= 0:
                global_limit = int(vv)
        except Exception:
            global_limit = None
    total_done = 0

    sigint_window_s = _env_float("SIGINT_WINDOW_S", 2.0)
    sig_state = {"last": 0.0}
    stop_flag = {"stop": False}

    def _request_stop(reason: str):
        try:
            touch_stop()
        except Exception:
            pass
        stop_flag["stop"] = True
        _print(f"HYBRID: STOP requested | reason={reason} | stop_file={_control_path(_stop_file())}")

    def _sigint_handler(signum, frame):
        try:
            now = time.time()
            if pause_requested():
                set_pause(False)
                _print(f"HYBRID: resume requested | pause_file={_control_path(_pause_file())}")
                return
            prev = float(sig_state.get("last") or 0.0)
            sig_state["last"] = float(now)
            if prev > 0.0 and (now - prev) <= float(sigint_window_s):
                _request_stop("SIGINT_DOUBLE")
                return
            set_pause(True)
            _print(f"HYBRID: pause requested | pause_file={_control_path(_pause_file())}")
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
                        if ch in ("\x04", "\x1a"):
                            _request_stop("CTRL_D")
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

    while True:
        if stop_flag.get("stop") or stop_requested():
            break

        if pause_requested():
            time.sleep(max(0.2, float(sleep_s)))
            continue

        if global_limit is not None and int(total_done) >= int(global_limit):
            _print(f"HYBRID: global MAX_IMAGES reached | total_done={int(total_done)} limit={int(global_limit)}")
            break

        remaining = None
        if global_limit is not None:
            remaining = max(0, int(global_limit) - int(total_done))

        try:
            rr = _run_requests_once(max_per_run=remaining) or {}
        except Exception as e:
            rr = {"mode": "process", "ingested": 0, "processed": 0, "err": str(e)}

        processed = 0
        try:
            processed = int(rr.get("processed") or 0)
        except Exception:
            processed = 0

        if processed > 0:
            total_done += int(processed)

        if global_limit is not None and int(total_done) >= int(global_limit):
            _print(f"HYBRID: global MAX_IMAGES reached | total_done={int(total_done)} limit={int(global_limit)}")
            break

        if normal_enabled and normal_when_idle and processed <= 0:
            normal_budget = None
            if remaining is not None:
                if int(remaining) <= 0:
                    break
                try:
                    bi = max(1, int(_env_int("HYBRID_NORMAL_BATCH_IMAGES", 3)))
                except Exception:
                    bi = 3
                normal_budget = int(min(int(remaining), int(bi)))
            rc = _run_normal_once(max_images_override=normal_budget)
            if remaining is not None and normal_budget is not None:
                # Conservative accounting: normal pipeline cannot exceed MAX_IMAGES.
                if int(normal_budget) > 0:
                    total_done += int(normal_budget)

        if once:
            break
        time.sleep(float(sleep_s))


def main():
    run_loop()


if __name__ == "__main__":
    main()
