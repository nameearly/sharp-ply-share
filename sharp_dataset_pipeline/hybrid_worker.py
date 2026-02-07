import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime

from . import requests_worker
from . import hf_utils

_env_str = hf_utils.env_str
_env_int = hf_utils.env_int
_env_float = hf_utils.env_float
_env_flag = hf_utils.env_flag

_control_dir = requests_worker._control_dir
_control_path = requests_worker._control_path
_pause_file = requests_worker._pause_file
_stop_file = requests_worker._stop_file
pause_requested = requests_worker.pause_requested
stop_requested = requests_worker.stop_requested
set_pause = requests_worker.set_pause
touch_stop = requests_worker.touch_stop


def _print(msg: str):
    try:
        print(msg, flush=True)
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
        normal_prefetch = (
            _env_int("HYBRID_NORMAL_MAX_CANDIDATES", 0)
            if os.getenv("HYBRID_NORMAL_MAX_CANDIDATES") is not None
            else _env_int("MAX_CANDIDATES", 8)
        )
    except Exception:
        normal_prefetch = 8
    normal_source = _env_str("HYBRID_NORMAL_SOURCE", "").strip().lower()

    run_id = _env_str("RUN_ID", "")
    if not run_id:
        run_id = "hybrid_" + datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    env = dict(os.environ)
    env["RUN_ID"] = run_id
    env["MAX_IMAGES"] = str(int(normal_max_images))
    env["DOWNLOAD_QUEUE_MAX"] = str(max(1, int(normal_prefetch)))
    env["CONTROL_DIR"] = _control_dir()
    if normal_source in ("search", "list"):
        env["SOURCE"] = normal_source

    try:
        p = subprocess.Popen([sys.executable, script], cwd=root, env=env)
        while True:
            if stop_requested():
                return 0
            if pause_requested():
                time.sleep(0.2)
                continue
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
                            _print(
                                f"HYBRID: {'pause' if (not paused) else 'resume'} requested | pause_file={_control_path(_pause_file())}"
                            )
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
