import os
import json
import time
import math
import random
import threading
import queue
import signal
import traceback
import glob
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Set, Callable, Tuple

from .progress import OrderedProgress
from . import hf_sync
from . import hf_upload
from . import metrics
from . import unsplash
from . import index_sync as hf_index_sync

from .index_sync import IndexSync


@dataclass
class PipelineConfig:
    save_dir: str
    control_dir: str | None
    pause_file: str
    stop_file: str
    idle_sleep_s: float

    source: str
    queries: list
    search_orders: list
    list_orders: list

    list_per_page: int
    list_auto_seek: bool
    list_seek_back_pages: int

    max_candidates: int
    max_images: int

    range_size: int

    stop_on_rate_limit: bool

    input_images_dir: str
    inject_exif: bool

    download_queue_max: int
    upload_queue_max: int
    upload_workers: int

    hf_upload: bool
    hf_repo_id: str | None
    hf_lock_stale_secs: float
    hf_squash_every: int

    ply_delete_after_upload: bool
    ply_keep_last: int
    gaussians_dir: str

    sigint_window_s: float

    hf_upload_batch_size: int
    hf_upload_batch_wait_ms: int

    pipeline_heartbeat_secs: float
    stall_warn_secs: float

    ant_enabled: bool
    ant_candidate_ranges: int
    ant_epsilon: float
    ant_fresh_secs: float


def _control_path(cfg: PipelineConfig, name: str) -> str:
    base = cfg.control_dir if cfg.control_dir else cfg.save_dir
    base_abs = os.path.abspath(str(base))
    n = str(name or "")
    try:
        if os.path.isabs(n):
            n = os.path.basename(n)
        p = os.path.normpath(os.path.join(base_abs, n))
        if os.path.commonpath([p, base_abs]) != base_abs:
            p = os.path.join(base_abs, os.path.basename(n))
        return p
    except Exception:
        return os.path.join(base_abs, os.path.basename(n))


def pause_requested(cfg: PipelineConfig) -> bool:
    try:
        return os.path.exists(_control_path(cfg, cfg.pause_file))
    except Exception:
        return False


def stop_requested(cfg: PipelineConfig) -> bool:
    try:
        return os.path.exists(_control_path(cfg, cfg.stop_file))
    except Exception:
        return False


def _pause_file_path(cfg: PipelineConfig) -> str:
    return _control_path(cfg, cfg.pause_file)


def _stop_file_path(cfg: PipelineConfig) -> str:
    return _control_path(cfg, cfg.stop_file)


def set_pause_file(cfg: PipelineConfig, paused: bool) -> bool:
    try:
        p = _pause_file_path(cfg)
        if paused:
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, 'a', encoding='utf-8'):
                pass
            return True
        if os.path.exists(p):
            os.remove(p)
        return False
    except Exception:
        return paused


def touch_stop_file(cfg: PipelineConfig) -> None:
    try:
        p = _stop_file_path(cfg)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, 'a', encoding='utf-8'):
            pass
    except Exception:
        return


def idle_sleep(cfg: PipelineConfig) -> None:
    time.sleep(max(0.1, float(cfg.idle_sleep_s or 0.5)))


def _sleep_with_gate(cfg: PipelineConfig, stop_event: threading.Event, seconds: float) -> bool:
    try:
        s = float(seconds)
    except Exception:
        s = 0.0
    if s <= 0:
        return True
    end_ts = time.time() + s
    while time.time() < end_ts:
        if not gate(cfg, stop_event):
            return False
        time.sleep(min(0.2, max(0.05, end_ts - time.time())))
    return True


def wait_if_paused(cfg: PipelineConfig, stop_event: threading.Event):
    while (not stop_event.is_set()) and (not stop_requested(cfg)) and pause_requested(cfg):
        idle_sleep(cfg)


def gate(cfg: PipelineConfig, stop_event: threading.Event) -> bool:
    if stop_event.is_set() or stop_requested(cfg):
        return False
    wait_if_paused(cfg, stop_event)
    if stop_event.is_set() or stop_requested(cfg):
        return False
    return True


def _wait_for_api_slot(cfg: PipelineConfig, stop_event: threading.Event) -> None:
    """Wait for a free slot in the Unsplash API rate limit."""
    try:
        if unsplash.is_rate_limited():
            wait_s = unsplash.rate_limit_wait_s(30.0)
            if wait_s > 0:
                _sleep_with_gate(cfg, stop_event, float(wait_s))
    except Exception:
        _sleep_with_gate(cfg, stop_event, 1.0)


def _log_exc(debug_fn: Optional[Callable[[str], None]], msg: str, e: Optional[BaseException] = None) -> None:
    if debug_fn is None:
        return
    try:
        err_msg = f"{msg} | err={type(e).__name__}: {str(e)}" if e else str(msg)
        debug_fn(err_msg)
        tb = traceback.format_exc() if e else ""
        if tb and tb.strip():
            debug_fn(tb.strip())
    except Exception:
        pass


def _debug(debug_fn: Optional[Callable[[str], None]], msg: str) -> None:
    if debug_fn is not None:
        try:
            debug_fn(str(msg))
        except Exception:
            pass


def _build_unsplash_meta(details: dict, *, photo_id: str) -> dict:
    if not isinstance(details, dict):
        return {"unsplash_id": str(photo_id)}
    
    try:
        tags = [
            t.get("title", "").strip()
            for t in details.get("tags", [])
            if isinstance(t, dict) and t.get("title")
        ]
        topics = [
            t.get("title", "").strip()
            for t in details.get("topics", [])
            if isinstance(t, dict) and t.get("title")
        ]
        user = details.get("user") or {}
        links = details.get("links") or {}
        
        return {
            "tags": tags,
            "topics": topics,
            "tags_text": ",".join(tags),
            "topics_text": ",".join(topics),
            "alt_description": details.get("alt_description"),
            "description": details.get("description"),
            "unsplash_id": str(photo_id),
            "unsplash_url": links.get("html"),
            "created_at": details.get("created_at"),
            "user_username": user.get("username"),
            "user_name": user.get("name"),
        }
    except Exception:
        return {"unsplash_id": str(photo_id)}


def _download_if_missing(download_location: str, out_path: str) -> bool:
    try:
        if os.path.isfile(str(out_path)):
            return True
        ok = bool(unsplash.download_image(str(download_location), str(out_path)))
        if not ok:
            return False
        return bool(os.path.isfile(str(out_path)))
    except Exception:
        return False


def _maybe_inject_focal_exif(
    cfg: PipelineConfig,
    *,
    photo_id: str,
    out_path: str,
    details: dict,
    local_has_focal_exif_fn,
    inject_focal_exif_if_missing_fn,
) -> None:
    # Strategy A: missing EXIF focal length is OK.
    # If `FocalLength` is missing, ml-sharp (`sharp predict`) falls back to a default focal length (currently 30mm).
    try:
        if (not cfg.inject_exif) or (local_has_focal_exif_fn is None) or local_has_focal_exif_fn(out_path):
            return
        details_exif = details or unsplash.fetch_photo_details(photo_id)
        focal_raw = ((details_exif or {}).get("exif") or {}).get("focal_length")
        _a, _b, _c, focal_avg = unsplash.parse_focal_length(focal_raw)
        if focal_avg is not None:
            inject_focal_exif_if_missing_fn(out_path, focal_avg)
    except Exception:
        return


def _enqueue_downloaded_image(
    cfg: PipelineConfig,
    *,
    stop_event: threading.Event,
    image_q: queue.Queue,
    photo_id: str,
    out_path: str,
    download_location: str | None,
    meta: dict,
) -> bool:
    try:
        if (not out_path) or (not os.path.isfile(str(out_path))):
            return False
    except Exception:
        return False
    while (not stop_event.is_set()) and image_q.full():
        if not gate(cfg, stop_event):
            return False
        idle_sleep(cfg)
    if gate(cfg, stop_event):
        image_q.put({"image_id": photo_id, "image_path": out_path, "download_location": download_location, "meta": meta})
        return True
    return False


def _cleanup_local_outputs(cfg: PipelineConfig, *, primary_path: str, debug_fn) -> None:
    try:
        ga = os.path.normcase(os.path.abspath(str(cfg.gaussians_dir)))
        ap = os.path.normcase(os.path.abspath(str(primary_path)))

        def _inside(p: str, root: str) -> bool:
            try:
                return os.path.commonpath([p, root]) == root
            except Exception:
                return False

        def _rm(path: str, root: str):
            try:
                pp = os.path.normcase(os.path.abspath(str(path)))
                rr = os.path.normcase(os.path.abspath(str(root)))
                if _inside(pp, rr) and os.path.isfile(pp):
                    os.remove(pp)
            except Exception as e:
                _log_exc(debug_fn, f"本地清理失败 | path={str(path)}", e)

        _rm(ap, ga)

        try:
            base = os.path.splitext(os.path.basename(ap))[0]
        except Exception:
            base = ""
        canon = str(base or "")
        try:
            changed = True
            while canon and changed:
                changed = False
                for suf in (".small.gsplat", ".vertexonly.binary"):
                    if canon.endswith(suf):
                        canon = canon[: -len(suf)]
                        changed = True
        except Exception:
            canon = str(base or "")

        if canon:
            for suf in (
                ".spz",
                ".vertexonly.binary.ply",
                ".small.gsplat.ply",
                ".small.gsplat.vertexonly.binary.ply",
            ):
                _rm(os.path.join(ga, canon + suf), ga)
            try:
                for p in glob.glob(os.path.join(ga, canon + ".small.gsplat.*.ply")):
                    _rm(p, ga)
            except Exception:
                pass

            try:
                img_root = os.path.normcase(os.path.abspath(str(cfg.input_images_dir)))
                save_root = os.path.normcase(os.path.abspath(str(cfg.save_dir)))
                if not _inside(img_root, save_root):
                    img_root = ""
                for ext in (".jpg", ".jpeg"):
                    if img_root:
                        _rm(os.path.join(img_root, canon + ext), img_root)
            except Exception:
                pass
    except Exception:
        return


def upload_worker(
    cfg: PipelineConfig,
    stop_event: threading.Event,
    upload_q: queue.Queue,
    counters: dict,
    lock: threading.Lock,
    checked_ids: set,
    coord: hf_sync.LockDoneSync | None,
    index_sync,
    upload_sample_pair_fn,
    upload_sample_pairs_fn,
    try_super_squash_fn,
    debug_fn,
):
    try:
        base_batch_size = int(getattr(cfg, "hf_upload_batch_size", 1))
    except Exception:
        base_batch_size = 1
    base_batch_size = max(1, min(int(base_batch_size), 64))
    last_batch_size_eff = None

    try:
        batch_wait_ms = int(getattr(cfg, "hf_upload_batch_wait_ms", 0))
    except Exception:
        batch_wait_ms = 0
    batch_wait_ms = max(0, min(int(batch_wait_ms), 5000))

    while not stop_event.is_set():
        if not gate(cfg, stop_event):
            break

        batch_size_eff = int(base_batch_size)
        try:
            if upload_sample_pairs_fn is not None:
                batch_size_eff = int(hf_upload.recommended_hf_upload_batch_size(int(base_batch_size)))
        except Exception:
            batch_size_eff = int(base_batch_size)
        batch_size_eff = max(1, min(int(batch_size_eff), 64))
        try:
            if last_batch_size_eff is None or int(last_batch_size_eff) != int(batch_size_eff):
                last_batch_size_eff = int(batch_size_eff)
                if debug_fn and int(batch_size_eff) != int(base_batch_size):
                    debug_fn(f"HF_UPLOAD_BATCH_SIZE 自适应调整 | base={int(base_batch_size)} -> eff={int(batch_size_eff)}")
        except Exception:
            pass
        try:
            task = upload_q.get(timeout=0.5)
        except Exception:
            continue
        if task is None:
            upload_q.task_done()
            break

        wait_if_paused(cfg, stop_event)
        if stop_event.is_set() or stop_requested(cfg):
            upload_q.task_done()
            break

        # Prepare a batch of tasks
        batch: List[Dict[str, Any]] = [task]
        if batch_size_eff > 1 and (upload_sample_pairs_fn is not None):
            try:
                # Optimized batching: try to collect more tasks from the queue
                # while respecting the batch_wait_ms timeout.
                end_ts = None
                if int(batch_wait_ms) > 0:
                    end_ts = time.time() + (float(batch_wait_ms) / 1000.0)

                while len(batch) < int(batch_size_eff):
                    if stop_event.is_set() or stop_requested(cfg):
                        break
                    
                    nxt = None
                    if end_ts is None:
                        try:
                            nxt = upload_q.get_nowait()
                        except Exception:
                            break # No more tasks available immediately
                    else:
                        remaining = float(end_ts) - time.time()
                        if remaining <= 0:
                            break # Timeout reached
                        try:
                            nxt = upload_q.get(timeout=min(0.05, remaining))
                        except Exception:
                            # This usually means a small wait didn't yield an item
                            if time.time() >= end_ts:
                                break
                            continue
                    
                    if nxt is None:
                        try:
                            upload_q.task_done()
                        except Exception:
                            pass
                        break
                    batch.append(nxt)
            except Exception as e:
                _log_exc(debug_fn, "Batch collection error", e)
                # Fallback to single task if batching fails
                if not batch:
                    batch = [task]

        try:
            with lock:
                counters["upload_inflight"] = int(counters.get("upload_inflight", 0)) + int(len(batch) or 1)
        except Exception:
            pass
        try:
            if not cfg.hf_upload:
                for _ in batch:
                    upload_q.task_done()
                continue
            if not cfg.hf_repo_id:
                debug_fn("HF_UPLOAD 开启但 HF_REPO_ID 为空，跳过上传")
                for _ in batch:
                    upload_q.task_done()
                continue

            valid = []
            invalid = []
            for item in batch:
                try:
                    ip = str(item.get("image_path") or "")
                    pp = str(item.get("ply_path") or "")
                    dl = item.get("download_location")
                    ok = True
                    if (not ip) or (not pp):
                        ok = False
                    if ok and (not os.path.isfile(ip)):
                        redl_ok = False
                        try:
                            if dl:
                                os.makedirs(os.path.dirname(ip), exist_ok=True)
                                redl_ok = bool(unsplash.download_image(str(dl), str(ip)))
                        except Exception:
                            redl_ok = False
                        if not redl_ok:
                            ok = False
                    if ok and (not os.path.isfile(pp)):
                        ok = False
                    if ok:
                        valid.append(item)
                    else:
                        invalid.append(item)
                except Exception:
                    invalid.append(item)

            if invalid:
                for item in invalid:
                    try:
                        iid = str(item.get("image_id") or "")
                        ip = str(item.get("image_path") or "")
                        pp = str(item.get("ply_path") or "")
                        missing_img = (not ip) or (not os.path.isfile(ip))
                        missing_ply = (not pp) or (not os.path.isfile(pp))
                        r = 0
                        try:
                            r = int(item.get("_missing_retries", 0) or 0)
                        except Exception:
                            r = 0
                        if missing_img and (not missing_ply) and r < 2 and item.get("download_location"):
                            item["_missing_retries"] = int(r) + 1
                            upload_q.put(item)
                        else:
                            debug_fn(
                                f"上传跳过 | id={iid} | missing_image={int(bool(missing_img))} missing_ply={int(bool(missing_ply))}"
                            )
                    except Exception:
                        pass

            if not valid:
                continue

            results = None
            if (upload_sample_pairs_fn is not None) and len(valid) > 1:
                results = upload_sample_pairs_fn(repo_id=cfg.hf_repo_id, tasks=valid)
            for item in valid:
                info = None
                if isinstance(results, dict) and results:
                    try:
                        info = results.get(str(item.get("image_id")))
                    except Exception:
                        info = None
                if info is None:
                    info = upload_sample_pair_fn(
                        repo_id=cfg.hf_repo_id,
                        image_id=item["image_id"],
                        image_path=item["image_path"],
                        ply_path=item["ply_path"],
                    )
                debug_fn(f"HF 上传完成 | image={info['image_url']} | ply={info['ply_url']}")

                if coord is not None:
                    try:
                        coord.mark_done(str(item["image_id"]))
                    except Exception:
                        pass

                try:
                    with lock:
                        checked_ids.add(str(item["image_id"]))
                except Exception:
                    pass

                if index_sync is not None:
                    try:
                        meta = item.get('meta') if isinstance(item, dict) else None
                        row = {
                            "image": info.get("image_url") if isinstance(info, dict) else None,
                            "image_id": str(item["image_id"]),
                            "image_url": info.get("image_url"),
                            "ply_url": info.get("ply_url"),
                        }
                        try:
                            if isinstance(info, dict):
                                for k, v in info.items():
                                    if k not in row:
                                        row[k] = v
                        except Exception:
                            pass
                        if isinstance(meta, dict):
                            row.update(meta)
                        index_sync.add_row(row)
                    except Exception as e:
                        _log_exc(debug_fn, f"写入 index 失败 | id={str(item.get('image_id'))}", e)

                to_delete = None
                if cfg.ply_delete_after_upload and int(cfg.ply_keep_last) > 0:
                    try:
                        with lock:
                            keep = counters.get("keep_plys")
                            if keep is None:
                                keep = deque()
                                counters["keep_plys"] = keep
                            keep.append(item["ply_path"])
                            if len(keep) > int(cfg.ply_keep_last):
                                to_delete = keep.popleft()
                    except Exception:
                        to_delete = None

                if to_delete:
                    _cleanup_local_outputs(cfg, primary_path=str(to_delete), debug_fn=debug_fn)

                do_squash = False
                with lock:
                    counters["uploaded"] = int(counters.get("uploaded", 0)) + 1
                    uploaded = int(counters["uploaded"])
                    if cfg.hf_squash_every and int(cfg.hf_squash_every) > 0 and uploaded % int(cfg.hf_squash_every) == 0:
                        do_squash = True
                if do_squash:
                    debug_fn(f"触发 HF super-squash | uploaded={uploaded}")
                    threading.Thread(target=try_super_squash_fn, args=(cfg.hf_repo_id,), daemon=True).start()
        except KeyboardInterrupt:
            try:
                if (not stop_event.is_set()) and (not stop_requested(cfg)):
                    max_retry = 2
                    try:
                        for it in (batch or []):
                            if not isinstance(it, dict):
                                continue
                            r = int(it.get("_kb_retries", 0) or 0)
                            if r < int(max_retry):
                                it["_kb_retries"] = int(r) + 1
                                upload_q.put(it)
                            else:
                                touch_stop_file(cfg)
                                stop_event.set()
                                break
                    except Exception:
                        try:
                            for it in (batch or []):
                                try:
                                    upload_q.put(it)
                                except Exception:
                                    pass
                        except Exception:
                            pass
            except Exception:
                pass
        except Exception as e:
            _log_exc(debug_fn, "HF 上传失败（可重试）", e)
        finally:
            try:
                with lock:
                    counters["upload_inflight"] = max(0, int(counters.get("upload_inflight", 0)) - int(len(batch) or 1))
            except Exception:
                pass
            try:
                for _ in batch:
                    upload_q.task_done()
            except Exception:
                try:
                    upload_q.task_done()
                except Exception:
                    pass


def predict_worker(
    cfg: PipelineConfig,
    stop_event: threading.Event,
    image_q: queue.Queue,
    upload_q: queue.Queue,
    counters: dict,
    lock: threading.Lock,
    run_sharp_predict_once_fn: Callable,
    index_sync: Optional[hf_index_sync.IndexSync],
    debug_fn: Optional[Callable[[str], None]],
):
    while not stop_event.is_set():
        wait_if_paused(cfg, stop_event)
        if stop_requested(cfg):
            break
        try:
            item = image_q.get(timeout=0.5)
        except Exception:
            continue
        if item is None:
            image_q.task_done()
            break

        wait_if_paused(cfg, stop_event)
        if stop_event.is_set() or stop_requested(cfg):
            image_q.task_done()
            break
        try:
            with lock:
                counters["predict_inflight"] = int(counters.get("predict_inflight", 0)) + 1
        except Exception:
            pass
        try:
            image_id = item["image_id"]
            image_path = item["image_path"]
            meta = item.get("meta") if isinstance(item, dict) else None
            dl = item.get("download_location") if isinstance(item, dict) else None
            _wait_for_api_slot(cfg, stop_event)
            
            # Persistent queue logic: save task before processing
            if index_sync is not None:
                try:
                    index_sync.add_to_queue(item)
                except Exception:
                    pass

            started_ts = time.time()
            try:
                with lock:
                    counters["predict_image_id"] = str(image_id)
                    counters["predict_started_ts"] = float(started_ts)
            except Exception:
                pass
            _debug(
                debug_fn,
                f"{'SKIP_PREDICT' if bool(getattr(run_sharp_predict_once_fn, '_skip_predict', False)) else 'ml-sharp'} | id={image_id} | input={image_path}",
            )
            try:
                try:
                    if (not os.path.isfile(str(image_path))) and dl:
                        try:
                            os.makedirs(os.path.dirname(str(image_path)), exist_ok=True)
                            unsplash.download_image(str(dl), str(image_path))
                        except Exception:
                            pass
                except Exception:
                    pass
                if not os.path.isfile(str(image_path)):
                    raise FileNotFoundError(str(image_path))
                plys = run_sharp_predict_once_fn(image_path)
            except KeyboardInterrupt:
                try:
                    if (not stop_event.is_set()) and (not stop_requested(cfg)):
                        max_retry = 2
                        try:
                            if isinstance(item, dict):
                                r = int(item.get("_kb_retries", 0) or 0)
                                if r < int(max_retry):
                                    item["_kb_retries"] = int(r) + 1
                                    image_q.put(item)
                                else:
                                    touch_stop_file(cfg)
                                    stop_event.set()
                        except Exception:
                            try:
                                image_q.put(item)
                            except Exception:
                                pass
                except Exception:
                    pass
                plys = []
            except Exception as e:
                _log_exc(debug_fn, f"predict 失败 | id={str(image_id)} | input={str(image_path)}", e)
                plys = []
            try:
                took_s = max(0.0, float(time.time()) - float(started_ts))
                _debug(debug_fn, f"predict done | id={image_id} | plys={int(len(plys or []))} | s={round(took_s, 2)}")
            except Exception:
                pass
            try:
                took_s = max(0.0, float(time.time()) - float(started_ts))
                metrics.emit(
                    "predict_done",
                    debug_fn=debug_fn,
                    image_id=str(image_id),
                    plys=int(len(plys or [])),
                    s=float(took_s),
                    **metrics.snapshot(),
                )
            except Exception:
                pass

            # Task level upload attribute check
            should_upload = cfg.hf_upload
            if isinstance(item, dict) and "hf_upload" in item:
                should_upload = bool(item["hf_upload"])

            for ply_path in plys or []:
                if not should_upload:
                    continue
                payload = {
                    "image_id": image_id,
                    "image_path": image_path,
                    "ply_path": ply_path,
                    "download_location": dl,
                    "meta": meta,
                }

                # Avoid deadlock when upload queue is full: enqueue with timeout
                # and keep checking stop/pause signals.
                while True:
                    if not gate(cfg, stop_event):
                        break
                    try:
                        upload_q.put(payload, timeout=0.5)
                        break
                    except Exception:
                        continue
        finally:
            try:
                with lock:
                    counters["predict_inflight"] = max(0, int(counters.get("predict_inflight", 0)) - 1)
                    counters["predict_image_id"] = ""
                    counters["predict_started_ts"] = 0.0
            except Exception:
                pass
            image_q.task_done()


def download_loop(
    cfg: PipelineConfig,
    stop_event: threading.Event,
    image_q: queue.Queue,
    checked_ids: set,
    lock: threading.Lock,
    coord: hf_sync.LockDoneSync | None,
    range_coord: hf_sync.RangeLockSync | None,
    remote_done_fn,
    local_has_focal_exif_fn,
    inject_focal_exif_if_missing_fn,
    debug_fn,
):
    scanned = 0
    downloaded_images = 0
    query_idx = 0
    order_idx = 0
    page = 1
    active_range = None
    active_range_start_page = None
    active_range_end_page = None
    range_progress = None
    active_range_acquired_ts = None
    last_range_meta_ts = 0.0

    try:
        ant_enabled = bool(getattr(cfg, "ant_enabled", True))
    except Exception:
        ant_enabled = True
    try:
        ant_candidates = int(getattr(cfg, "ant_candidate_ranges", 6))
    except Exception:
        ant_candidates = 6
    try:
        ant_epsilon = float(getattr(cfg, "ant_epsilon", 0.2))
    except Exception:
        ant_epsilon = 0.2
    try:
        ant_fresh_secs = float(getattr(cfg, "ant_fresh_secs", 90.0))
    except Exception:
        ant_fresh_secs = 90.0
    ant_candidates = max(1, min(int(ant_candidates), 20))
    ant_epsilon = max(0.0, min(float(ant_epsilon), 1.0))
    ant_fresh_secs = max(1.0, float(ant_fresh_secs))

    ant_scanned = 0
    ant_remote_done = 0
    ant_downloaded = 0
    ant_errors = 0

    no_photos_streak = 0
    last_no_photos_log_ts = 0.0

    last_downloaded_log_ts = 0.0
    no_download_streak = 0
    skipped_checked = 0
    skipped_remote_done = 0
    skipped_locked = 0
    skipped_details = 0

    def _clear_active_range(*, abandoned_reason: str | None = None) -> None:
        nonlocal active_range, active_range_end_page, range_progress, active_range_acquired_ts, active_range_start_page
        try:
            if abandoned_reason and (active_range is not None) and (range_coord is not None):
                a, b = active_range
                try:
                    range_coord.mark_abandoned_range(int(a), int(b), str(abandoned_reason))
                except Exception:
                    pass
        except Exception:
            pass
        active_range = None
        active_range_start_page = None
        active_range_end_page = None
        range_progress = None
        active_range_acquired_ts = None

    def _limit_reached(v: int, limit: int) -> bool:
        try:
            lim = int(limit)
        except Exception:
            lim = 0
        if lim < 0:
            return False
        try:
            return int(v) >= lim
        except Exception:
            return True

    while (not stop_event.is_set()):
        order_is_oldest = False
        if not gate(cfg, stop_event):
            break
        if _limit_reached(downloaded_images, int(cfg.max_images)):
            _debug(
                debug_fn,
                f"download_loop exit | reason=max_images | downloaded={int(downloaded_images)} max_images={int(cfg.max_images)} scanned={int(scanned)}",
            )
            break
        if cfg.stop_on_rate_limit and unsplash.is_rate_limited():
            _clear_active_range(abandoned_reason="rate_limited_sleep")

            wait_s = 3600.0
            try:
                wait_s = float(unsplash.rate_limit_wait_s(3600.0))
            except Exception:
                wait_s = 3600.0
            _debug(debug_fn, f"Unsplash rate limited：睡眠 {int(wait_s)}s 后继续")
            end_ts = time.time() + max(1.0, float(wait_s))
            while time.time() < end_ts:
                if not gate(cfg, stop_event):
                    return
                idle_sleep(cfg)
            try:
                unsplash.clear_rate_limited()
            except Exception:
                pass
            continue

        if str(cfg.source).strip().lower() == "list":
            order = cfg.list_orders[order_idx % len(cfg.list_orders)]
            pp = max(1, min(int(cfg.list_per_page), 30))
            order_is_oldest = str(order).strip().lower() == "oldest"

            if (
                order_is_oldest
                and (range_coord is not None)
                and bool(cfg.list_auto_seek)
                and int(page) == 1
            ):
                try:
                    done_prefix = int(range_coord.refresh_done_prefix() or 0)
                    page = int(done_prefix // pp) + 1
                except Exception:
                    pass

            if order_is_oldest and (range_coord is not None):
                if active_range is None:
                    try:
                        range_size = max(int(pp), int(cfg.range_size))
                        range_pages = int((range_size + int(pp) - 1) // int(pp))
                        range_size = int(range_pages * int(pp))
                        try:
                            range_coord.range_size = int(range_size)
                        except Exception:
                            pass
                        offset = max(0, (int(page) - 1) * int(pp))
                        base_idx = int(offset // range_size)

                        step = 1
                        try:
                            step = 1 + (abs(hash(str(getattr(range_coord, 'instance_id', '')))) % 3)
                        except Exception:
                            step = 1

                        acquired = False
                        last_end_page = None
                        candidates = []
                        now = time.time()
                        try:
                            seed = abs(hash(str(getattr(range_coord, 'instance_id', ''))))
                            seed = int(seed + int(now // 60))
                        except Exception:
                            seed = int(now // 60)
                        rng = random.Random(seed)

                        for i in range(0, int(ant_candidates)):
                            cand_idx = int(base_idx + i * step)
                            range_start = int(cand_idx * range_size)
                            range_end = int(range_start + range_size - 1)
                            start_page = int(range_start // int(pp)) + 1
                            end_page = int((range_end // int(pp)) + 1)
                            last_end_page = int(end_page)

                            score = 1.0
                            if ant_enabled and (range_coord is not None):
                                try:
                                    p = range_coord.read_progress(int(range_start), int(range_end))
                                except Exception:
                                    p = None
                                if isinstance(p, dict) and p:
                                    try:
                                        ps = int(p.get('ant_scanned') or 0)
                                    except Exception:
                                        ps = 0
                                    try:
                                        pd = int(p.get('ant_downloaded') or 0)
                                    except Exception:
                                        pd = 0
                                    try:
                                        pr = int(p.get('ant_remote_done') or 0)
                                    except Exception:
                                        pr = 0
                                    try:
                                        upd = float(p.get('_updated_at') or 0.0)
                                    except Exception:
                                        upd = 0.0
                                    age = float(now) - float(upd)
                                    yld = float(pd) / float(ps + pr + 1)
                                    score = float(yld) + 0.05 * math.log1p(float(pd))
                                    if age < float(ant_fresh_secs):
                                        score = float(score) - 1.0
                            candidates.append((float(score), int(i), int(range_start), int(range_end), int(start_page), int(end_page)))

                        if ant_enabled and candidates:
                            try:
                                if rng.random() < float(ant_epsilon):
                                    rng.shuffle(candidates)
                                else:
                                    candidates.sort(key=lambda x: (float(x[0]), -int(x[1])), reverse=True)
                            except Exception:
                                pass

                        for _score, _i, range_start, range_end, start_page, end_page in (candidates or []):
                            if range_coord.try_lock_range(int(range_start), int(range_end)):
                                active_range = (int(range_start), int(range_end))
                                active_range_start_page = int(start_page)
                                active_range_end_page = int(end_page)
                                active_range_acquired_ts = time.time()
                                range_progress = OrderedProgress(
                                    int(range_start),
                                    int(range_end),
                                    frontier_offset=int(range_start),
                                )

                                ant_scanned = 0
                                ant_remote_done = 0
                                ant_downloaded = 0
                                ant_errors = 0

                                try:
                                    p = range_coord.read_progress(int(range_start), int(range_end))
                                    if isinstance(p, dict) and p:
                                        range_progress.apply_dict(p)
                                        try:
                                            ant_scanned = int(p.get('ant_scanned') or 0)
                                        except Exception:
                                            ant_scanned = 0
                                        try:
                                            ant_remote_done = int(p.get('ant_remote_done') or 0)
                                        except Exception:
                                            ant_remote_done = 0
                                        try:
                                            ant_downloaded = int(p.get('ant_downloaded') or 0)
                                        except Exception:
                                            ant_downloaded = 0
                                        try:
                                            ant_errors = int(p.get('ant_errors') or 0)
                                        except Exception:
                                            ant_errors = 0
                                except Exception:
                                    pass

                                try:
                                    fp = int(range_progress.frontier // int(pp)) + 1
                                    if int(page) < int(fp):
                                        page = int(fp)
                                except Exception:
                                    if int(page) < int(start_page):
                                        page = int(start_page)

                                acquired = True
                                break

                        if not acquired:
                            page = int(last_end_page or int(page)) + 1
                            continue
                    except Exception:
                        _clear_active_range()

            if order_is_oldest and (range_coord is not None) and (active_range is not None) and (active_range_end_page is not None) and (range_progress is not None):
                try:
                    fp = int(range_progress.frontier // int(pp)) + 1
                    hp = None
                    ho = range_progress.next_hole_offset(time.time())
                    if ho is not None:
                        hp = int(int(ho) // int(pp)) + 1
                    desired = int(fp)
                    if hp is not None:
                        desired = min(int(desired), int(hp))
                    if active_range_start_page is not None:
                        desired = max(int(desired), int(active_range_start_page))
                    desired = min(int(desired), int(active_range_end_page))
                    if int(desired) < int(page):
                        page = int(desired)
                except Exception:
                    pass

                # Persist progress + keep the range lock alive.
                try:
                    now = time.time()
                    if (now - float(last_range_meta_ts)) >= 1.0:
                        last_range_meta_ts = float(now)
                        a, b = active_range
                        range_coord.heartbeat(
                            int(a),
                            int(b),
                            progress_obj={
                                **(range_progress.to_dict() if hasattr(range_progress, 'to_dict') else {}),
                                "page": int(page),
                                "pp": int(pp),
                                "active_range_start_page": int(active_range_start_page or 0),
                                "active_range_end_page": int(active_range_end_page or 0),
                                "acquired_at": float(active_range_acquired_ts or 0.0),
                                "ant_scanned": int(ant_scanned),
                                "ant_remote_done": int(ant_remote_done),
                                "ant_downloaded": int(ant_downloaded),
                                "ant_errors": int(ant_errors),
                            },
                        )
                except Exception:
                    pass

            photos = unsplash.fetch_list_photos(page=page, order_by=order)
        else:
            query = cfg.queries[query_idx % len(cfg.queries)]
            order = cfg.search_orders[order_idx % len(cfg.search_orders)]
            photos = unsplash.fetch_photos(query=query, page=page, order_by=order)

        if not photos:
            no_photos_streak = int(no_photos_streak) + 1
            try:
                now = time.time()
                if ((no_photos_streak in (1, 3, 10)) or ((now - float(last_no_photos_log_ts)) >= 60.0)):
                    last_no_photos_log_ts = float(now)
                    _debug(
                        debug_fn,
                        f"download_loop idle | reason=no_photos | source={str(cfg.source)} order={str(order)} query={str(locals().get('query', ''))} page={int(page)} streak={int(no_photos_streak)} rate_limited={int(unsplash.is_rate_limited())}",
                    )
            except Exception:
                pass
            if str(cfg.source).strip().lower() == "list":
                order_idx += 1
                page = 1
                _clear_active_range()
            else:
                query_idx += 1
                if query_idx % len(cfg.queries) == 0:
                    order_idx += 1
                page = 1
            time.sleep(min(30.0, max(1.0, 1.0 + 0.5 * float(no_photos_streak))))
            continue

        no_photos_streak = 0

        ordered_idxs = list(range(len(photos or [])))
        if order_is_oldest and (range_coord is not None) and (range_progress is not None):
            try:
                base = int((int(page) - 1) * int(pp))
                f = int(range_progress.frontier)
                if int(base) <= int(f) < int(base + len(ordered_idxs)):
                    start_idx = int(f - base)
                    ordered_idxs = ordered_idxs[start_idx:] + ordered_idxs[:start_idx]
            except Exception:
                pass

        for idx_in_page in ordered_idxs:
            photo = (photos or [])[idx_in_page]
            if not gate(cfg, stop_event):
                break
            if _limit_reached(downloaded_images, int(cfg.max_images)):
                try:
                    if debug_fn and _limit_reached(downloaded_images, int(cfg.max_images)):
                        debug_fn(
                            f"download_loop stop in-page | reason=max_images | downloaded={int(downloaded_images)} max_images={int(cfg.max_images)} scanned={int(scanned)} page={int(page)}"
                        )
                except Exception:
                    pass
                break

            photo_id = str((photo or {}).get("id"))
            if not photo_id:
                continue

            photo_offset = None
            if order_is_oldest and str(cfg.source).strip().lower() == "list":
                try:
                    photo_offset = int((int(page) - 1) * int(pp) + int(idx_in_page))
                except Exception:
                    photo_offset = None
            if (range_progress is not None) and (photo_offset is not None):
                try:
                    range_progress.remember(int(photo_offset), photo_id)
                except Exception:
                    pass

            try:
                with lock:
                    if photo_id in checked_ids:
                        skipped_checked += 1
                        continue
            except Exception:
                if photo_id in checked_ids:
                    skipped_checked += 1
                    continue

            scanned += 1
            try:
                if (order_is_oldest and (range_coord is not None) and (active_range is not None)):
                    ant_scanned = int(ant_scanned) + 1
            except Exception:
                pass

            if cfg.hf_repo_id:
                remote_done = False
                try:
                    if remote_done_fn is not None:
                        remote_done = bool(remote_done_fn(photo_id))
                except Exception:
                    remote_done = False
                if (not remote_done) and (remote_done_fn is None) and hf_sync.hf_file_exists_cached(
                    cfg.hf_repo_id, hf_sync.hf_done_repo_path(photo_id)
                ):
                    remote_done = True
                if remote_done:
                    skipped_remote_done += 1
                    try:
                        if (order_is_oldest and (range_coord is not None) and (active_range is not None)):
                            ant_remote_done = int(ant_remote_done) + 1
                    except Exception:
                        pass
                    if (range_progress is not None) and (photo_offset is not None):
                        try:
                            range_progress.mark_done(int(photo_offset))
                        except Exception:
                            pass
                    try:
                        with lock:
                            checked_ids.add(photo_id)
                    except Exception:
                        pass
                    continue

            if coord is not None:
                if order_is_oldest and (range_coord is not None) and (active_range is not None):
                    pass
                else:
                    extra = None
                    if order_is_oldest:
                        try:
                            extra = str(int((int(page) - 1) * int(pp) + int(idx_in_page)))
                        except Exception:
                            extra = None
                    st, retry_after = coord.try_lock_status(photo_id, extra=extra)
                    if st != "acquired":
                        skipped_locked += 1
                        if (range_progress is not None) and (photo_offset is not None):
                            try:
                                if st == "locked_by_other" and retry_after is not None:
                                    range_progress.mark_claimed_until(int(photo_offset), float(retry_after))
                                elif st == "error":
                                    range_progress.mark_error_retry(int(photo_offset), 30.0)
                            except Exception:
                                pass
                        continue
                    if (range_progress is not None) and (photo_offset is not None):
                        try:
                            if retry_after is not None:
                                range_progress.mark_claimed_until(int(photo_offset), float(retry_after))
                            else:
                                range_progress.mark_claimed(int(photo_offset), float(cfg.hf_lock_stale_secs))
                        except Exception:
                            pass

            details = unsplash.fetch_photo_details(photo_id)
            if not details:
                skipped_details += 1
                try:
                    if (order_is_oldest and (range_coord is not None) and (active_range is not None)):
                        ant_errors = int(ant_errors) + 1
                except Exception:
                    pass
                continue
            src = details
            download_location = ((src.get("links") or {}).get("download_location")) if isinstance(src, dict) else None
            if not download_location:
                try:
                    if (order_is_oldest and (range_coord is not None) and (active_range is not None)):
                        ant_errors = int(ant_errors) + 1
                except Exception:
                    pass
                continue

            meta = _build_unsplash_meta(details, photo_id=str(photo_id))

            if not os.path.exists(cfg.input_images_dir):
                os.makedirs(cfg.input_images_dir, exist_ok=True)
            out_path = os.path.join(cfg.input_images_dir, f"{photo_id}.jpg")

            ok = _download_if_missing(str(download_location), str(out_path))
            if not ok:
                try:
                    if (order_is_oldest and (range_coord is not None) and (active_range is not None)):
                        ant_errors = int(ant_errors) + 1
                except Exception:
                    pass
                continue

            _maybe_inject_focal_exif(
                cfg,
                photo_id=str(photo_id),
                out_path=str(out_path),
                details=details,
                local_has_focal_exif_fn=local_has_focal_exif_fn,
                inject_focal_exif_if_missing_fn=inject_focal_exif_if_missing_fn,
            )

            if not _enqueue_downloaded_image(
                cfg,
                stop_event=stop_event,
                image_q=image_q,
                photo_id=str(photo_id),
                out_path=str(out_path),
                download_location=str(download_location),
                meta=meta,
            ):
                if (not gate(cfg, stop_event)) or stop_event.is_set() or stop_requested(cfg):
                    break
                try:
                    if debug_fn:
                        debug_fn(
                            f"enqueue 跳过 | id={str(photo_id)} | path={str(out_path)} | exists={int(bool(os.path.exists(str(out_path))))} isfile={int(bool(os.path.isfile(str(out_path))))}"
                        )
                except Exception:
                    pass
                continue

            downloaded_images += 1
            try:
                if (order_is_oldest and (range_coord is not None) and (active_range is not None)):
                    ant_downloaded = int(ant_downloaded) + 1
            except Exception:
                pass

        try:
            now = time.time()
            if downloaded_images <= 0:
                no_download_streak = int(no_download_streak) + 1
            if downloaded_images > 0:
                no_download_streak = 0
            if (now - float(last_downloaded_log_ts)) >= 60.0 and int(no_download_streak) >= 3:
                last_downloaded_log_ts = float(now)
                _debug(
                    debug_fn,
                    f"download_loop idle | reason=no_downloads | scanned={int(scanned)} downloaded={int(downloaded_images)} page={int(page)} skipped_checked={int(skipped_checked)} skipped_remote_done={int(skipped_remote_done)} skipped_locked={int(skipped_locked)} skipped_details={int(skipped_details)}",
                )
        except Exception:
            pass

        page += 1

        if (active_range is not None) and (active_range_end_page is not None) and (range_coord is not None):
            try:
                if int(page) > int(active_range_end_page):
                    if _limit_reached(downloaded_images, int(cfg.max_images)) or stop_event.is_set() or stop_requested(cfg):
                        _clear_active_range(abandoned_reason="stopped_or_max_images")
                    else:
                        a, b = active_range
                        try:
                            if range_progress is not None:
                                range_coord.write_progress(int(a), int(b), {
                                    **(range_progress.to_dict() if hasattr(range_progress, 'to_dict') else {}),
                                    "final_page": int(page),
                                })
                        except Exception:
                            pass
                        range_coord.mark_done_range(int(a), int(b))
                        _clear_active_range()
            except Exception:
                _clear_active_range()

    try:
        reason = "loop_exit"
        if stop_event.is_set() or stop_requested(cfg):
            reason = "stopped"
        elif _limit_reached(downloaded_images, int(cfg.max_images)):
            reason = "max_images"
        elif cfg.stop_on_rate_limit and unsplash.is_rate_limited():
            reason = "rate_limited"
        _debug(
            debug_fn,
            f"download_loop done | reason={str(reason)} | downloaded={int(downloaded_images)} scanned={int(scanned)} page={int(page)} max_images={int(cfg.max_images)}",
        )
    except Exception:
        pass
    if (active_range is not None) and (range_coord is not None):
        try:
            a, b = active_range
            reason = "loop_exit"
            if _limit_reached(downloaded_images, int(cfg.max_images)):
                reason = "max_images"
            elif stop_event.is_set() or stop_requested(cfg):
                reason = "stopped"
            elif cfg.stop_on_rate_limit and unsplash.is_rate_limited():
                reason = "rate_limited"

            try:
                if range_progress is not None:
                    range_coord.write_progress(int(a), int(b), {
                        **(range_progress.to_dict() if hasattr(range_progress, 'to_dict') else {}),
                        "final_reason": str(reason),
                        "final_page": int(page),
                    })
            except Exception:
                pass
            range_coord.mark_abandoned_range(int(a), int(b), str(reason))
        except Exception:
            pass


def run(
    cfg: PipelineConfig,
    *,
    checked_ids: Set[str],
    coord: Optional[hf_sync.LockDoneSync],
    range_coord: Optional[hf_sync.RangeLockSync],
    remote_done_fn: Optional[Callable[[str], bool]],
    index_sync: Optional[hf_index_sync.IndexSync],
    upload_sample_pair_fn: Callable,
    upload_sample_pairs_fn: Optional[Callable],
    try_super_squash_fn: Callable,
    run_sharp_predict_once_fn: Callable,
    local_has_focal_exif_fn: Callable,
    inject_focal_exif_if_missing_fn: Callable,
    debug_fn: Optional[Callable[[str], None]],
):
    stop_event = threading.Event()
    image_q = queue.Queue(maxsize=int(cfg.download_queue_max or 8))
    upload_q = queue.Queue(maxsize=int(cfg.upload_queue_max or 256))
    counters = {
        "uploaded": 0,
        "predict_inflight": 0,
        "upload_inflight": 0,
        "predict_image_id": "",
        "predict_started_ts": 0.0,
    }
    lock = threading.Lock()

    # Recovery Logic: Absorbing residual queue
    if index_sync is not None:
        residual_tasks = index_sync.load_queue()
        if residual_tasks:
            _debug(debug_fn, f"发现残留队列任务: {len(residual_tasks)} 个，正在检查并吸收...")
            # Clear queue after loading to avoid duplicates if crash again before processing
            index_sync.clear_queue()
            
            for task in residual_tasks:
                image_id = task.get("image_id")
                if not image_id:
                    continue
                
                # Check if already done in HF or local
                is_done = False
                if coord is not None:
                    is_done = coord.is_done(image_id)
                elif image_id in checked_ids:
                    is_done = True
                
                if not is_done:
                    _debug(debug_fn, f"吸收残留任务 | id={image_id}")
                    try:
                        image_q.put(task, timeout=0.1)
                    except Exception:
                        try:
                            image_q.put_nowait(task)
                        except Exception:
                            _debug(debug_fn, f"残留队列过多，image_q 已满：将任务放回 pending_queue | id={image_id}")
                            try:
                                index_sync.add_to_queue(task)
                            except Exception:
                                pass
                else:
                    _debug(debug_fn, f"残留任务已完成，跳过 | id={image_id}")

    # ... (rest of the threads)

    def _snapshot():
        try:
            with lock:
                pi = int(counters.get("predict_inflight", 0))
                ui = int(counters.get("upload_inflight", 0))
                cur_pid = str(counters.get("predict_image_id", "") or "")
                cur_pts = float(counters.get("predict_started_ts", 0.0) or 0.0)
        except Exception:
            pi = 0
            ui = 0
            cur_pid = ""
            cur_pts = 0.0
        try:
            iq = int(getattr(image_q, "qsize", lambda: -1)())
        except Exception:
            iq = -1
        try:
            uq = int(getattr(upload_q, "qsize", lambda: -1)())
        except Exception:
            uq = -1
        return iq, uq, pi, ui, cur_pid, cur_pts

    def _write_stop_artifacts(reason: str) -> None:
        try:
            iq, uq, pi, ui, cur_pid, cur_pts = _snapshot()
            payload = {
                "ts": float(time.time()),
                "reason": str(reason),
                "stop_file": _stop_file_path(cfg),
                "pause_file": _pause_file_path(cfg),
                "image_q": int(iq),
                "upload_q": int(uq),
                "predict_inflight": int(pi),
                "upload_inflight": int(ui),
                "predict_image_id": str(cur_pid or ""),
                "predict_age_s": int(time.time()) - int(cur_pts) if cur_pts else 0,
            }
            p = _control_path(cfg, "stop_status.json")
            try:
                os.makedirs(os.path.dirname(p), exist_ok=True)
            except Exception:
                pass
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False, indent=2))
        except Exception:
            pass

        try:
            co = None
            try:
                co = {
                    "ts": float(time.time()),
                    "reason": str(reason),
                    "coord_type": type(coord).__name__ if coord is not None else None,
                    "owner": getattr(coord, "owner", None) if coord is not None else None,
                    "session_id": getattr(coord, "session_id", None) if coord is not None else None,
                }
            except Exception:
                co = None
            p2 = _control_path(cfg, "cowork_exit.json")
            try:
                os.makedirs(os.path.dirname(p2), exist_ok=True)
            except Exception:
                pass
            with open(p2, "w", encoding="utf-8") as f:
                f.write(json.dumps(co or {"ts": float(time.time()), "reason": str(reason)}, ensure_ascii=False, indent=2))
        except Exception:
            pass

    def _request_stop(reason: str):
        _write_stop_artifacts(str(reason))
        try:
            if coord is not None:
                coord.close()
        except Exception:
            pass
        try:
            touch_stop_file(cfg)
        except Exception:
            pass
        stop_event.set()
        try:
            if debug_fn:
                iq, uq, pi, ui, cur_pid, cur_pts = _snapshot()
                debug_fn(
                    f"STOP requested | reason={str(reason)} | stop_file={_stop_file_path(cfg)} | pause_file={_pause_file_path(cfg)} | image_q={iq} upload_q={uq} pi={pi} ui={ui} predict_id={cur_pid} predict_s={int(time.time()) - int(cur_pts)}"
                )
        except Exception:
            pass

    try:
        hb_s = float(getattr(cfg, "pipeline_heartbeat_secs", 10.0))
    except Exception:
        hb_s = 10.0
    hb_s = max(1.0, float(hb_s))

    def _heartbeat_loop():
        try:
            stall_warn_s = float(getattr(cfg, "stall_warn_secs", 120.0))
        except Exception:
            stall_warn_s = 120.0
        stall_warn_s = max(5.0, float(stall_warn_s))

        last_uploaded = -1
        last_progress_ts = float(time.time())
        last_work_sig = None
        while (not stop_event.is_set()) and (not stop_requested(cfg)):
            try:
                if pause_requested(cfg):
                    time.sleep(max(0.5, hb_s))
                    continue
                iq, uq, pi, ui, cur_pid, cur_pts = _snapshot()
                try:
                    with lock:
                        up = int(counters.get("uploaded", 0))
                except Exception:
                    up = -1
                now = float(time.time())
                has_work = False
                try:
                    has_work = bool((iq > 0) or (uq > 0) or (pi > 0) or (ui > 0))
                except Exception:
                    has_work = False

                work_sig = None
                try:
                    work_sig = (int(iq), int(uq), int(pi), int(ui), str(cur_pid or ""))
                except Exception:
                    work_sig = None

                progressed = False
                try:
                    if last_uploaded >= 0 and up != last_uploaded:
                        progressed = True
                except Exception:
                    pass
                try:
                    if last_work_sig is not None and work_sig is not None and work_sig != last_work_sig:
                        progressed = True
                except Exception:
                    pass
                if progressed:
                    last_progress_ts = float(now)

                stalled = ""
                try:
                    if has_work and (float(now) - float(last_progress_ts)) >= float(stall_warn_s):
                        stalled = " | stalled=1"
                except Exception:
                    stalled = ""
                pred_extra = ""
                try:
                    if cur_pid:
                        age_s = max(0.0, float(time.time()) - float(cur_pts or 0.0)) if float(cur_pts or 0.0) > 0 else 0.0
                        pred_extra = f" | predict_id={cur_pid} predict_s={int(age_s)}"
                except Exception:
                    pred_extra = ""
                try:
                    if debug_fn:
                        debug_fn(
                            f"HB | uploaded={up} image_q={iq} upload_q={uq} pi={pi} ui={ui} pause={int(pause_requested(cfg))} stop={int(stop_requested(cfg))}{stalled}{pred_extra}"
                        )
                except Exception:
                    pass
                last_uploaded = up
                last_work_sig = work_sig
            except Exception:
                pass
            time.sleep(hb_s)

    try:
        threading.Thread(target=_heartbeat_loop, daemon=True).start()
    except Exception:
        pass

    # Ctrl+C: stop.
    try:
        import signal

        prev_sigint_handler = signal.getsignal(signal.SIGINT)

        def _on_sigint(_signum, _frame):
            try:
                _request_stop("SIGINT")
            except Exception:
                _request_stop("sigint_handler_error")

        signal.signal(signal.SIGINT, _on_sigint)

        try:
            prev_sigterm_handler = signal.getsignal(signal.SIGTERM)

            def _on_sigterm(_signum, _frame):
                _request_stop("SIGTERM")

            signal.signal(signal.SIGTERM, _on_sigterm)
        except Exception:
            prev_sigterm_handler = None

        try:
            if hasattr(signal, "SIGBREAK"):
                prev_sigbreak_handler = signal.getsignal(signal.SIGBREAK)

                def _on_sigbreak(_signum, _frame):
                    _request_stop("SIGBREAK")

                signal.signal(signal.SIGBREAK, _on_sigbreak)
        except Exception:
            prev_sigbreak_handler = None
    except Exception:
        prev_sigint_handler = None
        prev_sigterm_handler = None
        prev_sigbreak_handler = None

    try:
        import msvcrt

        def _key_loop():
            while (not stop_event.is_set()) and (not stop_requested(cfg)):
                try:
                    if msvcrt.kbhit():
                        ch = msvcrt.getwch()
                        try:
                            c2 = str(ch or "")
                        except Exception:
                            c2 = ""
                        c2 = c2.lower()
                        if c2 == "p":
                            paused = bool(pause_requested(cfg))
                            set_pause_file(cfg, (not paused))
                        elif c2 == "q":
                            _request_stop("KEY_Q")
                            break
                    time.sleep(0.1)
                except Exception:
                    time.sleep(0.2)

        threading.Thread(target=_key_loop, daemon=True).start()
    except Exception:
        pass

    upload_threads = []
    if cfg.hf_upload:
        for _ in range(max(1, int(cfg.upload_workers))):
            t = threading.Thread(
                target=upload_worker,
                args=(
                    cfg,
                    stop_event,
                    upload_q,
                    counters,
                    lock,
                    checked_ids,
                    coord,
                    index_sync,
                    upload_sample_pair_fn,
                    upload_sample_pairs_fn,
                    try_super_squash_fn,
                    debug_fn,
                ),
                daemon=True,
            )
            t.start()
            upload_threads.append(t)

    predict_t = threading.Thread(
        target=predict_worker,
        args=(cfg, stop_event, image_q, upload_q, counters, lock, run_sharp_predict_once_fn, index_sync, debug_fn),
        daemon=True,
    )
    predict_t.start()

    try:
        while (not stop_event.is_set()) and (not stop_requested(cfg)):
            try:
                download_loop(
                    cfg,
                    stop_event,
                    image_q,
                    checked_ids,
                    lock,
                    coord,
                    range_coord,
                    remote_done_fn,
                    local_has_focal_exif_fn,
                    inject_focal_exif_if_missing_fn,
                    debug_fn,
                )
                while (not stop_event.is_set()) and (not stop_requested(cfg)):
                    wait_if_paused(cfg, stop_event)
                    try:
                        with lock:
                            pi = int(counters.get("predict_inflight", 0))
                            ui = int(counters.get("upload_inflight", 0))
                    except Exception:
                        pi = 0
                        ui = 0

                    try:
                        iq = int(getattr(image_q, "qsize", lambda: -1)())
                    except Exception:
                        iq = -1
                    try:
                        uq = int(getattr(upload_q, "qsize", lambda: -1)())
                    except Exception:
                        uq = -1

                    uploads_idle = (not cfg.hf_upload) or ((uq <= 0) and ui <= 0)
                    if (iq <= 0) and pi <= 0 and uploads_idle:
                        return
                    idle_sleep(cfg)
                return
            except KeyboardInterrupt:
                try:
                    _on_sigint(signal.SIGINT, None)
                except Exception as e:
                    _log_exc(debug_fn, "Ctrl+C 处理失败：停止并退出", e)
                    _request_stop("keyboardinterrupt_error")
                    break
                idle_sleep(cfg)
            except Exception as e:
                _log_exc(debug_fn, "run loop 未预料异常：停止并退出", e)
                try:
                    touch_stop_file(cfg)
                except Exception:
                    pass
                stop_event.set()
                break
    finally:
        stop_event.set()
        try:
            if prev_sigint_handler is not None:
                signal.signal(signal.SIGINT, prev_sigint_handler)
        except Exception:
            pass
        try:
            if prev_sigterm_handler is not None:
                signal.signal(signal.SIGTERM, prev_sigterm_handler)
        except Exception:
            pass
        try:
            if prev_sigbreak_handler is not None and hasattr(signal, "SIGBREAK"):
                signal.signal(signal.SIGBREAK, prev_sigbreak_handler)
        except Exception:
            pass
        try:
            image_q.put(None)
        except Exception:
            pass
        try:
            upload_q.put(None)
        except Exception:
            pass

        predict_t.join(timeout=5)
        for t in upload_threads:
            t.join(timeout=5)

        try:
            if index_sync is not None:
                index_sync.maybe_flush(True)
        except Exception:
            pass
