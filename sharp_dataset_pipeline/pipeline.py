import os
import time
import threading
import queue
import signal
import traceback
from collections import deque
from dataclasses import dataclass

from .progress import OrderedProgress
from . import hf_sync
from . import unsplash


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

    max_scan: int
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


def _log_exc(debug_fn, msg: str, e: BaseException | None = None) -> None:
    try:
        if debug_fn is None:
            return
        if e is not None:
            debug_fn(f"{msg} | err={type(e).__name__}: {str(e)}")
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        else:
            debug_fn(str(msg))
            tb = traceback.format_exc()
        if tb:
            debug_fn(tb.rstrip())
    except Exception:
        pass


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
    try_super_squash_fn,
    debug_fn,
):
    while not stop_event.is_set():
        if not gate(cfg, stop_event):
            break
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
        try:
            with lock:
                counters["upload_inflight"] = int(counters.get("upload_inflight", 0)) + 1
        except Exception:
            pass
        try:
            if not cfg.hf_upload:
                upload_q.task_done()
                continue
            if not cfg.hf_repo_id:
                debug_fn("HF_UPLOAD 开启但 HF_REPO_ID 为空，跳过上传")
                upload_q.task_done()
                continue

            info = upload_sample_pair_fn(
                repo_id=cfg.hf_repo_id,
                image_id=task["image_id"],
                image_path=task["image_path"],
                ply_path=task["ply_path"],
            )
            debug_fn(f"HF 上传完成 | image={info['image_url']} | ply={info['ply_url']}")

            if coord is not None:
                ok_done = coord.mark_done(str(task["image_id"]))
                if ok_done:
                    try:
                        with lock:
                            checked_ids.add(str(task["image_id"]))
                    except Exception:
                        pass

                    if index_sync is not None:
                        try:
                            meta = task.get('meta') if isinstance(task, dict) else None
                            row = {
                                "image_id": str(task["image_id"]),
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
                            _log_exc(debug_fn, f"写入 index 失败 | id={str(task.get('image_id'))}", e)

            to_delete = None
            if cfg.ply_delete_after_upload and int(cfg.ply_keep_last) > 0:
                try:
                    with lock:
                        keep = counters.get("keep_plys")
                        if keep is None:
                            keep = deque()
                            counters["keep_plys"] = keep
                        keep.append(task["ply_path"])
                        if len(keep) > int(cfg.ply_keep_last):
                            to_delete = keep.popleft()
                except Exception:
                    to_delete = None

            if to_delete:
                try:
                    ap = os.path.normcase(os.path.abspath(str(to_delete)))
                    ga = os.path.normcase(os.path.abspath(str(cfg.gaussians_dir)))
                    inside = False
                    try:
                        inside = os.path.commonpath([ap, ga]) == ga
                    except Exception:
                        inside = False
                    if inside and os.path.isfile(ap):
                        os.remove(ap)
                except Exception:
                    pass

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
                touch_stop_file(cfg)
            except Exception:
                pass
            stop_event.set()
            try:
                debug_fn(f"Ctrl+C: HF 上传中断，进入停止流程 | stop_file={_stop_file_path(cfg)}")
            except Exception:
                pass
        except Exception as e:
            _log_exc(debug_fn, "HF 上传失败（可重试）", e)
        finally:
            try:
                with lock:
                    counters["upload_inflight"] = max(0, int(counters.get("upload_inflight", 0)) - 1)
            except Exception:
                pass
            upload_q.task_done()


def predict_worker(
    cfg: PipelineConfig,
    stop_event: threading.Event,
    image_q: queue.Queue,
    upload_q: queue.Queue,
    counters: dict,
    lock: threading.Lock,
    run_sharp_predict_once_fn,
    debug_fn,
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
            try:
                if bool(getattr(run_sharp_predict_once_fn, "_skip_predict", False)):
                    debug_fn(f"SKIP_PREDICT | id={image_id} | input={image_path}")
                else:
                    debug_fn(f"ml-sharp | id={image_id} | input={image_path}")
            except Exception:
                debug_fn(f"ml-sharp | id={image_id} | input={image_path}")
            try:
                plys = run_sharp_predict_once_fn(image_path)
            except KeyboardInterrupt:
                try:
                    touch_stop_file(cfg)
                except Exception:
                    pass
                stop_event.set()
                try:
                    debug_fn(f"Ctrl+C: predict 中断，进入停止流程 | stop_file={_stop_file_path(cfg)}")
                except Exception:
                    pass
                plys = []
            except Exception as e:
                _log_exc(debug_fn, f"predict 失败 | id={str(image_id)} | input={str(image_path)}", e)
                plys = []
            for ply_path in plys or []:
                if cfg.hf_upload:
                    upload_q.put(
                        {
                            "image_id": image_id,
                            "image_path": image_path,
                            "ply_path": ply_path,
                            "meta": meta,
                        }
                    )
        finally:
            try:
                with lock:
                    counters["predict_inflight"] = max(0, int(counters.get("predict_inflight", 0)) - 1)
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

    while (not stop_event.is_set()) and scanned < int(cfg.max_scan):
        order_is_oldest = False
        if not gate(cfg, stop_event):
            break
        if downloaded_images >= int(cfg.max_images):
            break
        if cfg.stop_on_rate_limit and unsplash.is_rate_limited():
            break

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
                        offset = max(0, (int(page) - 1) * int(pp))
                        base_idx = int(offset // range_size)

                        # Smarter selection to reduce contention across multiple clients.
                        # Try several candidate ranges with a deterministic step based on instance id.
                        step = 1
                        try:
                            step = 1 + (abs(hash(str(getattr(range_coord, 'instance_id', '')))) % 3)
                        except Exception:
                            step = 1

                        acquired = False
                        last_end_page = None
                        for i in range(0, 6):
                            cand_idx = int(base_idx + i * step)
                            range_start = int(cand_idx * range_size)
                            range_end = int(range_start + range_size - 1)
                            start_page = int(range_start // int(pp)) + 1
                            end_page = int((range_end // int(pp)) + 1)
                            last_end_page = int(end_page)
                            if range_coord.try_lock_range(range_start, range_end):
                                active_range = (range_start, range_end)
                                active_range_start_page = int(start_page)
                                active_range_end_page = int(end_page)
                                active_range_acquired_ts = time.time()
                                range_progress = OrderedProgress(
                                    int(range_start),
                                    int(range_end),
                                    frontier_offset=int(range_start),
                                )

                                # Restore persisted progress if available.
                                try:
                                    p = range_coord.read_progress(int(range_start), int(range_end))
                                    if isinstance(p, dict) and p:
                                        range_progress.apply_dict(p)
                                except Exception:
                                    pass

                                # Best-effort: align page to restored frontier.
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
                            # Couldn't lock any candidates; skip forward.
                            page = int(last_end_page or int(page)) + 1
                            continue
                    except Exception:
                        active_range = None
                        active_range_start_page = None
                        active_range_end_page = None
                        range_progress = None
                        active_range_acquired_ts = None

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
            if str(cfg.source).strip().lower() == "list":
                order_idx += 1
                page = 1
                active_range = None
                active_range_end_page = None
                range_progress = None
                active_range_acquired_ts = None
            else:
                query_idx += 1
                if query_idx % len(cfg.queries) == 0:
                    order_idx += 1
                page = 1
            time.sleep(1.0)
            continue

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
            if scanned >= int(cfg.max_scan) or downloaded_images >= int(cfg.max_images):
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

            if order_is_oldest and (range_coord is not None) and cfg.hf_repo_id:
                try:
                    with lock:
                        local_checked = (photo_id in checked_ids)
                except Exception:
                    local_checked = (photo_id in checked_ids)
                if not local_checked:
                    if hf_sync.hf_file_exists_cached(cfg.hf_repo_id, hf_sync.hf_done_repo_path(photo_id)):
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

            try:
                with lock:
                    if photo_id in checked_ids:
                        continue
            except Exception:
                if photo_id in checked_ids:
                    continue

            if coord is not None:
                extra = None
                if order_is_oldest:
                    try:
                        extra = str(int((int(page) - 1) * int(pp) + int(idx_in_page)))
                    except Exception:
                        extra = None
                st, retry_after = coord.try_lock_status(photo_id, extra=extra)
                if st != "acquired":
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

            scanned += 1
            details = unsplash.fetch_photo_details(photo_id)
            if not details:
                continue
            src = details
            download_location = ((src.get("links") or {}).get("download_location")) if isinstance(src, dict) else None
            if not download_location:
                continue

            meta = {}
            try:
                tags = []
                for t in (details.get('tags') or []):
                    if isinstance(t, dict):
                        tt = (t.get('title') or '').strip()
                        if tt:
                            tags.append(tt)
                topics = []
                for t in (details.get('topics') or []):
                    if isinstance(t, dict):
                        tt = (t.get('title') or '').strip()
                        if tt:
                            topics.append(tt)
                alt_desc = details.get('alt_description')
                desc = details.get('description')
                user = details.get('user') if isinstance(details.get('user'), dict) else {}
                meta = {
                    "tags": tags,
                    "topics": topics,
                    "tags_text": ",".join(tags),
                    "topics_text": ",".join(topics),
                    "alt_description": alt_desc,
                    "description": desc,
                    "unsplash_id": photo_id,
                    "unsplash_url": (details.get('links') or {}).get('html'),
                    "created_at": details.get('created_at'),
                    "user_username": user.get('username') if isinstance(user, dict) else None,
                    "user_name": user.get('name') if isinstance(user, dict) else None,
                }
            except Exception:
                meta = {"unsplash_id": photo_id}

            if not os.path.exists(cfg.input_images_dir):
                os.makedirs(cfg.input_images_dir, exist_ok=True)
            out_path = os.path.join(cfg.input_images_dir, f"{photo_id}.jpg")

            ok = False
            if not os.path.exists(out_path):
                ok = unsplash.download_image(download_location, out_path)
            else:
                ok = True
            if not ok:
                continue

            if cfg.inject_exif and (local_has_focal_exif_fn is not None) and (not local_has_focal_exif_fn(out_path)):
                details_exif = details or unsplash.fetch_photo_details(photo_id)
                if details_exif:
                    focal_raw = (details_exif.get("exif") or {}).get("focal_length")
                    _, _, _, focal_avg = unsplash.parse_focal_length(focal_raw)
                    if focal_avg is not None:
                        try:
                            inject_focal_exif_if_missing_fn(out_path, focal_avg)
                        except Exception:
                            pass

            while (not stop_event.is_set()) and image_q.full():
                if not gate(cfg, stop_event):
                    break
                idle_sleep(cfg)

            image_q.put({"image_id": photo_id, "image_path": out_path, "meta": meta})
            try:
                with lock:
                    checked_ids.add(photo_id)
            except Exception:
                pass

            downloaded_images += 1

        page += 1

        if (active_range is not None) and (active_range_end_page is not None) and (range_coord is not None):
            try:
                if int(page) > int(active_range_end_page):
                    if downloaded_images >= int(cfg.max_images) or stop_event.is_set() or stop_requested(cfg):
                        try:
                            a, b = active_range
                            range_coord.mark_abandoned_range(int(a), int(b), "stopped_or_max_images")
                        except Exception:
                            pass
                        active_range = None
                        active_range_end_page = None
                        range_progress = None
                        active_range_acquired_ts = None
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
                        active_range = None
                        active_range_end_page = None
                        range_progress = None
                        active_range_acquired_ts = None
            except Exception:
                active_range = None
                active_range_end_page = None
                range_progress = None
                active_range_acquired_ts = None

    # If we exit early while holding a range (did not reach mark_done_range), record it for observability.
    if (active_range is not None) and (range_coord is not None):
        try:
            a, b = active_range
            reason = "loop_exit"
            if downloaded_images >= int(cfg.max_images):
                reason = "max_images"
            elif scanned >= int(cfg.max_scan):
                reason = "max_scan"
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
    checked_ids: set,
    coord: hf_sync.LockDoneSync | None,
    range_coord: hf_sync.RangeLockSync | None,
    index_sync,
    upload_sample_pair_fn,
    try_super_squash_fn,
    run_sharp_predict_once_fn,
    local_has_focal_exif_fn,
    inject_focal_exif_if_missing_fn,
    debug_fn,
):
    stop_event = threading.Event()

    prev_sigint_handler = None
    sigint_state = {"last": 0.0}
    try:
        prev_sigint_handler = signal.getsignal(signal.SIGINT)

        def _on_sigint(_signum, _frame):
            try:
                now = time.time()
                last = float(sigint_state.get("last", 0.0) or 0.0)
                sigint_state["last"] = float(now)
                if pause_requested(cfg) or ((now - last) <= float(cfg.sigint_window_s)):
                    touch_stop_file(cfg)
                    stop_event.set()
                    try:
                        if debug_fn:
                            debug_fn(f"SIGINT: stop requested | stop_file={_stop_file_path(cfg)}")
                    except Exception:
                        pass
                else:
                    set_pause_file(cfg, True)
                    try:
                        if debug_fn:
                            debug_fn(f"SIGINT: pause requested | pause_file={_pause_file_path(cfg)}")
                    except Exception:
                        pass
            except Exception:
                try:
                    touch_stop_file(cfg)
                    stop_event.set()
                except Exception:
                    pass

        signal.signal(signal.SIGINT, _on_sigint)
    except Exception:
        prev_sigint_handler = None

    image_q = queue.Queue(maxsize=max(1, int(cfg.download_queue_max)))
    upload_q = queue.Queue(maxsize=max(1, int(cfg.upload_queue_max)))
    counters = {"uploaded": 0, "keep_plys": deque(), "predict_inflight": 0, "upload_inflight": 0}
    lock = threading.Lock()

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
                    try_super_squash_fn,
                    debug_fn,
                ),
                daemon=True,
            )
            t.start()
            upload_threads.append(t)

    predict_t = threading.Thread(
        target=predict_worker,
        args=(cfg, stop_event, image_q, upload_q, counters, lock, run_sharp_predict_once_fn, debug_fn),
        daemon=True,
    )
    predict_t.start()

    last_sigint_ts = 0.0

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

                    uploads_idle = (not cfg.hf_upload) or (upload_q.empty() and ui <= 0)
                    if image_q.empty() and pi <= 0 and uploads_idle:
                        return
                    idle_sleep(cfg)
                return
            except KeyboardInterrupt:
                now = time.time()
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

                # If we're already paused, treat Ctrl+C as a stop request.
                if pause_requested(cfg):
                    debug_fn(
                        f"Ctrl+C：暂停中再次触发 -> 停止并退出（创建 STOP） | stop_file={_stop_file_path(cfg)} | pause_file={_pause_file_path(cfg)} | image_q={iq} upload_q={uq} pi={pi} ui={ui}"
                    )
                    touch_stop_file(cfg)
                    stop_event.set()
                    break

                if now - last_sigint_ts <= float(cfg.sigint_window_s):
                    debug_fn(
                        f"Ctrl+C 二次触发：停止并退出（创建 STOP） | stop_file={_stop_file_path(cfg)} | image_q={iq} upload_q={uq} pi={pi} ui={ui}"
                    )
                    touch_stop_file(cfg)
                    stop_event.set()
                    break
                last_sigint_ts = now
                set_pause_file(cfg, True)
                debug_fn(
                    f"Ctrl+C：暂停（创建 PAUSE） | pause_file={_pause_file_path(cfg)} | 再次 Ctrl+C 将停止 | image_q={iq} upload_q={uq} pi={pi} ui={ui}"
                )
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
