import base64
import json
import os
import random
import subprocess
import math
import shutil
import time


def make_small_ply(
    ply_path: str,
    *,
    splat_transform_bin: str,
    filter_visibility: int,
    debug_fn,
) -> str | None:
    try:
        src = os.path.abspath(str(ply_path))
        if not os.path.isfile(src):
            return None
        try:
            low = os.path.basename(src).lower()
            if ".small.gsplat" in low:
                return src
            if ".vertexonly.binary" in low:
                return src
        except Exception:
            pass
        base, _ = os.path.splitext(src)
        out = base + ".small.gsplat.ply"
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            return out
        cand = str(splat_transform_bin or "").strip() or "splat-transform"
        resolved = cand
        try:
            if (not os.path.isfile(resolved)) and shutil.which(resolved):
                resolved = shutil.which(resolved) or resolved
        except Exception:
            resolved = cand

        cmd = [
            str(resolved),
            "-w",
            src,
            "--filter-visibility",
            str(int(filter_visibility)),
            out,
        ]
        subprocess.run(cmd, check=True)
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            return out
        return None
    except Exception as e:
        try:
            if debug_fn:
                debug_fn(f"GSPLAT: splat-transform 生成小 PLY 失败 | err={str(e)}")
        except Exception:
            pass
        return None


def _trpc_extract_data(resp: dict | list | None):
    try:
        if resp is None:
            return None
        if isinstance(resp, list) and resp:
            item = (resp or [])[0] or {}
            data = ((item.get("result") or {}) or {}).get("data")
            if isinstance(data, dict) and "json" in data:
                return data.get("json")
            return data
        if isinstance(resp, dict):
            out = resp.get("result") or resp.get("data") or resp
            if isinstance(out, dict) and "json" in out:
                return out.get("json")
            return out
        return None
    except Exception:
        return None


def _trpc_extract_error(resp: dict | list | None):
    try:
        if resp is None:
            return None
        if isinstance(resp, list) and resp:
            item = (resp or [])[0] or {}
            err = item.get("error")
            if err:
                return err
            return None
        if isinstance(resp, dict):
            err = resp.get("error")
            if err:
                return err
            return None
        return None
    except Exception:
        return None


def _guess_gsplat_file_type(path: str) -> str:
    try:
        ext = os.path.splitext(str(path or "").strip())[-1].lower().lstrip(".")
        if ext in ("ply", "spz", "splat"):
            return ext
        return "ply"
    except Exception:
        return "ply"


def _deep_find_first(obj, keys: set[str], *, max_depth: int = 6):
    try:
        if max_depth <= 0:
            return None
        if obj is None:
            return None
        if isinstance(obj, dict):
            for k in keys:
                v = obj.get(k)
                if isinstance(v, str) and v.strip():
                    return v
            for v in obj.values():
                got = _deep_find_first(v, keys, max_depth=max_depth - 1)
                if got is not None:
                    return got
            return None
        if isinstance(obj, list):
            for it in obj:
                got = _deep_find_first(it, keys, max_depth=max_depth - 1)
                if got is not None:
                    return got
            return None
        return None
    except Exception:
        return None


def _chunked_upload_and_get_model_file_url(
    file_path: str,
    *,
    gsplat_base: str,
    title: str,
    description: str,
    expiration_type: str,
    chunk_size: int = 50 * 1024 * 1024,
    debug_fn,
) -> str | None:
    try:
        import requests

        src = os.path.abspath(str(file_path))
        if not os.path.isfile(src):
            return None
        file_size = int(os.path.getsize(src))
        if file_size <= 0:
            return None

        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            chunk_size = 50 * 1024 * 1024

        total_chunks = int(math.ceil(float(file_size) / float(chunk_size)))
        total_chunks = max(1, int(total_chunks))

        filename = os.path.basename(src)
        initiate_payload = {
            "0": {
                "filename": str(filename),
                "fileSize": int(file_size),
                "chunkSize": int(chunk_size),
                "contentType": "",
                "metadata": {
                    "title": str(title or ""),
                    "description": str(description or ""),
                    "expirationType": str(expiration_type or "1week"),
                },
            }
        }
        init_resp = trpc_post(
            gsplat_base,
            "/share/trpc/order.chunkedUploadInitiate?batch=1",
            initiate_payload,
            debug_fn=debug_fn,
        )
        if not init_resp:
            return None

        trpc_err = _trpc_extract_error(init_resp)
        if trpc_err is not None:
            try:
                if debug_fn:
                    debug_fn(f"GSPLAT: chunkedUploadInitiate 返回错误 | err={str(trpc_err)[:400]}")
            except Exception:
                pass
            return None

        init_data = _trpc_extract_data(init_resp)
        upload_id = None
        try:
            if isinstance(init_data, dict):
                upload_id = init_data.get("uploadId") or init_data.get("id")
            elif isinstance(init_data, str):
                upload_id = init_data
        except Exception:
            upload_id = None

        if not upload_id:
            try:
                if debug_fn:
                    debug_fn(f"GSPLAT: chunkedUploadInitiate 未返回 uploadId | resp={str(init_resp)[:400]}")
            except Exception:
                pass
            return None

        # Send chunks as base64 strings in JSON (same as gsplat.org frontend).
        with open(src, "rb") as f:
            for chunk_index in range(int(total_chunks)):
                raw = f.read(int(chunk_size))
                if not raw:
                    break
                data_b64 = base64.b64encode(raw).decode("ascii")
                chunk_payload = {
                    "0": {
                        "uploadId": str(upload_id),
                        "chunkIndex": int(chunk_index),
                        "totalChunks": int(total_chunks),
                        "data": data_b64,
                        "size": int(len(raw)),
                    }
                }
                chunk_resp = trpc_post(
                    gsplat_base,
                    "/share/trpc/order.chunkedUploadChunk?batch=1",
                    chunk_payload,
                    debug_fn=debug_fn,
                )
                if not chunk_resp:
                    return None

                trpc_err = _trpc_extract_error(chunk_resp)
                if trpc_err is not None:
                    try:
                        if debug_fn:
                            debug_fn(
                                f"GSPLAT: chunkedUploadChunk 返回错误 | idx={chunk_index} | err={str(trpc_err)[:400]}"
                            )
                    except Exception:
                        pass
                    return None

        finalize_payload = {
            "0": {
                "uploadId": str(upload_id),
                "totalChunks": int(total_chunks),
                "filename": str(filename),
                "metadata": {
                    "title": str(title or ""),
                    "description": str(description or ""),
                    "expirationType": str(expiration_type or "1week"),
                },
            }
        }
        fin_resp = trpc_post(
            gsplat_base,
            "/share/trpc/order.chunkedUploadFinalize?batch=1",
            finalize_payload,
            debug_fn=debug_fn,
        )
        if not fin_resp:
            return None

        trpc_err = _trpc_extract_error(fin_resp)
        if trpc_err is not None:
            try:
                if debug_fn:
                    debug_fn(f"GSPLAT: chunkedUploadFinalize 返回错误 | err={str(trpc_err)[:400]}")
            except Exception:
                pass
            return None

        fin_data = _trpc_extract_data(fin_resp)
        model_file_url = None
        try:
            if isinstance(fin_data, str):
                model_file_url = fin_data
            else:
                model_file_url = _deep_find_first(fin_data, {"modelFileUrl", "fileUrl", "url"})
        except Exception:
            model_file_url = None

        if not model_file_url:
            try:
                if debug_fn:
                    debug_fn(f"GSPLAT: chunkedUploadFinalize 未返回 modelFileUrl | resp={str(fin_resp)[:400]}")
            except Exception:
                pass
            return None
        return str(model_file_url)
    except Exception as e:
        try:
            if debug_fn:
                debug_fn(f"GSPLAT: chunked upload 异常 | err={str(e)}")
        except Exception:
            pass
        return None


def trpc_post(base_url: str, path: str, payload: dict, *, debug_fn) -> dict | None:
    def _sleep_backoff(attempt: int, *, retry_after_s: float | None = None):
        try:
            if retry_after_s is not None:
                wait_s = float(retry_after_s)
            else:
                wait_s = min(30.0, float(1.0 * (2**int(attempt))))
                wait_s = float(wait_s) * (0.5 + random.random())
            time.sleep(max(0.2, float(wait_s)))
        except Exception:
            time.sleep(0.5)

    try:
        tries = 3
        try:
            tries = int(os.getenv("GSPLAT_TRPC_RETRIES", "3") or "3")
        except Exception:
            tries = 3
        tries = max(1, min(10, int(tries)))

        import requests

        url = f"{str(base_url).rstrip('/')}{path}"
        last_err = None
        for attempt in range(0, int(tries)):
            try:
                r = requests.post(url, data=json.dumps(payload), headers={"content-type": "application/json"}, timeout=120)
                if int(r.status_code) != 200:
                    try:
                        if debug_fn:
                            debug_fn(
                                f"GSPLAT: tRPC 请求失败 | status={r.status_code} | url={url} | body={r.text[:400]}"
                            )
                    except Exception:
                        pass

                    st = int(r.status_code)
                    retry_after = None
                    try:
                        if st == 429:
                            ra = r.headers.get("retry-after") or r.headers.get("Retry-After")
                            if ra is not None and str(ra).strip():
                                retry_after = float(str(ra).strip())
                    except Exception:
                        retry_after = None

                    if attempt < (tries - 1) and st in (408, 425, 429, 500, 502, 503, 504):
                        _sleep_backoff(attempt, retry_after_s=retry_after)
                        continue
                    return None

                try:
                    return r.json()
                except Exception as e:
                    last_err = e
                    try:
                        if debug_fn:
                            debug_fn(f"GSPLAT: tRPC JSON 解析失败 | url={url} | text={r.text[:400]}")
                    except Exception:
                        pass
                    if attempt < (tries - 1):
                        _sleep_backoff(attempt)
                        continue
                    return None
            except Exception as e:
                last_err = e
                try:
                    if debug_fn:
                        debug_fn(f"GSPLAT: tRPC 请求异常（可重试） | url={url} | err={str(e)}")
                except Exception:
                    pass
                if attempt < (tries - 1):
                    _sleep_backoff(attempt)
                    continue
                return None

        if last_err is not None:
            raise last_err
        return None
    except Exception as e:
        try:
            if debug_fn:
                debug_fn(f"GSPLAT: tRPC 请求异常 | err={str(e)}")
        except Exception:
            pass
        return None


def upload_and_create_view(
    ply_path: str,
    *,
    gsplat_base: str,
    expiration_type: str,
    splat_transform_bin: str,
    filter_visibility: int,
    title: str,
    description: str = "",
    use_small_ply: bool = False,
    debug_fn,
) -> dict | None:
    try:
        src_ply = os.path.abspath(str(ply_path))
        if not os.path.isfile(src_ply):
            return None

        upload_ply = src_ply
        if bool(use_small_ply):
            small_ply = make_small_ply(
                src_ply,
                splat_transform_bin=str(splat_transform_bin),
                filter_visibility=int(filter_visibility),
                debug_fn=debug_fn,
            )
            if small_ply:
                upload_ply = small_ply
            else:
                try:
                    if debug_fn:
                        debug_fn("GSPLAT: small PLY 生成失败，将回退为直接上传原始 PLY")
                except Exception:
                    pass

        model_file_url = None
        try:
            sz = int(os.path.getsize(upload_ply))
        except Exception:
            sz = 0

        # If the payload is large, use chunked upload to avoid a huge single JSON request.
        if int(sz) >= int(20 * 1024 * 1024):
            model_file_url = _chunked_upload_and_get_model_file_url(
                upload_ply,
                gsplat_base=str(gsplat_base),
                title=str(title or ""),
                description=str(description or ""),
                expiration_type=str(expiration_type or "1week"),
                debug_fn=debug_fn,
            )
        else:
            with open(upload_ply, "rb") as f:
                raw = f.read()
            data_b64 = base64.b64encode(raw).decode("ascii")

            ftype = _guess_gsplat_file_type(upload_ply)

            up_payload = {
                "0": {
                    "gaussianSplatFile": {
                        "name": os.path.basename(upload_ply),
                        "data": data_b64,
                        "type": str(ftype),
                        "size": int(len(raw)),
                    }
                }
            }
            up_resp = trpc_post(gsplat_base, "/share/trpc/order.uploadGaussianSplat?batch=1", up_payload, debug_fn=debug_fn)
            if not up_resp:
                return None

            trpc_err = _trpc_extract_error(up_resp)
            if trpc_err is not None:
                try:
                    if debug_fn:
                        debug_fn(f"GSPLAT: uploadGaussianSplat 返回错误 | err={str(trpc_err)[:400]}")
                except Exception:
                    pass
                return None

            try:
                data = _trpc_extract_data(up_resp)
                if isinstance(data, str):
                    model_file_url = data
                else:
                    model_file_url = _deep_find_first(data, {"modelFileUrl", "fileUrl", "url"})
            except Exception:
                model_file_url = None

        if not model_file_url:
            try:
                if debug_fn:
                    debug_fn(f"GSPLAT: 未拿到 modelFileUrl，跳过 | resp={str(up_resp)[:400]}")
            except Exception:
                pass
            return None

        def _is_transient_trpc_error(err_obj) -> bool:
            try:
                s = str(err_obj or "").lower()
                return any(
                    x in s
                    for x in (
                        "timeout",
                        "timed out",
                        "temporar",
                        "rate",
                        "too many",
                        "429",
                        "502",
                        "503",
                        "504",
                        "gateway",
                        "service unavailable",
                        "network",
                    )
                )
            except Exception:
                return False

        tries = 4
        try:
            tries = int(os.getenv("GSPLAT_CREATE_ORDER_RETRIES", "4") or "4")
        except Exception:
            tries = 4
        tries = max(1, min(10, int(tries)))

        share_id = None
        order_id = None
        last_resp = None
        for attempt in range(0, int(tries)):
            order_payload = {
                "0": {
                    "modelFileUrl": str(model_file_url),
                    "title": str(title or ""),
                    "description": str(description or ""),
                    "expirationType": str(expiration_type or "1week"),
                }
            }
            order_resp = trpc_post(gsplat_base, "/share/trpc/order.createOrder?batch=1", order_payload, debug_fn=debug_fn)
            last_resp = order_resp
            if not order_resp:
                if attempt < (tries - 1):
                    try:
                        if debug_fn:
                            debug_fn(f"GSPLAT: createOrder 请求失败，将重试 | attempt={attempt + 1}/{tries}")
                    except Exception:
                        pass
                    try:
                        wait_s = min(30.0, float(1.0 * (2**int(attempt))))
                        wait_s = float(wait_s) * (0.5 + random.random())
                        time.sleep(max(0.2, float(wait_s)))
                    except Exception:
                        time.sleep(0.5)
                    continue
                return None

            trpc_err = _trpc_extract_error(order_resp)
            if trpc_err is not None:
                try:
                    if debug_fn:
                        debug_fn(f"GSPLAT: createOrder 返回错误 | err={str(trpc_err)[:400]}")
                except Exception:
                    pass
                if attempt < (tries - 1) and _is_transient_trpc_error(trpc_err):
                    try:
                        wait_s = min(30.0, float(1.0 * (2**int(attempt))))
                        wait_s = float(wait_s) * (0.5 + random.random())
                        time.sleep(max(0.2, float(wait_s)))
                    except Exception:
                        time.sleep(0.5)
                    continue
                return None

            try:
                data = _trpc_extract_data(order_resp)
                if isinstance(data, dict):
                    share_id = data.get("shareId") or _deep_find_first(data, {"shareId"})
                    order_id = data.get("id") or _deep_find_first(data, {"id"})
            except Exception:
                share_id = None

            if share_id:
                break

            if attempt < (tries - 1):
                try:
                    if debug_fn:
                        debug_fn(
                            f"GSPLAT: createOrder 未返回 shareId，将重试 | attempt={attempt + 1}/{tries} | resp={str(order_resp)[:300]}"
                        )
                except Exception:
                    pass
                try:
                    wait_s = min(30.0, float(1.0 * (2**int(attempt))))
                    wait_s = float(wait_s) * (0.5 + random.random())
                    time.sleep(max(0.2, float(wait_s)))
                except Exception:
                    time.sleep(0.5)
                continue

        if not share_id:
            try:
                if debug_fn:
                    debug_fn(f"GSPLAT: createOrder 重试后仍未返回 shareId | resp={str(last_resp)[:300]}")
            except Exception:
                pass
            return None

        gsplat_url = f"{str(gsplat_base).rstrip('/')}/viewer/{share_id}"
        return {
            "gsplat_url": gsplat_url,
            "gsplat_share_id": str(share_id),
            "gsplat_order_id": str(order_id) if order_id else None,
            "gsplat_model_file_url": str(model_file_url),
        }
    except Exception as e:
        try:
            if debug_fn:
                debug_fn(f"GSPLAT: 上传/建分享异常 | err={str(e)}")
        except Exception:
            pass
        return None
