import base64
import json
import os
import subprocess


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
        base, _ = os.path.splitext(src)
        out = base + ".small.gsplat.ply"
        if os.path.isfile(out) and os.path.getsize(out) > 0:
            return out
        cmd = [
            str(splat_transform_bin),
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


def trpc_post(base_url: str, path: str, payload: dict, *, debug_fn) -> dict | None:
    try:
        import requests

        url = f"{str(base_url).rstrip('/')}{path}"
        r = requests.post(url, data=json.dumps(payload), headers={"content-type": "application/json"}, timeout=120)
        if r.status_code != 200:
            try:
                if debug_fn:
                    debug_fn(f"GSPLAT: tRPC 请求失败 | status={r.status_code} | url={url} | body={r.text[:400]}")
            except Exception:
                pass
            return None
        try:
            return r.json()
        except Exception:
            try:
                if debug_fn:
                    debug_fn(f"GSPLAT: tRPC JSON 解析失败 | url={url} | text={r.text[:400]}")
            except Exception:
                pass
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
    debug_fn,
) -> dict | None:
    try:
        small_ply = make_small_ply(
            ply_path,
            splat_transform_bin=str(splat_transform_bin),
            filter_visibility=int(filter_visibility),
            debug_fn=debug_fn,
        )
        if not small_ply:
            return None

        with open(small_ply, "rb") as f:
            raw = f.read()
        data_b64 = base64.b64encode(raw).decode("ascii")

        up_payload = {
            "0": {
                "gaussianSplatFile": {
                    "name": os.path.basename(small_ply),
                    "data": data_b64,
                }
            }
        }
        up_resp = trpc_post(gsplat_base, "/share/trpc/order.uploadGaussianSplat?batch=1", up_payload, debug_fn=debug_fn)
        if not up_resp:
            return None

        model_file_url = None
        try:
            model_file_url = (((up_resp or [])[0] or {}).get("result") or {}).get("data")
            if isinstance(model_file_url, dict):
                model_file_url = model_file_url.get("modelFileUrl") or model_file_url.get("fileUrl")
        except Exception:
            model_file_url = None
        if not model_file_url:
            try:
                if debug_fn:
                    debug_fn(f"GSPLAT: uploadGaussianSplat 未返回 modelFileUrl | resp={str(up_resp)[:300]}")
            except Exception:
                pass
            return None

        order_payload = {
            "0": {
                "modelFileUrl": str(model_file_url),
                "title": str(title or ""),
                "description": str(description or ""),
                "expirationType": str(expiration_type or "1week"),
            }
        }
        order_resp = trpc_post(gsplat_base, "/share/trpc/order.createOrder?batch=1", order_payload, debug_fn=debug_fn)
        if not order_resp:
            return None

        share_id = None
        order_id = None
        try:
            data = (((order_resp or [])[0] or {}).get("result") or {}).get("data")
            if isinstance(data, dict):
                share_id = data.get("shareId")
                order_id = data.get("id")
        except Exception:
            share_id = None

        if not share_id:
            try:
                if debug_fn:
                    debug_fn(f"GSPLAT: createOrder 未返回 shareId | resp={str(order_resp)[:300]}")
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
