import os
import random
import threading
import time

from . import hf_utils
from . import gsplat_share
from . import metrics
from . import spz_export


_commit_lock = threading.Lock()


def _is_precondition_failed(err: Exception) -> bool:
    try:
        s = str(err)
        return (" 412" in s) or ("412" in s and "Precondition" in s) or ("Precondition Failed" in s) or (
            "A commit has happened since" in s
        )
    except Exception:
        return False


def _create_commit_retry(api, *, repo_id: str, repo_type: str, operations, commit_message: str, debug_fn):
    last_err = None
    for attempt in range(0, 6):
        try:
            with _commit_lock:
                api.create_commit(
                    repo_id=repo_id,
                    repo_type=repo_type,
                    operations=operations,
                    commit_message=commit_message,
                )
            return
        except Exception as e:
            last_err = e
            if hf_utils.should_retry_with_pr(e):
                raise
            if not _is_precondition_failed(e):
                raise
            if attempt >= 5:
                raise
            try:
                wait_s = min(8.0, float(0.5 * (2**attempt)))
                wait_s = float(wait_s) * (0.5 + random.random())
                if debug_fn:
                    debug_fn(f"HF 上传失败（可重试，冲突 412） | wait={wait_s:.2f}s | attempt={attempt + 1}/6")
                time.sleep(wait_s)
            except Exception:
                time.sleep(0.5)
    if last_err is not None:
        raise last_err


def upload_sample_pair(
    repo_id: str,
    image_id: str,
    image_path: str,
    ply_path: str,
    *,
    hf_subdir: str,
    repo_type: str,
    gsplat_enabled: bool = False,
    gsplat_base: str = "https://gsplat.org",
    gsplat_expiration_type: str = "1week",
    gsplat_filter_visibility: int = 20000,
    splat_transform_bin: str = "splat-transform",
    gsplat_use_small_ply: bool = False,
    spz_enabled: bool,
    spz_tool: str,
    gsbox_bin: str,
    gsbox_spz_quality: int,
    gsbox_spz_version: int,
    gsconverter_bin: str,
    gsconverter_compression_level: int,
    debug_fn,
) -> dict:
    from huggingface_hub import CommitOperationAdd, HfApi

    rel_dir = "/".join([p for p in [str(hf_subdir).strip().strip("/"), str(image_id)] if p])
    img_name = os.path.basename(image_path)
    ply_name = os.path.basename(ply_path)

    spz_t0 = float(time.time())
    spz_path = spz_export.maybe_export_from_ply(
        ply_path,
        enabled=bool(spz_enabled),
        tool=str(spz_tool or ""),
        gsbox_bin=str(gsbox_bin or "gsbox"),
        gsbox_spz_quality=int(gsbox_spz_quality),
        gsbox_spz_version=int(gsbox_spz_version),
        gsconverter_bin=str(gsconverter_bin or "3dgsconverter"),
        gsconverter_compression_level=int(gsconverter_compression_level),
        debug_fn=debug_fn,
    )
    try:
        spz_s = max(0.0, float(time.time()) - float(spz_t0))
        spz_bytes = None
        try:
            if spz_path and os.path.isfile(spz_path):
                spz_bytes = int(os.path.getsize(spz_path))
        except Exception:
            spz_bytes = None
        metrics.emit(
            "spz_export_done",
            debug_fn=debug_fn,
            image_id=str(image_id),
            ok=bool(spz_path),
            s=float(spz_s),
            spz_bytes=spz_bytes,
            tool=str(spz_tool or ""),
            **metrics.snapshot(),
        )
    except Exception:
        pass
    spz_name = os.path.basename(spz_path) if spz_path else None

    rel_img = f"{rel_dir}/{img_name}"
    rel_ply = f"{rel_dir}/{ply_name}"
    rel_spz = f"{rel_dir}/{spz_name}" if spz_name else None

    api = HfApi()
    ops = [
        CommitOperationAdd(path_in_repo=rel_img, path_or_fileobj=image_path),
        CommitOperationAdd(path_in_repo=rel_ply, path_or_fileobj=ply_path),
    ]
    if spz_path and rel_spz:
        ops.append(CommitOperationAdd(path_in_repo=rel_spz, path_or_fileobj=spz_path))

    commit_t0 = float(time.time())
    try:
        _create_commit_retry(
            api,
            repo_id=repo_id,
            repo_type=repo_type,
            operations=ops,
            commit_message=f"add {image_id}",
            debug_fn=debug_fn,
        )
    except Exception as e:
        if not hf_utils.should_retry_with_pr(e):
            raise
        with _commit_lock:
            api.create_commit(
                repo_id=repo_id,
                repo_type=repo_type,
                operations=ops,
                commit_message=f"add {image_id}",
                create_pr=True,
            )
    try:
        commit_s = max(0.0, float(time.time()) - float(commit_t0))
        metrics.emit(
            "hf_commit_done",
            debug_fn=debug_fn,
            image_id=str(image_id),
            s=float(commit_s),
            ops=int(len(ops or [])),
            **metrics.snapshot(),
        )
    except Exception:
        pass

    gsplat_meta = {}
    if bool(gsplat_enabled):
        try:
            gs_t0 = float(time.time())
            gsplat_meta = (
                gsplat_share.upload_and_create_view(
                    ply_path,
                    gsplat_base=str(gsplat_base),
                    expiration_type=str(gsplat_expiration_type),
                    splat_transform_bin=str(splat_transform_bin),
                    filter_visibility=int(gsplat_filter_visibility),
                    title=str(image_id),
                    description="",
                    use_small_ply=bool(gsplat_use_small_ply),
                    debug_fn=debug_fn,
                )
                or {}
            )
            try:
                gs_s = max(0.0, float(time.time()) - float(gs_t0))
                metrics.emit(
                    "gsplat_done",
                    debug_fn=debug_fn,
                    image_id=str(image_id),
                    ok=bool(gsplat_meta),
                    s=float(gs_s),
                    **metrics.snapshot(),
                )
            except Exception:
                pass
        except Exception as e:
            try:
                if debug_fn:
                    debug_fn(f"GSPLAT: upload_and_create_view 失败（跳过） | err={str(e)}")
            except Exception:
                pass
            try:
                metrics.emit(
                    "gsplat_done",
                    debug_fn=debug_fn,
                    image_id=str(image_id),
                    ok=False,
                    err=str(e)[:200],
                    **metrics.snapshot(),
                )
            except Exception:
                pass
            gsplat_meta = {}

    return {
        "image_url": hf_utils.build_resolve_url(repo_id, rel_img, repo_type=repo_type),
        "ply_url": hf_utils.build_resolve_url(repo_id, rel_ply, repo_type=repo_type),
        "spz_url": hf_utils.build_resolve_url(repo_id, rel_spz, repo_type=repo_type) if rel_spz else None,
        **gsplat_meta,
    }


def upload_sample_pairs(
    repo_id: str,
    tasks: list,
    *,
    hf_subdir: str,
    repo_type: str,
    gsplat_enabled: bool = False,
    gsplat_base: str = "https://gsplat.org",
    gsplat_expiration_type: str = "1week",
    gsplat_filter_visibility: int = 20000,
    splat_transform_bin: str = "splat-transform",
    gsplat_use_small_ply: bool = False,
    spz_enabled: bool = False,
    spz_tool: str = "",
    gsbox_bin: str = "gsbox",
    gsbox_spz_quality: int = 5,
    gsbox_spz_version: int = 0,
    gsconverter_bin: str = "3dgsconverter",
    gsconverter_compression_level: int = 6,
    debug_fn=None,
) -> dict:
    from huggingface_hub import CommitOperationAdd, HfApi

    if not isinstance(tasks, list) or (not tasks):
        return {}

    api = HfApi()
    ops = []
    per = {}

    for t in tasks:
        if not isinstance(t, dict):
            continue
        image_id = str(t.get("image_id") or "").strip()
        image_path = str(t.get("image_path") or "")
        ply_path = str(t.get("ply_path") or "")
        if (not image_id) or (not image_path) or (not ply_path):
            continue

        rel_dir = "/".join([p for p in [str(hf_subdir).strip().strip("/"), str(image_id)] if p])
        img_name = os.path.basename(image_path)
        ply_name = os.path.basename(ply_path)

        spz_t0 = float(time.time())
        spz_path = spz_export.maybe_export_from_ply(
            ply_path,
            enabled=bool(spz_enabled),
            tool=str(spz_tool or ""),
            gsbox_bin=str(gsbox_bin or "gsbox"),
            gsbox_spz_quality=int(gsbox_spz_quality),
            gsbox_spz_version=int(gsbox_spz_version),
            gsconverter_bin=str(gsconverter_bin or "3dgsconverter"),
            gsconverter_compression_level=int(gsconverter_compression_level),
            debug_fn=debug_fn,
        )
        try:
            spz_s = max(0.0, float(time.time()) - float(spz_t0))
            spz_bytes = None
            try:
                if spz_path and os.path.isfile(spz_path):
                    spz_bytes = int(os.path.getsize(spz_path))
            except Exception:
                spz_bytes = None
            metrics.emit(
                "spz_export_done",
                debug_fn=debug_fn,
                image_id=str(image_id),
                ok=bool(spz_path),
                s=float(spz_s),
                spz_bytes=spz_bytes,
                tool=str(spz_tool or ""),
                **metrics.snapshot(),
            )
        except Exception:
            pass
        spz_name = os.path.basename(spz_path) if spz_path else None

        rel_img = f"{rel_dir}/{img_name}"
        rel_ply = f"{rel_dir}/{ply_name}"
        rel_spz = f"{rel_dir}/{spz_name}" if (spz_path and spz_name) else None

        ops.append(CommitOperationAdd(path_in_repo=rel_img, path_or_fileobj=image_path))
        ops.append(CommitOperationAdd(path_in_repo=rel_ply, path_or_fileobj=ply_path))
        if spz_path and rel_spz:
            ops.append(CommitOperationAdd(path_in_repo=rel_spz, path_or_fileobj=spz_path))

        per[image_id] = {
            "rel_img": rel_img,
            "rel_ply": rel_ply,
            "rel_spz": rel_spz,
            "ply_path": ply_path,
        }

    if not ops:
        return {}

    commit_t0 = float(time.time())

    try:
        _create_commit_retry(
            api,
            repo_id=repo_id,
            repo_type=repo_type,
            operations=ops,
            commit_message=f"add batch {len(per)}",
            debug_fn=debug_fn,
        )
    except Exception as e:
        if not hf_utils.should_retry_with_pr(e):
            raise
        with _commit_lock:
            api.create_commit(
                repo_id=repo_id,
                repo_type=repo_type,
                operations=ops,
                commit_message=f"add batch {len(per)}",
                create_pr=True,
            )

    try:
        commit_s = max(0.0, float(time.time()) - float(commit_t0))
        metrics.emit(
            "hf_commit_batch_done",
            debug_fn=debug_fn,
            s=float(commit_s),
            ops=int(len(ops or [])),
            batch_n=int(len(per or {})),
            **metrics.snapshot(),
        )
    except Exception:
        pass

    out = {}
    for image_id, meta in (per or {}).items():
        rel_img = str(meta.get("rel_img") or "")
        rel_ply = str(meta.get("rel_ply") or "")
        rel_spz = meta.get("rel_spz")
        ply_path = str(meta.get("ply_path") or "")

        gsplat_meta = {}
        if bool(gsplat_enabled):
            try:
                gsplat_meta = (
                    gsplat_share.upload_and_create_view(
                        ply_path,
                        gsplat_base=str(gsplat_base),
                        expiration_type=str(gsplat_expiration_type),
                        splat_transform_bin=str(splat_transform_bin),
                        filter_visibility=int(gsplat_filter_visibility),
                        title=str(image_id),
                        description="",
                        use_small_ply=bool(gsplat_use_small_ply),
                        debug_fn=debug_fn,
                    )
                    or {}
                )
            except Exception as e:
                try:
                    if debug_fn:
                        debug_fn(f"GSPLAT: upload_and_create_view 失败（跳过） | err={str(e)}")
                except Exception:
                    pass
                gsplat_meta = {}

        out[str(image_id)] = {
            "image_url": hf_utils.build_resolve_url(repo_id, rel_img, repo_type=repo_type),
            "ply_url": hf_utils.build_resolve_url(repo_id, rel_ply, repo_type=repo_type),
            "spz_url": hf_utils.build_resolve_url(repo_id, rel_spz, repo_type=repo_type) if rel_spz else None,
            **gsplat_meta,
        }

    return out
