import os

from . import hf_utils
from . import gsplat_share
from . import spz_export


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

    try:
        api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=ops,
            commit_message=f"add {image_id}",
        )
    except Exception as e:
        if not hf_utils.should_retry_with_pr(e):
            raise
        api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=ops,
            commit_message=f"add {image_id}",
            create_pr=True,
        )

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

    return {
        "image_url": hf_utils.build_resolve_url(repo_id, rel_img, repo_type=repo_type),
        "ply_url": hf_utils.build_resolve_url(repo_id, rel_ply, repo_type=repo_type),
        "spz_url": hf_utils.build_resolve_url(repo_id, rel_spz, repo_type=repo_type) if rel_spz else None,
        **gsplat_meta,
    }
