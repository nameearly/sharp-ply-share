import json
import os
import sys
import time
from typing import Any

from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download

# When executed as `python scripts/...`, ensure repo root is importable.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from sharp_dataset_pipeline import hf_utils


def _is_precondition_failed(err: Exception) -> bool:
    try:
        s = str(err)
        return (
            (" 412" in s)
            or ("412" in s and "Precondition" in s)
            or ("Precondition Failed" in s)
            or ("A commit has happened since" in s)
        )
    except Exception:
        return False


def _create_commit_retry(api: HfApi, *, repo_id: str, repo_type: str, operations, commit_message: str) -> None:
    last_err: Exception | None = None
    for attempt in range(0, 6):
        try:
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
            wait_s = min(8.0, float(0.5 * (2**attempt)))
            time.sleep(wait_s)
    if last_err is not None:
        raise last_err


def _env_str(k: str, default: str = "") -> str:
    v = os.getenv(k)
    if v is None:
        return default
    return str(v)


def _env_flag(k: str, default: bool = False) -> bool:
    v = os.getenv(k)
    if v is None:
        return bool(default)
    return str(v).strip().lower() in ("1", "true", "yes", "y")


def _ordered_row(obj: dict) -> dict:
    # Ensure `image` then `image_id` are first keys.
    out = {"image": obj.get("image", ""), "image_id": obj.get("image_id", "")}
    for k, v in obj.items():
        if k in out:
            continue
        out[k] = v
    return out


def _normalize_to_image_url(*, repo_id: str, repo_type: str, obj: dict) -> str:
    # Prefer existing `image` then `image_url`, then derive from repo-relative path.
    for k in ("image", "image_url"):
        try:
            v = obj.get(k)
            if v is None:
                continue
            s = str(v).strip()
            if s:
                return s
        except Exception:
            continue

    # Try to derive from image_path (repo-relative path).
    try:
        p = str(obj.get("image_path") or "").strip().lstrip("/")
        if p:
            return hf_utils.build_resolve_url(repo_id, p, repo_type=repo_type)
    except Exception:
        pass

    # Default layout: <HF_SUBDIR>/<image_id>/<image_id>.jpg
    try:
        subdir = str(os.getenv("HF_SUBDIR", "unsplash") or "unsplash").strip().strip("/")
        pid = str(obj.get("image_id") or "").strip()
        if pid:
            rel = "/".join([p for p in [subdir, pid, f"{pid}.jpg"] if p])
            return hf_utils.build_resolve_url(repo_id, rel, repo_type=repo_type)
    except Exception:
        pass

    return ""


def migrate_file(*, repo_id: str, repo_type: str, src_path: str, dst_path: str, overwrite: bool) -> dict[str, Any]:
    total = 0
    changed = 0
    kept = 0
    bad = 0

    with open(src_path, "r", encoding="utf-8") as rf, open(dst_path, "w", encoding="utf-8") as wf:
        for line in rf:
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                obj = json.loads(s)
            except Exception:
                bad += 1
                continue
            if not isinstance(obj, dict):
                bad += 1
                continue

            pid = str(obj.get("image_id") or "").strip()
            if not pid:
                bad += 1
                continue
            obj["image_id"] = pid

            cur_image = str(obj.get("image") or "").strip()
            if (not cur_image) or overwrite:
                new_image = _normalize_to_image_url(repo_id=repo_id, repo_type=repo_type, obj=obj)
                if new_image and (new_image != cur_image):
                    obj["image"] = new_image
                    changed += 1
                else:
                    kept += 1
            else:
                kept += 1

            obj = _ordered_row(obj)
            wf.write(json.dumps(obj, ensure_ascii=False) + "\n")

    return {
        "total": int(total),
        "changed": int(changed),
        "kept": int(kept),
        "bad": int(bad),
        "src": str(src_path),
        "dst": str(dst_path),
    }


def main() -> int:
    repo_id = _env_str("HF_REPO_ID", "eatmorefruit/sharp-ply-share").strip()
    repo_type = _env_str("HF_REPO_TYPE", "dataset").strip() or "dataset"
    index_path = _env_str("HF_INDEX_REPO_PATH", "data/train.jsonl").strip().lstrip("/")

    overwrite = _env_flag("MIGRATE_OVERWRITE", False)
    upload = _env_flag("MIGRATE_UPLOAD", False)

    token = (_env_str("HF_TOKEN", "") or _env_str("HUGGINGFACE_HUB_TOKEN", "")).strip()
    if not token:
        print("Missing HF_TOKEN/HUGGINGFACE_HUB_TOKEN", file=sys.stderr)
        return 2

    src_local = hf_hub_download(repo_id=repo_id, repo_type=repo_type, filename=index_path, token=token)

    save_dir = os.path.abspath(os.path.join(os.getcwd(), "runs", "index_migrations"))
    os.makedirs(save_dir, exist_ok=True)
    dst_local = os.path.join(save_dir, os.path.basename(index_path) + ".migrated.jsonl")

    stats = migrate_file(repo_id=repo_id, repo_type=repo_type, src_path=src_local, dst_path=dst_local, overwrite=overwrite)
    print("Migration stats:")
    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if not upload:
        print("Dry-run complete. Set MIGRATE_UPLOAD=1 to upload the migrated index back to HF.")
        return 0

    api = HfApi(token=token)
    ops = [CommitOperationAdd(path_in_repo=index_path, path_or_fileobj=dst_local)]
    try:
        _create_commit_retry(
            api,
            repo_id=repo_id,
            repo_type=repo_type,
            operations=ops,
            commit_message="migrate index: add image preview url field",
        )
    except Exception as e:
        # In case the repo has moved forward or requires PR-based updates, fall back to a PR.
        if not hf_utils.should_retry_with_pr(e):
            raise
        api.create_commit(
            repo_id=repo_id,
            repo_type=repo_type,
            operations=ops,
            commit_message="migrate index: add image preview url field",
            create_pr=True,
        )
    print(f"Uploaded migrated index to HF: {repo_type}:{repo_id} | path={index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
