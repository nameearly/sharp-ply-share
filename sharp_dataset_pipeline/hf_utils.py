import os


def build_resolve_url(repo_id: str, rel_path: str, *, repo_type: str) -> str:
    rel_path = str(rel_path).lstrip("/")
    if str(repo_type).strip().lower() == "dataset":
        return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel_path}"
    return f"https://huggingface.co/{repo_id}/resolve/main/{rel_path}"


def ensure_repo(repo_id: str, *, repo_type: str, debug_fn=None) -> bool:
    try:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True, private=False)
        return True
    except Exception as e:
        try:
            if debug_fn:
                debug_fn(f"HF create_repo 失败 | err={str(e)}")
        except Exception:
            pass
        return False


def should_retry_with_pr(err: Exception) -> bool:
    try:
        s = str(err)
        return ("create_pr=1" in s) or ("create_pr" in s and "Pull Request" in s)
    except Exception:
        return False
