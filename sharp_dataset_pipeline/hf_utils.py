import os


def file_size(path: str) -> int:
    try:
        return int(os.path.getsize(str(path)))
    except Exception:
        return 0


def sha256_file(path: str) -> str:
    try:
        import hashlib

        h = hashlib.sha256()
        with open(str(path), "rb") as f:
            while True:
                b = f.read(1024 * 1024)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return ""


def env_str(name: str, default: str = "") -> str:
    try:
        v = os.getenv(name)
        if v is None:
            return str(default)
        return str(v)
    except Exception:
        return str(default)


def env_int(name: str, default: int = 0) -> int:
    try:
        v = os.getenv(name)
        if v is None:
            return int(default)
        return int(str(v).strip())
    except Exception:
        return int(default)


def env_float(name: str, default: float = 0.0) -> float:
    try:
        v = os.getenv(name)
        if v is None:
            return float(default)
        return float(str(v).strip())
    except Exception:
        return float(default)


def env_flag(name: str, default: bool = False) -> bool:
    try:
        v = os.getenv(name)
        if v is None:
            return bool(default)
        s = str(v).strip().lower()
        if not s:
            return bool(default)
        return s in ("1", "true", "yes", "y", "on")
    except Exception:
        return bool(default)


def build_resolve_url(repo_id: str, rel_path: str, *, repo_type: str) -> str:
    rel_path = str(rel_path).lstrip("/")
    if str(repo_type).strip().lower() == "dataset":
        return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{rel_path}"
    return f"https://huggingface.co/{repo_id}/resolve/main/{rel_path}"


def ensure_repo(repo_id: str, *, repo_type: str, debug_fn=None) -> bool:
    try:
        from huggingface_hub import HfApi

        token = str(
            os.getenv("HF_TOKEN", "")
            or os.getenv("HUGGINGFACE_HUB_TOKEN", "")
            or os.getenv("HUGGING_FACE_HUB_TOKEN", "")
            or ""
        ).strip()
        api = HfApi(token=token) if token else HfApi()
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
