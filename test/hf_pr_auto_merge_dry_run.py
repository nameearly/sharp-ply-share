import os
import re
import sys

from huggingface_hub import HfApi

try:
    from huggingface_hub import HfFolder
except Exception:
    HfFolder = None


def _env_str(k: str, default: str = "") -> str:
    v = os.getenv(k)
    if v is None:
        return default
    return str(v)


def _get_token() -> str:
    token = (_env_str("HF_TOKEN") or _env_str("HUGGINGFACE_HUB_TOKEN")).strip()
    if token:
        return token
    if HfFolder is None:
        return ""
    try:
        return (HfFolder.get_token() or "").strip()
    except Exception:
        return ""


def is_additive_only_diff(diff: str) -> bool:
    """Return True if the PR diff looks like it only adds new files.

    Conservative rules:
    - Each file section must contain `new file mode`.
    - Reject if any section contains deletions from an existing file (`--- a/...`).
    - Reject if any section contains `deleted file mode` or renames.
    """

    if not diff or not diff.strip():
        return False

    parts = diff.split("diff --git ")
    file_parts = [p for p in parts[1:] if p.strip()]
    if not file_parts:
        return False

    for p in file_parts:
        if "deleted file mode" in p:
            return False
        if "rename from" in p or "rename to" in p:
            return False
        if "new file mode" not in p:
            return False

        # For new files, old side must be /dev/null
        if re.search(r"^---\s+a/", p, flags=re.MULTILINE):
            return False
        if not re.search(r"^---\s+/dev/null\s*$", p, flags=re.MULTILINE):
            return False
        if not re.search(r"^\+\+\+\s+b/", p, flags=re.MULTILINE):
            return False

    return True


def main() -> int:
    repo_id = _env_str("HF_REPO_ID", "eatmorefruit/sharp-ply-share").strip()
    repo_type = (_env_str("HF_REPO_TYPE", "dataset").strip() or "dataset")

    token = _get_token()
    if not token:
        print("No HF token found. Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or run: hf auth login", file=sys.stderr)
        return 2

    api = HfApi(token=token)

    prs = list(
        api.get_repo_discussions(
            repo_id=repo_id,
            repo_type=repo_type,
            discussion_type="pull_request",
            discussion_status="open",
        )
    )

    print(f"Repo: {repo_type}:{repo_id} | open_prs={len(prs)}")

    would_merge = []
    skipped = []

    for d in prs:
        num = int(getattr(d, "num"))
        title = str(getattr(d, "title", "") or "")
        author = str(getattr(d, "author", "") or "")

        try:
            details = api.get_discussion_details(repo_id=repo_id, repo_type=repo_type, discussion_num=num)
        except Exception as e:
            skipped.append((num, title, f"details_error:{e}"))
            continue

        conflicting = getattr(details, "conflicting_files", None)
        if conflicting:
            skipped.append((num, title, f"conflicts:{len(conflicting)}"))
            continue

        diff = str(getattr(details, "diff", "") or "")
        if not is_additive_only_diff(diff):
            skipped.append((num, title, "non_additive"))
            continue

        would_merge.append((num, title, author))

    for num, title, author in would_merge:
        print(f"WOULD_MERGE PR#{num} by {author}: {title}")

    for num, title, reason in skipped:
        print(f"SKIP PR#{num}: {title} | reason={reason}")

    print(f"Summary | would_merge={len(would_merge)} skipped={len(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
