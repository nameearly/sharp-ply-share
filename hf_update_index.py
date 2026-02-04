import argparse
import os
import sys
import time
import tempfile
from pathlib import Path


def _find_latest_train_jsonl(repo_root: Path) -> Path | None:
    runs_dir = repo_root / "runs"
    if not runs_dir.exists():
        return None

    candidates: list[Path] = []
    try:
        for p in runs_dir.rglob("train.jsonl"):
            if p.is_file():
                candidates.append(p)
    except Exception:
        return None

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _should_retry_with_pr(err: Exception) -> bool:
    try:
        s = str(err)
        return ("create_pr=1" in s) or ("create_pr" in s and "Pull Request" in s)
    except Exception:
        return False


def _detect_format(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            while True:
                ch = f.read(1)
                if not ch:
                    break
                if ch.isspace():
                    continue
                if ch == "[":
                    return "json_array"
                return "jsonl"
    except Exception:
        return "jsonl"


def _rewrite_index_file(*, input_path: Path, output_path: Path, fmt: str, normalizer) -> tuple[int, int]:
    seen: set[str] = set()
    kept = 0
    dropped = 0

    def _handle_obj(obj):
        nonlocal kept, dropped
        norm = None
        try:
            norm = normalizer._normalize_row(obj)  # noqa: SLF001
        except Exception:
            norm = None
        if not norm:
            dropped += 1
            return None
        pid = str(norm.get("image_id") or "").strip()
        if not pid or pid in seen:
            dropped += 1
            return None
        seen.add(pid)
        kept += 1
        return norm

    if fmt == "json_array":
        import json

        data = None
        with open(input_path, "r", encoding="utf-8") as rf:
            data = json.load(rf)
        if not isinstance(data, list):
            raise ValueError("remote index is json but not a list")
        out_list = []
        for obj in data:
            if not isinstance(obj, dict):
                dropped += 1
                continue
            norm = _handle_obj(obj)
            if norm:
                out_list.append(norm)
        with open(output_path, "w", encoding="utf-8") as wf:
            json.dump(out_list, wf, ensure_ascii=False)
            wf.write("\n")
        return kept, dropped

    import json

    with open(input_path, "r", encoding="utf-8") as rf, open(output_path, "w", encoding="utf-8") as wf:
        for line in rf:
            s = line.strip()
            if not s:
                dropped += 1
                continue
            try:
                obj = json.loads(s)
            except Exception:
                dropped += 1
                continue
            if not isinstance(obj, dict):
                dropped += 1
                continue
            norm = _handle_obj(obj)
            if not norm:
                continue
            wf.write(json.dumps(norm, ensure_ascii=False) + "\n")
    return kept, dropped


def _preview_first_row(path: Path, *, fmt: str) -> None:
    try:
        import json

        obj = None
        if fmt == "json_array":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list) and data:
                obj = data[0] if isinstance(data[0], dict) else None
        else:
            with open(path, "r", encoding="utf-8") as f:
                line = f.readline().strip()
            if line:
                v = json.loads(line)
                obj = v if isinstance(v, dict) else None

        if not isinstance(obj, dict):
            print("Preview: no first row available")
            return

        keys = sorted(list(obj.keys()))
        def _has(k: str) -> str:
            return "Y" if k in obj else "-"

        print("Preview keys (first row):")
        print("  " + ", ".join(keys))
        print(
            "Preview presence | "
            f"tags={_has('tags')} topics={_has('topics')} tags_text={_has('tags_text')} topics_text={_has('topics_text')} "
            f"desc={_has('description')} alt_desc={_has('alt_description')} "
            f"image_url={_has('image_url')} ply_url={_has('ply_url')} spz_url={_has('spz_url')} "
            f"gsplat_url={_has('gsplat_url')} unsplash_url={_has('unsplash_url')} user_name={_has('user_name')}"
        )
    except Exception as e:
        print(f"Preview failed: {e}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Rewrite & upload HF index (download -> normalize -> upload) in a single commit")
    ap.add_argument(
        "--mode",
        default="remote",
        choices=["remote", "local"],
        help="remote: download HF index then rewrite & upload. local: upload a local file as-is.",
    )
    ap.add_argument("--local", default="", help="Local train.jsonl path (used in --mode local). If empty, picks newest runs/**/train.jsonl")
    ap.add_argument("--repo-id", default=os.getenv("HF_REPO_ID", "").strip(), help="HF repo id, e.g. user/repo")
    ap.add_argument("--repo-type", default=os.getenv("HF_REPO_TYPE", "dataset").strip().lower(), help="dataset | model")
    ap.add_argument(
        "--repo-path",
        default=os.getenv("HF_INDEX_REPO_PATH", "data/train.jsonl").strip().lstrip("/"),
        help="Target path in repo, default data/train.jsonl",
    )
    ap.add_argument(
        "--format",
        default="auto",
        choices=["auto", "jsonl", "json_array"],
        help="Index file format. auto detects json array vs jsonl.",
    )
    ap.add_argument(
        "--message",
        default="",
        help="Commit message (optional). Default: index update <ts>",
    )
    ap.add_argument("--dry-run", action="store_true", help="Rewrite locally but do not upload")
    ap.add_argument(
        "--token",
        default=(os.getenv("HF_TOKEN", "") or os.getenv("HUGGINGFACE_HUB_TOKEN", "")),
        help="HF token (optional; will also use default HF auth if empty)",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent

    if not args.repo_id:
        print("ERROR: HF repo id is empty. Set HF_REPO_ID or pass --repo-id.", file=sys.stderr)
        return 2

    repo_path = str(args.repo_path or "").strip().lstrip("/")
    if not repo_path:
        print("ERROR: repo path is empty. Set HF_INDEX_REPO_PATH or pass --repo-path.", file=sys.stderr)
        return 2

    repo_type = str(args.repo_type or "dataset").strip().lower()
    if repo_type not in ("dataset", "model"):
        print("ERROR: repo_type must be dataset|model", file=sys.stderr)
        return 2

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except Exception as e:
        print(f"ERROR: huggingface_hub not available: {e}", file=sys.stderr)
        return 2

    token = str(args.token or "").strip() or None

    local_path = None
    rewritten_path = None
    temp_dir_obj = None
    try:
        if str(args.mode) == "local":
            local_path = Path(args.local).expanduser().resolve() if str(args.local).strip() else None
            if local_path is None:
                local_path = _find_latest_train_jsonl(repo_root)
            if local_path is None or (not local_path.exists()):
                print("ERROR: cannot find local train.jsonl. Use --local to specify.", file=sys.stderr)
                return 2

            try:
                if str(args.format) == "auto":
                    fmt = _detect_format(local_path)
                else:
                    fmt = str(args.format)
                _preview_first_row(local_path, fmt=fmt)
            except Exception:
                pass
        else:
            try:
                from huggingface_hub import hf_hub_download
            except Exception as e:
                print(f"ERROR: huggingface_hub.hf_hub_download not available: {e}", file=sys.stderr)
                return 2

            temp_dir_obj = tempfile.TemporaryDirectory(prefix="hf_index_rewrite_")
            temp_dir = Path(temp_dir_obj.name)

            try:
                remote_local = hf_hub_download(repo_id=args.repo_id, repo_type=repo_type, filename=repo_path, token=token)
            except TypeError:
                remote_local = hf_hub_download(
                    repo_id=args.repo_id,
                    repo_type=repo_type,
                    filename=repo_path,
                    use_auth_token=token,
                )

            remote_local = Path(str(remote_local)).resolve()
            local_path = temp_dir / (Path(repo_path).name or "train.jsonl")
            try:
                local_path.write_bytes(remote_local.read_bytes())
            except Exception:
                import shutil

                shutil.copyfile(str(remote_local), str(local_path))

            try:
                from sharp_dataset_pipeline.index_sync import IndexSync
            except Exception as e:
                print(f"ERROR: cannot import IndexSync: {e}", file=sys.stderr)
                return 2

            normalizer = IndexSync(
                "",
                repo_type=repo_type,
                repo_path=repo_path,
                save_dir=str(temp_dir),
                hf_upload=False,
                hf_index_flush_every=999999,
                hf_index_flush_secs=999999.0,
                hf_index_refresh_secs=999999.0,
                debug_fn=None,
            )

            if str(args.format) == "auto":
                fmt = _detect_format(local_path)
            else:
                fmt = str(args.format)
            rewritten_path = temp_dir / ("rewritten_" + (Path(repo_path).name or "train.jsonl"))
            kept, dropped = _rewrite_index_file(input_path=local_path, output_path=rewritten_path, fmt=fmt, normalizer=normalizer)
            print(f"Rewrite done | kept={kept} dropped={dropped} fmt={fmt}")
            _preview_first_row(rewritten_path, fmt=fmt)
            local_path = rewritten_path
    finally:
        # temp_dir_obj is intentionally kept alive until after upload
        pass

    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    msg = str(args.message or "").strip() or f"index update {ts}"

    api = HfApi(token=token)

    ops = [CommitOperationAdd(path_in_repo=repo_path, path_or_fileobj=str(local_path))]

    if bool(args.dry_run):
        print("Dry-run: rewrite finished, skipping upload.")
        print(f"  local: {local_path}")
        return 0

    print("Uploading index to Hugging Face...")
    print(f"  local: {local_path}")
    print(f"  repo:  {repo_type}:{args.repo_id}")
    print(f"  path:  {repo_path}")

    try:
        api.create_commit(
            repo_id=args.repo_id,
            repo_type=repo_type,
            operations=ops,
            commit_message=msg,
        )
    except Exception as e:
        if not _should_retry_with_pr(e):
            raise
        api.create_commit(
            repo_id=args.repo_id,
            repo_type=repo_type,
            operations=ops,
            commit_message=msg,
            create_pr=True,
        )

    print("Done.")
    try:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
