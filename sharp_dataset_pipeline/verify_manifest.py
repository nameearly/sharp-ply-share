import argparse
import json
import os
import sys
from dataclasses import dataclass

from . import hf_utils


@dataclass
class VerifyStats:
    total: int = 0
    ok: int = 0
    missing: int = 0
    size_mismatch: int = 0
    sha_mismatch: int = 0
    bad_lines: int = 0


def _load_manifest_items(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
            except Exception:
                yield None
                continue
            if not isinstance(obj, dict):
                yield None
                continue
            p = str(obj.get("path") or "").strip().lstrip("/")
            if not p:
                yield None
                continue
            b = obj.get("bytes")
            try:
                b2 = int(b) if b is not None else None
            except Exception:
                b2 = None
            sha = str(obj.get("sha256") or "").strip().lower()
            yield {"path": p, "bytes": b2, "sha256": sha}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify downloaded assets against data/manifest.jsonl")
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--root", required=True, help="Root directory that contains the repo files (e.g. a snapshot of the dataset repo)")
    parser.add_argument("--fail-fast", action="store_true", help="Exit immediately on first mismatch")
    parser.add_argument("--limit", type=int, default=0, help="Only check first N items (0 = no limit)")
    parser.add_argument("--verbose", action="store_true", help="Print per-file results")
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="Skip sha256 verification (only check existence and size). Useful for quick checks.",
    )

    args = parser.parse_args(argv)

    manifest_path = os.path.abspath(str(args.manifest))
    root = os.path.abspath(str(args.root))

    if not os.path.isfile(manifest_path):
        print(f"manifest not found: {manifest_path}")
        return 2
    if not os.path.isdir(root):
        print(f"root is not a directory: {root}")
        return 2

    st = VerifyStats()
    checked = 0

    for it in _load_manifest_items(manifest_path):
        if args.limit and checked >= int(args.limit):
            break
        checked += 1
        if it is None:
            st.bad_lines += 1
            if args.verbose:
                print(f"[BAD] invalid manifest line | n={checked}")
            if args.fail_fast:
                return 1
            continue

        rel_path = str(it.get("path") or "").strip().lstrip("/")
        exp_bytes = it.get("bytes")
        exp_sha = str(it.get("sha256") or "").strip().lower()

        st.total += 1
        abs_path = os.path.join(root, rel_path)

        if not os.path.isfile(abs_path):
            st.missing += 1
            if args.verbose:
                print(f"[MISS] {rel_path}")
            if args.fail_fast:
                return 1
            continue

        size_ok = True
        if exp_bytes is not None and int(exp_bytes) > 0:
            got_bytes = hf_utils.file_size(abs_path)
            if int(got_bytes) != int(exp_bytes):
                size_ok = False
                st.size_mismatch += 1
                if args.verbose:
                    print(f"[SIZE] {rel_path} | expected={int(exp_bytes)} got={int(got_bytes)}")
                if args.fail_fast:
                    return 1

        sha_ok = True
        if (not args.skip_sha256) and exp_sha:
            got_sha = hf_utils.sha256_file(abs_path)
            if str(got_sha).lower() != str(exp_sha).lower():
                sha_ok = False
                st.sha_mismatch += 1
                if args.verbose:
                    print(f"[SHA ] {rel_path} | expected={exp_sha} got={got_sha}")
                if args.fail_fast:
                    return 1

        if size_ok and sha_ok:
            st.ok += 1
            if args.verbose:
                print(f"[OK  ] {rel_path}")

    print("\nSummary")
    print(f"- total: {st.total}")
    print(f"- ok: {st.ok}")
    print(f"- missing: {st.missing}")
    print(f"- size_mismatch: {st.size_mismatch}")
    print(f"- sha_mismatch: {st.sha_mismatch}")
    print(f"- bad_lines: {st.bad_lines}")

    if (st.missing + st.size_mismatch + st.sha_mismatch + st.bad_lines) > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
