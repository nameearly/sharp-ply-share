import argparse
import json
import math
import pathlib
import statistics
from collections import defaultdict


def _pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(float(x) for x in values)
    n = len(xs)
    if n == 1:
        return xs[0]
    k = (n - 1) * (float(p) / 100.0)
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return xs[f]
    return xs[f] * (c - k) + xs[c] * (k - f)


def _clamp_int(x: float, lo: int, hi: int) -> int:
    try:
        v = int(math.ceil(float(x)))
    except Exception:
        v = int(lo)
    return max(int(lo), min(int(v), int(hi)))


def _recommend_timeout_s(*, p95: float | None, p99: float | None, mx: float | None, floor_s: int, ceil_s: int) -> int:
    base = 0.0
    for v in (p99, p95, mx):
        if v is None:
            continue
        base = max(base, float(v))
    # heuristic headroom:
    # - protect against rare stalls and future variance
    # - but keep reasonably bounded to detect true hangs
    target = max(float(floor_s), base * 6.0)
    return _clamp_int(target, int(floor_s), int(ceil_s))


def _print_stage(name: str, values: list[float]) -> dict:
    out = {"n": len(values)}
    if not values:
        print(f"{name}: n=0")
        return out

    mean = statistics.mean(values)
    p50 = _pct(values, 50)
    p95 = _pct(values, 95)
    p99 = _pct(values, 99)
    mx = max(values)

    out.update({"mean": mean, "p50": p50, "p95": p95, "p99": p99, "max": mx})

    def _fmt(x: float | None) -> str:
        if x is None:
            return "-"
        return f"{x:.3f}"

    print(
        f"{name}: n={len(values)} mean={_fmt(mean)} p50={_fmt(p50)} p95={_fmt(p95)} p99={_fmt(p99)} max={_fmt(mx)}"
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="?", default="profile.jsonl")
    ap.add_argument("--print-env", action="store_true", help="print recommended PowerShell env lines")
    args = ap.parse_args()

    path = pathlib.Path(args.jsonl)
    if not path.is_file():
        raise SystemExit(f"profile jsonl not found: {path}")

    stage_s: dict[str, list[float]] = defaultdict(list)

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        ev = str(obj.get("event") or "")
        s = obj.get("s")
        if isinstance(s, (int, float)):
            stage_s[ev].append(float(s))

    print(f"Profile: {path} (events={len(stage_s)})")

    predict = _print_stage("predict_done", stage_s.get("predict_done", []))
    spz = _print_stage("spz_export_done", stage_s.get("spz_export_done", []))

    # commit metrics differ when batching is enabled
    commit = _print_stage("hf_commit_done", stage_s.get("hf_commit_done", []))
    commit_batch = _print_stage("hf_commit_batch_done", stage_s.get("hf_commit_batch_done", []))

    gs_total = _print_stage("gsplat_upload_total", stage_s.get("gsplat_upload_total", []))

    # Recommendations (heuristic, conservative)
    sharp_to = _recommend_timeout_s(
        p95=predict.get("p95"),
        p99=predict.get("p99"),
        mx=predict.get("max"),
        floor_s=600,
        ceil_s=7200,
    )
    spz_to = _recommend_timeout_s(
        p95=spz.get("p95"),
        p99=spz.get("p99"),
        mx=spz.get("max"),
        floor_s=300,
        ceil_s=3600,
    )
    gs_total_to = _recommend_timeout_s(
        p95=gs_total.get("p95"),
        p99=gs_total.get("p99"),
        mx=gs_total.get("max"),
        floor_s=900,
        ceil_s=7200,
    )

    # For per-request tRPC, we don't have per-chunk metrics; use a stable default.
    trpc_timeout = 180
    trpc_connect_timeout = 20

    # Upload batching: if you already enabled HF_UPLOAD_BATCH_SIZE, commit_done may disappear.
    batch_size = 8
    batch_wait_ms = 500

    chunk_mb = 16

    print("\nRecommended (next run):")
    print(f"- HF_UPLOAD_BATCH_SIZE={batch_size}")
    print(f"- HF_UPLOAD_BATCH_WAIT_MS={batch_wait_ms}")
    print(f"- SHARP_PREDICT_TIMEOUT_SECS_CUDA={sharp_to}")
    print(f"- SPZ_TIMEOUT_SECS_CUDA={spz_to}")
    print(f"- GSPLAT_TOTAL_TIMEOUT_SECS={gs_total_to}")
    print(f"- GSPLAT_CHUNK_SIZE_MB={chunk_mb}")
    print(f"- GSPLAT_TRPC_CONNECT_TIMEOUT_SECS={trpc_connect_timeout}")
    print(f"- GSPLAT_TRPC_TIMEOUT_SECS={trpc_timeout}")

    if args.print_env:
        print("\nPowerShell: ")
        print(f"$env:HF_UPLOAD_BATCH_SIZE=\"{batch_size}\"")
        print(f"$env:HF_UPLOAD_BATCH_WAIT_MS=\"{batch_wait_ms}\"")
        print(f"$env:SHARP_PREDICT_TIMEOUT_SECS_CUDA=\"{sharp_to}\"")
        print(f"$env:SPZ_TIMEOUT_SECS_CUDA=\"{spz_to}\"")
        print(f"$env:GSPLAT_TOTAL_TIMEOUT_SECS=\"{gs_total_to}\"")
        print(f"$env:GSPLAT_CHUNK_SIZE_MB=\"{chunk_mb}\"")
        print(f"$env:GSPLAT_TRPC_CONNECT_TIMEOUT_SECS=\"{trpc_connect_timeout}\"")
        print(f"$env:GSPLAT_TRPC_TIMEOUT_SECS=\"{trpc_timeout}\"")

    # Encourage larger sample sizes for stable percentiles
    if len(stage_s.get("predict_done", [])) < 50:
        print("\nNote: sample size < 50; consider MAX_IMAGES=100 for more stable p95/p99.")

    # Provide a hint when batching is enabled
    if commit_batch.get("n", 0) <= 0 and commit.get("n", 0) > 0:
        pass
    elif commit_batch.get("n", 0) > 0:
        print("Note: hf_commit_batch_done present => batching enabled; use that metric for commit tuning.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
