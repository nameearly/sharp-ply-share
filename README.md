---
license: other
configs:
- config_name: default
  data_files:
  - split: train
    path: "data/train.jsonl"
---
## Data files

- The Dataset Viewer is configured to read `data/train.jsonl` (see the `configs:` section in the YAML header above).
- The actual assets (JPG / PLY / SPZ) are stored under `unsplash/<image_id>/`.

## Links

- GitHub (pipeline code): https://github.com/nameearly/sharp-ply-share
- Hugging Face (dataset): https://huggingface.co/datasets/eatmorefruit/sharp-ply-share

## Pipeline coordination / dedup (for multi-client runs)

- Candidate limit: `MAX_CANDIDATES` limits how many photos are considered; `MAX_IMAGES` limits how many are actually downloaded/processed.
- Remote done check: `HF_DONE_BACKEND=index` uses the HF index file (`data/train.jsonl`) as a local in-memory done set, and periodically refreshes it for collaborator correctness (`HF_INDEX_REFRESH_SECS`).
- Range locks (list + oldest): range coordination is stored on HF under `ranges/locks`, `ranges/done`, and `ranges/progress`.
- Range done prefix: `ranges/progress/done_prefix.json` is used to avoid repo-wide listings of `ranges/done/`.
- Ant-style range selection (optional): `ANT_ENABLED=1` with `ANT_CANDIDATE_RANGES`, `ANT_EPSILON`, `ANT_FRESH_SECS` to reduce contention across multiple clients.
- HF upload batching (optional): `HF_UPLOAD_BATCH_SIZE=4` is recommended for throughput; small contributors can use `HF_UPLOAD_BATCH_SIZE=1`. `HF_UPLOAD_BATCH_WAIT_MS` controls the micro-batching wait window.

## Data fields

Each row in `data/train.jsonl` is a JSON object with stable (string) types for fields that commonly drift (to keep the Dataset Viewer working reliably).

| Field | Type | Description |
| --- | --- | --- |
| `image_id` | `string` | Unsplash photo id. Also used as the directory name for assets. |
| `image_url` | `string` | Hugging Face `resolve` URL for the JPG. |
| `ply_url` | `string` | Hugging Face `resolve` URL for the Gaussian Splat PLY (ml-sharp export). |
| `spz_url` | `string` | Hugging Face `resolve` URL for the SPZ (may be empty if not exported). |
| `gsplat_url` | `string` | Public viewer URL on `https://gsplat.org/viewer/<shareId>` (may be empty if not uploaded). |
| `gsplat_share_id` | `string` | Share id on gsplat.org (may be empty). |
| `gsplat_order_id` | `string` | Order id on gsplat.org (may be empty). |
| `gsplat_model_file_url` | `string` | gsplat.org internal file URL returned by upload endpoints (may be empty). |
| `tags` | `string` | Space-separated tags (derived from Unsplash tags). |
| `topics` | `string` | Space-separated topics (often empty). |
| `tags_text` | `string` | Same as `tags` (kept for backwards compatibility / full-text search). |
| `topics_text` | `string` | Same as `topics`. |
| `alt_description` | `string` | Unsplash `alt_description` (empty string if missing). |
| `description` | `string` | Unsplash `description` (empty string if missing). |
| `unsplash_id` | `string` | Same as `image_id`. |
| `unsplash_url` | `string` | Unsplash photo page URL. |
| `created_at` | `string` | Unsplash `created_at` timestamp (ISO8601). |
| `user_username` | `string` | Unsplash author username. |
| `user_name` | `string` | Unsplash author display name. |

## Quick usage

- **Load metadata**: download `data/train.jsonl` and parse JSONL.
- **Fetch assets**: use `image_url` / `ply_url` / `spz_url` (if non-empty) directly, or use `hf_hub_download` with the corresponding repo path.

`<image_id>.ply`: made by ml-sharp (https://github.com/apple/ml-sharp) from the corresponding Unsplash photo page `https://unsplash.com/photos/<image_id>`.

Unsplash photos are provided under the **Unsplash License** (not CC0): https://unsplash.com/license

TL;DR: you can do pretty much anything with Unsplash images (including commercial use), except:

- You can’t sell an image without significant modification.
- You can’t compile images from Unsplash to replicate a similar or competing service.

Attribution is not required, but appreciated.

One image to one ply, not high-quality, just for fun.

## Pipeline notes (gsplat)

- The pipeline can optionally upload a (potentially reduced) PLY to https://gsplat.org and record a public viewer link in `gsplat_url`.
- By default it uploads the original PLY.
- You can enable generating a smaller `*.small.gsplat.ply` via `splat-transform` by setting `GSPLAT_USE_SMALL_PLY=1`.

## Pipeline notes (Ctrl+C / pause / stop)

- `Ctrl+C` toggles cooperative pause/resume (creates or deletes `PAUSE`).
- `Ctrl+D` requests stop/exit (creates `STOP`). On Windows consoles, `Ctrl+Z` is also accepted as an exit shortcut.
- Unexpected exceptions (especially in worker threads) are logged with full tracebacks to simplify debugging.
