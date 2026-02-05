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
- The `image` field in `data/train.jsonl` stores the full HF `resolve` URL of the JPG for Dataset Viewer previews. `image_id` stays as the stable identifier.

## Links

- GitHub (pipeline code): https://github.com/nameearly/sharp-ply-share
- Hugging Face (dataset): https://huggingface.co/datasets/eatmorefruit/sharp-ply-share

## Pipeline coordination / dedup (for multi-client runs)

- Prefetch buffer: `MAX_CANDIDATES` controls how many images the downloader will try to keep queued for inference (same as `DOWNLOAD_QUEUE_MAX` unless you override it explicitly). `MAX_IMAGES` limits how many are actually downloaded/processed.
- Remote done check: `HF_DONE_BACKEND=index` uses the HF index file (`data/train.jsonl`) as a local in-memory done set, and periodically refreshes it for collaborator correctness (`HF_INDEX_REFRESH_SECS`).
- Range locks (list + oldest): range coordination is stored on HF under `ranges/locks`, `ranges/done`, and `ranges/progress`.
- Range done prefix: `ranges/progress/done_prefix.json` is used to avoid repo-wide listings of `ranges/done/`.
- Ant-style range selection (optional): `ANT_ENABLED=1` with `ANT_CANDIDATE_RANGES`, `ANT_EPSILON`, `ANT_FRESH_SECS` to reduce contention across multiple clients.
- HF upload batching (optional): `HF_UPLOAD_BATCH_SIZE=4` is recommended for throughput; small contributors can use `HF_UPLOAD_BATCH_SIZE=1`. `HF_UPLOAD_BATCH_WAIT_MS` controls the micro-batching wait window.

## Data fields

Each row in `data/train.jsonl` is a JSON object with stable (string) types for fields that commonly drift (to keep the Dataset Viewer working reliably).

| Field | Type | Description |
| --- | --- | --- |
| `image` | `string` | Full HF resolve URL for the JPG (used by Dataset Viewer to preview images). |
| `image_id` | `string` | Unsplash photo id. Also used as the directory name for assets. |
| `gsplat_share_id` | `string` | Share id on gsplat.org (may be empty). |
| `gsplat_order_id` | `string` | Order id on gsplat.org (may be empty). |
| `gsplat_model_file_url` | `string` | gsplat.org model file token (normalized): for example `1770129991964_T8LMLFAy` (may be empty). |
| `tags` | `string` | Space-separated tags (derived from Unsplash tags). |
| `topics` | `string` | Space-separated topics (often empty). |
| `tags_text` | `string` | Same as `tags` (kept for backwards compatibility / full-text search). |
| `topics_text` | `string` | Same as `topics`. |
| `alt_description` | `string` | Unsplash `alt_description` (empty string if missing). |
| `description` | `string` | Unsplash `description` (empty string if missing). |
| `created_at` | `string` | Unsplash `created_at` timestamp (ISO8601). |
| `user_username` | `string` | Unsplash author username. |

## Quick usage

- **Load metadata**: download `data/train.jsonl` and parse JSONL.

- **Fetch assets**: assets are stored under `unsplash/<image_id>/`:
  - `unsplash/<image_id>/<image_id>.jpg`
  - `unsplash/<image_id>/<image_id>.ply`
  - `unsplash/<image_id>/<image_id>.spz`

You can reconstruct URLs from ids:

- Unsplash photo page: `https://unsplash.com/photos/<image_id>`
- HF resolve URL (dataset): `https://huggingface.co/datasets/eatmorefruit/sharp-ply-share/resolve/main/unsplash/<image_id>/<image_id>.<ext>`
- gsplat viewer URL (if `gsplat_share_id` present): `https://gsplat.org/viewer/<gsplat_share_id>`

If you need the original gsplat share file path, reconstruct:

- `gsplat_model_file_url_raw`: `/share/file/<gsplat_model_file_url>.ply`

`<image_id>.ply`: made by ml-sharp (https://github.com/apple/ml-sharp) from the corresponding Unsplash photo page `https://unsplash.com/photos/<image_id>`.

Unsplash photos are provided under the **Unsplash License** (not CC0): https://unsplash.com/license

TL;DR: you can do pretty much anything with Unsplash images (including commercial use), except:

- You can’t sell an image without significant modification.
- You can’t compile images from Unsplash to replicate a similar or competing service.

One image to one ply, not high-quality, just for fun.

## Pipeline notes (gsplat)

- The pipeline can optionally upload a (potentially reduced) PLY to https://gsplat.org and record a public viewer link in `gsplat_share_id`.
- By default it uploads the original PLY.
- You can enable generating a smaller `*.small.gsplat.ply` via `splat-transform` by setting `GSPLAT_USE_SMALL_PLY=1`.

## Pipeline notes (Ctrl+C / pause / stop)

- The pipeline supports cooperative pause/stop via flag files under `CONTROL_DIR` (defaults to the run folder): `PAUSE` and `STOP`.
- On Windows consoles, press `p` to toggle pause/resume (create/delete `PAUSE`), and press `q` to request stop (create `STOP`).
- `Ctrl+C` requests stop (safe-point semantics): the pipeline will stop at the next check without hard-killing in-flight work.
- Unexpected exceptions (especially in worker threads) are logged with full tracebacks to simplify debugging.
