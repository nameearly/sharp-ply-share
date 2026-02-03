# SHARP Dataset Pipeline

This pipeline script:

- Downloads images from Unsplash (search or list mode)
- Runs `sharp predict` (ml-sharp) to produce 3D Gaussian `.ply` files
- Optionally uploads results + an index file to a Hugging Face Hub repo (typically a `dataset` repo)
- Supports multi-client coordination via Hugging Face-based locks/done markers

Recommended entrypoint:

- `sharp_dataset_pipeline_main.py`


## Quick start

1. Create and install the ml-sharp environment (see https://github.com/apple/ml-sharp).

2. Set required env vars:

- `UNSPLASH_ACCESS_KEY`: your Unsplash access key

3. (Optional) Hugging Face upload:

- Make sure you have a valid Hugging Face token available for `huggingface_hub` (e.g. `HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`).

4. Run:

```bash
python sharp_dataset_pipeline_main.py
```


## Output layout

Outputs are organized under:

- `runs/<RUN_ID>/images/` downloaded JPEGs
- `runs/<RUN_ID>/gaussians/` generated `.ply` files
- `runs/<RUN_ID>/gpu_mem_log.csv` (optional, when `LOG_GPU_MEM=1`)


## Pause / Stop control

This pipeline supports cooperative pause/stop through files under `CONTROL_DIR`:

- Pause: create a `PAUSE` file (configurable via `PAUSE_FILE`)
- Stop: create a `STOP` file (configurable via `STOP_FILE`)

Defaults:

- `CONTROL_DIR=<SAVE_DIR>`
- `PAUSE_FILE=PAUSE`
- `STOP_FILE=STOP`


## Important environment variables

### Unsplash

- `UNSPLASH_ACCESS_KEY` (required)
- `UNSPLASH_APP_NAME` (default: `your_app_name`)
- `SOURCE`: `search` or `list` (default: `search`)
- `PER_PAGE`: per_page for search endpoint (default: `10`, max 30)
- `LIST_AUTO_SEEK`: auto-seek optimization for `list`+`oldest` (default: `1`)
- `LIST_SEEK_BACK_PAGES`: seek-back pages when estimating start page (default: `2`)
- `STOP_ON_RATE_LIMIT`: stop the pipeline when Unsplash rate limit is hit (default: `1`)

### Scan limits

- `MAX_SCAN`: max photos to scan per cycle (default: `200`)
- `MAX_IMAGES`: max images to download per run (default: `50`)

### SHARP / ml-sharp

- `ML_SHARP_DIR`: path to your local ml-sharp project folder (recommended: set this explicitly)
- `CONDA_ENV_NAME`: conda env name (default: `sharp`)
- `SHARP_DEVICE`: `default` / `cuda` / `cpu` / `mps` (default: `default`)
- `SHARP_VERBOSE`: verbose `sharp` output (default: `0`)
- `SHARP_INPUT_DIR`: override input images directory (default: empty)

### Queues / concurrency

- `DOWNLOAD_QUEUE_MAX` (default: `8`)
- `UPLOAD_QUEUE_MAX` (default: `256`)
- `UPLOAD_WORKERS` (default: `2`)

### Hugging Face upload

- `HF_UPLOAD`: enable upload (default: `1`)
- `HF_REPO_ID`: target repo (default: `eatmorefruit/sharp-ply-share`)
- `HF_REPO_TYPE`: `dataset` or `model` (default: `dataset`)
- `HF_SUBDIR`: subdir within repo (default: `unsplash`)
- `HF_USE_LOCKS`: enable per-image locks (default: `1`)
- `HF_LOCKS_DIR`: lock dir in repo (default: `locks`)
- `HF_DONE_DIR`: done dir in repo (default: `done`)
- `HF_LOCK_STALE_SECS`: stale lock TTL (default: `21600`)

### HF index (optional)

- `HF_WRITE_INDEX`: write/update index file (default: `1`)
- `HF_INDEX_REPO_PATH`: index path in repo (default: `data/train.jsonl`)
- `HF_INDEX_FLUSH_EVERY`: flush every N rows (default: `20`)
- `HF_INDEX_FLUSH_SECS`: flush at least every N seconds (default: `30`)

### gsplat.org share (optional)

- `GSPLAT_UPLOAD`: enable uploading to gsplat.org and recording `gsplat_url` (default: `1`)
- `GSPLAT_BASE`: gsplat base URL (default: `https://gsplat.org`)
- `GSPLAT_EXPIRATION_TYPE`: `1week` / etc (default: `1week`)
- `GSPLAT_FILTER_VISIBILITY`: visibility filter passed to `splat-transform` (default: `20000`)
- `SPLAT_TRANSFORM_BIN`: `splat-transform` path (default: `splat-transform`)
- `GSPLAT_USE_SMALL_PLY`: generate `*.small.gsplat.ply` before upload (default: `1`); set `0` to upload the original PLY

### Range locks (for `SOURCE=list` + `order_by=oldest`)

- `RANGE_LOCKS_ENABLED`: enable range locks (default: `1`)
- `RANGE_LOCKS_DIR`: range lock dir in repo (default: `ranges/locks`)
- `RANGE_DONE_DIR`: range done dir in repo (default: `ranges/done`)
- `RANGE_LOCK_STALE_SECS`: range lock TTL (default: `21600`)
- `RANGE_SIZE`: range size in items (default: `300`)
- `RANGE_LOCK_MIN_IMAGES`: do not enable range-lock if `MAX_IMAGES` is smaller (default: `30`)


## Module layout

- `sharp_dataset_pipeline/unsplash.py`
  - Unsplash API calls, download, rate-limit handling
- `sharp_dataset_pipeline/hf_sync.py`
  - Hugging Face locks/done markers, range locks, cached existence checks
- `sharp_dataset_pipeline/progress.py`
  - `OrderedProgress` (frontier + holes tracking for oldest mode)
- `sharp_dataset_pipeline/pipeline.py`
  - Thread/queue orchestration and the main processing loop


## Licensing / attribution note

This project downloads photos from Unsplash and uses them as inputs to generate derived 3D Gaussian splat assets (e.g. `.ply`, optionally `.spz`) via ml-sharp.

Unsplash photos are **not CC0**. They are provided under the **Unsplash License**:

- https://unsplash.com/license

For each sample, the index includes `unsplash_url`, `user_name`, and `user_username` so consumers can attribute and link back to the original photo/author.
