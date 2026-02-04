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

Notes:

- `Ctrl+C` is handled cooperatively (safe-point semantics): the pipeline will avoid hard-killing in-flight work and will stop as soon as it reaches the next check.
- `Ctrl+C` toggles pause/resume by creating or deleting `PAUSE`.
- `Ctrl+D` requests stop/exit by creating `STOP`. On Windows consoles, `Ctrl+Z` is also accepted as an exit shortcut.
- `STOP` is terminal for the current run: if `STOP` exists, the pipeline will exit. To continue processing, delete `STOP` and restart the pipeline.

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

- `MAX_CANDIDATES`: max candidate photos to consider per run (default: `200`).
- `MAX_IMAGES`: max images to download per run (default: `50`)

Notes:

- `MAX_IMAGES` is the main limiter for how many samples will be produced/processed.
- Use `-1` for unlimited.

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
- `HF_UPLOAD_BATCH_SIZE`: optional batch size for HF uploads (default: `1`). When >1, each upload worker will try to group multiple samples into a single HF commit for better throughput.
- `HF_UPLOAD_BATCH_WAIT_MS`: micro-batching wait window in milliseconds (default: `200` when batch is enabled; otherwise `0`).

Recommended:

- `HF_UPLOAD_BATCH_SIZE=4` for higher throughput.
- `HF_UPLOAD_BATCH_SIZE=1` for small contributors.

### Logging / debugging

- The pipeline logs unexpected exceptions with full tracebacks (especially inside worker threads) to help debug rare/unanticipated issues.

### Hugging Face upload

- `HF_UPLOAD`: enable upload (default: `1`)
- `HF_REPO_ID`: target repo (default: `eatmorefruit/sharp-ply-share`)
- `HF_REPO_TYPE`: `dataset` or `model` (default: `dataset`)
- `HF_SUBDIR`: subdir for assets (default: `unsplash`)
- `HF_USE_LOCKS`: enable per-image locks (default: `1`)
- `HF_LOCKS_DIR`: lock dir in repo (default: `locks`)
- `HF_DONE_DIR`: done dir in repo (default: `done`)
- `HF_LOCK_STALE_SECS`: stale lock TTL (default: `21600`)
- `HF_INDEX_REPO_PATH`: JSONL index path in repo (default: `data/train.jsonl`)
- `HF_INDEX_REFRESH_SECS`: refresh interval (index backend) (default: `300`)

Index JSONL compact mode:

- `HF_INDEX_COMPACT=1`: write compact rows to local `train.jsonl` by dropping redundant fields (e.g. `unsplash_id`, `tags_text/topics_text`) and not forcing optional keys if not present.
- `HF_INDEX_COMPACT_DROP_EMPTY=1`: when compact mode is enabled, also drop optional fields that are empty (e.g. empty `gsplat_*`, empty `topics`, empty `description`).
- `HF_INDEX_ASSET_MODE`: how to store asset references in index rows:
  - `url` (default when not compact): store `image_url/ply_url/spz_url` as full resolve URLs.
  - `path` (default when compact): store `image_path/ply_path/spz_path` as repo-relative paths and drop the long URL fields.
  - `both`: store both URL and path fields.
  - `none`: drop both URL and path asset fields (downstream reconstructs from `image_id`).
- `HF_INDEX_TEXT_MODE`: how to store text fields in index rows:
  - `full`: keep `tags/topics/description/alt_description`.
  - `minimal` (default when compact): drop empty optional text fields (e.g. empty `topics`, empty `description`).
  - `none`: drop text fields for maximum compression.
- `HF_INDEX_DROP_DERIVABLE_URLS=1`: (optional) drop URL fields that can be derived from other fields (e.g. `unsplash_url`, `gsplat_url`, `image_url/ply_url/spz_url`). Default is `0` when not in compact mode.
- `HF_INDEX_DROP_USER_NAME=1`: drop `user_name` because it duplicates `user_username`.
- `HF_INDEX_DROP_UNSPLASH_ID=1`: drop `unsplash_id` because it duplicates `image_id`.

If you store only repo-relative paths (asset_mode=`path`), downstream can reconstruct the resolve URLs:

- Dataset repo: `https://huggingface.co/datasets/<HF_REPO_ID>/resolve/<REV>/<path>`
- Model repo: `https://huggingface.co/models/<HF_REPO_ID>/resolve/<REV>/<path>`

Notes:

- When only `*_path` fields are stored (no `*_url`), the Hugging Face Dataset Viewer will not be able to directly render assets from the index without additional processing.

If `HF_INDEX_DROP_DERIVABLE_URLS=1` and/or `HF_INDEX_ASSET_MODE=none` is used, the index may not contain any URL/path fields. Downstream can reconstruct the URLs/paths:

- HF repo relative asset path (default layout): `<HF_SUBDIR>/<image_id>/<image_id>.<ext>`
  - ext: `jpg`, `ply`, `spz`
- HF resolve URL:
  - dataset repo: `https://huggingface.co/datasets/<HF_REPO_ID>/resolve/<REV>/<HF_SUBDIR>/<image_id>/<image_id>.<ext>`
  - model repo: `https://huggingface.co/<HF_REPO_ID>/resolve/<REV>/<HF_SUBDIR>/<image_id>/<image_id>.<ext>`
- Unsplash photo page URL: `https://unsplash.com/photos/<image_id>`
- gsplat viewer URL (when `gsplat_share_id` is present): `<GSPLAT_BASE>/viewer/<gsplat_share_id>`

### HF index (optional)

- `HF_WRITE_INDEX`: write/update index file (default: `1`)
- `HF_INDEX_REPO_PATH`: index path in repo (default: `data/train.jsonl`)
- `HF_INDEX_FLUSH_EVERY`: flush every N rows (default: `20`)
- `HF_INDEX_FLUSH_SECS`: flush at least every N seconds (default: `30`)
- `HF_INDEX_REFRESH_SECS`: refresh remote index for collaborator correctness when `HF_DONE_BACKEND=index` (default: `300`)

### Remote done check backend (optional)

To reduce Hub small-file requests, you can choose how the pipeline checks whether an `image_id` has already been processed remotely:

- `HF_DONE_BACKEND`:
  - `index` (default): use the local in-memory set populated from the HF index file (`HF_INDEX_REPO_PATH`) when available.
  - `parquet` / `viewer`: use datasets-server Parquet filter queries.
  - `duckdb`: query the dataset Parquet exports locally via DuckDB (requires `duckdb` installed).
  - `none`: disable remote done checking.

Related variables (for `parquet`/`duckdb`):

- `HF_DONE_DATASET` (default: `HF_REPO_ID`)
- `HF_DONE_CONFIG`, `HF_DONE_SPLIT` (optional; auto-resolved for `parquet` if omitted)
- `HF_DONE_COLUMN` (default: `image_id`)

### gsplat.org share (optional)

- `GSPLAT_UPLOAD`: enable uploading to gsplat.org and recording `gsplat_url` (default: `1`)
- `GSPLAT_BASE`: gsplat base URL (default: `https://gsplat.org`)
- `GSPLAT_EXPIRATION_TYPE`: `1week` / etc (default: `1week`)
- `GSPLAT_FILTER_VISIBILITY`: visibility filter passed to `splat-transform` (default: `20000`)
- `SPLAT_TRANSFORM_BIN`: `splat-transform` path (default: `splat-transform`)
- `GSPLAT_USE_SMALL_PLY`: generate `*.small.gsplat.ply` before upload (default: `0`); set `1` to enable

Index notes:

- `gsplat_model_file_url` in the index is normalized to store only the file token (e.g. `1770129991964_T8LMLFAy`), stripping `/share/file/` and `.ply`.
- If needed, reconstruct the raw share path as `/share/file/<token>.ply`.

### Range locks (for `SOURCE=list` + `order_by=oldest`)

- `RANGE_LOCKS_ENABLED`: enable range locks (default: `1`)
- `RANGE_LOCKS_DIR`: range lock dir in repo (default: `ranges/locks`)
- `RANGE_DONE_DIR`: range done dir in repo (default: `ranges/done`)
- `RANGE_LOCK_STALE_SECS`: range lock TTL (default: `21600`)
- `RANGE_SIZE`: range size in items (default: `300`)
- `RANGE_LOCK_MIN_IMAGES`: do not enable range-lock if `MAX_IMAGES` is smaller (default: `30`)

Notes:

- Range done detection uses a shared prefix file at `ranges/progress/done_prefix.json` to avoid repo-wide listings of `ranges/done/`.

### Ant-style range selection (optional, for multi-client coordination)

- `ANT_ENABLED`: enable ant-style range selection (default: `1`)
- `ANT_CANDIDATE_RANGES`: number of candidate ranges to score before locking (default: `6`)
- `ANT_EPSILON`: probability of random exploration vs greedy choice (default: `0.2`)
- `ANT_FRESH_SECS`: treat recently-updated ranges as “busy” and de-prioritize them (default: `90`)


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
