import os
from datetime import datetime
from . import hf_utils

def load_dotenv_if_present() -> None:
    try:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    except Exception:
        base_dir = os.getcwd()

    loaded_keys = set()

    def _apply_one(path: str, *, allow_override_loaded: bool) -> None:
        try:
            if not path or not os.path.isfile(path):
                return
            with open(path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    if line.lower().startswith("export "):
                        line = line[7:].lstrip()
                    if "=" not in line:
                        continue
                    k, v = map(str.strip, line.split("=", 1))
                    if not k:
                        continue
                    if k not in os.environ or (allow_override_loaded and k in loaded_keys):
                        if len(v) >= 2 and v[0] == v[-1] and v[0] in ("\"", "'"):
                            v = v[1:-1]
                        os.environ[k] = v
                        loaded_keys.add(k)
        except Exception as e:
            print(f"Error loading dotenv file {path}: {e}")

    dirs = [base_dir]
    try:
        cwd = os.getcwd()
        if cwd and os.path.abspath(cwd) != os.path.abspath(base_dir):
            dirs.append(cwd)
    except Exception:
        pass

    for d in dirs:
        _apply_one(os.path.join(d, ".env"), allow_override_loaded=False)
        _apply_one(os.path.join(d, ".env.local"), allow_override_loaded=True)

# Load configuration initially
load_dotenv_if_present()

_env_flag = hf_utils.env_flag
_env_int = hf_utils.env_int
_env_str = hf_utils.env_str

# ===================== Configuration =====================
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")
APP_NAME = os.getenv("UNSPLASH_APP_NAME", "sharp-ply-share")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ML_SHARP_DIR = os.getenv("ML_SHARP_DIR", os.path.join(OUTPUT_DIR, "ml-sharp-main"))
RUN_ID = os.getenv("RUN_ID", datetime.now().strftime("unsplash_%Y%m%d_%H%M%S"))
SAVE_DIR = os.path.normpath(os.path.join(OUTPUT_DIR, "runs", RUN_ID))
IMAGES_DIR = os.path.join(SAVE_DIR, "images")
GAUSSIANS_DIR = os.path.join(SAVE_DIR, "gaussians")
SOURCE = _env_str("SOURCE", "list").strip().lower()
MAX_CANDIDATES = _env_int("MAX_CANDIDATES", 8)
MAX_IMAGES = _env_int("MAX_IMAGES", -1)
INJECT_EXIF = _env_flag("INJECT_EXIF", True)
STOP_ON_RATE_LIMIT = _env_flag("STOP_ON_RATE_LIMIT", True)
DEBUG_MODE = _env_flag("DEBUG_MODE", True)

CONDA_ENV_NAME = os.getenv("CONDA_ENV_NAME", "sharp")
SHARP_DEVICE = os.getenv("SHARP_DEVICE", "default")
FORBID_CPU = _env_flag("FORBID_CPU", False)
SHARP_VERBOSE = _env_flag("SHARP_VERBOSE", False)
SHARP_PER_IMAGE = _env_flag("SHARP_PER_IMAGE", False)
SHARP_BATCH_SIZE = _env_int("SHARP_BATCH_SIZE", 0)
SHARP_INPUT_DIR = os.getenv("SHARP_INPUT_DIR", "").strip()
SKIP_PREDICT = _env_flag("SKIP_PREDICT", False)

HF_UPLOAD = _env_flag("HF_UPLOAD", True)
REQUIRE_HF_UPLOAD = _env_flag("REQUIRE_HF_UPLOAD", False)
HF_REPO_ID = _env_str("HF_REPO_ID", "eatmorefruit/sharp-ply-share").strip()
HF_SUBDIR = _env_str("HF_SUBDIR", "unsplash").strip().strip("/")
HF_INDEX_REPO_PATH = _env_str("HF_INDEX_REPO_PATH", "data/train.jsonl").strip().lstrip("/")

GSPLAT_UPLOAD = _env_flag("GSPLAT_UPLOAD", True)
SPZ_EXPORT = _env_flag("SPZ_EXPORT", True)
SPZ_TOOL = _env_str("SPZ_TOOL", "").strip().lower()
GSBOX_BIN = _env_str("GSBOX_BIN", "gsbox").strip()
GSCONVERTER_BIN = _env_str("GSCONVERTER_BIN", "3dgsconverter").strip()

PLY_DELETE_AFTER_UPLOAD = _env_flag("PLY_DELETE_AFTER_UPLOAD", True)
VALID_FOCAL_FILE = _env_str("VALID_FOCAL_FILE", "").strip()

PER_PAGE = _env_int("PER_PAGE", 10)
LIST_PER_PAGE = 30
LIST_AUTO_SEEK = _env_flag("LIST_AUTO_SEEK", True)
LIST_SEEK_BACK_PAGES = _env_int("LIST_SEEK_BACK_PAGES", 2)

RANGE_LOCKS_ENABLED = _env_flag("RANGE_LOCKS_ENABLED", True)
RANGE_LOCKS_DIR = _env_str("RANGE_LOCKS_DIR", "ranges/locks").strip().strip("/")
RANGE_DONE_DIR = _env_str("RANGE_DONE_DIR", "ranges/done").strip().strip("/")
RANGE_LOCK_STALE_SECS = float(os.getenv("RANGE_LOCK_STALE_SECS", "21600"))
RANGE_SIZE = _env_int("RANGE_SIZE", 300)
RANGE_SEEK_BACK_ITEMS = _env_int("RANGE_SEEK_BACK_ITEMS", 0)
RANGE_LOCK_MIN_IMAGES = _env_int("RANGE_LOCK_MIN_IMAGES", 30)
RANGE_HEARTBEAT_SECS = float(os.getenv("RANGE_HEARTBEAT_SECS", "600"))
RANGE_PROGRESS_SECS = float(os.getenv("RANGE_PROGRESS_SECS", "300"))
RANGE_ABANDONED_SECS = float(os.getenv("RANGE_ABANDONED_SECS", "60"))

LOG_GPU_MEM = _env_flag("LOG_GPU_MEM", False)
GPU_LOG_FILE = os.path.join(SAVE_DIR, "gpu_mem_log.csv")
DOWNLOAD_QUEUE_MAX = _env_int("DOWNLOAD_QUEUE_MAX", 8)
UPLOAD_QUEUE_MAX = _env_int("UPLOAD_QUEUE_MAX", 256)
UPLOAD_WORKERS = _env_int("UPLOAD_WORKERS", 2)
IDLE_SLEEP_S = float(os.getenv("IDLE_SLEEP_S", "0.5"))
CONTROL_DIR = _env_str("CONTROL_DIR", SAVE_DIR)
PAUSE_FILE = _env_str("PAUSE_FILE", "PAUSE")
STOP_FILE = _env_str("STOP_FILE", "STOP")

HF_REPO_TYPE = _env_str("HF_REPO_TYPE", "dataset").strip().lower()
HF_SQUASH_EVERY = _env_int("HF_SQUASH_EVERY", 0)
HF_USE_LOCKS = _env_flag("HF_USE_LOCKS", True)
HF_LOCK_BACKEND = _env_str("HF_LOCK_BACKEND", "auto").strip().lower() or "hf"
HF_LOCKS_DIR = _env_str("HF_LOCKS_DIR", "locks").strip().strip("/")
HF_DONE_DIR = _env_str("HF_DONE_DIR", "done").strip().strip("/")
HF_LOCK_STALE_SECS = float(os.getenv("HF_LOCK_STALE_SECS", "21600"))
HF_WRITE_INDEX = _env_flag("HF_WRITE_INDEX", True)
HF_INDEX_FLUSH_EVERY = _env_int("HF_INDEX_FLUSH_EVERY", 20)
HF_INDEX_FLUSH_SECS = float(os.getenv("HF_INDEX_FLUSH_SECS", "30"))
HF_INDEX_REFRESH_SECS = float(os.getenv("HF_INDEX_REFRESH_SECS", "300"))

GSPLAT_BASE = _env_str("GSPLAT_BASE", "https://gsplat.org").strip().rstrip("/")
GSPLAT_EXPIRATION_TYPE = _env_str("GSPLAT_EXPIRATION_TYPE", "1week").strip()
GSPLAT_FILTER_VISIBILITY = _env_int("GSPLAT_FILTER_VISIBILITY", 20000)
SPLAT_TRANSFORM_BIN = _env_str("SPLAT_TRANSFORM_BIN", "splat-transform").strip()
GSPLAT_USE_SMALL_PLY = _env_flag("GSPLAT_USE_SMALL_PLY", False)

GSBOX_SPZ_QUALITY = _env_int("GSBOX_SPZ_QUALITY", 5)
GSBOX_SPZ_VERSION = _env_int("GSBOX_SPZ_VERSION", 0)
GSCONVERTER_COMPRESSION_LEVEL = _env_int("GSCONVERTER_COMPRESSION_LEVEL", 6)

PLY_KEEP_LAST = _env_int("PLY_KEEP_LAST", 10)

HF_UPLOAD_BATCH_SIZE = _env_int("HF_UPLOAD_BATCH_SIZE", 1)
try:
    _default_wait = 200 if int(HF_UPLOAD_BATCH_SIZE) > 1 else 0
except Exception:
    _default_wait = 0
HF_UPLOAD_BATCH_WAIT_MS = _env_int("HF_UPLOAD_BATCH_WAIT_MS", int(_default_wait))

PIPELINE_HEARTBEAT_SECS = float(os.getenv("PIPELINE_HEARTBEAT_SECS", "10"))
STALL_WARN_SECS = float(os.getenv("STALL_WARN_SECS", "120"))

ANT_ENABLED = _env_flag("ANT_ENABLED", True)
ANT_CANDIDATE_RANGES = _env_int("ANT_CANDIDATE_RANGES", 6)
try:
    ANT_EPSILON = float(os.getenv("ANT_EPSILON", "0.2"))
except Exception:
    ANT_EPSILON = 0.2
try:
    ANT_FRESH_SECS = float(os.getenv("ANT_FRESH_SECS", "90"))
except Exception:
    ANT_FRESH_SECS = 90.0

UNSPLASH_API_BASE = "https://api.unsplash.com"
INPUT_IMAGES_DIR = SHARP_INPUT_DIR if SHARP_INPUT_DIR else IMAGES_DIR

QUERIES = [
    "camera", "dslr", "lens", "nikon", "canon", "sony", "fujifilm",
    "photography", "portrait", "street", "landscape",
]
SEARCH_ORDERS = ["latest", "relevant"]
LIST_ORDERS = ["oldest", "latest", "popular"]
