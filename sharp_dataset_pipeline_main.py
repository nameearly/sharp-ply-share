import os
import time
from datetime import datetime
import subprocess
import shutil
import csv
import json
import re

import sharp_dataset_pipeline.hf_sync as hf_sync
import sharp_dataset_pipeline.unsplash as unsplash
import sharp_dataset_pipeline.pipeline as pipeline
import sharp_dataset_pipeline.hf_utils as hf_utils
import sharp_dataset_pipeline.hf_upload as hf_upload
import sharp_dataset_pipeline.index_sync as hf_index_sync
import sharp_dataset_pipeline.parquet_tools as parquet_tools

def _env_flag(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return bool(default)

    return str(v).strip().lower() in ("1", "true", "yes", "y")

def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    if v is None:
        return int(default)
    try:
        return int(str(v).strip())
    except Exception:
        return int(default)

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    if v is None:
        return str(default)
    return str(v)


def _load_unsplash_key_pool(json_path: str):
    try:
        p = str(json_path or "").strip()
        if not p:
            return []
        if not os.path.exists(p):
            return []
        raw = open(p, "r", encoding="utf-8").read()
    except Exception:
        return []

    def _normalize_items(obj):
        if obj is None:
            return []
        if isinstance(obj, dict):
            obj = [obj]
        if not isinstance(obj, list):
            return []
        out = []
        for it in obj:
            if not isinstance(it, dict):
                continue
            k = it.get("UNSPLASH_ACCESS_KEY") or it.get("unsplash_access_key") or it.get("access_key")
            k = str(k or "").strip()
            if not k:
                continue
            an = it.get("UNSPLASH_APP_NAME") or it.get("unsplash_app_name") or it.get("app_name")
            an = str(an or "").strip()
            if not an:
                an = str(APP_NAME or "").strip() or "sharp-ply-share"
            out.append({"UNSPLASH_ACCESS_KEY": k, "UNSPLASH_APP_NAME": an})
        return out

    try:
        return _normalize_items(json.loads(raw))
    except Exception:
        pass

    try:
        s = str(raw or "").strip()
        if not s:
            return []
        # Tolerate the current file style (not valid JSON):
        # {
        #   { UNSPLASH_APP_NAME:"...", UNSPLASH_ACCESS_KEY:"..." },
        #   ...
        # }
        if s.startswith("{") and s.endswith("}"):
            s = "[" + s[1:-1] + "]"
        s = re.sub(r"(?m)\b(UNSPLASH_APP_NAME|UNSPLASH_ACCESS_KEY)\s*:", r'"\\1":', s)
        s = re.sub(r",\s*([}\]])", r"\\1", s)
        obj = json.loads(s)
        return _normalize_items(obj)
    except Exception:
        return []

def _choose_bucket_per_page(remaining: int, hard_max: int) -> int:
    try:
        hard_max = int(hard_max)
    except Exception:
        hard_max = 30
    hard_max = max(1, hard_max)
    try:
        remaining = int(remaining)
    except Exception:
        remaining = 0
    remaining = max(0, remaining)

    candidates = [10, 20, 30]
    filtered = [x for x in candidates if x <= hard_max]
    if not filtered:
        return min(10, hard_max)

    for x in filtered:
        if remaining <= x:
            return x
    return filtered[-1]

# ===================== 配置（常用/建议修改）=====================
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")
APP_NAME = os.getenv("UNSPLASH_APP_NAME", "your_app_name")
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.dirname(__file__))
ML_SHARP_DIR = os.getenv("ML_SHARP_DIR", os.path.join(os.path.dirname(__file__), "ml-sharp-main"))
RUN_ID = os.getenv("RUN_ID", datetime.now().strftime("unsplash_%Y%m%d_%H%M%S"))
SAVE_DIR = os.path.join(OUTPUT_DIR, "runs", RUN_ID)
IMAGES_DIR = os.path.join(SAVE_DIR, "images")
GAUSSIANS_DIR = os.path.join(SAVE_DIR, "gaussians")
SOURCE = _env_str("SOURCE", "list").strip().lower()  # search | list
MAX_CANDIDATES = _env_int("MAX_CANDIDATES", 200)
MAX_IMAGES = _env_int("MAX_IMAGES", 50)
INJECT_EXIF = _env_flag("INJECT_EXIF", True)
STOP_ON_RATE_LIMIT = _env_flag("STOP_ON_RATE_LIMIT", True)
DEBUG_MODE = True       # 调试开关

CONDA_ENV_NAME = "sharp"
SHARP_DEVICE = os.getenv("SHARP_DEVICE", "default")

FORBID_CPU = str(os.getenv("FORBID_CPU", "0")).strip().lower() in ("1", "true", "yes", "y")
SHARP_VERBOSE = str(os.getenv("SHARP_VERBOSE", "0")).strip().lower() in ("1", "true", "yes", "y")
SHARP_PER_IMAGE = str(os.getenv("SHARP_PER_IMAGE", "0")).strip().lower() in ("1", "true", "yes", "y")
SHARP_BATCH_SIZE = int(os.getenv("SHARP_BATCH_SIZE", "0"))
SHARP_INPUT_DIR = os.getenv("SHARP_INPUT_DIR", "").strip()
SKIP_PREDICT = _env_flag("SKIP_PREDICT", False)

HF_UPLOAD = _env_flag("HF_UPLOAD", True)
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

# ===================== 配置（高级/一般不建议修改）=====================
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
# ==========================================================

UNSPLASH_API_BASE = "https://api.unsplash.com"

INPUT_IMAGES_DIR = SHARP_INPUT_DIR if SHARP_INPUT_DIR else IMAGES_DIR

QUERIES = [
    "camera",
    "dslr",
    "lens",
    "nikon",
    "canon",
    "sony",
    "fujifilm",
    "photography",
    "portrait",
    "street",
    "landscape",
]
SEARCH_ORDERS = ["latest", "relevant"]
LIST_ORDERS = ["oldest", "latest", "popular"]

def print_debug(msg):
    """调试输出（带时间戳）"""
    if DEBUG_MODE:
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        print(f"{timestamp} [DEBUG] {msg}")

def _query_nvidia_smi_rows():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, check=True)
        out = (p.stdout or "").strip()
        if not out:
            return []
        rows = []
        for line in out.splitlines():
            parts = [x.strip() for x in line.split(",")]
            if len(parts) < 6:
                continue
            rows.append(parts[:6])
        return rows
    except Exception:
        return []

def _append_gpu_log(event: str, input_path: str):
    if not LOG_GPU_MEM:
        return
    rows = _query_nvidia_smi_rows()
    if not rows:
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    write_header = not os.path.exists(GPU_LOG_FILE)
    try:
        with open(GPU_LOG_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(
                    [
                        "timestamp",
                        "gpu_index",
                        "gpu_name",
                        "mem_used_mib",
                        "mem_total_mib",
                        "util_gpu_pct",
                        "event",
                        "run_id",
                        "input_path",
                    ]
                )
            for ts, idx, name, mem_used, mem_total, util in rows:
                w.writerow([ts, idx, name, mem_used, mem_total, util, event, RUN_ID, input_path])
    except Exception:
        return

def init_environment():
    """初始化目录和记录文件，读取已处理的ID（防重基础）"""
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        print_debug(f"创建保存目录：{SAVE_DIR}")
    else:
        print_debug(f"保存目录已存在：{SAVE_DIR}")

    checked_ids = set()
    print_debug(f"已处理的ID数量：{len(checked_ids)}")
    return checked_ids

def _hf_try_super_squash(repo_id: str) -> bool:
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.super_squash_history(repo_id=repo_id, repo_type=HF_REPO_TYPE)
        return True
    except Exception as e:
        print_debug(f"HF super_squash_history 失败（可忽略） | err={str(e)}")
        return False


def _list_gaussian_plys():
    try:
        if not os.path.exists(GAUSSIANS_DIR):
            return set()
        out = set()
        for f in os.listdir(GAUSSIANS_DIR):
            low = str(f).lower()
            if not low.endswith(".ply"):
                continue
            if ".small.gsplat" in low:
                continue
            if ".vertexonly.binary" in low:
                continue
            out.add(os.path.join(GAUSSIANS_DIR, f))
        return out
    except Exception:
        return set()


def _run_sharp_predict_once(input_path):
    extra = ["-v"] if SHARP_VERBOSE else []
    plys_before = _list_gaussian_plys()

    def _sig(p: str):
        try:
            st = os.stat(p)
            return (int(st.st_size), int(getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))))
        except Exception:
            return None

    before_sig = {}
    try:
        before_sig = {p: _sig(p) for p in (plys_before or set())}
    except Exception:
        before_sig = {}

    _append_gpu_log("before_predict", input_path)

    cmd = [
        "conda",
        "run",
        "-n",
        CONDA_ENV_NAME,
        "sharp",
        "predict",
        "-i",
        input_path,
        "-o",
        GAUSSIANS_DIR,
        "--device",
        SHARP_DEVICE,
        *extra,
    ]

    popen_kw = {}
    try:
        if os.name == "nt" and hasattr(subprocess, "CREATE_NEW_PROCESS_GROUP"):
            popen_kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    except Exception:
        popen_kw = {}

    try:
        try:
            subprocess.run(cmd, cwd=ML_SHARP_DIR, check=True, **popen_kw)
        except subprocess.CalledProcessError as e:
            print_debug(f"ml-sharp 通过 'sharp predict' 执行失败，将尝试源码方式运行 | err={str(e)}")

            src_dir = os.path.join(ML_SHARP_DIR, "src")
            argv = [
                "sharp",
                "predict",
                "-i",
                input_path,
                "-o",
                GAUSSIANS_DIR,
                "--device",
                SHARP_DEVICE,
            ]
            if SHARP_VERBOSE:
                argv.append("-v")

            code = (
                "import sys; "
                f"sys.path.insert(0, {repr(src_dir)}); "
                "from sharp.cli import main_cli; "
                f"sys.argv = {repr(argv)}; "
                "main_cli()"
            )
            cmd2 = ["conda", "run", "-n", CONDA_ENV_NAME, "python", "-c", code]
            subprocess.run(cmd2, cwd=ML_SHARP_DIR, check=True, **popen_kw)
    finally:
        _append_gpu_log("after_predict", input_path)

    produced = []
    try:
        plys_after = _list_gaussian_plys()
        after_sig = {p: _sig(p) for p in (plys_after or set())}
        candidates = []
        for p, s in after_sig.items():
            if p not in before_sig:
                candidates.append(p)
                continue
            if (s is not None) and (before_sig.get(p) != s):
                candidates.append(p)
        produced = sorted(candidates) if candidates else []

        if not produced:
            try:
                base = os.path.splitext(os.path.basename(str(input_path)))[0]
                exp = os.path.join(GAUSSIANS_DIR, base + ".ply")
                low = os.path.basename(exp).lower()
                if (
                    os.path.isfile(exp)
                    and os.path.getsize(exp) > 0
                    and (".small.gsplat" not in low)
                    and (".vertexonly.binary" not in low)
                ):
                    produced = [exp]
            except Exception:
                produced = []
    except Exception:
        produced = []

    return produced

def _extract_pil_exif_dict(img_pil):
    try:
        from PIL import ExifTags
        from PIL import TiffTags
    except Exception:
        return {}

    exif_dict = {}
    try:
        img_exif = img_pil.getexif().get_ifd(0x8769)
        exif_dict = {ExifTags.TAGS[k]: v for k, v in img_exif.items() if k in ExifTags.TAGS}
    except Exception:
        exif_dict = {}

    try:
        tiff_tags = img_pil.getexif()
        tiff_dict = {TiffTags.TAGS_V2[k].name: v for k, v in tiff_tags.items() if k in TiffTags.TAGS_V2}
    except Exception:
        tiff_dict = {}

    return {**exif_dict, **tiff_dict}

def _to_float_maybe(v):
    try:
        if isinstance(v, tuple) and len(v) == 2 and v[1] != 0:
            return float(v[0]) / float(v[1])
        return float(v)
    except Exception:
        return None

def _extract_focal_from_exif(exif_dict):
    f35 = exif_dict.get("FocalLengthIn35mmFilm", exif_dict.get("FocalLenIn35mmFilm", None))
    f35 = _to_float_maybe(f35) if f35 is not None else None
    focal = exif_dict.get("FocalLength", None)
    focal = _to_float_maybe(focal) if focal is not None else None
    return focal, f35

def write_focal_compare_report():
    if not os.path.exists(VALID_FOCAL_FILE):
        return None
    if not os.path.exists(INPUT_IMAGES_DIR):
        return None

    try:
        from PIL import Image
    except Exception:
        print_debug("未安装 Pillow，跳过本地EXIF焦距对账。")
        return None

    out_path = os.path.join(SAVE_DIR, "focal_compare.tsv")
    with open(VALID_FOCAL_FILE, "r", encoding="utf-8") as f_in, open(
        out_path, "w", encoding="utf-8"
    ) as f_out:
        _ = f_in.readline()
        f_out.write(
            "photo_id\tunsplash_focal_raw\tunsplash_focal_avg_mm\t"
            "local_exif_focal_mm\tlocal_exif_focal_35mm_mm\t"
            "local_image_path\tlocal_exif_found\n"
        )
        for line in f_in:
            line = line.strip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 5:
                continue
            photo_id = parts[0]
            unsplash_focal_raw = parts[1]
            unsplash_focal_avg = parts[4]
            local_path = os.path.join(INPUT_IMAGES_DIR, f"{photo_id}.jpg")
            if not os.path.exists(local_path):
                continue

            local_focal = None
            local_f35 = None
            found = False
            try:
                img = Image.open(local_path)
                exif_dict = _extract_pil_exif_dict(img)
                local_focal, local_f35 = _extract_focal_from_exif(exif_dict)
                found = (local_focal is not None) or (local_f35 is not None)
            except Exception:
                found = False

            f_out.write(
                f"{photo_id}\t{unsplash_focal_raw}\t{unsplash_focal_avg}\t"
                f"{local_focal}\t{local_f35}\t{local_path}\t{int(found)}\n"
            )
    return out_path

def _mm_to_rational(mm_value: float):
    try:
        if mm_value is None:
            return None
        v = float(mm_value)
        if v <= 0:
            return None
        denom = 10 if abs(v - round(v)) > 1e-6 else 1
        num = int(round(v * denom))
        return (num, denom)
    except Exception:
        return None

def inject_focal_exif_if_missing(image_path: str, focal_mm: float) -> bool:
    try:
        from PIL import Image
    except Exception:
        return False

    try:
        img = Image.open(image_path)
        exif = img.getexif()
    except Exception:
        return False

    try:
        try:
            has_focal = (37386 in exif) or (41989 in exif)
        except Exception:
            has_focal = False
        if has_focal:
            return False

        rat = _mm_to_rational(focal_mm)
        if not rat:
            return False

        exif[37386] = rat
        try:
            exif[41989] = int(round(float(focal_mm)))
        except Exception:
            pass

        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(image_path, exif=exif)
        return True
    except Exception:
        return False

def _local_has_focal_exif(image_path: str) -> bool:
    try:
        from PIL import Image
    except Exception:
        return False
    try:
        img = Image.open(image_path)
        exif = img.getexif()
        return (37386 in exif) or (41989 in exif)
    except Exception:
        return False

def _get_local_focal_mm(image_path: str):
    try:
        from PIL import Image
    except Exception:
        return None

    try:
        img = Image.open(image_path)
        exif_dict = _extract_pil_exif_dict(img)
        focal, f35 = _extract_focal_from_exif(exif_dict)
        if f35 is not None:
            return float(f35)
        if focal is not None:
            return float(focal)
        return None
    except Exception:
        return None


def run_pipeline():
    unsplash_key_pool = None
    if not str(UNSPLASH_ACCESS_KEY or "").strip():
        try:
            keys_path = os.getenv(
                "UNSPLASH_ACCESS_KEY_JSON",
                os.path.join(os.path.dirname(__file__), "UNSPLASH_ACCESS_KEY.json"),
            )
            unsplash_key_pool = _load_unsplash_key_pool(keys_path)
        except Exception:
            unsplash_key_pool = None

    if (not str(UNSPLASH_ACCESS_KEY or "").strip()) and (not unsplash_key_pool):
        raise RuntimeError("UNSPLASH_ACCESS_KEY 为空，且未能从 UNSPLASH_ACCESS_KEY.json 加载")
    os.makedirs(SAVE_DIR, exist_ok=True)
    os.makedirs(INPUT_IMAGES_DIR, exist_ok=True)
    os.makedirs(GAUSSIANS_DIR, exist_ok=True)

    try:
        max_candidates_eff = int(_env_int("MAX_CANDIDATES", int(MAX_CANDIDATES)))
    except Exception:
        max_candidates_eff = int(MAX_CANDIDATES)
    try:
        max_images_eff = int(_env_int("MAX_IMAGES", int(MAX_IMAGES)))
    except Exception:
        max_images_eff = int(MAX_IMAGES)

    try:
        if int(max_images_eff) < 0 and os.getenv("MAX_CANDIDATES") is None:
            max_candidates_eff = -1
            try:
                print_debug("Config | MAX_IMAGES=-1 detected and MAX_CANDIDATES not set: defaulting MAX_CANDIDATES to -1 (unlimited)")
            except Exception:
                pass
    except Exception:
        pass

    try:
        print_debug(
            f"Config | SOURCE={str(SOURCE)} | MAX_IMAGES={int(max_images_eff)} | MAX_CANDIDATES={int(max_candidates_eff)} | LIST_PER_PAGE={int(LIST_PER_PAGE)} | LIST_AUTO_SEEK={bool(LIST_AUTO_SEEK)}"
        )
    except Exception:
        pass

    hf_sync.configure_hf_sync(
        hf_upload=HF_UPLOAD,
        repo_type=HF_REPO_TYPE,
        hf_locks_dir=HF_LOCKS_DIR,
        hf_done_dir=HF_DONE_DIR,
        range_locks_dir=RANGE_LOCKS_DIR,
        range_done_dir=RANGE_DONE_DIR,
        hf_lock_stale_secs=float(HF_LOCK_STALE_SECS),
        range_lock_stale_secs=float(RANGE_LOCK_STALE_SECS),
        debug_fn=print_debug,
    )

    unsplash.configure_unsplash(
        access_key=(unsplash_key_pool if unsplash_key_pool else UNSPLASH_ACCESS_KEY),
        app_name=APP_NAME,
        api_base=UNSPLASH_API_BASE,
        per_page=int(PER_PAGE),
        list_per_page=int(LIST_PER_PAGE),
        stop_on_rate_limit=STOP_ON_RATE_LIMIT,
        debug_fn=print_debug,
    )

    checked_ids = init_environment()

    if HF_UPLOAD:
        if not HF_REPO_ID:
            raise RuntimeError("HF_UPLOAD 开启但 HF_REPO_ID 为空")
        hf_utils.ensure_repo(HF_REPO_ID, repo_type=HF_REPO_TYPE, debug_fn=print_debug)

    coord = None
    if (not SKIP_PREDICT) and HF_UPLOAD and HF_USE_LOCKS and HF_REPO_ID:
        coord = hf_sync.LockDoneSync(HF_REPO_ID)

    range_coord = None
    # 小贡献者保护：MAX_IMAGES 太小就不启用 range-lock，避免锁住大区间。
    if (
        HF_UPLOAD
        and HF_REPO_ID
        and SOURCE == "list"
        and bool(RANGE_LOCKS_ENABLED)
        and (int(max_images_eff) < 0 or int(max_images_eff) >= int(RANGE_LOCK_MIN_IMAGES))
    ):
        try:
            range_coord = hf_sync.RangeLockSync(HF_REPO_ID)
            try:
                if float(RANGE_HEARTBEAT_SECS) > 0:
                    range_coord.heartbeat_secs = float(RANGE_HEARTBEAT_SECS)
            except Exception:
                pass
            try:
                if float(RANGE_PROGRESS_SECS) > 0:
                    range_coord.progress_secs = float(RANGE_PROGRESS_SECS)
            except Exception:
                pass
            try:
                if float(RANGE_ABANDONED_SECS) > 0:
                    range_coord.abandoned_secs = float(RANGE_ABANDONED_SECS)
            except Exception:
                pass
        except Exception:
            range_coord = None

    try:
        os.environ.setdefault("HF_INDEX_COMPACT", "1")
        os.environ.setdefault("HF_INDEX_COMPACT_DROP_EMPTY", "1")
        os.environ.setdefault("HF_INDEX_TEXT_MODE", "full")
        os.environ.setdefault("HF_INDEX_ASSET_MODE", "none")
        os.environ.setdefault("HF_INDEX_DROP_DERIVABLE_URLS", "1")
        os.environ.setdefault("HF_INDEX_DROP_USER_NAME", "1")
        os.environ.setdefault("HF_INDEX_DROP_UNSPLASH_ID", "1")
    except Exception:
        pass

    index_sync_obj = None
    if HF_UPLOAD and HF_WRITE_INDEX and HF_REPO_ID and HF_INDEX_REPO_PATH:
        try:
            index_sync_obj = hf_index_sync.IndexSync(
                HF_REPO_ID,
                repo_type=HF_REPO_TYPE,
                repo_path=HF_INDEX_REPO_PATH,
                save_dir=SAVE_DIR,
                hf_upload=HF_UPLOAD,
                hf_index_flush_every=HF_INDEX_FLUSH_EVERY,
                hf_index_flush_secs=HF_INDEX_FLUSH_SECS,
                hf_index_refresh_secs=HF_INDEX_REFRESH_SECS,
                debug_fn=print_debug,
            )
        except Exception:
            index_sync_obj = None

    remote_done_fn = None
    try:
        backend = str(os.getenv("HF_DONE_BACKEND", "index") or "").strip().lower()
    except Exception:
        backend = "index"

    if backend in ("none", "disabled"):
        remote_done_fn = None
    elif backend in ("index", "jsonl"):
        try:
            if index_sync_obj is not None:
                def _remote_done_index(pid):
                    try:
                        index_sync_obj.maybe_refresh(False)
                    except Exception:
                        pass
                    return str(pid) in (index_sync_obj.indexed or set())

                remote_done_fn = _remote_done_index
        except Exception:
            remote_done_fn = None
    elif backend in ("parquet", "viewer"):
        try:
            ds = str(os.getenv("HF_DONE_DATASET", HF_REPO_ID) or "").strip()
            tok = str(os.getenv("HF_TOKEN", "") or "").strip() or None
            cfg_name = str(os.getenv("HF_DONE_CONFIG", "") or "").strip()
            split_name = str(os.getenv("HF_DONE_SPLIT", "") or "").strip()
            col = str(os.getenv("HF_DONE_COLUMN", "image_id") or "image_id").strip() or "image_id"
            if (not cfg_name) or (not split_name):
                sp = parquet_tools.viewer_splits(dataset=ds, token=tok)
                try:
                    first = ((sp.get("splits") or [])[0]) if isinstance(sp, dict) else None
                    cfg_name = cfg_name or str((first or {}).get("config") or "").strip()
                    split_name = split_name or str((first or {}).get("split") or "").strip()
                except Exception:
                    pass
            if ds and cfg_name and split_name:
                remote_done_fn = lambda pid: parquet_tools.viewer_filter_contains(
                    dataset=ds,
                    config=cfg_name,
                    split=split_name,
                    column=col,
                    value=str(pid),
                    token=tok,
                )
        except Exception:
            remote_done_fn = None
    elif backend in ("duckdb",):
        try:
            ds = str(os.getenv("HF_DONE_DATASET", HF_REPO_ID) or "").strip()
            tok = str(os.getenv("HF_TOKEN", "") or "").strip() or None
            cfg_name = str(os.getenv("HF_DONE_CONFIG", "") or "").strip() or None
            split_name = str(os.getenv("HF_DONE_SPLIT", "") or "").strip() or None
            col = str(os.getenv("HF_DONE_COLUMN", "image_id") or "image_id").strip() or "image_id"
            if ds:
                remote_done_fn = lambda pid: parquet_tools.duckdb_contains(
                    dataset=ds,
                    config=cfg_name,
                    split=split_name,
                    column=col,
                    value=str(pid),
                    token=tok,
                )
        except Exception:
            remote_done_fn = None

    def _upload_sample_pair(repo_id: str, image_id: str, image_path: str, ply_path: str):
        return hf_upload.upload_sample_pair(
            repo_id=repo_id,
            image_id=image_id,
            image_path=image_path,
            ply_path=ply_path,
            hf_subdir=HF_SUBDIR,
            repo_type=HF_REPO_TYPE,
            gsplat_enabled=GSPLAT_UPLOAD,
            gsplat_base=GSPLAT_BASE,
            gsplat_expiration_type=GSPLAT_EXPIRATION_TYPE,
            gsplat_filter_visibility=GSPLAT_FILTER_VISIBILITY,
            splat_transform_bin=SPLAT_TRANSFORM_BIN,
            gsplat_use_small_ply=GSPLAT_USE_SMALL_PLY,
            spz_enabled=SPZ_EXPORT,
            spz_tool=SPZ_TOOL,
            gsbox_bin=GSBOX_BIN,
            gsbox_spz_quality=GSBOX_SPZ_QUALITY,
            gsbox_spz_version=GSBOX_SPZ_VERSION,
            gsconverter_bin=GSCONVERTER_BIN,
            gsconverter_compression_level=GSCONVERTER_COMPRESSION_LEVEL,
            debug_fn=print_debug,
        )

    def _upload_sample_pairs(repo_id: str, tasks: list):
        return hf_upload.upload_sample_pairs(
            repo_id=repo_id,
            tasks=tasks,
            hf_subdir=HF_SUBDIR,
            repo_type=HF_REPO_TYPE,
            gsplat_enabled=GSPLAT_UPLOAD,
            gsplat_base=GSPLAT_BASE,
            gsplat_expiration_type=GSPLAT_EXPIRATION_TYPE,
            gsplat_filter_visibility=GSPLAT_FILTER_VISIBILITY,
            splat_transform_bin=SPLAT_TRANSFORM_BIN,
            gsplat_use_small_ply=GSPLAT_USE_SMALL_PLY,
            spz_enabled=SPZ_EXPORT,
            spz_tool=SPZ_TOOL,
            gsbox_bin=GSBOX_BIN,
            gsbox_spz_quality=GSBOX_SPZ_QUALITY,
            gsbox_spz_version=GSBOX_SPZ_VERSION,
            gsconverter_bin=GSCONVERTER_BIN,
            gsconverter_compression_level=GSCONVERTER_COMPRESSION_LEVEL,
            debug_fn=print_debug,
        )

    cfg = pipeline.PipelineConfig(
        save_dir=SAVE_DIR,
        control_dir=CONTROL_DIR,
        pause_file=PAUSE_FILE,
        stop_file=STOP_FILE,
        idle_sleep_s=float(IDLE_SLEEP_S or 0.5),
        source=SOURCE,
        queries=QUERIES,
        search_orders=SEARCH_ORDERS,
        list_orders=LIST_ORDERS,
        list_per_page=int(LIST_PER_PAGE),
        list_auto_seek=bool(LIST_AUTO_SEEK),
        list_seek_back_pages=int(LIST_SEEK_BACK_PAGES),
        max_candidates=int(max_candidates_eff),
        max_images=int(max_images_eff),
        range_size=int(RANGE_SIZE),
        stop_on_rate_limit=bool(STOP_ON_RATE_LIMIT),
        input_images_dir=INPUT_IMAGES_DIR,
        inject_exif=bool(INJECT_EXIF),
        download_queue_max=int(DOWNLOAD_QUEUE_MAX),
        upload_queue_max=int(UPLOAD_QUEUE_MAX),
        upload_workers=int(UPLOAD_WORKERS),
        hf_upload=bool(HF_UPLOAD),
        hf_repo_id=HF_REPO_ID,
        hf_lock_stale_secs=float(HF_LOCK_STALE_SECS),
        hf_squash_every=int(HF_SQUASH_EVERY or 0),
        ply_delete_after_upload=bool(PLY_DELETE_AFTER_UPLOAD),
        ply_keep_last=int(PLY_KEEP_LAST),
        gaussians_dir=GAUSSIANS_DIR,
        sigint_window_s=float(os.getenv("SIGINT_WINDOW_S", "2.0")),
    )

    run_sharp_predict_once_fn = _run_sharp_predict_once
    if SKIP_PREDICT:
        def _skip_predict(_input_path: str):
            return []
        try:
            _skip_predict._skip_predict = True
        except Exception:
            pass
        run_sharp_predict_once_fn = _skip_predict

    pipeline.run(
        cfg,
        checked_ids=checked_ids,
        coord=coord,
        range_coord=range_coord,
        remote_done_fn=remote_done_fn,
        index_sync=index_sync_obj,
        upload_sample_pair_fn=_upload_sample_pair,
        upload_sample_pairs_fn=_upload_sample_pairs,
        try_super_squash_fn=_hf_try_super_squash,
        run_sharp_predict_once_fn=run_sharp_predict_once_fn,
        local_has_focal_exif_fn=_local_has_focal_exif,
        inject_focal_exif_if_missing_fn=inject_focal_exif_if_missing,
        debug_fn=print_debug,
    )



def main():
    print_debug("===== 脚本启动：全量验焦+本地防重 =====")
    pmode = str(os.getenv("PARQUET_MODE", "") or "").strip().lower()
    if pmode:
        dataset = str(os.getenv("PARQUET_DATASET", HF_REPO_ID) or "").strip()
        cfg = str(os.getenv("PARQUET_CONFIG", "") or "").strip()
        split = str(os.getenv("PARQUET_SPLIT", "") or "").strip()
        token = str(os.getenv("HF_TOKEN", "") or "").strip() or None
        try:
            if pmode in ("list", "parquet"):
                try:
                    obj = parquet_tools.viewer_list_parquet_files(dataset=dataset, token=token)
                except Exception:
                    obj = parquet_tools.hub_list_parquet_urls(dataset=dataset, token=token)
                print_debug(json.dumps(obj, ensure_ascii=False, indent=2))
                return

            if pmode in ("rows", "jump"):
                offset = int(str(os.getenv("PARQUET_OFFSET", "0") or "0").strip())
                length = int(str(os.getenv("PARQUET_LENGTH", "100") or "100").strip())
                obj = parquet_tools.viewer_rows(
                    dataset=dataset,
                    config=cfg,
                    split=split,
                    offset=offset,
                    length=length,
                    token=token,
                )
                print_debug(json.dumps(obj, ensure_ascii=False, indent=2))
                return

            if pmode == "search":
                q = str(os.getenv("PARQUET_QUERY", "") or "").strip()
                offset = int(str(os.getenv("PARQUET_OFFSET", "0") or "0").strip())
                length = int(str(os.getenv("PARQUET_LENGTH", "100") or "100").strip())
                obj = parquet_tools.viewer_search(
                    dataset=dataset,
                    config=cfg,
                    split=split,
                    query=q,
                    offset=offset,
                    length=length,
                    token=token,
                )
                print_debug(json.dumps(obj, ensure_ascii=False, indent=2))
                return

            if pmode == "filter":
                w = str(os.getenv("PARQUET_WHERE", "") or "").strip()
                offset = int(str(os.getenv("PARQUET_OFFSET", "0") or "0").strip())
                length = int(str(os.getenv("PARQUET_LENGTH", "100") or "100").strip())
                obj = parquet_tools.viewer_filter(
                    dataset=dataset,
                    config=cfg,
                    split=split,
                    where=w,
                    offset=offset,
                    length=length,
                    token=token,
                )
                print_debug(json.dumps(obj, ensure_ascii=False, indent=2))
                return
        except Exception as e:
            print_debug(f"PARQUET_MODE failed | mode={pmode} | err={type(e).__name__}: {str(e)}")
            return

    run_pipeline()

if __name__ == "__main__":
    main()

