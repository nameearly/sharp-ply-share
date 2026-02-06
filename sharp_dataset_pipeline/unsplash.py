import asyncio
import os
import random
import threading
import subprocess
from typing import List, Dict, Optional, Any, Callable, Union

import requests

# For auto-registration
_AUTO_REG_LOCK = threading.Lock()
_IS_REGISTERING = False


_UNSPLASH_ACCESS_KEY = None
_APP_NAME = "sharp-ply-share"
_API_BASE = "https://api.unsplash.com"
_PER_PAGE = 10
_LIST_PER_PAGE = 30
_STOP_ON_RATE_LIMIT = True

_KEY_POOL: list[dict] = []
_KEY_NEXT_API_ALLOWED_TS: list[float] = []
_KEY_API_BACKOFF_SECONDS: list[float] = []
_KEY_RATE_LIMITED: list[bool] = []
_ACTIVE_KEY_IDX: int = 0

_session = requests.Session()
_next_api_allowed_ts = 0.0
_api_backoff_seconds = 0.0
_rate_limited = False
_debug = None


def configure_unsplash(
    *,
    access_key,
    app_name: str,
    api_base: str,
    per_page: int,
    list_per_page: int,
    stop_on_rate_limit: bool,
    debug_fn=None,
):
    global _UNSPLASH_ACCESS_KEY, _APP_NAME, _API_BASE
    global _PER_PAGE, _LIST_PER_PAGE, _STOP_ON_RATE_LIMIT, _debug
    global _KEY_POOL, _KEY_NEXT_API_ALLOWED_TS, _KEY_API_BACKOFF_SECONDS, _KEY_RATE_LIMITED, _ACTIVE_KEY_IDX
    global _next_api_allowed_ts, _api_backoff_seconds, _rate_limited

    pool: list[dict] = []
    try:
        if isinstance(access_key, (list, tuple)):
            for it in (access_key or []):
                if it is None:
                    continue
                if isinstance(it, dict):
                    k = (
                        it.get("UNSPLASH_ACCESS_KEY")
                        or it.get("unsplash_access_key")
                        or it.get("access_key")
                        or it.get("key")
                    )
                    k = str(k or "").strip()
                    if not k:
                        continue
                    an = (
                        it.get("UNSPLASH_APP_NAME")
                        or it.get("unsplash_app_name")
                        or it.get("app_name")
                        or app_name
                    )
                    an = str(an or "").strip() or str(app_name or "").strip() or _APP_NAME
                    pool.append({"access_key": k, "app_name": an})
                else:
                    k = str(it or "").strip()
                    if k:
                        an = str(app_name or "").strip() or _APP_NAME
                        pool.append({"access_key": k, "app_name": an})
        else:
            k = str(access_key or "").strip()
            if k:
                an = str(app_name or "").strip() or _APP_NAME
                pool.append({"access_key": k, "app_name": an})
    except Exception:
        pool = []

    _KEY_POOL = list(pool)
    _KEY_NEXT_API_ALLOWED_TS = [0.0 for _ in _KEY_POOL]
    _KEY_API_BACKOFF_SECONDS = [0.0 for _ in _KEY_POOL]
    _KEY_RATE_LIMITED = [False for _ in _KEY_POOL]
    _ACTIVE_KEY_IDX = 0

    _UNSPLASH_ACCESS_KEY = str((_KEY_POOL[0].get("access_key") if _KEY_POOL else "") or "").strip() or None
    _APP_NAME = str(app_name or '').strip() or _APP_NAME
    _API_BASE = str(api_base or '').strip() or "https://api.unsplash.com"
    try:
        _PER_PAGE = max(1, int(per_page))
    except Exception:
        _PER_PAGE = 10
    try:
        _LIST_PER_PAGE = max(1, min(30, int(list_per_page)))
    except Exception:
        _LIST_PER_PAGE = 30
    _STOP_ON_RATE_LIMIT = bool(stop_on_rate_limit)
    _debug = debug_fn

    _next_api_allowed_ts = 0.0
    _api_backoff_seconds = 0.0
    _rate_limited = False


def load_unsplash_key_pool(json_path: str, *, default_app_name: Optional[str] = None) -> List[Dict[str, str]]:
    try:
        p = str(json_path or "").strip()
        if not p or not os.path.exists(p):
            return []
        with open(p, "r", encoding="utf-8") as f:
            raw = f.read()
    except Exception:
        return []

    def _normalize_items(obj: Any) -> List[Dict[str, str]]:
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
            if not an and default_app_name:
                an = str(default_app_name).strip()
            out.append({"access_key": k, "app_name": an or "sharp-ply-share"})
        return out

    try:
        import json

        return _normalize_items(json.loads(raw))
    except Exception:
        pass

    try:
        import json

        s = str(raw or "").strip()
        if not s:
            return []
        if s.startswith("{") and s.endswith("}"):
            s = "[" + s[1:-1] + "]"
        s = re.sub(r"(?m)\b(UNSPLASH_APP_NAME|UNSPLASH_ACCESS_KEY)\s*:", r'"\\1":', s)
        s = re.sub(r",\s*([}\]])", r"\\1", s)
        return _normalize_items(json.loads(s))
    except Exception:
        pass

    try:
        s = str(raw or "")
        try:
            blocks = re.findall(r"\{[^{}]*\}", s, flags=re.DOTALL)
        except Exception:
            blocks = []

        out = []
        for b in blocks or []:
            try:
                km = re.search(r"\bUNSPLASH_ACCESS_KEY\s*:\s*['\"]([^'\"]+)['\"]", b)
                if not km:
                    km = re.search(r"\bunsplash_access_key\s*:\s*['\"]([^'\"]+)['\"]", b)
                if not km:
                    km = re.search(r"\baccess_key\s*:\s*['\"]([^'\"]+)['\"]", b)
                if not km:
                    continue
                k = str(km.group(1) or "").strip()
                if not k:
                    continue

                am = re.search(r"\bUNSPLASH_APP_NAME\s*:\s*['\"]([^'\"]+)['\"]", b)
                if not am:
                    am = re.search(r"\bunsplash_app_name\s*:\s*['\"]([^'\"]+)['\"]", b)
                if not am:
                    am = re.search(r"\bapp_name\s*:\s*['\"]([^'\"]+)['\"]", b)
                an = str(am.group(1) if am else "").strip() or str(default_app_name or "").strip()

                out.append({"UNSPLASH_ACCESS_KEY": k, "UNSPLASH_APP_NAME": an})
            except Exception:
                continue

        out = _normalize_items(out)
        if out:
            return out
    except Exception:
        pass

    return []


def _active_key_obj() -> dict | None:
    try:
        if not _KEY_POOL:
            return None
        idx = max(0, min(int(_ACTIVE_KEY_IDX), len(_KEY_POOL) - 1))
        return _KEY_POOL[idx]
    except Exception:
        return None


def _active_app_name() -> str:
    try:
        obj = _active_key_obj()
        if isinstance(obj, dict):
            s = str(obj.get("app_name") or "").strip()
            if s:
                return s
        return str(_APP_NAME or "").strip() or "sharp-ply-share"
    except Exception:
        return str(_APP_NAME or "").strip() or "sharp-ply-share"


def _trigger_auto_registration():
    """Trigger the external registration script and wait for it to finish."""
    global _IS_REGISTERING
    
    # 检查是否允许自动化注册
    if str(os.getenv("ALLOW_AUTO_REG", "0")).strip() != "1":
        _d("[AUTO_REG] 未获得自动化注册许可，跳过。请设置 ALLOW_AUTO_REG=1")
        return

    with _AUTO_REG_LOCK:
        if _IS_REGISTERING:
            return
        _IS_REGISTERING = True
    
    try:
        app_name = f"sharp-ply-share-{random.randint(1000, 9999)}"
        _d(f"[AUTO_REG] 检测到 Key 池耗尽，尝试注册新应用: {app_name}")
        
        # Run the script using the current python interpreter
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts", "register_unsplash_app.py")
        
        # Use a separate process to avoid event loop conflicts if we are in one
        # Also, we want to run it in a way that might allow the user to see the browser if needed, 
        # but since this is a background pipeline, we try headless=False first but it might fail 
        # if no X server/UI is available. 
        # The script is designed to handle this.
        cmd = [sys.executable, script_path, app_name]
        
        # We need sys for sys.executable
        import sys
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            # Extract access key from stdout if possible
            match = re.search(r"Access Key: (\S+)", stdout)
            if match:
                new_key = match.group(1).strip()
                _d(f"[AUTO_REG] 新应用注册成功! Key: {new_key}")
                
                # Add to pool
                with _AUTO_REG_LOCK:
                    _KEY_POOL.append({"access_key": new_key, "app_name": app_name})
                    _KEY_NEXT_API_ALLOWED_TS.append(0.0)
                    _KEY_API_BACKOFF_SECONDS.append(0.0)
                    _KEY_RATE_LIMITED.append(False)
                
                # Try to append to .env or config if possible (optional but good)
                _update_env_file(new_key)
            else:
                _d(f"[AUTO_REG] 脚本运行成功但未在输出中找到 Key. Output: {stdout[:200]}")
        else:
            _d(f"[AUTO_REG] 脚本运行失败 (code={process.returncode}). Error: {stderr}")
            
    except Exception as e:
        _d(f"[AUTO_REG] 自动注册发生异常: {str(e)}")
    finally:
        with _AUTO_REG_LOCK:
            _IS_REGISTERING = False


def _update_env_file(new_key: str):
    """Attempt to append the new key to the .env file."""
    try:
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
        if not os.path.exists(env_path):
            return
            
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            
        updated = False
        new_lines = []
        for line in lines:
            if line.strip().startswith("UNSPLASH_API_KEYS="):
                parts = line.strip().split("=", 1)
                val = parts[1].strip()
                if new_key not in val:
                    if val:
                        new_val = f"{val},{new_key}"
                    else:
                        new_val = new_key
                    new_lines.append(f"UNSPLASH_API_KEYS={new_val}\n")
                    updated = True
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)
        
        if not updated:
            # Maybe the key wasn't there at all
            has_keys_var = any(l.strip().startswith("UNSPLASH_API_KEYS=") for l in lines)
            if not has_keys_var:
                new_lines.append(f"\nUNSPLASH_API_KEYS={new_key}\n")
                updated = True
                
        if updated:
            with open(env_path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
            _d(f"[AUTO_REG] 已更新 .env 文件并添加新 Key.")
    except Exception as e:
        _d(f"[AUTO_REG] 更新 .env 失败: {str(e)}")


def _ensure_key_for_request() -> bool:
    global _rate_limited, _next_api_allowed_ts, _ACTIVE_KEY_IDX
    try:
        if not _KEY_POOL:
            _rate_limited = True
            _next_api_allowed_ts = time.time() + 3600.0
            return False

        now = time.time()
        try:
            cur = max(0, min(int(_ACTIVE_KEY_IDX), len(_KEY_POOL) - 1))
        except Exception:
            cur = 0

        def _ready(i: int) -> bool:
            try:
                return float(_KEY_NEXT_API_ALLOWED_TS[i]) <= float(now)
            except Exception:
                return True

        if _ready(cur):
            try:
                _KEY_RATE_LIMITED[cur] = False
            except Exception:
                pass
            _rate_limited = False
            _next_api_allowed_ts = float(now)
            return True

        best = None
        best_ts = None
        for i in range(len(_KEY_POOL)):
            try:
                ts = float(_KEY_NEXT_API_ALLOWED_TS[i])
            except Exception:
                ts = 0.0
            if ts <= now:
                best = i
                best_ts = ts
                break
            
            # 允许提前 30 分钟（1800秒）尝试已被限速的 key
            # Unsplash 限速重置通常是一小时，但半小时可能已经恢复部分额度或重置
            effective_ts = ts
            if _KEY_RATE_LIMITED[i]:
                effective_ts = ts - 1800.0

            if best_ts is None or effective_ts < best_ts:
                best = i
                best_ts = effective_ts

        if best is None:
            # Trigger auto-registration if all keys are exhausted
            _trigger_auto_registration()
            
            # Re-check pool after registration attempt
            if _KEY_POOL:
                for i in range(len(_KEY_POOL)):
                    ts = float(_KEY_NEXT_API_ALLOWED_TS[i])
                    if ts <= now:
                        best = i
                        best_ts = ts
                        break
            
            if best is None:
                best = 0
                best_ts = float(now) + 3600.0

        prev_idx = int(cur)
        _ACTIVE_KEY_IDX = int(best)
        try:
            if _debug is not None and int(prev_idx) != int(best):
                _d(
                    f"Unsplash key rotate | from_idx={int(prev_idx)} to_idx={int(best)} | next_allowed_in_s={round(max(0.0, float(best_ts or 0.0) - float(now)), 2)}"
                )
        except Exception:
            pass
        try:
            if float(best_ts or 0.0) <= float(now):
                _KEY_RATE_LIMITED[best] = False
        except Exception:
            pass

        if float(best_ts or 0.0) <= float(now):
            _rate_limited = False
            _next_api_allowed_ts = float(now)
            return True

        _rate_limited = bool(_STOP_ON_RATE_LIMIT)
        _next_api_allowed_ts = float(best_ts or (now + 3600.0))
        return False
    except Exception:
        _rate_limited = bool(_STOP_ON_RATE_LIMIT)
        try:
            _next_api_allowed_ts = time.time() + 3600.0
        except Exception:
            _next_api_allowed_ts = 0.0
        return False


def is_rate_limited() -> bool:
    try:
        _ensure_key_for_request()
    except Exception:
        pass
    return bool(_rate_limited)


def rate_limit_wait_s(default_s: float = 3600.0) -> float:
    try:
        _ensure_key_for_request()
        if not _rate_limited:
            return 0.0
        now = time.time()
        wait_s = max(0.0, float(_next_api_allowed_ts) - float(now))
        if wait_s <= 0.0:
            return float(default_s)
        return float(wait_s)
    except Exception:
        return float(default_s)


def clear_rate_limited() -> None:
    global _rate_limited, _api_backoff_seconds, _next_api_allowed_ts
    try:
        _rate_limited = False
        _api_backoff_seconds = 0.0
        _next_api_allowed_ts = min(float(_next_api_allowed_ts), time.time() + 0.1)
        try:
            for i in range(len(_KEY_POOL)):
                _KEY_RATE_LIMITED[i] = False
                _KEY_API_BACKOFF_SECONDS[i] = 0.0
                _KEY_NEXT_API_ALLOWED_TS[i] = min(float(_KEY_NEXT_API_ALLOWED_TS[i]), time.time() + 0.1)
        except Exception:
            pass
    except Exception:
        return


def _d(msg: str) -> None:
    if _debug is None:
        return
    try:
        _debug(msg)
    except Exception:
        pass


def _wait_for_api_slot():
    global _next_api_allowed_ts
    _ensure_key_for_request()
    try:
        if _KEY_POOL:
            idx = max(0, min(int(_ACTIVE_KEY_IDX), len(_KEY_POOL) - 1))
            _next_api_allowed_ts = float(_KEY_NEXT_API_ALLOWED_TS[idx])
    except Exception:
        pass
    now = time.time()
    wait_s = max(0.0, _next_api_allowed_ts - now)
    if wait_s > 0:
        _d(f"API节流：等待 {round(wait_s, 2)}s")
        time.sleep(wait_s)


def _note_api_request_done(min_interval_s=0.75):
    global _next_api_allowed_ts, _api_backoff_seconds
    now = time.time()
    _api_backoff_seconds = 0.0
    _next_api_allowed_ts = now + float(min_interval_s)
    try:
        if _KEY_POOL:
            idx = max(0, min(int(_ACTIVE_KEY_IDX), len(_KEY_POOL) - 1))
            _KEY_API_BACKOFF_SECONDS[idx] = 0.0
            _KEY_NEXT_API_ALLOWED_TS[idx] = max(float(_KEY_NEXT_API_ALLOWED_TS[idx]), float(_next_api_allowed_ts))
            _KEY_RATE_LIMITED[idx] = False
    except Exception:
        pass


def _note_api_rate_limited(response):
    global _next_api_allowed_ts, _api_backoff_seconds
    retry_after = response.headers.get("Retry-After")
    wait_s = None
    if retry_after:
        try:
            wait_s = float(str(retry_after).strip())
        except Exception:
            wait_s = None

    if wait_s is None:
        if _STOP_ON_RATE_LIMIT:
            wait_s = 3600.0
            _api_backoff_seconds = float(wait_s)
        else:
            if _api_backoff_seconds <= 0:
                _api_backoff_seconds = 2.0
            else:
                _api_backoff_seconds = min(120.0, _api_backoff_seconds * 2.0)
            wait_s = _api_backoff_seconds
    else:
        if _STOP_ON_RATE_LIMIT:
            wait_s = max(3600.0, float(wait_s))
        _api_backoff_seconds = max(_api_backoff_seconds, float(wait_s))

    _next_api_allowed_ts = time.time() + float(wait_s)
    try:
        if _KEY_POOL:
            idx = max(0, min(int(_ACTIVE_KEY_IDX), len(_KEY_POOL) - 1))
            _KEY_NEXT_API_ALLOWED_TS[idx] = max(float(_KEY_NEXT_API_ALLOWED_TS[idx]), float(_next_api_allowed_ts))
            _KEY_API_BACKOFF_SECONDS[idx] = float(_api_backoff_seconds)
            _KEY_RATE_LIMITED[idx] = True
            _ensure_key_for_request()
    except Exception:
        pass
    try:
        hdr_retry_after = response.headers.get("Retry-After")
        hdr_remaining = response.headers.get("X-Ratelimit-Remaining")
        _d(
            f"Unsplash rate limit hit | status={response.status_code} | retry_after={hdr_retry_after} | remaining={hdr_remaining} | url={response.url}"
            + (response.text[:200] if response.text else "")
        )
    except Exception:
        pass


def _is_rate_limit_exceeded(response):
    try:
        if response is None:
            return False
        if response.status_code == 429:
            return True
        if response.status_code == 403 and response.text and "rate limit" in response.text.lower():
            return True
        remaining = response.headers.get("X-Ratelimit-Remaining")
        if remaining is not None and str(remaining).strip() == "0" and response.status_code != 200:
            return True
        return False
    except Exception:
        return False


def _mark_rate_limited(response):
    global _rate_limited
    _note_api_rate_limited(response)
    if _STOP_ON_RATE_LIMIT:
        _rate_limited = True


def _build_utm_query():
    return urlencode({"utm_source": _active_app_name(), "utm_medium": "referral"})


def add_utm(url):
    if not url:
        return url
    if "?" in url:
        return url + "&" + _build_utm_query()
    return url + "?" + _build_utm_query()


def parse_focal_length(value):
    if value is None:
        return None, None, None, None

    if isinstance(value, (int, float)):
        v = float(value)
        if v > 0:
            return f"{v}mm", v, v, v
        return None, None, None, None

    s = str(value).strip().lower()
    if not s:
        return None, None, None, None

    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)]
    if not nums:
        return None, None, None, None

    if len(nums) >= 2 and ("-" in s or " to " in s or s.startswith("to ") or s.endswith(" to")):
        mn = min(nums[0], nums[1])
        mx = max(nums[0], nums[1])
        avg = (mn + mx) / 2.0
        raw = f"{mn}-{mx}mm"
        return raw, mn, mx, avg

    v = nums[0]
    if v > 0:
        return f"{v}mm", v, v, v
    return None, None, None, None


def _headers():
    _ensure_key_for_request()
    obj = _active_key_obj()
    k = None
    if isinstance(obj, dict):
        k = obj.get("access_key")
    if not k:
        k = _UNSPLASH_ACCESS_KEY
    if not k:
        raise RuntimeError("Missing UNSPLASH_ACCESS_KEY env var")
    return {
        "Authorization": f"Client-ID {str(k).strip()}",
        "Accept-Version": "v1",
        "User-Agent": f"{_active_app_name()}",
    }


def fetch_photos(query, page=1, order_by="latest"):
    params = {
        "query": query,
        "page": page,
        "per_page": min(int(_PER_PAGE), 30),
        "order_by": order_by,
        "content_filter": "high",
    }

    tries = 0
    while True:
        tries += 1
        try:
            _ensure_key_for_request()
            if _STOP_ON_RATE_LIMIT and _rate_limited:
                return None
            _wait_for_api_slot()
            response = _session.get(
                f"{_API_BASE}/search/photos",
                headers=_headers(),
                params=params,
                timeout=20,
            )
            if _is_rate_limit_exceeded(response):
                _mark_rate_limited(response)
                if tries >= 8:
                    return None
                continue
            if response.status_code != 200:
                _note_api_request_done()
                _d(f"API请求失败 | 状态码={response.status_code} | 响应={response.text}")
                return None
            _note_api_request_done(min_interval_s=1.2)
            return (response.json() or {}).get("results", [])
        except Exception as e:
            _note_api_request_done()
            _d(f"拉取图片失败 | query={query} | 页码={page} | 错误={str(e)}")
            return None


def fetch_list_photos(page=1, order_by="latest"):
    params = {
        "page": page,
        "per_page": max(1, min(int(_LIST_PER_PAGE), 30)),
        "order_by": order_by,
    }

    tries = 0
    while True:
        tries += 1
        try:
            _ensure_key_for_request()
            if _STOP_ON_RATE_LIMIT and _rate_limited:
                return None
            _wait_for_api_slot()
            response = _session.get(
                f"{_API_BASE}/photos",
                headers=_headers(),
                params=params,
                timeout=20,
            )
            if _is_rate_limit_exceeded(response):
                _mark_rate_limited(response)
                if tries >= 8:
                    return None
                continue
            if response.status_code != 200:
                _note_api_request_done()
                _d(f"API请求失败 | 状态码={response.status_code} | 响应={response.text}")
                return None
            _note_api_request_done(min_interval_s=1.2)
            return response.json() or []
        except Exception as e:
            _note_api_request_done()
            _d(f"拉取图片失败 | list/photos | 页码={page} | 错误={str(e)}")
            return None


def fetch_photo_details(photo_id):
    if not photo_id:
        return None

    tries = 0
    while True:
        tries += 1
        try:
            _ensure_key_for_request()
            if _STOP_ON_RATE_LIMIT and _rate_limited:
                return None
            _wait_for_api_slot()
            response = _session.get(
                f"{_API_BASE}/photos/{photo_id}",
                headers=_headers(),
                timeout=20,
            )
            if _is_rate_limit_exceeded(response):
                _mark_rate_limited(response)
                if tries >= 8:
                    return None
                continue
            if response.status_code != 200:
                _note_api_request_done()
                _d(f"拉取详情失败 | id={photo_id} | 状态码={response.status_code} | 响应={response.text}")
                return None
            _note_api_request_done(min_interval_s=1.2)
            return response.json()
        except Exception as e:
            _note_api_request_done()
            if tries >= 3:
                _d(f"拉取详情异常 | id={photo_id} | 错误={str(e)}")
                return None
            time.sleep(1.0)


def _get_download_url(download_location):
    if not download_location:
        return None

    tries = 0
    while True:
        tries += 1
        try:
            _ensure_key_for_request()
            if _STOP_ON_RATE_LIMIT and _rate_limited:
                return None
            _wait_for_api_slot()
            response = _session.get(download_location, headers=_headers(), timeout=20)
            if _is_rate_limit_exceeded(response):
                _mark_rate_limited(response)
                if tries >= 8:
                    return None
                continue
            if response.status_code != 200:
                _note_api_request_done()
                _d(
                    f"download_location 请求失败 | status={response.status_code} | url={download_location} | resp={response.text[:200]}"
                )
                return None
            download_url = (response.json() or {}).get("url")
            _note_api_request_done(min_interval_s=1.2)
            return download_url
        except Exception:
            _note_api_request_done()
            if tries >= 3:
                return None
            time.sleep(1.0)


def download_image(download_location, output_path):
    if not download_location:
        _d(f"下载失败：download_location 为空 | path={output_path}")
        return False

    download_url = _get_download_url(download_location)
    if not download_url:
        _d(f"下载失败：无法获取 download url | download_location={download_location}")
        return False

    tries = 0
    while True:
        tries += 1
        try:
            r = _session.get(download_url, timeout=30, stream=True)
            if r.status_code != 200:
                _d(f"下载失败 | status={r.status_code} | url={download_url}")
                if tries >= 3:
                    return False
                time.sleep(1.0)
                continue
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if not chunk:
                        continue
                    f.write(chunk)
            return True
        except Exception:
            if tries >= 3:
                return False
            time.sleep(1.0)
