import asyncio
import os
import random
import threading
import subprocess
import time
import re
from urllib.parse import urlencode, urlparse
from typing import List, Dict, Optional, Any, Callable, Union
from datetime import datetime

import requests

try:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except Exception:  # pragma: no cover
    HTTPAdapter = None
    Retry = None

# For auto-registration
_AUTO_REG_LOCK = threading.Lock()
_IS_REGISTERING = False

_AUTO_REG_DISABLED_UNTIL_TS: float = 0.0
_AUTO_REG_ACCOUNT_RATE_LIMITED: bool = False


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
_all_keys_rate_limited_last_log_ts = 0.0
_API_REQ_LOCK = threading.Lock()
_debug = None


def _configure_session_retries() -> None:
    global _session
    if HTTPAdapter is None or Retry is None:
        return
    try:
        total = int(os.getenv("UNSPLASH_HTTP_RETRIES", "6") or "6")
    except Exception:
        total = 6
    try:
        backoff = float(os.getenv("UNSPLASH_HTTP_BACKOFF", "0.6") or "0.6")
    except Exception:
        backoff = 0.6
    try:
        retry = Retry(
            total=max(0, total),
            connect=max(0, total),
            read=max(0, total),
            status=max(0, total),
            backoff_factor=max(0.0, float(backoff)),
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
        _session.mount("https://", adapter)
        _session.mount("http://", adapter)
    except Exception:
        return


def _sleep_retry(tries: int) -> None:
    try:
        base = float(os.getenv("UNSPLASH_NET_RETRY_BASE", "0.8") or "0.8")
    except Exception:
        base = 0.8
    try:
        cap = float(os.getenv("UNSPLASH_NET_RETRY_CAP", "20") or "20")
    except Exception:
        cap = 20.0
    try:
        t = min(float(cap), float(base) * (2 ** max(0, int(tries) - 1)))
        t = float(t) * (0.7 + 0.6 * random.random())
        time.sleep(max(0.2, float(t)))
    except Exception:
        time.sleep(1.0)


def _is_unsplash_api_url(url: str) -> bool:
    try:
        s = str(url or "").strip()
    except Exception:
        return False
    if not s:
        return False
    try:
        p = urlparse(s)
        host = str(p.netloc or "").split("@")[-1]
        host = host.split(":", 1)[0].lower()
        if not host:
            return False
        return host == "api.unsplash.com" or host.endswith(".api.unsplash.com")
    except Exception:
        return False


def _session_get(url: str, **kwargs):
    if _is_unsplash_api_url(url):
        with _API_REQ_LOCK:
            return _session.get(url, **kwargs)
    return _session.get(url, **kwargs)


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

    try:
        _configure_session_retries()
    except Exception:
        pass


def resolve_unsplash_keys_json_path(*, base_dir: Optional[str] = None) -> str:
    try:
        p = str(os.getenv("UNSPLASH_ACCESS_KEY_JSON", "") or "").strip()
        if p:
            return os.path.abspath(p)
    except Exception:
        pass

    try:
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
        return os.path.abspath(os.path.join(str(base_dir), "UNSPLASH_ACCESS_KEY.json"))
    except Exception:
        return os.path.abspath("UNSPLASH_ACCESS_KEY.json")


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


def _update_persistence(new_key: str, app_name: str):
    """Update both .env and UNSPLASH_ACCESS_KEY.json with the new key."""
    # 1. Update .env (existing logic)
    _update_env_file(new_key)
    
    # 2. Update UNSPLASH_ACCESS_KEY.json
    try:
        json_path = resolve_unsplash_keys_json_path()
        data = []
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        import json
                        data = json.loads(content)
                        if isinstance(data, dict):
                            data = [data]
            except Exception:
                data = []
        
        # Check if already exists
        exists = any(item.get("access_key") == new_key or item.get("UNSPLASH_ACCESS_KEY") == new_key for item in data)
        if not exists:
            data.append({
                "UNSPLASH_ACCESS_KEY": new_key,
                "UNSPLASH_APP_NAME": app_name
            })
            with open(json_path, "w", encoding="utf-8") as f:
                import json
                json.dump(data, f, indent=2, ensure_ascii=False)
            _d(f"[AUTO_REG] 已更新 UNSPLASH_ACCESS_KEY.json 并添加新 Key.")
    except Exception as e:
        _d(f"[AUTO_REG] 更新 JSON 失败: {str(e)}")


def _trigger_auto_registration():
    """Trigger the external registration script and wait for it to finish."""
    global _IS_REGISTERING

    now = time.time()
    try:
        if float(_AUTO_REG_DISABLED_UNTIL_TS) > float(now):
            _d(
                f"[AUTO_REG] 已暂停自动注册（疑似账号级限流），需等待 {int(float(_AUTO_REG_DISABLED_UNTIL_TS) - float(now))}s"
            )
            return
    except Exception:
        pass
    
    # 检查是否允许自动化注册 - 增加从 .env 实时读取逻辑，并确保区分注释和实际值
    allow_reg = str(os.getenv("ALLOW_AUTO_REG", "0")).strip()
    if allow_reg != "1":
        try:
            env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if line.startswith("ALLOW_AUTO_REG=1"):
                            allow_reg = "1"
                            os.environ["ALLOW_AUTO_REG"] = "1"
                            break
                        if "=" in line:
                            parts = line.split("=", 1)
                            if parts[0].strip() == "ALLOW_AUTO_REG" and parts[1].strip() == "1":
                                allow_reg = "1"
                                os.environ["ALLOW_AUTO_REG"] = "1"
                                break
        except Exception:
            pass

    if allow_reg != "1":
        # 调试信息：输出当前环境变量状态，帮助定位为什么没读到
        _d(f"[AUTO_REG] 未获得许可 (ALLOW_AUTO_REG={os.getenv('ALLOW_AUTO_REG')})")
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
        import sys
        cmd = [sys.executable, script_path, app_name]
        
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        if process.returncode == 0:
            # Extract access key from stdout if possible
            import re
            new_key = None
            try:
                patterns = [
                    r"Access\s*Key\s*:\s*(\S+)",
                    r"应用注册成功[！!]?\s*Key\s*:\s*(\S+)",
                    r"\bKey\s*:\s*(\S+)",
                ]
                for pat in patterns:
                    m = re.search(pat, stdout)
                    if m:
                        new_key = m.group(1).strip()
                        break
            except Exception:
                new_key = None

            if new_key:
                _d(f"[AUTO_REG] 新应用注册成功! Key: {new_key}")
                
                # Add to pool
                with _AUTO_REG_LOCK:
                    global _ACTIVE_KEY_IDX
                    _KEY_POOL.append(
                        {
                            "access_key": new_key,
                            "app_name": app_name,
                            "_auto_registered_ts": time.time(),
                            "_auto_registered_new": True,
                        }
                    )
                    _KEY_NEXT_API_ALLOWED_TS.append(0.0)
                    _KEY_API_BACKOFF_SECONDS.append(0.0)
                    _KEY_RATE_LIMITED.append(False)
                    try:
                        _ACTIVE_KEY_IDX = max(0, int(len(_KEY_POOL) - 1))
                    except Exception:
                        _ACTIVE_KEY_IDX = 0
                    # 关键修复：确保全局速率限制标志被清除，以便程序能立即使用新 Key 继续
                    global _rate_limited
                    _rate_limited = False
                
                # Persistent storage
                _update_persistence(new_key, app_name)
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
    now = time.time()
    try:
        if not _KEY_POOL:
            _rate_limited = True
            _next_api_allowed_ts = now + 3600.0
            # 池子为空，触发注册
            _trigger_auto_registration()
            if _KEY_POOL:
                # 注册成功后，递归调用一次以初始化状态
                return _ensure_key_for_request()
            return False

        def _ready(i: int) -> bool:
            try:
                return float(_KEY_NEXT_API_ALLOWED_TS[i]) <= now
            except Exception:
                return True

        # 1. 检查当前 Key 是否可用
        cur = max(0, min(int(_ACTIVE_KEY_IDX), len(_KEY_POOL) - 1))
        if _ready(cur):
            _KEY_RATE_LIMITED[cur] = False
            _rate_limited = False
            _next_api_allowed_ts = now
            return True

        # 2. 寻找池中其他可用的 Key
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
            if best_ts is None or ts < best_ts:
                best = i
                best_ts = ts

        # 3. 如果找到了现在就能用的 Key
        if best is not None and best_ts <= now:
            _ACTIVE_KEY_IDX = best
            _KEY_RATE_LIMITED[best] = False
            _rate_limited = False
            _next_api_allowed_ts = now
            _KEY_API_BACKOFF_SECONDS[best] = 0.0
            return True

        # 4. 如果所有 Key 都不可用，判断是否触发自动注册
        if best is not None:
            wait_time = best_ts - now
            all_rl = False
            try:
                all_rl = bool(_KEY_POOL) and bool(_KEY_RATE_LIMITED) and all(bool(x) for x in _KEY_RATE_LIMITED)
            except Exception:
                all_rl = False

            # Log wording: distinguish "real" rate limit vs local throttling interval.
            try:
                global _all_keys_rate_limited_last_log_ts
                last = float(_all_keys_rate_limited_last_log_ts or 0.0)
                if float(now) - float(last) >= 2.0:
                    if all_rl:
                        _d(f"[AUTO_REG] 现有 Key 均被限速 (需等待 {int(wait_time)}s)")
                    else:
                        _d(f"API节流：等待 {round(max(0.0, float(wait_time)), 2)}s")
                    _all_keys_rate_limited_last_log_ts = float(now)
            except Exception:
                if all_rl:
                    _d(f"[AUTO_REG] 现有 Key 均被限速 (需等待 {int(wait_time)}s)")
                else:
                    _d(f"API节流：等待 {round(max(0.0, float(wait_time)), 2)}s")

            # If the best key will be available soon, prefer waiting rather than creating a new app.
            try:
                short_wait_s = float(os.getenv("AUTO_REG_SHORT_WAIT_SECS", "30") or "30")
            except Exception:
                short_wait_s = 30.0
            if wait_time <= float(short_wait_s):
                _rate_limited = True
                _next_api_allowed_ts = float(best_ts)
                try:
                    # Break potential busy-loop callers (e.g. repeated is_rate_limited checks)
                    sleep_s = max(0.0, min(float(wait_time), 1.0))
                    if sleep_s > 0:
                        time.sleep(float(sleep_s))
                except Exception:
                    pass
                return False

            # If not all keys are actually rate-limited, do NOT attempt auto-registration.
            if not all_rl:
                _rate_limited = True
                _next_api_allowed_ts = float(best_ts)
                return False

            allow_reg = str(os.getenv("ALLOW_AUTO_REG", "0") or "0").strip()
            if allow_reg != "1":
                _rate_limited = True
                _next_api_allowed_ts = float(best_ts)
                return False

            _d("[AUTO_REG] 等待时间较长，尝试自动注册新 Key...")
            try:
                if float(_AUTO_REG_DISABLED_UNTIL_TS) > float(now):
                    _d("[AUTO_REG] 自动注册已暂停（疑似账号级限流），改为等待现有 Key 复活")
                else:
                    _trigger_auto_registration()
            except Exception:
                _trigger_auto_registration()
            
            # 注册后立即重新检查
            now_after = time.time()
            for i in range(len(_KEY_POOL)):
                try:
                    ts_after = float(_KEY_NEXT_API_ALLOWED_TS[i])
                except Exception:
                    ts_after = 0.0
                if ts_after <= now_after:
                    _ACTIVE_KEY_IDX = i
                    _KEY_RATE_LIMITED[i] = False
                    _rate_limited = False
                    _next_api_allowed_ts = now_after
                    return True
            
            # 如果注册后的新 Key 也还没生效，或者注册失败，再进入短时间休眠
            _rate_limited = True
            _next_api_allowed_ts = best_ts
            return False

        # 兜底
        _rate_limited = True
        _next_api_allowed_ts = now + 3600.0
        return False
    except Exception as e:
        _d(f"_ensure_key_for_request 发生异常: {e}")
        _rate_limited = True
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
            try:
                obj = _KEY_POOL[idx]
                if isinstance(obj, dict) and obj.get("_auto_registered_new") is True:
                    obj["_auto_registered_new"] = False
            except Exception:
                pass
    except Exception:
        pass


def _note_api_rate_limited(response):
    global _next_api_allowed_ts, _api_backoff_seconds
    stop_wait_default_s = 1800.0
    try:
        stop_wait_default_s = float(os.getenv("UNSPLASH_RATE_LIMIT_WAIT_SECS", "1800") or "1800")
    except Exception:
        stop_wait_default_s = 1800.0
    retry_after = response.headers.get("Retry-After")
    wait_s = None
    if retry_after:
        try:
            wait_s = float(str(retry_after).strip())
        except Exception:
            wait_s = None

    if wait_s is None:
        if _STOP_ON_RATE_LIMIT:
            wait_s = float(stop_wait_default_s)
            _api_backoff_seconds = float(wait_s)
        else:
            if _api_backoff_seconds <= 0:
                _api_backoff_seconds = 2.0
            else:
                _api_backoff_seconds = min(120.0, _api_backoff_seconds * 2.0)
            wait_s = _api_backoff_seconds
    else:
        if _STOP_ON_RATE_LIMIT:
            wait_s = max(float(stop_wait_default_s), float(wait_s))
        _api_backoff_seconds = max(_api_backoff_seconds, float(wait_s))

    _next_api_allowed_ts = time.time() + float(wait_s)
    try:
        if _KEY_POOL:
            idx = max(0, min(int(_ACTIVE_KEY_IDX), len(_KEY_POOL) - 1))
            _KEY_NEXT_API_ALLOWED_TS[idx] = max(float(_KEY_NEXT_API_ALLOWED_TS[idx]), float(_next_api_allowed_ts))
            _KEY_API_BACKOFF_SECONDS[idx] = float(_api_backoff_seconds)
            _KEY_RATE_LIMITED[idx] = True

            try:
                obj = _KEY_POOL[idx]
                if isinstance(obj, dict) and obj.get("_auto_registered_new") is True:
                    reg_ts = float(obj.get("_auto_registered_ts") or 0.0)
                    age_s = time.time() - reg_ts
                    if reg_ts > 0 and age_s <= 180.0:
                        global _AUTO_REG_DISABLED_UNTIL_TS, _AUTO_REG_ACCOUNT_RATE_LIMITED
                        _AUTO_REG_ACCOUNT_RATE_LIMITED = True
                        _AUTO_REG_DISABLED_UNTIL_TS = max(float(_AUTO_REG_DISABLED_UNTIL_TS), float(_next_api_allowed_ts))
                        _d(
                            f"[AUTO_REG] 新注册 Key 立即被限速（age={int(age_s)}s），疑似账号级限流：暂停自动注册直至 {datetime.fromtimestamp(float(_AUTO_REG_DISABLED_UNTIL_TS))}"
                        )
            except Exception:
                pass

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


def build_download_location(photo_id: str) -> str | None:
    try:
        pid = str(photo_id or "").strip()
        if not pid:
            return None
        return add_utm(f"{_API_BASE}/photos/{pid}/download")
    except Exception:
        return None


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
                _wait_for_api_slot()
                clear_rate_limited()
            _wait_for_api_slot()
            response = _session_get(
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
            if tries < 8:
                _d(f"拉取图片失败（可重试） | query={query} | 页码={page} | err={str(e)}")
                _sleep_retry(tries)
                continue
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
                _wait_for_api_slot()
                clear_rate_limited()
            _wait_for_api_slot()
            response = _session_get(
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
            if tries < 8:
                _d(f"拉取图片失败（可重试） | list/photos | 页码={page} | err={str(e)}")
                _sleep_retry(tries)
                continue
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
                _wait_for_api_slot()
                clear_rate_limited()
            _wait_for_api_slot()
            response = _session_get(
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
                try:
                    now = time.time()
                    wait_s = max(0.0, float(_next_api_allowed_ts) - float(now))
                    _d(
                        f"download_location waiting due to rate limit | wait={int(wait_s)}s | url={str(download_location)[:120]}"
                    )
                except Exception:
                    wait_s = 0.0
                _wait_for_api_slot()
                clear_rate_limited()
                continue
            _wait_for_api_slot()
            response = _session_get(download_location, headers=_headers(), timeout=20)
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
            try:
                if _is_unsplash_api_url(download_url):
                    _ensure_key_for_request()
                    if _STOP_ON_RATE_LIMIT and _rate_limited:
                        return False
                    _wait_for_api_slot()
                    r = _session_get(download_url, headers=_headers(), timeout=30, stream=True)
                else:
                    r = _session_get(download_url, timeout=30, stream=True)
            except Exception:
                r = _session_get(download_url, timeout=30, stream=True)
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
