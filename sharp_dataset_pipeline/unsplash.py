import re
import time
from urllib.parse import urlencode

import requests


_UNSPLASH_ACCESS_KEY = None
_APP_NAME = "sharp-ply-share"
_API_BASE = "https://api.unsplash.com"
_PER_PAGE = 10
_LIST_PER_PAGE = 30
_STOP_ON_RATE_LIMIT = True

_session = requests.Session()
_next_api_allowed_ts = 0.0
_api_backoff_seconds = 0.0
_rate_limited = False
_debug = None


def configure_unsplash(
    *,
    access_key: str,
    app_name: str,
    api_base: str,
    per_page: int,
    list_per_page: int,
    stop_on_rate_limit: bool,
    debug_fn=None,
):
    global _UNSPLASH_ACCESS_KEY, _APP_NAME, _API_BASE
    global _PER_PAGE, _LIST_PER_PAGE, _STOP_ON_RATE_LIMIT, _debug

    _UNSPLASH_ACCESS_KEY = str(access_key or '').strip() or None
    _APP_NAME = str(app_name or '').strip() or _APP_NAME
    _API_BASE = str(api_base or '').strip() or "https://api.unsplash.com"
    try:
        _PER_PAGE = int(per_page)
    except Exception:
        _PER_PAGE = 10
    try:
        _LIST_PER_PAGE = int(list_per_page)
    except Exception:
        _LIST_PER_PAGE = 30
    _STOP_ON_RATE_LIMIT = bool(stop_on_rate_limit)
    _debug = debug_fn


def is_rate_limited() -> bool:
    return bool(_rate_limited)


def rate_limit_wait_s(default_s: float = 3600.0) -> float:
    try:
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
    now = time.time()
    wait_s = max(0.0, _next_api_allowed_ts - now)
    if wait_s > 0:
        _d(f"API节流：等待 {round(wait_s, 2)}s")
        time.sleep(wait_s)


def _note_api_request_done(min_interval_s=0.75):
    global _next_api_allowed_ts, _api_backoff_seconds
    _api_backoff_seconds = 0.0
    _next_api_allowed_ts = time.time() + float(min_interval_s)


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
    return urlencode({"utm_source": _APP_NAME, "utm_medium": "referral"})


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
    if not _UNSPLASH_ACCESS_KEY:
        raise RuntimeError("Missing UNSPLASH_ACCESS_KEY env var")
    return {
        "Authorization": f"Client-ID {_UNSPLASH_ACCESS_KEY}",
        "Accept-Version": "v1",
        "User-Agent": f"{_APP_NAME}",
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
