import os
import time

import requests


_DEFAULT_TIMEOUT_S = 20.0


def _auth_headers(token: str | None) -> dict:
    t = str(token or "").strip()
    if not t:
        return {}
    return {"Authorization": f"Bearer {t}"}


def _get_token(token: str | None) -> str | None:
    if token is not None:
        t = str(token).strip()
        return t or None
    t = str(os.getenv("HF_TOKEN", "") or "").strip()
    return t or None


def hub_list_parquet_urls(*, dataset: str, token: str | None = None, timeout_s: float = _DEFAULT_TIMEOUT_S):
    ds = str(dataset or "").strip()
    if not ds:
        raise ValueError("dataset is required")

    tok = _get_token(token)
    url = f"https://huggingface.co/api/datasets/{ds}/parquet"
    r = requests.get(url, headers=_auth_headers(tok), timeout=float(timeout_s))
    if r.status_code != 200:
        raise RuntimeError(f"hub parquet api failed | status={r.status_code} | body={r.text[:500]}")
    return r.json()


def viewer_list_parquet_files(*, dataset: str, token: str | None = None, timeout_s: float = _DEFAULT_TIMEOUT_S):
    ds = str(dataset or "").strip()
    if not ds:
        raise ValueError("dataset is required")

    tok = _get_token(token)
    url = "https://datasets-server.huggingface.co/parquet"
    r = requests.get(url, params={"dataset": ds}, headers=_auth_headers(tok), timeout=float(timeout_s))
    if r.status_code != 200:
        raise RuntimeError(f"datasets-server parquet failed | status={r.status_code} | body={r.text[:500]}")
    return r.json()


def viewer_splits(*, dataset: str, token: str | None = None, timeout_s: float = _DEFAULT_TIMEOUT_S):
    ds = str(dataset or "").strip()
    if not ds:
        raise ValueError("dataset is required")

    tok = _get_token(token)
    url = "https://datasets-server.huggingface.co/splits"
    r = requests.get(url, params={"dataset": ds}, headers=_auth_headers(tok), timeout=float(timeout_s))
    if r.status_code != 200:
        raise RuntimeError(f"datasets-server splits failed | status={r.status_code} | body={r.text[:500]}")
    return r.json()


def viewer_rows(*, dataset: str, config: str, split: str, offset: int = 0, length: int = 100, token: str | None = None, timeout_s: float = _DEFAULT_TIMEOUT_S):
    ds = str(dataset or "").strip()
    cfg = str(config or "").strip()
    sp = str(split or "").strip()
    if not ds or not cfg or not sp:
        raise ValueError("dataset/config/split are required")

    tok = _get_token(token)
    url = "https://datasets-server.huggingface.co/rows"
    params = {
        "dataset": ds,
        "config": cfg,
        "split": sp,
        "offset": int(offset),
        "length": int(length),
    }
    r = requests.get(url, params=params, headers=_auth_headers(tok), timeout=float(timeout_s))
    if r.status_code != 200:
        raise RuntimeError(f"datasets-server rows failed | status={r.status_code} | body={r.text[:500]}")
    return r.json()


def viewer_search(*, dataset: str, config: str, split: str, query: str, offset: int = 0, length: int = 100, token: str | None = None, timeout_s: float = _DEFAULT_TIMEOUT_S):
    ds = str(dataset or "").strip()
    cfg = str(config or "").strip()
    sp = str(split or "").strip()
    q = str(query or "").strip()
    if not ds or not cfg or not sp or not q:
        raise ValueError("dataset/config/split/query are required")

    tok = _get_token(token)
    url = "https://datasets-server.huggingface.co/search"
    params = {
        "dataset": ds,
        "config": cfg,
        "split": sp,
        "query": q,
        "offset": int(offset),
        "length": int(length),
    }
    r = requests.get(url, params=params, headers=_auth_headers(tok), timeout=float(timeout_s))
    if r.status_code != 200:
        raise RuntimeError(f"datasets-server search failed | status={r.status_code} | body={r.text[:500]}")
    return r.json()


def viewer_filter(*, dataset: str, config: str, split: str, where: str, offset: int = 0, length: int = 100, token: str | None = None, timeout_s: float = _DEFAULT_TIMEOUT_S):
    ds = str(dataset or "").strip()
    cfg = str(config or "").strip()
    sp = str(split or "").strip()
    w = str(where or "").strip()
    if not ds or not cfg or not sp or not w:
        raise ValueError("dataset/config/split/where are required")

    tok = _get_token(token)
    url = "https://datasets-server.huggingface.co/filter"
    params = {
        "dataset": ds,
        "config": cfg,
        "split": sp,
        "where": w,
        "offset": int(offset),
        "length": int(length),
    }
    r = requests.get(url, params=params, headers=_auth_headers(tok), timeout=float(timeout_s))
    if r.status_code != 200:
        raise RuntimeError(f"datasets-server filter failed | status={r.status_code} | body={r.text[:500]}")
    return r.json()


def probe_datasets_server(*, timeout_s: float = 5.0) -> bool:
    try:
        t0 = time.time()
        r = requests.get("https://datasets-server.huggingface.co/healthcheck", timeout=float(timeout_s))
        if r.status_code != 200:
            return False
        if (time.time() - t0) > float(timeout_s):
            return False
        return True
    except Exception:
        return False


def _viewer_rows_has_any(obj) -> bool:
    try:
        rows = None
        if isinstance(obj, dict):
            rows = obj.get("rows")
        if isinstance(rows, list) and len(rows) > 0:
            return True
    except Exception:
        pass
    return False


def viewer_filter_contains(
    *,
    dataset: str,
    config: str,
    split: str,
    column: str,
    value: str,
    token: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> bool:
    col = str(column or "").strip()
    val = str(value or "").strip()
    if not col or not val:
        return False

    safe = val.replace("\\", "\\\\").replace("'", "\\'")
    where = f"{col} = '{safe}'"
    obj = viewer_filter(
        dataset=str(dataset),
        config=str(config),
        split=str(split),
        where=str(where),
        offset=0,
        length=1,
        token=token,
        timeout_s=timeout_s,
    )
    return _viewer_rows_has_any(obj)


def duckdb_contains(
    *,
    dataset: str,
    column: str,
    value: str,
    config: str | None = None,
    split: str | None = None,
    token: str | None = None,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    max_files: int = 64,
) -> bool:
    try:
        import duckdb
    except Exception as e:
        raise RuntimeError(f"duckdb not available: {str(e)}")

    meta = hub_list_parquet_urls(dataset=str(dataset), token=token, timeout_s=timeout_s)
    urls = []
    try:
        for it in meta or []:
            if not isinstance(it, dict):
                continue
            if config is not None and str(it.get("config") or "") != str(config):
                continue
            if split is not None and str(it.get("split") or "") != str(split):
                continue
            u = str(it.get("url") or "").strip()
            if u:
                urls.append(u)
    except Exception:
        urls = []

    if not urls:
        try:
            for it in meta or []:
                if isinstance(it, dict):
                    u = str(it.get("url") or "").strip()
                    if u:
                        urls.append(u)
        except Exception:
            urls = []

    urls = urls[: max(1, int(max_files))]
    if not urls:
        return False

    col = str(column or "").strip()
    if not col:
        return False
    v = str(value or "")

    con = duckdb.connect(database=":memory:")
    try:
        q = f"SELECT 1 FROM read_parquet(?) WHERE {col} = ? LIMIT 1"
        res = con.execute(q, [urls, v]).fetchone()
        return res is not None
    finally:
        try:
            con.close()
        except Exception:
            pass
