# stock_v2.py – Taiwan Stock Technical Analysis + AI Features JSON1.16
# Supports two modes: human (fast, skip network) / ai (full data)
# *** OPTIMIZED VERSION – key changes marked with # [OPT] ***
# *** DATA CLEANED VERSION – Added auto_adjust=True and ffill() for dirty data ***
# *** + ADDED ATR FLOOR & WHIPSAW RISK ALERT ***

import io
import json
import math
import re
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass

# ===================================================================
# 0. UTILS
# ===================================================================

# [OPT] Reusable HTTP session for connection pooling across all requests
_http_session: Optional[requests.Session] = None

# [FIX] yf.download() is NOT thread-safe — concurrent calls cause data
# cross-contamination between tickers. This lock serializes all yfinance
# calls while allowing external data fetches to remain fully parallel.
_yf_lock = threading.Lock()

# ===================================================================
# 0a. CONFIGURABLE PARAMETERS (Phase 2/3/4)
# ===================================================================

# KD cross anti-whipsaw filter (Phase 3 Item 1)
KD_ZONE_OVERSOLD = 40       # Golden cross only valid when K below this
KD_ZONE_OVERBOUGHT = 60     # Death cross only valid when K above this
KD_SPREAD_STD_MULT = 0.5    # Min K-D spread = recent_std * this multiplier

# Volatility regime hysteresis (Phase 3 Item 2)
VOL_REGIME_CONFIRM_DAYS = 3  # Days a regime must persist before switching

# SuperTrend smoothing (Phase 3 Item 3)
SUPERTREND_SMOOTHING = "wilder"  # "wilder" (stable) or "ema" (early entry)

# Entry trigger config (Phase 4 Item 1)
ENTRY_MIN_RR = 1.5            # Minimum R:R for entry trigger

# Position sizing (Phase 4 Item 2)
PORTFOLIO_RISK_PCT = 1.0      # % of total capital risked per trade
MAX_POSITION_PCT = 25.0       # Maximum single-stock allocation %

# R:R anomaly threshold (Phase 4 Item 3)
RR_ANOMALY_THRESHOLD = 5.0

# ===================================================================
# 0b. CACHES (Phase 2)
# ===================================================================

# [P1] Benchmark cache — download 0050.TW once, reuse for all stocks
_bench_cache: Dict[str, Any] = {"df": None, "ts": None}
_bench_cache_lock = threading.Lock()
_BENCH_TTL_SECONDS = 300  # 5-minute TTL

# [P1] Institutional full-market CSV cache — one download per (market, date)
_inst_csv_cache: Dict[tuple, Any] = {}
_inst_cache_lock = threading.Lock()


def _get_cached_benchmark(ttl_seconds: int = _BENCH_TTL_SECONDS) -> pd.DataFrame:
    """Download 0050.TW benchmark once, cache for `ttl_seconds`."""
    now = datetime.now()
    with _bench_cache_lock:
        if (_bench_cache["df"] is not None
                and _bench_cache["ts"] is not None
                and (now - _bench_cache["ts"]).total_seconds() < ttl_seconds):
            return _bench_cache["df"]
    # Download outside lock to avoid blocking other threads
    df = _download_yf("0050.TW", "2y", "1d")
    with _bench_cache_lock:
        _bench_cache["df"] = df
        _bench_cache["ts"] = now
    return df


# ===================================================================
# 0c. REGULATORY STATUS CACHE — Attention / Disposition lists
# ===================================================================

_reg_cache: Dict[str, Any] = {
    "attention": set(),
    "disposition": set(),
    "ts": None,
}
_reg_cache_lock = threading.Lock()
_REG_TTL_SECONDS = 600  # 10-minute TTL


def _extract_stock_codes_from_text(text: str) -> set:
    """Extract 4-digit Taiwan stock codes from raw text/CSV/HTML content."""
    # Match 4-digit stock codes (possibly followed by a letter suffix for KY stocks etc.)
    return set(re.findall(r'\b(\d{4}[A-Z]?)\b', text))


def _fetch_twse_attention() -> set:
    """Fetch TWSE (listed) attention list stock codes."""
    codes = set()
    sess = _get_session()
    try:
        # TWSE attention list CSV endpoint
        url = "https://www.twse.com.tw/rwd/zh/announcement/notice?response=csv"
        resp = sess.get(url, timeout=10)
        resp.raise_for_status()
        text = resp.text
        codes |= _extract_stock_codes_from_text(text)
    except Exception:
        pass

    if not codes:
        try:
            # Fallback: JSON endpoint
            url = "https://www.twse.com.tw/rwd/zh/announcement/notice?response=json"
            resp = sess.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for row in data.get("data", []):
                if row and len(row) > 0:
                    codes |= _extract_stock_codes_from_text(str(row[0]))
        except Exception:
            pass
    return codes


def _fetch_twse_disposition() -> set:
    """Fetch TWSE (listed) disposition list stock codes."""
    codes = set()
    sess = _get_session()
    try:
        url = "https://www.twse.com.tw/rwd/zh/announcement/punish?response=csv"
        resp = sess.get(url, timeout=10)
        resp.raise_for_status()
        codes |= _extract_stock_codes_from_text(resp.text)
    except Exception:
        pass

    if not codes:
        try:
            url = "https://www.twse.com.tw/rwd/zh/announcement/punish?response=json"
            resp = sess.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            for row in data.get("data", []):
                if row and len(row) > 0:
                    codes |= _extract_stock_codes_from_text(str(row[0]))
        except Exception:
            pass
    return codes


def _fetch_tpex_attention() -> set:
    """Fetch TPEx (OTC) attention list stock codes."""
    codes = set()
    sess = _get_session()
    try:
        url = "https://www.tpex.org.tw/www/zh-tw/announce/attention"
        resp = sess.get(url, timeout=10)
        resp.raise_for_status()
        codes |= _extract_stock_codes_from_text(resp.text)
    except Exception:
        pass

    if not codes:
        try:
            # Fallback: legacy CSV endpoint
            url = "https://www.tpex.org.tw/web/stock/attention/attention_result.php?l=zh-tw"
            resp = sess.get(url, timeout=10)
            resp.raise_for_status()
            codes |= _extract_stock_codes_from_text(resp.text)
        except Exception:
            pass
    return codes


def _fetch_tpex_disposition() -> set:
    """Fetch TPEx (OTC) disposition list stock codes."""
    codes = set()
    sess = _get_session()
    try:
        url = "https://www.tpex.org.tw/www/zh-tw/announce/disposal"
        resp = sess.get(url, timeout=10)
        resp.raise_for_status()
        codes |= _extract_stock_codes_from_text(resp.text)
    except Exception:
        pass

    if not codes:
        try:
            url = "https://www.tpex.org.tw/web/stock/disposal/disposal_result.php?l=zh-tw"
            resp = sess.get(url, timeout=10)
            resp.raise_for_status()
            codes |= _extract_stock_codes_from_text(resp.text)
        except Exception:
            pass
    return codes


def _refresh_regulatory_cache() -> None:
    """Fetch all 4 lists in parallel, merge into cache."""
    attention = set()
    disposition = set()

    with ThreadPoolExecutor(max_workers=4) as pool:
        fut_att_twse = pool.submit(_fetch_twse_attention)
        fut_att_tpex = pool.submit(_fetch_tpex_attention)
        fut_dis_twse = pool.submit(_fetch_twse_disposition)
        fut_dis_tpex = pool.submit(_fetch_tpex_disposition)

        for fut, target in [
            (fut_att_twse, attention), (fut_att_tpex, attention),
            (fut_dis_twse, disposition), (fut_dis_tpex, disposition),
        ]:
            try:
                target |= fut.result(timeout=15)
            except Exception:
                pass

    with _reg_cache_lock:
        _reg_cache["attention"] = attention
        _reg_cache["disposition"] = disposition
        _reg_cache["ts"] = datetime.now()


def get_regulatory_status(stock_no: str) -> List[str]:
    """
    Return status tags for a stock code (e.g. '2330').
    Returns a subset of: ['ATTENTION', 'DISPOSITION'] or [].
    Uses a cached copy of the official TWSE + TPEx lists.
    """
    stock_no = _strip_suffix(stock_no).strip().upper()
    now = datetime.now()

    # Check if cache needs refresh
    with _reg_cache_lock:
        needs_refresh = (
            _reg_cache["ts"] is None
            or (now - _reg_cache["ts"]).total_seconds() > _REG_TTL_SECONDS
        )

    if needs_refresh:
        try:
            _refresh_regulatory_cache()
        except Exception:
            pass  # Use stale cache on failure

    with _reg_cache_lock:
        att = _reg_cache["attention"]
        dis = _reg_cache["disposition"]

    tags = []
    if stock_no in att:
        tags.append("ATTENTION")
    if stock_no in dis:
        tags.append("DISPOSITION")
    return tags


def _get_session() -> requests.Session:
    """Return a module-level requests.Session with connection pooling."""
    global _http_session
    if _http_session is None:
        _http_session = requests.Session()
        _http_session.headers.update(
            {"User-Agent": "Mozilla/5.0", "Accept-Language": "zh-TW,zh;q=0.9"}
        )
        # Increase pool size for concurrent requests
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10, pool_maxsize=20, max_retries=1
        )
        _http_session.mount("https://", adapter)
        _http_session.mount("http://", adapter)
    return _http_session


def clean_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        lv1 = df.columns.get_level_values(1)
        if "Close" in set(lv0):
            df.columns = lv0
        elif "Close" in set(lv1):
            df.columns = lv1
        else:
            df.columns = df.columns.droplevel(1)
    return df


def _safe_int(x):
    try:
        s = str(x).strip().replace(",", "")
        if s in ("", "–", "—", "NaN", "nan", "None"):
            return None
        return int(float(s))
    except Exception:
        return None


def _resolve_yf_ticker(stock_id: str) -> str:
    s = stock_id.strip().upper()
    if any(s.endswith(sfx) for sfx in [".TW", ".TWO", ".KS", ".T", ".HK"]):
        return s
    if len(s) > 0 and s[0].isdigit():
        return f"{s}.TW"
    return s


def _strip_suffix(stock_id: str) -> str:
    return (
        stock_id.strip()
        .upper()
        .replace(".TW", "")
        .replace(".TWO", "")
        .replace(".KS", "")
        .replace(".T", "")
        .replace(".HK", "")
    )


def _norm_col(s: str) -> str:
    s = str(s).replace("\ufeff", "").replace("\u3000", "")
    s = s.replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", "", s)


def _ensure_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        if getattr(df.index, "tz", None) is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
    except Exception:
        pass
    return df


def _nearest_trading_ts(df: pd.DataFrame, target_date):
    if df is None or df.empty:
        return None
    if target_date is None:
        target_date = datetime.now().date()
    target = pd.Timestamp(target_date)
    idx = df.index
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
    except Exception:
        pass
    pos = idx.searchsorted(target + pd.Timedelta(days=1)) - 1
    return None if pos < 0 else idx[pos]


def _parse_roc_date(date_str: str) -> str:
    try:
        parts = str(date_str).split("/")
        if len(parts) == 3:
            year = int(parts[0]) + 1911
            return f"{year}-{parts[1]}-{parts[2]}"
        return str(date_str)
    except Exception:
        return str(date_str)


# -------------------------------------------------------------------
# FIX: _sanitize_numpy — 統一清洗 numpy 型別，確保 JSON 可序列化
# -------------------------------------------------------------------
def _sanitize_numpy(obj):
    """遞迴清洗 dict / list 中的 numpy 型別，確保 JSON 可序列化。"""
    if isinstance(obj, dict):
        return {k: _sanitize_numpy(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_numpy(v) for v in obj]
    if isinstance(obj, tuple):
        return [_sanitize_numpy(v) for v in obj]

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, (np.integer,)):
        return int(obj)

    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or math.isinf(v):
            return None
        return v

    if isinstance(obj, pd.Timestamp):
        return str(obj)

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    return obj


# – stat utils –
def percentile_rank(series: pd.Series, window: int = 252) -> Optional[float]:
    s = series.dropna()
    if len(s) < 10:
        return None
    current = s.iloc[-1]
    hist = s.iloc[-window:]
    return round((hist < current).sum() / len(hist), 4)


def zscore_last(series: pd.Series, window: int = 252) -> Optional[float]:
    s = series.dropna()
    if len(s) < 20:
        return None
    hist = s.iloc[-window:]
    mu, sigma = hist.mean(), hist.std()
    if sigma == 0 or pd.isna(sigma):
        return 0.0
    return round((s.iloc[-1] - mu) / sigma, 4)


def slope_n(series: pd.Series, n: int = 5) -> Optional[float]:
    """Normalized slope: percent change per day relative to the series mean."""
    s = series.dropna().iloc[-n:]
    if len(s) < n:
        return None
    x = np.arange(len(s), dtype=float)
    y = s.values.astype(float)
    if np.all(np.isnan(y)):
        return None
    m, _ = np.polyfit(x, y, 1)
    # Normalize: slope per day as % of series mean
    mean_val = np.mean(y)
    if mean_val == 0:
        return 0.0
    return round(float(m / abs(mean_val) * 100), 4)


def max_drawdown(series: pd.Series, window: int = 20) -> Optional[float]:
    s = series.dropna().iloc[-window:]
    if len(s) < 2:
        return None
    peak = s.expanding().max()
    dd = (s - peak) / peak
    return round(float(dd.min()), 4)


# ===================================================================
# 1. CANDLE
# ===================================================================


def classify_candle(o, h, l, c) -> str:
    o, h, l, c = float(o), float(h), float(l), float(c)
    rng = h - l
    if rng <= 0:
        return "doji"
    body = abs(c - o)
    body_r = body / rng
    bullish = c > o
    if body_r <= 0.10:
        return "doji"
    if body_r >= 0.60:
        return "long_bull" if bullish else "long_bear"
    return "small_bull" if bullish else "small_bear"


def detect_momentum_candle_patterns(df: pd.DataFrame, n: int = 5) -> Dict[str, Any]:
    if df is None or len(df) < n:
        return {
            "consecutive_bull_bars": 0,
            "consecutive_bear_bars": 0,
            "gap_up_breakout": False,
            "gap_down_breakdown": False,
        }

    tail = df.tail(n)

    # [OPT] Vectorised candle classification – no iterrows()
    o_arr = tail["Open"].values.astype(float)
    h_arr = tail["High"].values.astype(float)
    l_arr = tail["Low"].values.astype(float)
    c_arr = tail["Close"].values.astype(float)

    rng = h_arr - l_arr
    body = np.abs(c_arr - o_arr)
    safe_rng = np.where(rng > 0, rng, 1.0)
    body_r = body / safe_rng
    bullish = c_arr > o_arr

    types = []
    for i in range(len(tail)):
        if rng[i] <= 0 or body_r[i] <= 0.10:
            types.append("doji")
        elif body_r[i] >= 0.60:
            types.append("long_bull" if bullish[i] else "long_bear")
        else:
            types.append("small_bull" if bullish[i] else "small_bear")

    cons_bull = 0
    for t in reversed(types):
        if t in ("long_bull", "small_bull"):
            cons_bull += 1
        else:
            break

    cons_bear = 0
    for t in reversed(types):
        if t in ("long_bear", "small_bear"):
            cons_bear += 1
        else:
            break

    gap_up = False
    gap_down = False
    if len(df) >= 2:
        last, prev = df.iloc[-1], df.iloc[-2]
        if float(last["Low"]) > float(prev["High"]):
            gap_up = True
        if float(last["High"]) < float(prev["Low"]):
            gap_down = True

    return {
        "consecutive_bull_bars": int(cons_bull),
        "consecutive_bear_bars": int(cons_bear),
        "gap_up_breakout": bool(gap_up),
        "gap_down_breakdown": bool(gap_down),
    }


# ===================================================================
# 2. CORE INDICATORS
# ===================================================================


def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    """RSI using Wilder's exponential smoothing (alpha=1/period)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_adx(df: pd.DataFrame, period: int = 14):
    df2 = df.copy()
    df2["H-L"] = df2["High"] - df2["Low"]
    df2["H-PC"] = abs(df2["High"] - df2["Close"].shift(1))
    df2["L-PC"] = abs(df2["Low"] - df2["Close"].shift(1))
    df2["TR"] = df2[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df2["UpMove"] = df2["High"] - df2["High"].shift(1)
    df2["DownMove"] = df2["Low"].shift(1) - df2["Low"]
    df2["+DM"] = np.where(
        (df2["UpMove"] > df2["DownMove"]) & (df2["UpMove"] > 0),
        df2["UpMove"],
        0.0,
    )
    df2["-DM"] = np.where(
        (df2["DownMove"] > df2["UpMove"]) & (df2["DownMove"] > 0),
        df2["DownMove"],
        0.0,
    )
    alpha = 1 / period
    df2["TR_s"] = df2["TR"].ewm(alpha=alpha, adjust=False).mean()
    df2["+DM_s"] = df2["+DM"].ewm(alpha=alpha, adjust=False).mean()
    df2["-DM_s"] = df2["-DM"].ewm(alpha=alpha, adjust=False).mean()
    df2["+DI"] = 100 * df2["+DM_s"] / df2["TR_s"].replace(0, np.nan)
    df2["-DI"] = 100 * df2["-DM_s"] / df2["TR_s"].replace(0, np.nan)
    df2["DX"] = (
            100
            * abs(df2["+DI"] - df2["-DI"])
            / (df2["+DI"] + df2["-DI"]).replace(0, np.nan)
    )
    df2["ADX"] = df2["DX"].ewm(alpha=alpha, adjust=False).mean()
    return df2["ADX"], df2["+DI"], df2["-DI"]


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR using Wilder's exponential smoothing (alpha=1/period)."""
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(
        axis=1
    )
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def calculate_bbands(df: pd.DataFrame, period: int = 20, std_dev: int = 2):
    ma = df["Close"].rolling(window=period).mean()
    std = df["Close"].rolling(window=period).std()
    upper = ma + std * std_dev
    lower = ma - std * std_dev
    bbw = (upper - lower) / ma * 100
    return bbw, upper, lower


def calculate_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    rmf = tp * df["Volume"]
    pos = pd.Series(np.where(tp > tp.shift(1), rmf, 0.0), index=df.index).rolling(
        period
    ).sum()
    neg = pd.Series(np.where(tp < tp.shift(1), rmf, 0.0), index=df.index).rolling(
        period
    ).sum()
    return 100 - (100 / (1 + pos / neg.abs().replace(0, np.nan)))


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()


# [OPT] SuperTrend: use numpy arrays inside the loop to avoid pandas iloc overhead
# [P2] Added smoothing parameter: "wilder" (default, matches ATR) or "ema" (faster response)
def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0,
                         smoothing: str = None):
    if smoothing is None:
        smoothing = SUPERTREND_SMOOTHING  # Use global config
    n = len(df)
    high = df["High"].values.astype(float)
    low = df["Low"].values.astype(float)
    close = df["Close"].values.astype(float)

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1)),
        ),
    )
    tr[0] = high[0] - low[0]

    # EWM ATR — smoothing toggle (Phase 3 Item 3)
    atr = np.empty(n, dtype=float)
    atr[0] = tr[0]
    if smoothing == "wilder":
        alpha = 1.0 / period  # Wilder's smoothing — matches calculate_atr(), more stable
    else:
        alpha = 2.0 / (period + 1)  # Standard EMA — faster response, earlier entry signals
    for i in range(1, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

    hl2 = (high + low) / 2.0
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    supertrend = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)

    supertrend[0] = upper_band[0]
    direction[0] = -1

    for i in range(1, n):
        ub_p = upper_band[i - 1]
        lb_p = lower_band[i - 1]
        c_p = close[i - 1]

        if c_p <= ub_p:
            upper_band[i] = min(upper_band[i], ub_p)
        if c_p >= lb_p:
            lower_band[i] = max(lower_band[i], lb_p)

        c_now = close[i]
        st_p = supertrend[i - 1]

        if np.isnan(st_p) or st_p >= ub_p:
            if c_now > upper_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = 1
            else:
                supertrend[i] = upper_band[i]
                direction[i] = -1
        else:
            if c_now < lower_band[i]:
                supertrend[i] = upper_band[i]
                direction[i] = -1
            else:
                supertrend[i] = lower_band[i]
                direction[i] = 1

    return (
        pd.Series(supertrend, index=df.index),
        pd.Series(direction, index=df.index),
    )


def calculate_avwap(df: pd.DataFrame, anchor_date) -> Optional[pd.Series]:
    mask = df.index >= anchor_date
    if not mask.any():
        return None
    sub = df.loc[mask].copy()
    tp = (sub["High"] + sub["Low"] + sub["Close"]) / 3
    return (tp * sub["Volume"]).cumsum() / sub["Volume"].cumsum().replace(0, np.nan)


# ===================================================================
# 3. VOLUME PROFILE (POC) – [OPT] fully vectorised with NumPy
# ===================================================================


def calculate_volume_profile(
        df: pd.DataFrame, lookback: int = 60, n_bins: int = 50
) -> Dict[str, Any]:
    sub = df.tail(lookback)
    if sub.empty:
        return {"poc_price": None, "poc_volume_k": None, "price_vs_poc_pct": None}

    lo = float(sub["Low"].min())
    hi = float(sub["High"].max())
    if hi <= lo:
        return {"poc_price": None, "poc_volume_k": None, "price_vs_poc_pct": None}

    bins = np.linspace(lo, hi, n_bins + 1)

    # [OPT] Vectorised volume distribution – replaces double-nested Python loop
    row_lo = sub["Low"].values.astype(float)
    row_hi = sub["High"].values.astype(float)
    row_vol = sub["Volume"].values.astype(float)

    # Mask out invalid rows
    valid = (row_hi > row_lo) & (row_vol > 0)
    row_lo = row_lo[valid]
    row_hi = row_hi[valid]
    row_vol = row_vol[valid]

    if len(row_lo) == 0:
        return {"poc_price": None, "poc_volume_k": None, "price_vs_poc_pct": None}

    row_range = row_hi - row_lo  # (R,)

    # bins_lo[b], bins_hi[b] for each bin
    bins_lo = bins[:-1]  # (B,)
    bins_hi = bins[1:]  # (B,)

    # Broadcasting: overlap[row, bin] = max(0, min(row_hi, bin_hi) - max(row_lo, bin_lo))
    # Shapes: row arrays (R,1), bin arrays (1,B)
    overlap = np.maximum(
        0.0,
        np.minimum(row_hi[:, None], bins_hi[None, :])
        - np.maximum(row_lo[:, None], bins_lo[None, :]),
    )  # (R, B)

    # Weighted volume per bin
    weights = row_vol[:, None] * overlap / row_range[:, None]  # (R, B)
    vol_by_bin = weights.sum(axis=0)  # (B,)

    poc_idx = int(np.argmax(vol_by_bin))
    poc_price = round((bins[poc_idx] + bins[poc_idx + 1]) / 2, 2)
    poc_volume = int(vol_by_bin[poc_idx] / 1000)

    c_now = float(sub["Close"].iloc[-1])
    price_vs_poc = round((c_now - poc_price) / poc_price * 100, 4) if poc_price > 0 else None

    # Value Area: expand from POC until 70% of total volume is captured
    total_vol = vol_by_bin.sum()
    va_vol = vol_by_bin[poc_idx]
    lo_idx, hi_idx = poc_idx, poc_idx
    while total_vol > 0 and va_vol / total_vol < 0.70:
        expand_lo = vol_by_bin[lo_idx - 1] if lo_idx > 0 else 0
        expand_hi = vol_by_bin[hi_idx + 1] if hi_idx < n_bins - 1 else 0
        if expand_lo >= expand_hi and lo_idx > 0:
            lo_idx -= 1
            va_vol += expand_lo
        elif hi_idx < n_bins - 1:
            hi_idx += 1
            va_vol += expand_hi
        else:
            break
    va_high = round(float((bins[hi_idx] + bins[hi_idx + 1]) / 2), 2)
    va_low = round(float((bins[lo_idx] + bins[lo_idx + 1]) / 2), 2)

    return {
        "poc_price": poc_price,
        "poc_volume_k": poc_volume,
        "price_vs_poc_pct": price_vs_poc,
        "va_high": va_high,
        "va_low": va_low,
    }


def detect_short_term_sr(df: pd.DataFrame, lookback: int = 120, pivot_lr: int = 3, cluster_pct: float = 0.015) -> Dict[
    str, Any]:
    """
    Detect support/resistance by clustering pivot highs and lows.
    Levels where 2+ pivots cluster within `cluster_pct` of each other are S/R.
    Falls back to range-based min/max if insufficient pivots.
    """
    sub = df.tail(lookback)
    if sub.empty:
        return {"st_support": None, "st_resistance": None}

    c_now = float(sub["Close"].iloc[-1])
    hp, lp = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    all_pivot_prices = [p[2] for p in hp] + [p[2] for p in lp]

    if len(all_pivot_prices) < 3:
        # Fallback to range
        return {
            "st_support": round(float(sub["Low"].min()), 2),
            "st_resistance": round(float(sub["High"].max()), 2),
        }

    # Cluster nearby pivot prices
    all_pivot_prices.sort()
    clusters = []
    current_cluster = [all_pivot_prices[0]]
    for i in range(1, len(all_pivot_prices)):
        if abs(all_pivot_prices[i] - current_cluster[0]) / abs(current_cluster[0]) <= cluster_pct:
            current_cluster.append(all_pivot_prices[i])
        else:
            if len(current_cluster) >= 2:
                clusters.append(round(float(np.mean(current_cluster)), 2))
            current_cluster = [all_pivot_prices[i]]
    if len(current_cluster) >= 2:
        clusters.append(round(float(np.mean(current_cluster)), 2))

    supports = sorted([lv for lv in clusters if lv < c_now], reverse=True)
    resistances = sorted([lv for lv in clusters if lv >= c_now])

    return {
        "st_support": supports[0] if supports else round(float(sub["Low"].min()), 2),
        "st_resistance": resistances[0] if resistances else round(float(sub["High"].max()), 2),
    }


# ===================================================================
# 4. DIVERGENCE DETECTION
# ===================================================================


def detect_divergence(
        price: pd.Series, indicator: pd.Series, lookback: int = 60,
        pivot_left: int = 5, pivot_right: int = 3,
) -> Dict[str, bool]:
    """
    [P0 FIX] Detect divergence using TIMESTAMP-based indicator lookup
    instead of positional indexing. This prevents false divergence signals
    caused by dropped NaN rows creating index misalignment.
    - Bearish divergence: price makes higher high, indicator makes lower high
    - Bullish divergence: price makes lower low, indicator makes higher low
    """
    result = {"bearish_divergence": False, "bullish_divergence": False}
    p = price.dropna().iloc[-lookback:]
    # Reindex indicator to price index but do NOT dropna — keep alignment
    ind = indicator.reindex(p.index)
    if len(p) < 20:
        return result

    p_vals = p.values.astype(float)
    p_dates = p.index
    n = len(p_vals)

    # Find pivot highs and lows in price
    pivot_highs = []  # (position, timestamp, price_value)
    pivot_lows = []
    for i in range(pivot_left, n - pivot_right):
        window = p_vals[i - pivot_left: i + pivot_right + 1]
        if np.isfinite(p_vals[i]):
            if p_vals[i] == np.nanmax(window):
                if not pivot_highs or pivot_highs[-1][0] < i - pivot_right:
                    pivot_highs.append((i, p_dates[i], p_vals[i]))
            if p_vals[i] == np.nanmin(window):
                if not pivot_lows or pivot_lows[-1][0] < i - pivot_right:
                    pivot_lows.append((i, p_dates[i], p_vals[i]))

    # Bearish divergence: last two pivot highs — price HH, indicator LH
    if len(pivot_highs) >= 2:
        _, ts1, ph1_price = pivot_highs[-2]
        _, ts2, ph2_price = pivot_highs[-1]
        # [FIX] Use timestamp lookup instead of positional index
        ind_at_ph1 = ind.get(ts1)
        ind_at_ph2 = ind.get(ts2)
        if (ind_at_ph1 is not None and ind_at_ph2 is not None
                and np.isfinite(ind_at_ph1) and np.isfinite(ind_at_ph2)
                and ph2_price > ph1_price and ind_at_ph2 < ind_at_ph1):
            result["bearish_divergence"] = True

    # Bullish divergence: last two pivot lows — price LL, indicator HL
    if len(pivot_lows) >= 2:
        _, ts1, pl1_price = pivot_lows[-2]
        _, ts2, pl2_price = pivot_lows[-1]
        ind_at_pl1 = ind.get(ts1)
        ind_at_pl2 = ind.get(ts2)
        if (ind_at_pl1 is not None and ind_at_pl2 is not None
                and np.isfinite(ind_at_pl1) and np.isfinite(ind_at_pl2)
                and pl2_price < pl1_price and ind_at_pl2 > ind_at_pl1):
            result["bullish_divergence"] = True

    return result


# ===================================================================
# 5. VOLUME QUALITY
# ===================================================================


def calculate_volume_quality(df: pd.DataFrame, period: int = 20) -> Dict[str, Any]:
    sub = df.tail(max(period, 60)).copy()
    vol = sub["Volume"]
    vol_pct = percentile_rank(vol, window=252)

    tail = sub.tail(period)
    up_mask = tail["Close"] > tail["Close"].shift(1)
    down_mask = tail["Close"] < tail["Close"].shift(1)
    up_vol = float(tail.loc[up_mask, "Volume"].sum()) if up_mask.any() else 0
    down_vol = float(tail.loc[down_mask, "Volume"].sum()) if down_mask.any() else 0
    up_down_ratio = round(up_vol / down_vol, 4) if down_vol > 0 else None

    obv = calculate_obv(df)
    obv_slope_5 = slope_n(obv, 5)
    obv_slope_20 = slope_n(obv, 20)
    obv_pct = percentile_rank(obv, 252)

    return {
        "vol_percentile_252d": vol_pct,
        "up_down_vol_ratio_20d": up_down_ratio,
        "obv_slope_5d": obv_slope_5,
        "obv_slope_20d": obv_slope_20,
        "obv_percentile_252d": obv_pct,
    }


# ===================================================================
# 6. FIBONACCI
# ===================================================================


def calculate_fibonacci_summary(df: pd.DataFrame, lookback: int = 120, trend_state: str = "consolidation") -> Dict[
    str, Any]:
    """
    Direction-aware Fibonacci retracement.
    - Uptrend: measure from swing low → swing high (retracement = pullback support)
    - Downtrend: measure from swing high → swing low (retracement = bounce resistance)
    Adds extension levels (1.272, 1.618) for breakout targets.
    """
    sub = df.tail(lookback)
    high = float(sub["High"].max())
    low = float(sub["Low"].min())
    diff = high - low
    c_now = float(sub["Close"].iloc[-1])

    # [P3] bottom_bounce excluded from uptrend — in still-bearish context (MA20 < MA60),
    # uptrend Fib produces misleading support levels. Consolidation Fib is more appropriate.
    is_uptrend = trend_state in (
        "uptrend", "strong_uptrend", "weak_uptrend", "uptrend_pullback",
    )

    if is_uptrend:
        # Uptrend: retracement from high, extensions above high
        levels = {
            "fib_0236": round(high - 0.236 * diff, 2),
            "fib_0382": round(high - 0.382 * diff, 2),
            "fib_0500": round(high - 0.500 * diff, 2),
            "fib_0618": round(high - 0.618 * diff, 2),
        }
        extensions = {
            "fib_ext_1272": round(low + 1.272 * diff, 2),
            "fib_ext_1618": round(low + 1.618 * diff, 2),
        }
    else:
        # Downtrend: retracement from low upward, extensions below low
        levels = {
            "fib_0236": round(low + 0.236 * diff, 2),
            "fib_0382": round(low + 0.382 * diff, 2),
            "fib_0500": round(low + 0.500 * diff, 2),
            "fib_0618": round(low + 0.618 * diff, 2),
        }
        extensions = {
            "fib_ext_1272": round(high - 1.272 * diff, 2),
            "fib_ext_1618": round(high - 1.618 * diff, 2),
        }

    resistances = sorted([v for v in levels.values() if v > c_now])
    supports = sorted([v for v in levels.values() if v <= c_now], reverse=True)

    result = {
        "fib_high": high,
        "fib_low": low,
        "fib_direction": "up" if is_uptrend else "down",
        "fib_nearest_support_1": supports[0] if len(supports) > 0 else None,
        "fib_nearest_support_2": supports[1] if len(supports) > 1 else None,
        "fib_nearest_resistance_1": resistances[0] if len(resistances) > 0 else None,
        "fib_nearest_resistance_2": resistances[1] if len(resistances) > 1 else None,
    }
    result.update(extensions)
    return result


# ===================================================================
# 7. GAPS – [OPT] vectorised gap detection
# ===================================================================


def detect_gaps_summary(df: pd.DataFrame, lookback: int = 30) -> List[Dict]:
    if df is None or df.empty or len(df) < 2:
        return []
    sub = df.tail(lookback + 1)
    if len(sub) < 2:
        return []

    # [OPT] vectorised: extract arrays once
    highs = sub["High"].values.astype(float)
    lows = sub["Low"].values.astype(float)
    dates = sub.index

    gaps = []
    for i in range(1, len(sub)):
        date_str = dates[i].strftime("%Y-%m-%d")

        if lows[i] > highs[i - 1]:
            # Gap up – check fill using vectorised comparison
            future_lows = lows[i + 1:]
            filled = bool(np.any(future_lows <= highs[i - 1])) if len(future_lows) > 0 else False
            if not filled:
                gaps.append(
                    {
                        "date": date_str,
                        "type": "up",
                        "lower": float(highs[i - 1]),
                        "upper": float(lows[i]),
                    }
                )

        elif highs[i] < lows[i - 1]:
            # Gap down – check fill using vectorised comparison
            future_highs = highs[i + 1:]
            filled = bool(np.any(future_highs >= lows[i - 1])) if len(future_highs) > 0 else False
            if not filled:
                gaps.append(
                    {
                        "date": date_str,
                        "type": "down",
                        "lower": float(highs[i]),
                        "upper": float(lows[i - 1]),
                    }
                )

    return gaps[-2:] if len(gaps) > 2 else gaps


# ===================================================================
# 8. PATTERN DETECTION
# ===================================================================


def _find_pivots(df, left=3, right=3):
    if df is None or df.empty or len(df) < left + right + 5:
        return [], []
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    idx = df.index
    hp, lp = [], []
    for i in range(left, len(df) - right):
        hw = highs[i - left: i + right + 1]
        lw = lows[i - left: i + right + 1]
        if np.isfinite(highs[i]) and highs[i] == np.nanmax(hw):
            if not hp or hp[-1][0] < i - right:
                hp.append((i, idx[i], highs[i]))
        if np.isfinite(lows[i]) and lows[i] == np.nanmin(lw):
            if not lp or lp[-1][0] < i - right:
                lp.append((i, idx[i], lows[i]))
    return hp, lp


def _pct_diff(a, b):
    if a == 0 or not np.isfinite(a) or not np.isfinite(b):
        return np.inf
    return abs(a - b) / abs(a)


def detect_double_top(
        df, lookback=120, pivot_lr=3, peak_tol=0.018, min_gap=8, max_gap=60, confirm_margin=0.003
):
    if df is None or df.empty:
        return None
    sub = df.tail(lookback).copy()
    hp, _ = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(hp) < 2:
        return None
    p2 = hp[-1]
    p1 = next((c for c in reversed(hp[:-1]) if min_gap <= p2[0] - c[0] <= max_gap), None)
    if p1 is None or _pct_diff(float(p1[2]), float(p2[2])) > peak_tol:
        return None
    lo, hi = min(p1[0], p2[0]), max(p1[0], p2[0])
    neckline = float(sub.iloc[lo: hi + 1]["Low"].min())
    confirmed = bool(float(sub["Close"].iloc[-1]) < neckline * (1 - confirm_margin))
    height = max(float(p1[2]), float(p2[2])) - neckline
    return {
        "pattern": "double_top",
        "confirmed": confirmed,
        "neckline": neckline,
        "target": round(neckline - height, 2),
        "bias": "bearish",
    }


def detect_double_bottom(
        df, lookback=160, pivot_lr=3, trough_tol=0.020, min_gap=8, max_gap=80, confirm_margin=0.003
):
    if df is None or df.empty:
        return None
    sub = df.tail(lookback).copy()
    _, lp = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(lp) < 2:
        return None
    t2 = lp[-1]
    t1 = next((c for c in reversed(lp[:-1]) if min_gap <= t2[0] - c[0] <= max_gap), None)
    if t1 is None or _pct_diff(float(t1[2]), float(t2[2])) > trough_tol:
        return None
    lo, hi = min(t1[0], t2[0]), max(t1[0], t2[0])
    neckline = float(sub.iloc[lo: hi + 1]["High"].max())
    confirmed = bool(float(sub["Close"].iloc[-1]) > neckline * (1 + confirm_margin))
    height = neckline - min(float(t1[2]), float(t2[2]))
    return {
        "pattern": "double_bottom",
        "confirmed": confirmed,
        "neckline": neckline,
        "target": round(neckline + height, 2),
        "bias": "bullish",
    }


def detect_head_and_shoulders(df, lookback=180, pivot_lr=3, shoulder_tol=0.025, confirm_margin=0.003):
    if df is None or df.empty:
        return None
    sub = df.tail(lookback).copy()
    hp, lp = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(hp) < 3 or len(lp) < 2:
        return None
    ls, head, rs = hp[-3], hp[-2], hp[-1]
    if not (float(head[2]) > float(ls[2]) and float(head[2]) > float(rs[2])):
        return None
    if _pct_diff(float(ls[2]), float(rs[2])) > shoulder_tol:
        return None
    mid_lows = [p for p in lp if ls[0] <= p[0] <= rs[0]]
    if len(mid_lows) < 2:
        return None
    neckline = float(np.mean([p[2] for p in mid_lows]))
    confirmed = bool(float(sub["Close"].iloc[-1]) < neckline * (1 - confirm_margin))
    height = float(head[2]) - neckline
    return {
        "pattern": "head_and_shoulders_top",
        "confirmed": confirmed,
        "neckline": round(neckline, 2),
        "target": round(neckline - height, 2),
        "bias": "bearish",
    }


def detect_inverse_head_and_shoulders(
        df, lookback=180, pivot_lr=3, shoulder_tol=0.025, confirm_margin=0.003
):
    if df is None or df.empty:
        return None
    sub = df.tail(lookback).copy()
    hp, lp = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(lp) < 3 or len(hp) < 2:
        return None
    ls, head, rs = lp[-3], lp[-2], lp[-1]
    if not (float(head[2]) < float(ls[2]) and float(head[2]) < float(rs[2])):
        return None
    if _pct_diff(float(ls[2]), float(rs[2])) > shoulder_tol:
        return None
    mid_highs = [p for p in hp if ls[0] <= p[0] <= rs[0]]
    if len(mid_highs) < 2:
        return None
    neckline = float(np.mean([p[2] for p in mid_highs]))
    confirmed = bool(float(sub["Close"].iloc[-1]) > neckline * (1 + confirm_margin))
    height = neckline - float(head[2])
    return {
        "pattern": "inv_head_and_shoulders",
        "confirmed": confirmed,
        "neckline": round(neckline, 2),
        "target": round(neckline + height, 2),
        "bias": "bullish",
    }


def detect_wedge(df, lookback=120, pivot_lr=3, breakout_margin=0.003):
    if df is None or df.empty:
        return None
    sub = df.tail(lookback).copy()
    hp, lp = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(hp) < 3 or len(lp) < 3:
        return None
    highs, lows = hp[-3:], lp[-3:]
    xh = np.array([p[0] for p in highs], float)
    yh = np.array([p[2] for p in highs], float)
    xl = np.array([p[0] for p in lows], float)
    yl = np.array([p[2] for p in lows], float)
    ah, bh = np.polyfit(xh, yh, 1)
    al, bl = np.polyfit(xl, yl, 1)
    n = len(sub) - 1
    upper_now = ah * n + bh
    lower_now = al * n + bl
    close_now = float(sub["Close"].iloc[-1])

    if ah > 0 and al > 0 and al > ah:
        confirmed = bool(close_now < lower_now * (1 - breakout_margin))
        return {"pattern": "rising_wedge", "confirmed": confirmed, "bias": "bearish"}
    if ah < 0 and al < 0 and ah < al:
        confirmed = bool(close_now > upper_now * (1 + breakout_margin))
        return {"pattern": "falling_wedge", "confirmed": confirmed, "bias": "bullish"}
    return None


def detect_patterns(df) -> List[Dict]:
    results = []
    for fn in [
        detect_double_top,
        detect_double_bottom,
        detect_head_and_shoulders,
        detect_inverse_head_and_shoulders,
        detect_wedge,
    ]:
        try:
            p = fn(df)
            if p:
                results.append(p)
        except Exception:
            pass
    return results


_PATTERN_PRIORITY = {
    "head_and_shoulders_top": 1,
    "inv_head_and_shoulders": 1,
    "double_top": 2,
    "double_bottom": 2,
    "rising_wedge": 3,
    "falling_wedge": 3,
}


def _prioritize_patterns(raw_patterns: List[Dict]) -> List[Dict]:
    """
    將 pattern 分成 bearish / bullish 兩組，
    每組取 confirmed 優先、priority 最高的 1 個，最多輸出 2 個。
    """
    bearish = [p for p in raw_patterns if p.get("bias") == "bearish"]
    bullish = [p for p in raw_patterns if p.get("bias") == "bullish"]

    def pick_best(group: List[Dict]) -> Optional[Dict]:
        if not group:
            return None
        group.sort(
            key=lambda p: (
                0 if p.get("confirmed") else 1,
                _PATTERN_PRIORITY.get(p.get("pattern", ""), 99),
            )
        )
        return group[0]

    result = []
    b = pick_best(bearish)
    if b:
        result.append(b)
    u = pick_best(bullish)
    if u:
        result.append(u)
    return result


# ===================================================================
# 9. INSTITUTIONAL DATA – [OPT] uses pooled session
# ===================================================================


def _parse_twse_t86_csv(text: str):
    text = text.replace("\r", "").replace("=", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    start = next(
        (
            i
            for i, ln in enumerate(lines)
            if "證券代號" in ln and "證券名稱" in ln
        ),
        None,
    )
    if start is None:
        return None
    end = next(
        (
            j
            for j in range(start + 1, len(lines))
            if lines[j].startswith("說明") or lines[j].startswith("備註")
        ),
        len(lines),
    )
    try:
        return pd.read_csv(io.StringIO("\n".join(lines[start:end])))
    except Exception:
        return None


def _parse_tpex_csv(text: str):
    text = text.replace("\ufeff", "").replace("\r", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    start = next((i for i, ln in enumerate(lines) if "代號" in ln and "名稱" in ln), None)
    if start is None:
        return None
    end = next(
        (
            j
            for j in range(start + 1, len(lines))
            if lines[j].startswith("說明") or lines[j].startswith("備註")
        ),
        len(lines),
    )
    try:
        return pd.read_csv(io.StringIO("\n".join(lines[start:end])))
    except Exception:
        return None


def get_institutional_data(stock_id: str, trade_date, market_hint=None, max_back: int = 10):
    stock_no = _strip_suffix(stock_id)
    sess = _get_session()  # [OPT] connection pooling
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "zh-TW,zh;q=0.9"}
    prefer_twse = not (isinstance(market_hint, str) and market_hint.upper().endswith(".TWO"))
    last_error = None

    def try_twse(d_):
        # [P1] Use cached full-market CSV — one download per date for ALL stocks
        cache_key = ("twse_t86", d_.strftime('%Y%m%d'))
        with _inst_cache_lock:
            if cache_key in _inst_csv_cache:
                cached = _inst_csv_cache[cache_key]
                if cached is None:
                    return None, "TWSE cached no-data"
                return cached, None

        url = f"https://www.twse.com.tw/fund/T86?response=csv&date={d_.strftime('%Y%m%d')}&selectType=ALLBUT0999"
        r = sess.get(url, headers=headers, timeout=15)
        if r.status_code != 200 or len(r.text) < 200:
            with _inst_cache_lock:
                _inst_csv_cache[cache_key] = None
            return None, f"TWSE HTTP {r.status_code}"
        if "沒有符合條件的資料" in r.text or "很抱歉" in r.text:
            with _inst_cache_lock:
                _inst_csv_cache[cache_key] = None
            return None, "TWSE no data"
        df = _parse_twse_t86_csv(r.text)
        with _inst_cache_lock:
            _inst_csv_cache[cache_key] = df
        return (df, None) if df is not None else (None, "TWSE parse failed")

    def try_tpex(d_):
        roc_year = d_.year - 1911
        roc_date = f"{roc_year:03d}/{d_.month:02d}/{d_.day:02d}"
        # [P1] Use cached full-market CSV
        cache_key = ("tpex_inst", roc_date)
        with _inst_cache_lock:
            if cache_key in _inst_csv_cache:
                cached = _inst_csv_cache[cache_key]
                if cached is None:
                    return None, "TPEx cached no-data"
                return cached, None

        url = (
            "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"
            f"?l=zh-tw&o=csv&se=EW&t=D&d={roc_date}&s=0,asc"
        )
        h2 = dict(headers)
        h2["Referer"] = "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge.php"
        r = sess.get(url, headers=h2, timeout=15)
        if r.status_code != 200 or len(r.text) < 200:
            with _inst_cache_lock:
                _inst_csv_cache[cache_key] = None
            return None, f"TPEx HTTP {r.status_code}"
        if "沒有符合條件的資料" in r.text or "很抱歉" in r.text:
            with _inst_cache_lock:
                _inst_csv_cache[cache_key] = None
            return None, "TPEx no data"
        df = _parse_tpex_csv(r.text)
        with _inst_cache_lock:
            _inst_csv_cache[cache_key] = df
        return (df, None) if df is not None else (None, "TPEx parse failed")

    markets = [("TWSE", try_twse), ("TPEx", try_tpex)]
    if not prefer_twse:
        markets = [("TPEx", try_tpex), ("TWSE", try_twse)]

    def find_col(colmap, pred):
        return next((orig for nk, orig in colmap.items() if pred(nk)), None)

    def is_foreign(nk):
        if "外資自營商" in nk and "不含外資自營商" not in nk:
            return False
        return ("外陸資" in nk) or ("外資及陸資" in nk) or ("外資" in nk)

    for back in range(max_back + 1):
        d = trade_date - timedelta(days=back)
        for mkt_name, fn in markets:
            df, err = fn(d)
            if df is None:
                last_error = err
                continue
            cols = list(df.columns)
            colmap = {_norm_col(c): c for c in cols}
            code_col = next((c for c in cols if _norm_col(c) in ("證券代號", "代號")), None)
            if code_col is None:
                last_error = f"{mkt_name} no code col"
                continue
            row = df[df[code_col].astype(str).str.strip() == stock_no]
            if row.empty:
                last_error = f"{mkt_name} no {stock_no}"
                continue

            foreign_col = find_col(colmap, lambda nk: is_foreign(nk) and "買賣超" in nk)
            trust_col = find_col(colmap, lambda nk: "投信" in nk and "買賣超" in nk)
            dealer_total = find_col(
                colmap,
                lambda nk: "自營商" in nk
                           and "買賣超" in nk
                           and "外資" not in nk
                           and "自營買賣" not in nk
                           and "避險" not in nk,
            )
            dealer_self = find_col(
                colmap,
                lambda nk: "自營商" in nk
                           and "自營買賣" in nk
                           and "買賣超" in nk
                           and "外資" not in nk,
            )
            dealer_hedge = find_col(
                colmap,
                lambda nk: "自營商" in nk and "避險" in nk and "買賣超" in nk and "外資" not in nk,
            )

            foreign = _safe_int(row.iloc[0][foreign_col]) if foreign_col else None
            trust = _safe_int(row.iloc[0][trust_col]) if trust_col else None
            dealer = None
            if dealer_total:
                dealer = _safe_int(row.iloc[0][dealer_total])
            elif dealer_self or dealer_hedge:
                a = _safe_int(row.iloc[0][dealer_self]) if dealer_self else 0
                b = _safe_int(row.iloc[0][dealer_hedge]) if dealer_hedge else 0
                dealer = (a or 0) + (b or 0)

            if foreign is None:
                buy_c = find_col(
                    colmap,
                    lambda nk: is_foreign(nk)
                               and ("買進" in nk or "買入" in nk)
                               and "買賣超" not in nk,
                )
                sell_c = find_col(
                    colmap, lambda nk: is_foreign(nk) and "賣出" in nk and "買賣超" not in nk
                )
                bv = _safe_int(row.iloc[0][buy_c]) if buy_c else None
                sv = _safe_int(row.iloc[0][sell_c]) if sell_c else None
                if bv is not None and sv is not None:
                    foreign = bv - sv

            yf_suffix = ".TW" if mkt_name == "TWSE" else ".TWO"
            return {
                "id": f"{stock_no}{yf_suffix}",
                "date": d.strftime("%Y-%m-%d"),
                "foreign": foreign,
                "trust": trust,
                "dealer": dealer,
                "error": None,
            }

    return {"error": last_error or "unknown", "id": f"{stock_no}.TW"}


def get_institutional_multi_days(stock_id: str, end_date, market_hint=None, days=20):
    results, d, attempts = [], end_date, 0
    while len(results) < days and attempts < days * 4:
        attempts += 1
        rec = get_institutional_data(stock_id, d, market_hint=market_hint, max_back=0)
        if rec.get("error") is None:
            results.insert(0, rec)
            d = datetime.strptime(rec["date"], "%Y-%m-%d").date() - timedelta(days=1)
        else:
            d = d - timedelta(days=1)
    return results


def compute_institutional_features(chips_multi: List[Dict], price_now: float, price_20d_ago: float,
                                   avg_daily_vol: float = 0) -> Dict[str, Any]:
    feat: Dict[str, Any] = {}
    if not chips_multi:
        return {"inst_data_available": False}

    feat["inst_data_available"] = True

    def _lots(v):
        v2 = _safe_int(v)
        return int(round(v2 / 1000)) if v2 is not None else 0

    f_vals = [_lots(r.get("foreign")) for r in chips_multi]
    t_vals = [_lots(r.get("trust")) for r in chips_multi]
    d_vals = [_lots(r.get("dealer")) for r in chips_multi]

    feat["foreign_5d_net"] = sum(f_vals[-5:])
    feat["foreign_20d_net"] = sum(f_vals)
    feat["trust_5d_net"] = sum(t_vals[-5:])
    feat["trust_20d_net"] = sum(t_vals)
    feat["dealer_5d_net"] = sum(d_vals[-5:])
    feat["dealer_20d_net"] = sum(d_vals)

    # Normalized institutional flow as % of average daily volume (Item #14)
    if avg_daily_vol > 0:
        feat["foreign_20d_net_pct_adv"] = round(
            sum(f_vals) * 1000 / avg_daily_vol * 100, 2
        )
        feat["trust_20d_net_pct_adv"] = round(
            sum(t_vals) * 1000 / avg_daily_vol * 100, 2
        )
    else:
        feat["foreign_20d_net_pct_adv"] = None
        feat["trust_20d_net_pct_adv"] = None

    feat["foreign_slope_20d"] = slope_n(pd.Series(f_vals), len(f_vals))
    feat["trust_slope_20d"] = slope_n(pd.Series(t_vals), len(t_vals))

    def _consec(vals):
        if not vals or vals[-1] == 0:
            return 0
        direction = 1 if vals[-1] > 0 else -1
        count = 0
        for v in reversed(vals):
            if (direction > 0 and v > 0) or (direction < 0 and v < 0):
                count += 1
            else:
                break
        return count * direction

    feat["foreign_consecutive_days"] = _consec(f_vals)
    feat["trust_consecutive_days"] = _consec(t_vals)

    price_up_20d = price_now > price_20d_ago
    feat["flag_foreign_divergence"] = bool(price_up_20d and feat["foreign_20d_net"] < 0)
    feat["flag_inst_consensus_buy"] = bool(feat["foreign_20d_net"] > 0 and feat["trust_20d_net"] > 0)
    feat["flag_inst_consensus_sell"] = bool(feat["foreign_20d_net"] < 0 and feat["trust_20d_net"] < 0)

    return feat



# ===================================================================
# 10. MARGIN TRADING – [OPT] uses pooled session
# ===================================================================


def _parse_twse_json_table(obj):
    if not isinstance(obj, dict):
        return None
    if "fields" in obj and "data" in obj:
        try:
            return pd.DataFrame(obj["data"], columns=obj["fields"])
        except Exception:
            pass
    if "tables" in obj and isinstance(obj["tables"], list):
        for tbl in obj["tables"]:
            fields = tbl.get("fields", [])
            data = tbl.get("data", [])
            if not fields or not data:
                continue
            if any("代號" in str(f) or "代碼" in str(f) for f in fields):
                try:
                    return pd.DataFrame(data, columns=fields)
                except Exception:
                    continue
    best_table, max_cols = None, 0
    if "tables" in obj:
        for tbl in obj["tables"]:
            if "fields" in tbl and "data" in tbl:
                if len(tbl["fields"]) > max_cols:
                    max_cols = len(tbl["fields"])
                    best_table = tbl
    if best_table and max_cols > 5:
        try:
            return pd.DataFrame(best_table["data"], columns=best_table["fields"])
        except Exception:
            pass
    return None


def _twse_margin_json(date_yyyymmdd: str, headers: dict):
    sess = _get_session()  # [OPT]
    urls = [
        f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?response=json&date={date_yyyymmdd}&selectType=ALL",
        f"https://www.twse.com.tw/exchangeReport/MI_MARGN?response=json&date={date_yyyymmdd}&selectType=ALL",
    ]
    h2 = dict(headers)
    h2["Referer"] = "https://www.twse.com.tw/zh/page/trading/exchange/MI_MARGN.html"
    last_err = None
    for url in urls:
        try:
            r = sess.get(url, headers=h2, timeout=15)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                continue
            j = r.json()
            df = pd.DataFrame(j) if isinstance(j, list) else _parse_twse_json_table(j)
            if df is not None and not df.empty:
                return df, None
            last_err = "empty"
        except Exception as e:
            last_err = str(e)
    return None, last_err or "failed"


def _parse_tpex_margin_csv(text: str):
    if not text:
        return None
    text = text.replace("\ufeff", "").replace("\r", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    start = next(
        (
            i
            for i, ln in enumerate(lines)
            if "代號" in ln and "名稱" in ln and ("資" in ln or "融資" in ln)
        ),
        None,
    )
    if start is None:
        return None
    end = next(
        (
            j
            for j in range(start + 1, len(lines))
            if lines[j].startswith("*****") or lines[j].startswith("說明") or lines[j].startswith("備註")
        ),
        len(lines),
    )
    try:
        return pd.read_csv(io.StringIO("\n".join(lines[start:end])))
    except Exception:
        return None


def get_margin_short_data(stock_id: str, trade_date, market_hint: str = None, max_back: int = 10):
    stock_no = _strip_suffix(stock_id)
    sess = _get_session()  # [OPT]
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "zh-TW,zh;q=0.9"}
    is_two = isinstance(market_hint, str) and market_hint.upper().endswith(".TWO")
    last_error = None

    def find_col(colmap, contains):
        return next((orig for nk, orig in colmap.items() if all(kw in nk for kw in contains)), None)

    for back in range(max_back + 1):
        d = trade_date - timedelta(days=back)

        if not is_two:
            df, err = _twse_margin_json(d.strftime("%Y%m%d"), headers=headers)
            if df is not None and not df.empty:
                colmap = {_norm_col(c): c for c in df.columns}
                code_col = next(
                    (
                        v
                        for k, v in colmap.items()
                        if k in ("股票代號", "證券代號", "證券代碼", "Code", "代號")
                    ),
                    None,
                )
                if not code_col:
                    code_col = next((v for k, v in colmap.items() if "代號" in k or "代碼" in k), None)
                if not code_col and len(df.columns) > 0:
                    first_col = df.columns[0]
                    sample = df[first_col].astype(str).str.strip().head(10)
                    if sample.str.match(r"^\d{4,6}[A-Z]?$").any():
                        code_col = first_col

                if code_col:
                    sub = df[df[code_col].astype(str).str.strip().str.strip('"') == stock_no]
                    if not sub.empty:
                        row = sub.iloc[0]
                        m_bal_c = find_col(colmap, ["融資", "餘額"]) or find_col(colmap, ["資", "餘額"])
                        m_chg_c = find_col(colmap, ["融資", "增減"]) or find_col(colmap, ["資", "增減"])
                        m_lim_c = find_col(colmap, ["融資", "限額"]) or find_col(colmap, ["資", "限額"])
                        s_bal_c = find_col(colmap, ["融券", "餘額"]) or find_col(colmap, ["券", "餘額"])
                        s_chg_c = find_col(colmap, ["融券", "增減"]) or find_col(colmap, ["券", "增減"])
                        m_bal = _safe_int(row[m_bal_c]) if m_bal_c else None
                        m_chg = _safe_int(row[m_chg_c]) if m_chg_c else None
                        m_lim = _safe_int(row[m_lim_c]) if m_lim_c else None
                        s_bal = _safe_int(row[s_bal_c]) if s_bal_c else None
                        s_chg = _safe_int(row[s_chg_c]) if s_chg_c else None
                        usage = round(m_bal / m_lim * 100, 1) if (m_bal and m_lim and m_lim > 0) else None
                        return {
                            "id": f"{stock_no}.TW",
                            "date": d.strftime("%Y-%m-%d"),
                            "margin_balance": m_bal,
                            "margin_change": m_chg,
                            "margin_limit": m_lim,
                            "margin_usage_rate": usage,
                            "short_balance": s_bal,
                            "short_change": s_chg,
                            "error": None,
                        }
                    last_error = f"TWSE margin no {stock_no}"
                else:
                    last_error = "TWSE margin no code col"
            else:
                last_error = err

        roc_year = d.year - 1911
        roc_date = f"{roc_year:03d}/{d.month:02d}/{d.day:02d}"
        url_csv = (
            "https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php"
            f"?l=zh-tw&d={roc_date}&o=csv&s=0,asc"
        )
        try:
            r = sess.get(url_csv, headers=headers, timeout=15)
            df = _parse_tpex_margin_csv(r.text) if r.status_code == 200 and len(r.text) > 200 else None
            if df is not None and not df.empty:
                colmap = {_norm_col(c): c for c in df.columns}
                code_col = colmap.get("代號") or colmap.get("股票代號")
                if code_col:
                    sub = df[df[code_col].astype(str).str.strip() == stock_no]
                    if not sub.empty:
                        row = sub.iloc[0]
                        m_bal_c = find_col(colmap, ["資", "餘額"])
                        m_chg_c = find_col(colmap, ["資", "增減"])
                        s_bal_c = find_col(colmap, ["券", "餘額"])
                        s_chg_c = find_col(colmap, ["券", "增減"])
                        return {
                            "id": f"{stock_no}.TWO",
                            "date": d.strftime("%Y-%m-%d"),
                            "margin_balance": _safe_int(row[m_bal_c]) if m_bal_c else None,
                            "margin_change": _safe_int(row[m_chg_c]) if m_chg_c else None,
                            "margin_limit": None,
                            "margin_usage_rate": None,
                            "short_balance": _safe_int(row[s_bal_c]) if s_bal_c else None,
                            "short_change": _safe_int(row[s_chg_c]) if s_chg_c else None,
                            "error": None,
                        }
        except Exception as e:
            last_error = f"TPEx margin: {e}"

    return {"error": last_error or "unknown"}


# ===================================================================
# 11. TDCC – [OPT] uses pooled session
# ===================================================================


def get_tdcc_distribution(stock_no: str, weeks_back: int = 2) -> Dict[str, Any]:
    stock_no = _strip_suffix(stock_no)
    sess = _get_session()  # [OPT]
    headers = {"User-Agent": "Mozilla/5.0"}
    result = {"error": None, "data": [], "as_of_date": None, "data_lag_days": None}

    try:
        url = (
            "https://www.tdcc.com.tw/portal/zh/smWeb/QryStockAJ?"
            "scaDates=&scaDate=&SqlMethod=StockNo&StockNo={}"
            "&radioStockNo=&StockName=&REession_SCA_150=&clession_SCA_150=".format(stock_no)
        )
        r = sess.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            result["error"] = f"TDCC HTTP {r.status_code}"
            return result

        try:
            data = r.json()
        except Exception:
            result["error"] = "TDCC JSON parse failed"
            return result

        if not data:
            result["error"] = "TDCC no data"
            return result

        dates = sorted(set(str(d.get("SCA_DATE", "")) for d in data if d.get("SCA_DATE")))
        if not dates:
            result["error"] = "TDCC no dates"
            return result

        latest_dates = dates[-weeks_back:] if len(dates) >= weeks_back else dates

        for date_str in latest_dates:
            day_data = [d for d in data if str(d.get("SCA_DATE", "")) == date_str]
            if not day_data:
                continue

            total_holders = 0
            total_shares = 0
            retail_shares = 0
            whale_400_shares = 0
            whale_1000_shares = 0

            for row in day_data:
                hold_com = str(row.get("HOLD_COM", ""))
                holders = _safe_int(row.get("HOLD_NUM")) or 0
                shares = _safe_int(row.get("HOLD_UNIT")) or 0
                total_holders += holders
                total_shares += shares

                nums = re.findall(r"[\d,]+", hold_com.replace(",", ""))
                if nums:
                    try:
                        lower_bound = int(nums[0])
                    except Exception:
                        lower_bound = 0
                    if lower_bound <= 10000:
                        retail_shares += shares
                    if lower_bound >= 400000:
                        whale_400_shares += shares
                    if lower_bound >= 1000000:
                        whale_1000_shares += shares

            retail_pct = round(retail_shares / total_shares * 100, 2) if total_shares > 0 else None
            whale_400_pct = round(whale_400_shares / total_shares * 100, 2) if total_shares > 0 else None
            whale_1000_pct = round(whale_1000_shares / total_shares * 100, 2) if total_shares > 0 else None

            try:
                dt = datetime.strptime(date_str, "%Y%m%d")
                date_fmt = dt.strftime("%Y-%m-%d")
            except Exception:
                date_fmt = date_str

            result["data"].append(
                {
                    "date": date_fmt,
                    "total_holders": total_holders,
                    "retail_pct": retail_pct,
                    "whale_400_pct": whale_400_pct,
                    "whale_1000_pct": whale_1000_pct,
                }
            )

        if result["data"]:
            result["as_of_date"] = result["data"][-1]["date"]
            try:
                last_data_date = datetime.strptime(result["as_of_date"], "%Y-%m-%d").date()
                result["data_lag_days"] = (datetime.now().date() - last_data_date).days
            except Exception:
                result["data_lag_days"] = None

    except Exception as e:
        result["error"] = str(e)

    return result


def compute_tdcc_features(tdcc_data: Dict) -> Dict[str, Any]:
    feat: Dict[str, Any] = {}

    if tdcc_data.get("error") or not tdcc_data.get("data"):
        feat["tdcc_available"] = False
        return feat

    feat["tdcc_available"] = True
    feat["tdcc_as_of_date"] = tdcc_data.get("as_of_date")
    feat["tdcc_data_lag_days"] = tdcc_data.get("data_lag_days")

    records = tdcc_data["data"]
    latest = records[-1]

    feat["tdcc_total_holders"] = latest.get("total_holders")
    feat["tdcc_retail_pct"] = latest.get("retail_pct")
    feat["tdcc_whale_400_pct"] = latest.get("whale_400_pct")
    feat["tdcc_whale_1000_pct"] = latest.get("whale_1000_pct")

    if len(records) >= 2:
        prev = records[-2]
        feat["tdcc_holders_change"] = (latest.get("total_holders") or 0) - (prev.get("total_holders") or 0)
        feat["tdcc_retail_pct_change"] = round((latest.get("retail_pct") or 0) - (prev.get("retail_pct") or 0), 2)
        feat["tdcc_whale_400_pct_change"] = round((latest.get("whale_400_pct") or 0) - (prev.get("whale_400_pct") or 0),
                                                  2)
        feat["tdcc_whale_1000_pct_change"] = round(
            (latest.get("whale_1000_pct") or 0) - (prev.get("whale_1000_pct") or 0), 2)
        feat["flag_whale_up_retail_down"] = bool(
            (feat.get("tdcc_whale_400_pct_change") or 0) > 0 and (feat.get("tdcc_retail_pct_change") or 0) < 0
        )
        feat["flag_holders_decreasing"] = bool((feat.get("tdcc_holders_change") or 0) < 0)
    else:
        feat["tdcc_holders_change"] = None
        feat["tdcc_retail_pct_change"] = None
        feat["tdcc_whale_400_pct_change"] = None
        feat["tdcc_whale_1000_pct_change"] = None
        feat["flag_whale_up_retail_down"] = False
        feat["flag_holders_decreasing"] = False

    return feat


# ===================================================================
# 12. RELATIVE STRENGTH
# ===================================================================


def calc_relative_strength(stock_df, benchmark_ticker="0050.TW", period=20):
    try:
        with _yf_lock:
            # [FIX] auto_adjust=True and ffill() to avoid missing data/unadjusted price bugs
            bench = yf.download(benchmark_ticker, period="3mo", progress=False, auto_adjust=True)
            bench = clean_yf_columns(_ensure_naive_index(bench))
            if not bench.empty:
                bench = bench.ffill().dropna(subset=["Close"])

        if bench.empty or len(stock_df) < period:
            return None
        s_ret = (float(stock_df["Close"].iloc[-1]) / float(stock_df["Close"].iloc[-period]) - 1) * 100
        b_ret = (float(bench["Close"].iloc[-1]) / float(bench["Close"].iloc[-period]) - 1) * 100
        return {
            "stock_ret_20d": round(s_ret, 2),
            "bench_ret_20d": round(b_ret, 2),
            "rs_20d": round(s_ret - b_ret, 2),
        }
    except Exception:
        return None


# [OPT] Pre-fetch benchmark with provided df to avoid extra yf.download call
def _calc_relative_strength_with_bench(stock_df, bench_df, period=20):
    """Compute relative strength using pre-fetched benchmark data."""
    try:
        if bench_df is None or bench_df.empty or len(stock_df) < period:
            return None
        s_ret = (float(stock_df["Close"].iloc[-1]) / float(stock_df["Close"].iloc[-period]) - 1) * 100
        b_ret = (float(bench_df["Close"].iloc[-1]) / float(bench_df["Close"].iloc[-period]) - 1) * 100
        return {
            f"stock_ret_{period}d": round(s_ret, 2),
            f"bench_ret_{period}d": round(b_ret, 2),
            f"rs_{period}d": round(s_ret - b_ret, 2),
        }
    except Exception:
        return None


# ===================================================================
# 13. DECISION FIELDS (MODIFIED: ATR FLOOR & WHIPSAW WARNING)
# ===================================================================

# [P3] Unified trend classifier — ADX qualification in one pass, no intermediate states
def classify_trend(c: float, ma5: float, ma20: float, ma60: float, adx_val: float) -> tuple:
    """
    Returns (trend_state, trend_strength) in a single pass.
    Eliminates the bug where ADX qualification could be bypassed
    by exceptions between the initial classification and the qualifier.
    """
    # Determine base trend from MA cascade
    if c > ma20 > ma60 and ma5 > ma20:
        base = "uptrend"
    elif c > ma20 > ma60 and ma5 <= ma20:
        return "uptrend_pullback", _adx_strength(adx_val)
    elif c < ma20 < ma60 and ma5 < ma20:
        base = "downtrend"
    elif c < ma20 < ma60 and ma5 >= ma20:
        return "downtrend_bounce", _adx_strength(adx_val)
    elif c > ma20 and ma20 < ma60:
        # [FIX] bottom_bounce also gets ADX qualification
        prefix = "strong_" if adx_val >= 25 else "weak_"
        return prefix + "bottom_bounce", _adx_strength(adx_val)
    elif c < ma20 and ma20 > ma60:
        return "top_pullback", _adx_strength(adx_val)
    else:
        return "consolidation", _adx_strength(adx_val)

    # Qualify directional trends with ADX strength
    prefix = "strong_" if adx_val >= 25 else "weak_"
    return prefix + base, _adx_strength(adx_val)


def _adx_strength(adx_val: float) -> str:
    if adx_val >= 25:
        return "strong"
    elif adx_val >= 20:
        return "moderate"
    return "weak"


def compute_decision_fields(
        c_now: float,
        atr_now: float,
        resistance: Optional[float],
        support: Optional[float],
        supertrend_dir: int,
        bb_squeeze: bool = False,
        trend_state: str = "consolidation",
        volatility_regime: str = "normal",
        # [P4] Additional inputs for event-based entry trigger
        bb_squeeze_fire: bool = False,
        vol_ratio: Optional[float] = None,
        supertrend_flip_bull: bool = False,
        kd_golden_cross: bool = False,
        macd_golden_cross: bool = False,
        # [P1-B] Veto rule inputs
        above_bb_upper: bool = False,
        price_down_vol_up: bool = False,
        close_gt_open: bool = False,
) -> Dict[str, Any]:
    feat: Dict[str, Any] = {}

    if atr_now is None or atr_now <= 0 or not np.isfinite(atr_now):
        return {"decision_available": False}

    feat["decision_available"] = True

    is_bullish = supertrend_dir == 1
    is_bearish_trend = trend_state in ("downtrend", "strong_downtrend", "weak_downtrend")

    # Adaptive stop-loss multiplier based on context
    if is_bearish_trend:
        multiplier = 2.5
        feat["stop_loss_context"] = "bearish_trend_caution"
    elif bb_squeeze and volatility_regime not in ("high_volatility", "expansion", "squeeze_breakout"):
        multiplier = 1.5
        feat["stop_loss_context"] = "bb_squeeze_tight"
    elif volatility_regime in ("high_volatility", "expansion"):
        multiplier = 2.5
        feat["stop_loss_context"] = "high_volatility_wide"
    elif volatility_regime == "squeeze_breakout":
        multiplier = 2.0
        feat["stop_loss_context"] = "squeeze_breakout"
    else:
        multiplier = 2.0
        feat["stop_loss_context"] = "normal"

    raw_stop = c_now - multiplier * atr_now

    # --- ATR FLOOR MECHANISM ---
    min_stop_distance = 1.5 * atr_now
    max_stop_price = c_now - min_stop_distance

    if support and support < c_now and support > raw_stop:
        if support <= max_stop_price:
            stop_loss = round(support, 2)
            feat["stop_loss_context"] += "_support_anchored"
        else:
            stop_loss = round(max_stop_price, 2)
            feat["stop_loss_context"] += "_atr_floor_enforced"
    else:
        stop_loss = round(raw_stop, 2)

    feat["atr_stop_loss"] = stop_loss
    feat["atr_stop_loss_pct"] = round((stop_loss - c_now) / c_now * 100, 2)
    feat["atr_stop_multiplier"] = multiplier

    if resistance and resistance > c_now:
        feat["target_resistance"] = resistance
        feat["target_distance_pct"] = round((resistance - c_now) / c_now * 100, 2)
        risk = c_now - stop_loss
        reward = resistance - c_now
        rr = round(reward / risk, 2) if risk > 0 else None
        feat["risk_reward_ratio"] = rr

        # [P3 FIX] Tighter anomaly threshold
        if rr and rr > RR_ANOMALY_THRESHOLD:
            feat["flag_rr_anomaly"] = True
            feat["stop_loss_context"] += "_[Whipsaw_Risk]"
        else:
            feat["flag_rr_anomaly"] = False
    else:
        feat["target_resistance"] = None
        feat["target_distance_pct"] = None
        feat["risk_reward_ratio"] = None
        feat["flag_rr_anomaly"] = False

    if support and support < c_now:
        feat["nearest_support"] = support
        feat["support_distance_pct"] = round((support - c_now) / c_now * 100, 2)
    else:
        feat["nearest_support"] = None
        feat["support_distance_pct"] = None

    # ---------------------------------------------------------------
    # [P4 Item 1] EVENT-BASED ENTRY TRIGGER (v2)
    # Now includes:
    #   - Veto rules for adverse tape conditions (P1-B)
    #   - Single-event confirmation requirement
    #   - Price-action quality check
    # ---------------------------------------------------------------
    rr_val = feat.get("risk_reward_ratio")
    rr_ok = bool(rr_val is not None and rr_val >= ENTRY_MIN_RR)
    vol_surge = bool(vol_ratio is not None and vol_ratio > 1.5)

    # "Today events" — instantaneous signals that only fire on transition day
    today_events = []
    if bb_squeeze_fire:
        today_events.append("bb_squeeze_fire")
    if supertrend_flip_bull:
        today_events.append("supertrend_flip_bull")
    if kd_golden_cross:
        today_events.append("kd_golden_cross")
    if macd_golden_cross:
        today_events.append("macd_golden_cross")
    if vol_surge:
        today_events.append("volume_surge")

    has_event = len(today_events) >= 1
    not_bearish = trend_state not in ("downtrend", "strong_downtrend", "weak_downtrend", "top_pullback")

    # [P1-B] VETO RULES — hard disqualifiers that prevent trigger regardless of events
    veto_reasons = []
    if price_down_vol_up and not supertrend_flip_bull:
        # Distribution bar (down + volume) — only override if a strong reversal signal is present
        veto_reasons.append("price_down_vol_up_distribution")
    if above_bb_upper and volatility_regime in ("high_volatility", "expansion"):
        # Overextended + high vol — pullback risk too high
        veto_reasons.append("above_bb_upper_in_high_vol")
    if not close_gt_open and not supertrend_flip_bull and not kd_golden_cross:
        # Red candle without strong reversal confirmation
        veto_reasons.append("red_candle_no_reversal_confirm")

    vetoed = len(veto_reasons) > 0

    # [P1-B] CONFIRMATION REQUIREMENT — single-event triggers need extra support
    # bb_squeeze_fire alone is a "candidate", not a trigger. It needs at least one of:
    #   - volume confirmation (vol_surge)
    #   - another event (ST flip, KD cross, MACD cross)
    #   - green candle in favorable regime
    needs_confirmation = (len(today_events) == 1 and today_events[0] == "bb_squeeze_fire")
    has_confirmation = vol_surge or close_gt_open
    single_event_blocked = needs_confirmation and not has_confirmation

    # Final trigger decision
    trigger_pass = bool(
        has_event and rr_ok and not_bearish and is_bullish
        and not vetoed and not single_event_blocked
    )

    feat["flag_entry_trigger"] = trigger_pass
    feat["entry_trigger_events"] = today_events if trigger_pass else []
    feat["entry_trigger_veto"] = veto_reasons if veto_reasons else []

    if trigger_pass:
        feat["entry_trigger_reason"] = f"Events: {', '.join(today_events)} | R:R={rr_val}"
    elif vetoed:
        feat["entry_trigger_reason"] = f"vetoed: {', '.join(veto_reasons)}"
    elif single_event_blocked:
        feat["entry_trigger_reason"] = f"needs_confirmation: {today_events[0]} alone insufficient"
    elif has_event and not rr_ok:
        feat["entry_trigger_reason"] = f"rr_insufficient: R:R={rr_val} < {ENTRY_MIN_RR}"
    else:
        feat["entry_trigger_reason"] = "no_trigger"

    # ---------------------------------------------------------------
    # [P4 Item 2] SUGGESTED POSITION SIZE
    # Formula: Position % = Portfolio Risk % / |ATR Stop Loss %|
    # Capped at MAX_POSITION_PCT to prevent concentration.
    # ---------------------------------------------------------------
    stop_pct = abs(feat.get("atr_stop_loss_pct", -3.0))
    if stop_pct > 0:
        raw_size = PORTFOLIO_RISK_PCT / stop_pct * 100
        feat["position_size_pct"] = round(min(raw_size, MAX_POSITION_PCT), 1)
    else:
        feat["position_size_pct"] = None

    return feat


# ===================================================================
# 14. MAIN: BUILD AI FEATURES JSON
# [OPT] Sequential yf downloads (thread-unsafe) + parallel external data
# ===================================================================


def _download_yf(ticker: str, period: str, interval: str):
    """Helper for threaded yf.download — serialised via _yf_lock."""
    with _yf_lock:
        try:
            # [FIX] auto_adjust=True and ffill() to avoid missing data/unadjusted price bugs
            df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
            df = clean_yf_columns(_ensure_naive_index(df))
            if not df.empty:
                df = df.ffill().dropna(subset=["Close"])
            return df
        except Exception:
            return pd.DataFrame()


def build_ai_features(stock_id: str, as_of_date=None, mode: str = "ai") -> Dict[str, Any]:
    stock_id = stock_id.strip().upper()
    yf_ticker = _resolve_yf_ticker(stock_id)
    stock_no = _strip_suffix(stock_id)

    # -- download daily (sequential — yf.download is NOT thread-safe) --
    df_daily = _download_yf(yf_ticker, "2y", "1d")

    if df_daily.empty and yf_ticker.endswith(".TW"):
        yf_ticker = yf_ticker.replace(".TW", ".TWO")
        df_daily = _download_yf(yf_ticker, "2y", "1d")

    if df_daily.empty:
        return {"error": f"not found: {stock_id}", "symbol": stock_id}

    chosen_ts = _nearest_trading_ts(df_daily, as_of_date)
    if chosen_ts is None:
        return {"error": "no data before date", "symbol": stock_id}

    df = df_daily.loc[:chosen_ts].copy()
    if len(df) < 60:
        return {"error": "less than 60 days data", "symbol": stock_id}

    latest = df.iloc[-1]
    trade_date = chosen_ts.date()
    query_date = as_of_date if as_of_date else datetime.now().date()

    # -- download weekly + benchmark sequentially (AI only) --
    df_weekly_upto = None
    bench_df = None
    if mode == "ai":
        df_weekly = _download_yf(yf_ticker, "2y", "1wk")
        df_weekly_upto = df_weekly.loc[:chosen_ts] if df_weekly is not None and not df_weekly.empty else None
        # [P1] Use cached benchmark — downloaded once, reused for all stocks
        bench_df = _get_cached_benchmark()

    # -- basic price --
    close = df["Close"]
    c_now = float(close.iloc[-1])
    o_now = float(latest["Open"])
    h_now = float(latest["High"])
    l_now = float(latest["Low"])
    vol_now = float(latest["Volume"])

    feat: Dict[str, Any] = {"symbol": yf_ticker, "price_date": str(trade_date), "query_date": str(query_date)}

    feat["close"] = c_now
    feat["open"] = o_now
    feat["high"] = h_now
    feat["low"] = l_now
    feat["volume"] = int(vol_now)
    feat["volume_k"] = int(vol_now / 1000)

    # -- MA + deviation --
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    ma5_now = float(ma5.iloc[-1])
    ma20_now = float(ma20.iloc[-1])
    ma60_now = float(ma60.iloc[-1])

    feat["ma5"] = round(ma5_now, 2)
    feat["ma20"] = round(ma20_now, 2)
    feat["ma60"] = round(ma60_now, 2)

    ma20_dev = (c_now - ma20_now) / ma20_now * 100
    ma60_dev = (c_now - ma60_now) / ma60_now * 100
    feat["ma20_dev_pct"] = round(ma20_dev, 4)
    feat["ma60_dev_pct"] = round(ma60_dev, 4)

    ma20_dev_series = (close - ma20) / ma20 * 100
    ma60_dev_series = (close - ma60) / ma60 * 100
    feat["ma20_dev_percentile_252d"] = percentile_rank(ma20_dev_series.dropna(), 252)
    feat["ma60_dev_percentile_252d"] = percentile_rank(ma60_dev_series.dropna(), 252)

    feat["ma5_slope_5d"] = slope_n(ma5, 5)
    feat["ma20_slope_5d"] = slope_n(ma20, 5)

    # [P3 FIX] Preliminary MA-only trend classification for sections that run
    # before ADX is available. Will be overwritten by classify_trend() after ADX.
    if c_now > ma20_now > ma60_now and ma5_now > ma20_now:
        trend_state = "uptrend"
    elif c_now > ma20_now > ma60_now and ma5_now <= ma20_now:
        trend_state = "uptrend_pullback"
    elif c_now < ma20_now < ma60_now and ma5_now < ma20_now:
        trend_state = "downtrend"
    elif c_now < ma20_now < ma60_now and ma5_now >= ma20_now:
        trend_state = "downtrend_bounce"
    elif c_now > ma20_now and ma20_now < ma60_now:
        trend_state = "bottom_bounce"
    elif c_now < ma20_now and ma20_now > ma60_now:
        trend_state = "top_pullback"
    else:
        trend_state = "consolidation"
    # Note: trend_state will be REFINED with ADX qualification by classify_trend() below

    # 52w position
    high_52 = float(df.tail(252)["High"].max())
    low_52 = float(df.tail(252)["Low"].min())
    feat["pos_52w_pct"] = round((c_now - low_52) / (high_52 - low_52) * 100, 2) if high_52 != low_52 else 50.0

    # weekly vs 20w MA + multi-timeframe analysis (AI only, Item #12)
    if mode == "ai" and df_weekly_upto is not None and not df_weekly_upto.empty and len(df_weekly_upto) >= 20:
        w_close = df_weekly_upto["Close"]
        wma20 = float(w_close.rolling(20).mean().iloc[-1])
        feat["weekly_above_ma20"] = bool(float(w_close.iloc[-1]) > wma20)

        # Weekly trend state
        wma5 = w_close.rolling(5).mean()
        wma60_s = w_close.rolling(60).mean()
        wma5_now = float(wma5.iloc[-1]) if len(wma5.dropna()) >= 1 else None
        wma60_now = float(wma60_s.iloc[-1]) if len(wma60_s.dropna()) >= 1 else None
        wc_now = float(w_close.iloc[-1])

        if wma5_now and wma60_now:
            if wc_now > wma20 > wma60_now and wma5_now > wma20:
                weekly_trend = "uptrend"
            elif wc_now < wma20 < wma60_now and wma5_now < wma20:
                weekly_trend = "downtrend"
            else:
                weekly_trend = "consolidation"
        else:
            weekly_trend = "insufficient_data"
        feat["weekly_trend_state"] = weekly_trend

        # Weekly RSI
        if len(w_close.dropna()) >= 20:
            w_rsi = calculate_rsi(w_close, 14)
            feat["weekly_rsi14"] = round(float(w_rsi.iloc[-1]), 2)
        else:
            feat["weekly_rsi14"] = None

        # Multi-timeframe alignment — upgraded taxonomy (P2-C)
        # States: aligned_bull, aligned_bear, mixed_bull_bias, mixed_bear_bias, conflicting
        daily_strong_bull = trend_state in ("uptrend", "strong_uptrend")
        daily_bull_bias = trend_state in (
            "uptrend", "strong_uptrend", "weak_uptrend", "uptrend_pullback",
            "bottom_bounce", "strong_bottom_bounce", "weak_bottom_bounce",
        )
        daily_strong_bear = trend_state in ("downtrend", "strong_downtrend")
        daily_bear_bias = trend_state in (
            "downtrend", "strong_downtrend", "weak_downtrend",
            "downtrend_bounce", "top_pullback",
        )
        weekly_bull = weekly_trend == "uptrend"
        weekly_bear = weekly_trend == "downtrend"
        weekly_neutral = weekly_trend in ("consolidation", "insufficient_data")

        if daily_bull_bias and weekly_bull:
            feat["mtf_alignment"] = "aligned_bull"
        elif daily_bear_bias and weekly_bear:
            feat["mtf_alignment"] = "aligned_bear"
        elif daily_bull_bias and weekly_neutral:
            feat["mtf_alignment"] = "mixed_bull_bias"
        elif daily_bear_bias and weekly_neutral:
            feat["mtf_alignment"] = "mixed_bear_bias"
        elif daily_strong_bull and weekly_bear:
            feat["mtf_alignment"] = "conflicting"  # True directional disagreement
        elif daily_strong_bear and weekly_bull:
            feat["mtf_alignment"] = "conflicting"
        else:
            feat["mtf_alignment"] = "mixed_neutral"

        # Strict agreement (unchanged — directional only)
        feat["daily_weekly_trend_agree"] = bool(
            (daily_bull_bias and weekly_bull) or (daily_bear_bias and weekly_bear)
        )
        # [P2-C] Bias agreement (looser — includes neutral weekly)
        feat["daily_weekly_bias_agree"] = bool(
            (daily_bull_bias and (weekly_bull or weekly_neutral))
            or (daily_bear_bias and (weekly_bear or weekly_neutral))
        )
    else:
        feat["weekly_above_ma20"] = None
        feat["weekly_trend_state"] = None
        feat["weekly_rsi14"] = None
        feat["mtf_alignment"] = None
        feat["daily_weekly_trend_agree"] = None
        feat["daily_weekly_bias_agree"] = None

    # [OPT] relative strength using pre-fetched benchmark (no extra download)
    # Multi-period RS (Item #18): 20d, 63d, 252d
    if mode == "ai":
        for rs_period in [20, 63, 252]:
            rs_data = _calc_relative_strength_with_bench(df, bench_df, period=rs_period)
            if rs_data:
                feat[f"rs_vs_bench_{rs_period}d"] = rs_data.get(f"rs_{rs_period}d")
                if rs_period == 20:
                    feat["stock_ret_20d"] = rs_data.get("stock_ret_20d")
                    feat["bench_ret_20d"] = rs_data.get("bench_ret_20d")
            else:
                feat[f"rs_vs_bench_{rs_period}d"] = None
                if rs_period == 20:
                    feat["stock_ret_20d"] = None
                    feat["bench_ret_20d"] = None

        # RS acceleration: is short-term RS improving vs medium-term?
        rs_20 = feat.get("rs_vs_bench_20d")
        rs_63 = feat.get("rs_vs_bench_63d")
        feat["rs_improving"] = bool(rs_20 is not None and rs_63 is not None and rs_20 > rs_63)
    else:
        feat["rs_vs_bench_20d"] = None

    # volume analysis
    # [P0 FIX] Use PREVIOUS 5 days as baseline, excluding today's volume.
    # Including today dampens the ratio (today is 1/5 of the denominator).
    avg_vol_5_prev = float(df["Volume"].iloc[-6:-1].mean()) if len(df) >= 6 else float(df["Volume"].tail(5).mean())
    feat["vol_ratio_5d"] = round(vol_now / avg_vol_5_prev, 4) if avg_vol_5_prev > 0 else None
    prev_close = float(df["Close"].iloc[-2]) if len(df) >= 2 else c_now
    feat["flag_price_up_vol_up"] = bool(c_now > prev_close and vol_now > avg_vol_5_prev)
    feat["flag_price_up_vol_down"] = bool(c_now > prev_close and vol_now < avg_vol_5_prev)
    feat["flag_price_down_vol_up"] = bool(c_now < prev_close and vol_now > avg_vol_5_prev)

    vq = calculate_volume_quality(df)
    feat.update(vq)

    # RSI
    rsi14 = calculate_rsi(close, 14)
    rsi14_now = float(rsi14.iloc[-1])
    feat["rsi14"] = round(rsi14_now, 2)
    feat["rsi14_percentile_252d"] = percentile_rank(rsi14, 252)
    feat["rsi14_slope_5d"] = slope_n(rsi14, 5)

    # KD
    low_min = df["Low"].rolling(9).min()
    high_max = df["High"].rolling(9).max()
    rsv = (close - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    k_val = rsv.ewm(com=2).mean()
    d_val = k_val.ewm(com=2).mean()
    feat["kd_k"] = round(float(k_val.iloc[-1]), 2)
    feat["kd_d"] = round(float(d_val.iloc[-1]), 2)

    # [P2] KD cross with anti-whipsaw filter (zone + dynamic spread)
    if len(k_val) >= 2 and len(d_val) >= 2:
        k_prev = float(k_val.iloc[-2])
        d_prev = float(d_val.iloc[-2])
        k_now_v = float(k_val.iloc[-1])
        d_now_v = float(d_val.iloc[-1])
        raw_cross_up = k_prev <= d_prev and k_now_v > d_now_v
        raw_cross_down = k_prev >= d_prev and k_now_v < d_now_v
        kd_spread = abs(k_now_v - d_now_v)

        # Dynamic spread threshold: use recent KD volatility
        kd_diff_series = (k_val - d_val).dropna().tail(60)
        kd_std = float(kd_diff_series.std()) if len(kd_diff_series) >= 20 else 3.0
        min_spread = max(kd_std * KD_SPREAD_STD_MULT, 1.0)  # Floor at 1.0

        feat["flag_kd_golden_cross"] = bool(
            raw_cross_up and k_now_v < KD_ZONE_OVERSOLD and kd_spread > min_spread
        )
        feat["flag_kd_death_cross"] = bool(
            raw_cross_down and k_now_v > KD_ZONE_OVERBOUGHT and kd_spread > min_spread
        )
        # Raw crosses (unfiltered) — for the AI consumer
        feat["flag_kd_cross_up_raw"] = bool(raw_cross_up)
        feat["flag_kd_cross_down_raw"] = bool(raw_cross_down)
    else:
        feat["flag_kd_golden_cross"] = False
        feat["flag_kd_death_cross"] = False
        feat["flag_kd_cross_up_raw"] = False
        feat["flag_kd_cross_down_raw"] = False

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    osc = dif - dea
    feat["macd_dif"] = round(float(dif.iloc[-1]), 4)
    feat["macd_dea"] = round(float(dea.iloc[-1]), 4)
    feat["macd_osc"] = round(float(osc.iloc[-1]), 4)
    feat["macd_osc_slope_5d"] = slope_n(osc, 5)
    feat["macd_osc_percentile_252d"] = percentile_rank(osc, 252)
    feat["flag_macd_golden_cross"] = bool(
        len(dif) >= 2 and len(dea) >= 2 and dif.iloc[-2] < dea.iloc[-2] and dif.iloc[-1] > dea.iloc[-1]
    )
    feat["flag_macd_death_cross"] = bool(
        len(dif) >= 2 and len(dea) >= 2 and dif.iloc[-2] > dea.iloc[-2] and dif.iloc[-1] < dea.iloc[-1]
    )
    # [P0] Raw MACD crosses (unfiltered) — ghost field implementation
    feat["flag_macd_cross_up_raw"] = feat["flag_macd_golden_cross"]
    feat["flag_macd_cross_down_raw"] = feat["flag_macd_death_cross"]

    # ADX
    adx, pdi, mdi = calculate_adx(df)
    feat["adx"] = round(float(adx.iloc[-1]), 2)
    feat["plus_di"] = round(float(pdi.iloc[-1]), 2)
    feat["minus_di"] = round(float(mdi.iloc[-1]), 2)
    feat["flag_strong_trend"] = bool(float(adx.iloc[-1]) > 25)
    feat["flag_di_bullish"] = bool(float(pdi.iloc[-1]) > float(mdi.iloc[-1]))

    # [P3 FIX] Unified trend classification — ADX + MA in one atomic call
    # This eliminates the bug where intermediate exceptions could leave
    # trend_state in an unqualified state like bare "uptrend".
    adx_now_val = float(adx.iloc[-1])
    trend_state, trend_strength = classify_trend(c_now, ma5_now, ma20_now, ma60_now, adx_now_val)
    feat["trend_state"] = trend_state
    feat["trend_strength"] = trend_strength

    # ATR
    atr_series = calculate_atr(df, 14)
    atr_now = float(atr_series.iloc[-1])
    feat["atr14"] = round(atr_now, 2)
    feat["atr14_pct"] = round(atr_now / c_now * 100, 4)
    feat["atr14_percentile_252d"] = percentile_rank(atr_series, 252)

    # Volatility
    # Volatility (annualized)
    returns_20d = close.pct_change().tail(20)
    if len(returns_20d) >= 10:
        daily_std = float(returns_20d.std())
        feat["volatility_20d_ann"] = round(daily_std * math.sqrt(252) * 100, 4)
    else:
        feat["volatility_20d_ann"] = None
    feat["max_drawdown_20d"] = max_drawdown(close, 20)
    feat["max_drawdown_60d"] = max_drawdown(close, 60)

    # Bollinger Bands
    bbw, bb_upper, bb_lower = calculate_bbands(df)
    bb_upper_now = float(bb_upper.iloc[-1])
    bb_lower_now = float(bb_lower.iloc[-1])
    bb_range = bb_upper_now - bb_lower_now
    feat["bb_width_pct"] = round(float(bbw.iloc[-1]), 4)
    feat["bb_position_pct"] = round((c_now - bb_lower_now) / bb_range * 100, 2) if bb_range > 0 else 50.0
    feat["bb_width_percentile_252d"] = percentile_rank(bbw, 252)
    feat["flag_bb_squeeze"] = bool(
        feat["bb_width_percentile_252d"] is not None and feat["bb_width_percentile_252d"] < 0.15)
    feat["flag_above_bb_upper"] = bool(c_now > bb_upper_now)

    # Volatility regime classifier WITH HYSTERESIS (Phase 3 Item 2)
    # Requires VOL_REGIME_CONFIRM_DAYS consecutive days in new regime before switching
    def _classify_vol_regime_single(bb_p, atr_p):
        """Classify a single day's regime (raw, no hysteresis)."""
        if bb_p is None or atr_p is None or not np.isfinite(bb_p) or not np.isfinite(atr_p):
            return "unknown"
        if bb_p < 0.15 and atr_p < 0.25:
            return "compression"
        if bb_p < 0.25 and atr_p > 0.50:
            return "squeeze_breakout"
        if bb_p > 0.75 and atr_p > 0.75:
            return "expansion"
        if atr_p > 0.65:
            return "high_volatility"
        return "normal"

    def _classify_vol_regime_with_hysteresis(bbw_series, atr_series, confirm_days=VOL_REGIME_CONFIRM_DAYS):
        """
        Apply hysteresis: only switch regime if the NEW regime has been
        consistent for `confirm_days` consecutive days. This prevents the
        stop-loss multiplier from jumping erratically on single-day spikes.
        """
        if bbw_series is None or atr_series is None:
            return "unknown"
        n_check = confirm_days + 1
        if len(bbw_series.dropna()) < n_check or len(atr_series.dropna()) < n_check:
            # Not enough data for hysteresis — fall back to single-day
            bb_p = percentile_rank(bbw_series.dropna(), 252)
            atr_p = percentile_rank(atr_series.dropna(), 252)
            return _classify_vol_regime_single(bb_p, atr_p)

        # Compute regime for each of the last N days
        recent_regimes = []
        for offset in range(confirm_days):
            idx = -(offset + 1)
            bb_slice = bbw_series.iloc[:idx] if idx < -1 else bbw_series
            atr_slice = atr_series.iloc[:idx] if idx < -1 else atr_series
            bb_p = percentile_rank(bb_slice.dropna(), 252)
            atr_p = percentile_rank(atr_slice.dropna(), 252)
            recent_regimes.append(_classify_vol_regime_single(bb_p, atr_p))

        # Current day's raw classification
        bb_p_now = percentile_rank(bbw_series.dropna(), 252)
        atr_p_now = percentile_rank(atr_series.dropna(), 252)
        current_raw = _classify_vol_regime_single(bb_p_now, atr_p_now)

        # Hysteresis: only adopt new regime if ALL recent days agree
        if all(r == current_raw for r in recent_regimes):
            return current_raw
        # Otherwise stay with the most common recent regime (conservative)
        from collections import Counter
        counts = Counter(recent_regimes + [current_raw])
        return counts.most_common(1)[0][0]

    feat["volatility_regime"] = _classify_vol_regime_with_hysteresis(bbw, atr_series)

    # Also store the raw single-day classification for transparency
    feat["volatility_regime_raw"] = _classify_vol_regime_single(
        feat.get("bb_width_percentile_252d"),
        feat.get("atr14_percentile_252d"),
    )

    # SuperTrend
    st_direction = 0
    st_dir_series = None
    try:
        st_line, st_dir = calculate_supertrend(df)
        st_val = float(st_line.iloc[-1])
        st_direction = int(st_dir.iloc[-1])
        st_dir_series = st_dir
        feat["supertrend_bullish"] = bool(st_direction == 1)
        feat["supertrend_distance_pct"] = round((c_now - st_val) / st_val * 100, 4)
        # [P0] SuperTrend flip flags — fire only on the transition day
        if len(st_dir) >= 2:
            feat["flag_supertrend_flip_bull"] = bool(
                int(st_dir.iloc[-2]) == -1 and int(st_dir.iloc[-1]) == 1
            )
            feat["flag_supertrend_flip_bear"] = bool(
                int(st_dir.iloc[-2]) == 1 and int(st_dir.iloc[-1]) == -1
            )
        else:
            feat["flag_supertrend_flip_bull"] = False
            feat["flag_supertrend_flip_bear"] = False
    except Exception:
        feat["supertrend_bullish"] = None
        feat["supertrend_distance_pct"] = None
        feat["flag_supertrend_flip_bull"] = False
        feat["flag_supertrend_flip_bear"] = False

    # [P0] BB Squeeze Fire — bandwidth expanding after being in squeeze territory
    if len(bbw.dropna()) >= 6:
        recent_5_bbw = bbw.iloc[-6:-1]
        squeeze_threshold = bbw.quantile(0.15)
        was_squeezed = bool((recent_5_bbw < squeeze_threshold).any())
        feat["flag_bb_squeeze_fire"] = bool(was_squeezed and not feat.get("flag_bb_squeeze", False))
    else:
        feat["flag_bb_squeeze_fire"] = False

    # MFI
    mfi_s = calculate_mfi(df)
    feat["mfi14"] = round(float(mfi_s.iloc[-1]), 2)

    # AVWAP
    avwap = calculate_avwap(df, f"{chosen_ts.year}-01-01")
    if avwap is not None and not avwap.empty and np.isfinite(float(avwap.iloc[-1])):
        avwap_now = float(avwap.iloc[-1])
        feat["avwap_ytd"] = round(avwap_now, 2)
        feat["avwap_dev_pct"] = round((c_now - avwap_now) / avwap_now * 100, 4)
    else:
        feat["avwap_ytd"] = None
        feat["avwap_dev_pct"] = None

    # Volume Profile / POC
    vp = calculate_volume_profile(df, lookback=60)
    feat.update(vp)

    # short term S/R (pivot-clustered)
    sr = detect_short_term_sr(df, lookback=120)
    feat.update(sr)

    # Fibonacci (direction-aware)
    fib = calculate_fibonacci_summary(df, trend_state=trend_state)
    feat.update(fib)

    # Gaps
    gaps = detect_gaps_summary(df)
    feat["unfilled_gaps_count"] = len(gaps)
    if gaps:
        feat["nearest_gap_type"] = gaps[-1]["type"]
        feat["nearest_gap_lower"] = gaps[-1]["lower"]
        feat["nearest_gap_upper"] = gaps[-1]["upper"]
    else:
        feat["nearest_gap_type"] = None
        feat["nearest_gap_lower"] = None
        feat["nearest_gap_upper"] = None

    # Divergence
    div_rsi = detect_divergence(close, rsi14, lookback=60)
    feat["flag_bearish_divergence_rsi"] = div_rsi["bearish_divergence"]
    feat["flag_bullish_divergence_rsi"] = div_rsi["bullish_divergence"]

    div_macd = detect_divergence(close, osc, lookback=60)
    feat["flag_bearish_divergence_macd"] = div_macd["bearish_divergence"]
    feat["flag_bullish_divergence_macd"] = div_macd["bullish_divergence"]

    # Candle patterns
    candle_patterns = detect_momentum_candle_patterns(df, n=5)
    feat.update(candle_patterns)

    # Chart patterns
    raw_chart_patterns = detect_patterns(df)
    prioritized = _prioritize_patterns(raw_chart_patterns)

    feat["chart_pattern_count"] = len(raw_chart_patterns)

    if len(prioritized) >= 1:
        feat["chart_pattern_1"] = prioritized[0].get("pattern")
        feat["chart_pattern_1_confirmed"] = bool(prioritized[0].get("confirmed", False))
        feat["chart_pattern_1_bias"] = prioritized[0].get("bias")
    else:
        feat["chart_pattern_1"] = None
        feat["chart_pattern_1_confirmed"] = False
        feat["chart_pattern_1_bias"] = None

    if len(prioritized) >= 2:
        feat["chart_pattern_2"] = prioritized[1].get("pattern")
        feat["chart_pattern_2_confirmed"] = bool(prioritized[1].get("confirmed", False))
        feat["chart_pattern_2_bias"] = prioritized[1].get("bias")
    else:
        feat["chart_pattern_2"] = None
        feat["chart_pattern_2_confirmed"] = False
        feat["chart_pattern_2_bias"] = None

    # [OPT] External data (AI only) – run all 4 fetches in PARALLEL
    if mode == "ai":
        latest_overall_ts = df_daily.index[-1]
        price_20d_ago = float(df["Close"].iloc[-20]) if len(df) >= 20 else c_now
        avg_daily_vol_20d = float(df["Volume"].tail(20).mean()) if len(df) >= 20 else 0

        # Define tasks for parallel execution
        ext_results = {}

        def _fetch_institutional():
            try:
                chips_multi = get_institutional_multi_days(stock_id, trade_date, market_hint=yf_ticker, days=20)
                return compute_institutional_features(chips_multi, c_now, price_20d_ago, avg_daily_vol_20d)
            except Exception as e:
                return {"inst_data_available": False, "inst_error": str(e)}

        def _fetch_margin():
            try:
                margin = get_margin_short_data(
                    stock_id,
                    trade_date=latest_overall_ts.date(),
                    market_hint=yf_ticker,
                    max_back=7,
                )
                if isinstance(margin, dict) and margin.get("error") is None:
                    m_bal = margin.get("margin_balance")
                    s_bal = margin.get("short_balance")
                    m_chg = margin.get("margin_change")
                    s_chg = margin.get("short_change")
                    m_usage = margin.get("margin_usage_rate")
                    # [P0-A FIX] Only mark available if at least one core field is non-null
                    has_real_data = any(v is not None for v in [m_bal, s_bal, m_chg, s_chg, m_usage])
                    return {
                        "margin_balance": m_bal,
                        "margin_change": m_chg,
                        "margin_usage_rate": m_usage,
                        "short_balance": s_bal,
                        "short_change": s_chg,
                        "short_margin_ratio": round(s_bal / m_bal * 100, 2) if m_bal and s_bal and m_bal > 0 else None,
                        "margin_data_available": has_real_data,
                    }
                return {"margin_data_available": False}
            except Exception:
                return {"margin_data_available": False}

        def _fetch_tdcc():
            try:
                tdcc = get_tdcc_distribution(stock_no, weeks_back=2)
                return compute_tdcc_features(tdcc)
            except Exception:
                return {"tdcc_available": False}

        # [OPT] Run all 4 external data fetches in parallel threads
        with ThreadPoolExecutor(max_workers=4) as pool:
            fut_inst = pool.submit(_fetch_institutional)
            fut_margin = pool.submit(_fetch_margin)
            fut_tdcc = pool.submit(_fetch_tdcc)

        feat.update(fut_inst.result())
        feat.update(fut_margin.result())
        feat.update(fut_tdcc.result())

    else:
        feat["inst_data_available"] = False
        feat["margin_data_available"] = False
        feat["tdcc_available"] = False

    # ===================================================================
    # [P0] GHOST FIELD IMPLEMENTATIONS — Beta, Liquidity, Divergence Net
    # ===================================================================

    # Beta (60-day vs benchmark)
    if mode == "ai" and bench_df is not None and not bench_df.empty and len(df) >= 60:
        try:
            stock_ret = df["Close"].pct_change().iloc[-60:].dropna()
            bench_close_reb = bench_df["Close"].reindex(stock_ret.index, method="nearest")
            bench_ret_s = bench_close_reb.pct_change().dropna()
            common_idx = stock_ret.index.intersection(bench_ret_s.index)
            if len(common_idx) >= 30:
                cov_mat = np.cov(
                    stock_ret.loc[common_idx].values.astype(float),
                    bench_ret_s.loc[common_idx].values.astype(float),
                )
                var_bench = cov_mat[1, 1]
                feat["beta_60d"] = round(float(cov_mat[0, 1] / var_bench), 3) if var_bench > 0 else None
            else:
                feat["beta_60d"] = None
        except Exception:
            feat["beta_60d"] = None
    else:
        feat["beta_60d"] = None

    # Liquidity metrics
    avg_vol_20 = float(df["Volume"].tail(20).mean()) if len(df) >= 20 else 0
    feat["avg_daily_volume_20d"] = int(avg_vol_20)
    feat["avg_daily_turnover_20d_m"] = round(avg_vol_20 * c_now / 1_000_000, 2)
    feat["is_liquid"] = bool(feat["avg_daily_turnover_20d_m"] >= 50)  # >= 50M TWD/day

    # Divergence net signal (resolve conflicting RSI vs MACD divergences)
    bull_div = feat.get("flag_bullish_divergence_rsi") or feat.get("flag_bullish_divergence_macd")
    bear_div = feat.get("flag_bearish_divergence_rsi") or feat.get("flag_bearish_divergence_macd")
    if bull_div and bear_div:
        feat["divergence_net_signal"] = "conflicting"
    elif bull_div:
        feat["divergence_net_signal"] = "bullish"
    elif bear_div:
        feat["divergence_net_signal"] = "bearish"
    else:
        feat["divergence_net_signal"] = "none"

    # ===================================================================
    # Decision fields (WITH ATR FLOOR, EVENT-BASED TRIGGERS, POSITION SIZING)
    # ===================================================================
    resistance = feat.get("fib_nearest_resistance_1") or feat.get("st_resistance")
    # For breakout stocks at highs with no retracement resistance, use extension levels
    if resistance and resistance <= c_now * 1.005:
        ext_resistance = feat.get("fib_ext_1272")
        if ext_resistance and ext_resistance > c_now:
            resistance = ext_resistance
    support = feat.get("fib_nearest_support_1") or feat.get("st_support")

    # [P4] Pass event-based trigger inputs to decision engine
    decision = compute_decision_fields(
        c_now, atr_now, resistance, support, st_direction,
        bb_squeeze=bool(feat.get("flag_bb_squeeze")),
        trend_state=trend_state,
        volatility_regime=feat.get("volatility_regime", "normal"),
        # Event-based trigger inputs
        bb_squeeze_fire=bool(feat.get("flag_bb_squeeze_fire")),
        vol_ratio=feat.get("vol_ratio_5d"),
        supertrend_flip_bull=bool(feat.get("flag_supertrend_flip_bull")),
        kd_golden_cross=bool(feat.get("flag_kd_golden_cross")),
        macd_golden_cross=bool(feat.get("flag_macd_golden_cross")),
        # [P1-B] Veto rule inputs
        above_bb_upper=bool(feat.get("flag_above_bb_upper")),
        price_down_vol_up=bool(feat.get("flag_price_down_vol_up")),
        close_gt_open=bool(c_now > o_now),
    )
    feat.update(decision)

    # [P0] Poor risk/reward flag
    rr = feat.get("risk_reward_ratio")
    feat["flag_poor_risk_reward"] = bool(rr is not None and rr < 1.0)

    # --- Regulatory Status Tags (ATTENTION / DISPOSITION) ---
    try:
        feat["status_tags"] = get_regulatory_status(stock_no)
    except Exception:
        feat["status_tags"] = []

    # Data quality metadata (Item #19)
    # [P0-A] Data quality metadata — now includes external source reliability
    feat["data_quality"] = {
        "price_data_days": len(df),
        "volume_zero_pct": round(float((df["Volume"] == 0).sum()) / len(df) * 100, 1) if len(df) > 0 else None,
        "stale_warning": bool((datetime.now().date() - trade_date).days > 3),
    }
    if mode == "ai":
        feat["data_quality"]["inst_data_coverage"] = None
        # [P0-A] External source reliability gates
        feat["data_quality"]["external_sources"] = {
            "margin": {
                "available": feat.get("margin_data_available", False),
                "has_real_data": feat.get("margin_data_available", False),
            },
            "institutional": {
                "available": feat.get("inst_data_available", False),
            },
        }

    # Final sanitize
    feat = _sanitize_numpy(feat)
    return feat


# ===================================================================
# 15. TEXT REPORT
# ===================================================================


def format_text_report(feat: Dict[str, Any]) -> str:
    if "error" in feat and feat["error"]:
        return "X {}：{}".format(feat.get("symbol", "？"), feat["error"])

    lines = []
    SEP = "=" * 30

    lines.append(SEP)
    lines.append("  {}  Technical Summary".format(feat["symbol"]))
    lines.append("  Data：{}  Query：{}".format(feat["price_date"], feat["query_date"]))
    status_tags = feat.get("status_tags", [])
    if status_tags:
        tag_str = ", ".join(status_tags)
        lines.append("  *** REGULATORY: {} ***".format(tag_str))
    lines.append(SEP)

    lines.append("  Close：{}  Trend：{}".format(feat["close"], feat["trend_state"]))
    lines.append("  MA20 dev：{:+.2f}% (pctl={})".format(feat["ma20_dev_pct"], feat.get("ma20_dev_percentile_252d")))
    lines.append("  MA60 dev：{:+.2f}% (pctl={})".format(feat["ma60_dev_pct"], feat.get("ma60_dev_percentile_252d")))
    lines.append("  52w pos：{:.1f}%".format(feat["pos_52w_pct"]))

    lines.append("  RSI14：{}  KD：{}/{}".format(feat["rsi14"], feat["kd_k"], feat["kd_d"]))
    lines.append("  MACD OSC：{:.4f}  ADX：{}".format(feat["macd_osc"], feat["adx"]))
    lines.append("  ATR14：{} ({:.2f}%)".format(feat["atr14"], feat["atr14_pct"]))

    lines.append("  Vol ratio：{}  OBV slope 5d：{}".format(feat.get("vol_ratio_5d"), feat.get("obv_slope_5d")))

    flags = [k for k, v in feat.items() if k.startswith("flag_") and v is True]
    if flags:
        lines.append("  Flags：{}".format("，".join(f.replace("flag_", "") for f in flags)))

    cp1 = feat.get("chart_pattern_1")
    cp2 = feat.get("chart_pattern_2")
    if cp1 or cp2:
        parts = []
        if cp1:
            conf = "✓" if feat.get("chart_pattern_1_confirmed") else "?"
            parts.append("{}[{}]".format(cp1, conf))
        if cp2:
            conf = "✓" if feat.get("chart_pattern_2_confirmed") else "?"
            parts.append("{}[{}]".format(cp2, conf))
        lines.append("  Patterns：{}".format("，".join(parts)))

    if feat.get("decision_available"):
        lines.append("  Stop：{} ({}%)".format(feat.get("atr_stop_loss"), feat.get("atr_stop_loss_pct")))
        lines.append("  Target：{}  R:R={}".format(feat.get("target_resistance"), feat.get("risk_reward_ratio")))
        if feat.get("position_size_pct") is not None:
            lines.append("  Position Size：{}%".format(feat.get("position_size_pct")))
        if feat.get("flag_entry_trigger"):
            lines.append("  *** ENTRY TRIGGER: {} ***".format(feat.get("entry_trigger_reason", "")))

    # Vol regime & liquidity
    lines.append("  Vol Regime：{}  Liquid：{}".format(
        feat.get("volatility_regime", "?"), "Y" if feat.get("is_liquid") else "N"))
    if feat.get("beta_60d") is not None:
        lines.append("  Beta(60d)：{}".format(feat["beta_60d"]))

    lines.append(SEP)
    return "\n".join(lines)


# ===================================================================
# 16. SECTOR ANALYSIS – [OPT] parallel stock analysis
# ===================================================================

SECTOR_DICT = {
    "半導體權值": ["2330.TW", "2454.TW", "3711.TW", "3034.TW", "2303.TW"],
    "AI 伺服器": ["2382.TW", "3231.TW", "6669.TW", "2356.TW", "2376.TW"],
    "IC 設計高價": ["3661.TW", "3443.TW", "3529.TW", "3035.TW", "4966.TW"],
    "金融保險": ["2881.TW", "2882.TW", "2886.TW", "2891.TW", "5880.TW"],
    "航運貨櫃": ["2603.TW", "2609.TW", "2615.TW", "2637.TW", "2618.TW"],
    "重電綠能": ["1513.TW", "1519.TW", "1503.TW", "1514.TW", "6806.TW"],
    "散熱模組": ["3017.TW", "3324.TW", "3338.TW", "6230.TW", "2421.TW"],
}


def _analyze_one_stock_for_sector(stock_id, as_of_date, mode):
    """Helper for parallel sector analysis."""
    try:
        feat = build_ai_features(stock_id, as_of_date=as_of_date, mode=mode)
        if feat.get("error"):
            return {
                "Symbol": _strip_suffix(stock_id),
                "Close": "-",
                "Trend": "Error",
                "Vol_R": "-",
                "KD": "-/-",
                "Score": 0,
            }
        score = 0
        if feat.get("trend_state") == "uptrend":
            score += 2
        if feat.get("flag_price_up_vol_up"):
            score += 1
        if feat.get("flag_kd_golden_cross"):
            score += 1
        if feat.get("flag_inst_consensus_buy"):
            score += 2
        return {
            "Symbol": _strip_suffix(feat.get("symbol", stock_id)),
            "Close": feat.get("close", "-"),
            "Trend": feat.get("trend_state", "-"),
            "MA20_Dev": "{:+.2f}%".format(feat.get("ma20_dev_pct", 0) or 0),
            "Vol_R": feat.get("vol_ratio_5d", "-"),
            "KD": "{:.0f}/{:.0f}".format(feat.get("kd_k", 0) or 0, feat.get("kd_d", 0) or 0),
            "Score": score,
            "Status": ",".join(feat.get("status_tags", [])) or "-",
        }
    except Exception:
        return {"Symbol": stock_id, "Trend": "Exception", "Score": -1}


def analyze_sector_performance(sector_name: str, as_of_date=None, custom_tickers=None, mode: str = "human") -> str:
    target_list = custom_tickers if custom_tickers else SECTOR_DICT.get(sector_name, [])

    if not target_list:
        return f"No stocks in sector {sector_name}."

    # [OPT] Parallel analysis of all stocks in sector
    results = []
    with ThreadPoolExecutor(max_workers=min(len(target_list), 5)) as pool:
        futures = {
            pool.submit(_analyze_one_stock_for_sector, sid, as_of_date, mode): sid
            for sid in target_list
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda x: x.get("Score", 0), reverse=True)

    lines = []
    lines.append(f"Sector Scan：{sector_name}")
    d_str = str(as_of_date) if as_of_date else "Today"
    lines.append(f"Date：{d_str}")
    lines.append("-" * 50)
    lines.append(
        "{:<10} {:<8} {:<12} {:<10} {:<6} {:<8} {:<4} {:<12}".format(
            "Symbol", "Price", "Trend", "MA20Dev", "VolR", "KD", "Score", "Status"))
    lines.append("-" * 50)

    for r in results:
        lines.append(
            "{:<10} {:<8} {:<12} {:<10} {:<6} {:<8} {:<4} {:<12}".format(
                str(r.get("Symbol", "")),
                str(r.get("Close", "")),
                str(r.get("Trend", "")),
                str(r.get("MA20_Dev", "")),
                str(r.get("Vol_R", "")),
                str(r.get("KD", "")),
                str(r.get("Score", "")),
                str(r.get("Status", "-")),
            )
        )

    lines.append("-" * 50)
    lines.append("（Score：uptrend+2，price_up_vol_up+1，KD_golden+1，inst_consensus+2）")

    return "\n".join(lines)


# ===================================================================
# 17. ENTRY POINT
# mode="human" => fast, text only
# mode="ai"    => full JSON
# ===================================================================


def analyze_stock_technical(stock_id: str, as_of_date=None, mode: str = "human") -> dict:
    """
    Main entry point.
    mode="human" => returns {"human_report": str, "status_tags": list}
    mode="ai"    => returns {"ai_report": dict, "status_tags": list}
    status_tags: e.g. ["ATTENTION"], ["DISPOSITION"], ["ATTENTION","DISPOSITION"], or []
    """
    feat = build_ai_features(stock_id, as_of_date, mode=mode)
    tags = feat.get("status_tags", [])

    if mode == "ai":
        return {"ai_report": feat, "status_tags": tags}
    else:
        text = format_text_report(feat)
        return {"human_report": text, "status_tags": tags}


if __name__ == "__main__":
    import sys

    sid = sys.argv[1] if len(sys.argv) > 1 else "2330"
    m = sys.argv[2] if len(sys.argv) > 2 else "human"
    result = analyze_stock_technical(sid, mode=m)
    if "human_report" in result:
        print(result["human_report"])
    if "ai_report" in result:
        print(json.dumps(result["ai_report"], ensure_ascii=False, default=str))
