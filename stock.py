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
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import requests
import logging
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

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

# [FIX] Margin full-market cache — same pattern as institutional cache.
# One download per (market, date) for ALL stocks, avoids repeated TWSE requests.
_margin_csv_cache: Dict[tuple, Any] = {}
_margin_cache_lock = threading.Lock()

# [FIX] FinMind per-stock margin cache — keyed by (stock_no, date).
# FinMind API is rate-limited at 600/hour with token, so cache aggressively.
_finmind_margin_cache: Dict[tuple, Any] = {}
_finmind_cache_lock = threading.Lock()

# [NEW] FinMind extended data caches — monthly revenue, EPS, major holders, securities lending
# Monthly revenue and EPS are full-market (keyed by date), major holders and lending are per-stock
_finmind_revenue_cache: Dict[tuple, Any] = {}   # (stock_no, date) — per-stock
_finmind_eps_cache: Dict[tuple, Any] = {}       # (stock_no, date) — per-stock
_finmind_holders_cache: Dict[tuple, Any] = {}   # (stock_no, date) — per-stock (now used for foreign shareholding)
_finmind_lending_cache: Dict[tuple, Any] = {}   # (stock_no, date) — per-stock
_finmind_daytrade_cache: Dict[tuple, Any] = {}  # (stock_no, date) — per-stock (day-trading ratio)
_finmind_ext_cache_lock = threading.Lock()

# [FIX] Serialize TWSE/TPEx API requests to avoid rate-limiting
# Only 1 thread can make an uncached TWSE request at a time
_twse_request_semaphore = threading.Semaphore(1)


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

        # [FIX] Rate-limit guard: sleep before TWSE request to avoid 429/block
        # [FIX] Serialize TWSE requests across all threads
        with _twse_request_semaphore:
            time.sleep(0.35)
            url = f"https://www.twse.com.tw/fund/T86?response=csv&date={d_.strftime('%Y%m%d')}&selectType=ALLBUT0999"
            # [FIX] Double-check cache inside semaphore (another thread may have fetched it)
            with _inst_cache_lock:
                if cache_key in _inst_csv_cache:
                    cached = _inst_csv_cache[cache_key]
                    if cached is None:
                        return None, "TWSE cached no-data"
                    return cached, None
            try:
                r = sess.get(url, headers=headers, timeout=15)
            except Exception as e:
                # [FIX] Network error → do NOT cache, allow retry on next stock
                return None, f"TWSE request error: {e}"
        if r.status_code != 200 or len(r.text) < 200:
            # [FIX] HTTP error (rate-limit etc.) → do NOT cache as None
            # Only cache legitimate "no data" responses, not transient failures
            return None, f"TWSE HTTP {r.status_code}"
        if "沒有符合條件的資料" in r.text or "很抱歉" in r.text:
            # This IS a legitimate no-data response → safe to cache
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

        # [FIX] Rate-limit guard + serialize
        with _twse_request_semaphore:
            time.sleep(0.2)
            url = (
                "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"
                f"?l=zh-tw&o=csv&se=EW&t=D&d={roc_date}&s=0,asc"
            )
            h2 = dict(headers)
            h2["Referer"] = "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge.php"
            # [FIX] Double-check cache inside semaphore
            with _inst_cache_lock:
                if cache_key in _inst_csv_cache:
                    cached = _inst_csv_cache[cache_key]
                    if cached is None:
                        return None, "TPEx cached no-data"
                    return cached, None
            try:
                r = sess.get(url, headers=h2, timeout=15)
            except Exception as e:
                return None, f"TPEx request error: {e}"
        if r.status_code != 200 or len(r.text) < 200:
            # [FIX] HTTP error → do NOT cache, allow retry
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

    # [V2] Surface the latest T86 data date for staleness checks downstream
    try:
        feat["inst_date"] = chips_multi[-1].get("date") if chips_multi else None
    except Exception:
        feat["inst_date"] = None

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


def _finmind_margin_fetch(stock_no: str, trade_date, max_back: int = 10):
    """Fetch margin/short sale data from FinMind API.

    [FIX] FinMind is a free Taiwan financial data API that proxies TWSE/TPEx data.
    Used as primary source because Streamlit Cloud's IP can't reach TWSE MI_MARGN
    endpoint (both JSON and CSV variants are blocked), but TWSE T86 works.

    Returns normalized dict matching get_margin_short_data's return schema,
    or None if FinMind doesn't have data (caller should fall back to TWSE/TPEx).

    Requires FINMIND_TOKEN env variable (optional — works without but lower rate limit).
    """
    import os
    token = os.environ.get("FINMIND_TOKEN", "").strip()

    # Cache key — per stock, per requested date
    date_str = trade_date.strftime("%Y-%m-%d")
    cache_key = (stock_no, date_str)
    with _finmind_cache_lock:
        if cache_key in _finmind_margin_cache:
            return _finmind_margin_cache[cache_key]

    sess = _get_session()
    url = "https://api.finmindtrade.com/api/v4/data"
    # Query window: go back max_back days to handle non-trading days
    start_date = (trade_date - timedelta(days=max_back)).strftime("%Y-%m-%d")
    end_date = date_str

    params = {
        "dataset": "TaiwanStockMarginPurchaseShortSale",
        "data_id": stock_no,
        "start_date": start_date,
        "end_date": end_date,
    }
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        r = sess.get(url, params=params, headers=headers, timeout=20)
        if r.status_code != 200:
            logger.warning(f"FinMind HTTP {r.status_code} for {stock_no}")
            with _finmind_cache_lock:
                _finmind_margin_cache[cache_key] = None
            return None
        payload = r.json()
        if payload.get("status") != 200:
            # e.g. rate limited, bad token
            logger.warning(f"FinMind status {payload.get('status')} msg={payload.get('msg')}")
            with _finmind_cache_lock:
                _finmind_margin_cache[cache_key] = None
            return None

        rows = payload.get("data") or []
        if not rows:
            with _finmind_cache_lock:
                _finmind_margin_cache[cache_key] = None
            return None

        # Rows sorted by date ascending — take the latest one
        rows_sorted = sorted(rows, key=lambda x: x.get("date", ""))
        latest = rows_sorted[-1]

        # Map FinMind fields to our schema
        m_bal = latest.get("MarginPurchaseTodayBalance")
        m_yes = latest.get("MarginPurchaseYesterdayBalance")
        m_lim = latest.get("MarginPurchaseLimit")
        s_bal = latest.get("ShortSaleTodayBalance")
        s_yes = latest.get("ShortSaleYesterdayBalance")

        # Compute change from today - yesterday
        m_chg = (m_bal - m_yes) if (m_bal is not None and m_yes is not None) else None
        s_chg = (s_bal - s_yes) if (s_bal is not None and s_yes is not None) else None
        usage = round(m_bal / m_lim * 100, 1) if (m_bal and m_lim and m_lim > 0) else None

        result = {
            "id": f"{stock_no}.TW",
            "date": latest.get("date"),
            "margin_balance": _safe_int(m_bal),
            "margin_change": _safe_int(m_chg),
            "margin_limit": _safe_int(m_lim),
            "margin_usage_rate": usage,
            "short_balance": _safe_int(s_bal),
            "short_change": _safe_int(s_chg),
            "error": None,
        }
        with _finmind_cache_lock:
            _finmind_margin_cache[cache_key] = result
        return result
    except Exception as e:
        logger.warning(f"FinMind fetch error for {stock_no}: {e}")
        with _finmind_cache_lock:
            _finmind_margin_cache[cache_key] = None
        return None


def _finmind_api_get(dataset: str, params: dict, timeout: int = 20):
    """Unified FinMind API caller. Returns list of row dicts or None.

    Handles: token auth, HTTP errors, status checks, rate limit detection.
    """
    import os
    token = os.environ.get("FINMIND_TOKEN", "").strip()
    sess = _get_session()
    url = "https://api.finmindtrade.com/api/v4/data"
    q = dict(params)
    q["dataset"] = dataset
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        r = sess.get(url, params=q, headers=headers, timeout=timeout)
        if r.status_code != 200:
            logger.warning(f"FinMind {dataset} HTTP {r.status_code}")
            return None
        payload = r.json()
        if payload.get("status") != 200:
            logger.warning(f"FinMind {dataset} status {payload.get('status')}: {payload.get('msg')}")
            return None
        return payload.get("data") or []
    except Exception as e:
        logger.warning(f"FinMind {dataset} error: {e}")
        return None


def _finmind_revenue_fetch(stock_no: str, trade_date):
    """Fetch monthly revenue data for a stock.

    Returns dict with revenue YoY/MoM trend analysis, or None.
    """
    date_str = trade_date.strftime("%Y-%m-%d")
    cache_key = (stock_no, date_str)
    with _finmind_ext_cache_lock:
        if cache_key in _finmind_revenue_cache:
            return _finmind_revenue_cache[cache_key]

    # Fetch past 14 months to have enough for 3m rolling YoY + prior year comparison
    start_date = (trade_date - timedelta(days=450)).strftime("%Y-%m-%d")
    rows = _finmind_api_get("TaiwanStockMonthRevenue", {
        "data_id": stock_no,
        "start_date": start_date,
    })

    result = None
    if rows:
        try:
            # Sort by revenue_year + revenue_month ascending
            def _rev_key(r):
                return (int(r.get("revenue_year", 0)), int(r.get("revenue_month", 0)))
            rows_sorted = sorted(rows, key=_rev_key)

            if len(rows_sorted) >= 1:
                latest = rows_sorted[-1]
                latest_rev = latest.get("revenue")
                latest_yoy = latest.get("revenue_year_ago_yoy") or latest.get("YoY")
                # FinMind field name varies; compute manually if missing
                if latest_yoy is None and len(rows_sorted) >= 13:
                    prev_year = rows_sorted[-13]  # same month last year
                    pv = prev_year.get("revenue")
                    if pv and pv > 0 and latest_rev:
                        latest_yoy = round((latest_rev - pv) / pv * 100, 2)

                # MoM — current vs last month
                latest_mom = None
                if len(rows_sorted) >= 2:
                    prev_month = rows_sorted[-2]
                    pm = prev_month.get("revenue")
                    if pm and pm > 0 and latest_rev:
                        latest_mom = round((latest_rev - pm) / pm * 100, 2)

                # 3-month rolling YoY average (last 3 months YoY averaged)
                yoy_3m_avg = None
                yoys = []
                for i in range(1, min(4, len(rows_sorted) + 1)):
                    idx = -i
                    r = rows_sorted[idx]
                    y = r.get("revenue_year_ago_yoy") or r.get("YoY")
                    if y is None:
                        # try manual compute
                        try:
                            yr = int(r.get("revenue_year"))
                            mo = int(r.get("revenue_month"))
                            # find same month previous year
                            for pr in rows_sorted:
                                if int(pr.get("revenue_year", 0)) == yr - 1 and int(pr.get("revenue_month", 0)) == mo:
                                    pv = pr.get("revenue")
                                    cv = r.get("revenue")
                                    if pv and pv > 0 and cv is not None:
                                        y = (cv - pv) / pv * 100
                                    break
                        except Exception:
                            pass
                    if y is not None:
                        yoys.append(float(y))
                if yoys:
                    yoy_3m_avg = round(sum(yoys) / len(yoys), 2)

                # Momentum classification
                momentum = "unknown"
                if latest_yoy is not None and yoy_3m_avg is not None:
                    if latest_yoy > yoy_3m_avg + 5:
                        momentum = "accelerating"
                    elif latest_yoy < yoy_3m_avg - 5:
                        momentum = "decelerating"
                    else:
                        momentum = "stable"
                elif latest_yoy is not None:
                    momentum = "stable" if abs(latest_yoy) < 10 else ("accelerating" if latest_yoy > 0 else "decelerating")

                result = {
                    "revenue_month_label": f"{latest.get('revenue_year')}/{latest.get('revenue_month'):02d}" if latest.get('revenue_month') else None,
                    "revenue_latest": _safe_int(latest_rev),
                    "revenue_yoy_latest": round(float(latest_yoy), 2) if latest_yoy is not None else None,
                    "revenue_mom_latest": latest_mom,
                    "revenue_yoy_3m_avg": yoy_3m_avg,
                    "revenue_momentum": momentum,
                    "revenue_data_available": True,
                }
        except Exception as e:
            logger.warning(f"FinMind revenue parse error for {stock_no}: {e}")
            result = None

    with _finmind_ext_cache_lock:
        _finmind_revenue_cache[cache_key] = result
    return result


def _finmind_eps_fetch(stock_no: str, trade_date):
    """Fetch EPS data from financial statements.

    Returns dict with latest quarter EPS + trend, or None.
    """
    date_str = trade_date.strftime("%Y-%m-%d")
    cache_key = (stock_no, date_str)
    with _finmind_ext_cache_lock:
        if cache_key in _finmind_eps_cache:
            return _finmind_eps_cache[cache_key]

    # Fetch past ~2 years of financial statements to get trend
    start_date = (trade_date - timedelta(days=800)).strftime("%Y-%m-%d")
    rows = _finmind_api_get("TaiwanStockFinancialStatements", {
        "data_id": stock_no,
        "start_date": start_date,
    })

    result = None
    if rows:
        try:
            # Filter to EPS rows only
            eps_rows = [r for r in rows if r.get("type") == "EPS"]
            if eps_rows:
                eps_sorted = sorted(eps_rows, key=lambda r: r.get("date", ""))
                latest = eps_sorted[-1]
                latest_eps = latest.get("value")
                latest_date = latest.get("date")

                # EPS YoY — same quarter last year
                eps_yoy = None
                if len(eps_sorted) >= 5:
                    prev_year = eps_sorted[-5]  # 4 quarters ago
                    pv = prev_year.get("value")
                    if pv is not None and pv != 0 and latest_eps is not None:
                        eps_yoy = round((latest_eps - pv) / abs(pv) * 100, 2)

                # EPS QoQ — last quarter
                eps_qoq = None
                if len(eps_sorted) >= 2:
                    prev_q = eps_sorted[-2]
                    pv = prev_q.get("value")
                    if pv is not None and pv != 0 and latest_eps is not None:
                        eps_qoq = round((latest_eps - pv) / abs(pv) * 100, 2)

                # Trailing 4Q sum (TTM EPS)
                eps_ttm = None
                if len(eps_sorted) >= 4:
                    last4 = eps_sorted[-4:]
                    vals = [r.get("value") for r in last4 if r.get("value") is not None]
                    if len(vals) == 4:
                        eps_ttm = round(sum(vals), 2)

                # Trend: last 4 quarters direction
                trend = "unknown"
                if len(eps_sorted) >= 4:
                    last4 = [r.get("value") for r in eps_sorted[-4:] if r.get("value") is not None]
                    if len(last4) == 4:
                        if last4[-1] > last4[0] and last4[-1] > last4[-2]:
                            trend = "improving"
                        elif last4[-1] < last4[0] and last4[-1] < last4[-2]:
                            trend = "deteriorating"
                        else:
                            trend = "mixed"

                result = {
                    "eps_quarter_label": latest_date,
                    "eps_latest": round(float(latest_eps), 2) if latest_eps is not None else None,
                    "eps_yoy_pct": eps_yoy,
                    "eps_qoq_pct": eps_qoq,
                    "eps_ttm": eps_ttm,
                    "eps_trend_4q": trend,
                    "eps_data_available": True,
                }
        except Exception as e:
            logger.warning(f"FinMind EPS parse error for {stock_no}: {e}")
            result = None

    with _finmind_ext_cache_lock:
        _finmind_eps_cache[cache_key] = result
    return result


def _finmind_holders_fetch(stock_no: str, trade_date):
    """Fetch foreign shareholding ratio from TaiwanStockShareholding.

    NOTE: Originally designed for major holder (千張大戶) concentration via
    TaiwanStockHoldingSharesPer, but that dataset requires FinMind backer/sponsor
    membership. Switched to TaiwanStockShareholding (free) which gives foreign
    investment ratio — a different but equally useful chip signal for Taiwan stocks.

    Foreign holding ratio trend analysis:
      - Rising trend = foreign accumulation (strong chip signal)
      - Falling trend = foreign distribution
      - >50% ratio = foreign-dominated stock (e.g. 2330)

    Schema fields (Chinese → English):
      外資及陸資共用法令投資上限比率 → ChineseInvestmentUpperLimitRatio (overall foreign cap %)
      外資尚可投資股數 → ForeignInvestmentRemainingShares
      全體外資持有股數 → ForeignInvestmentShares (what we want)
      發行股數 → NumberOfSharesIssued (for computing ratio if needed)
    """
    date_str = trade_date.strftime("%Y-%m-%d")
    cache_key = (stock_no, date_str)
    with _finmind_ext_cache_lock:
        if cache_key in _finmind_holders_cache:
            return _finmind_holders_cache[cache_key]

    # Daily-ish updates — fetch ~35 days for 4-week delta
    start_date = (trade_date - timedelta(days=45)).strftime("%Y-%m-%d")
    rows = _finmind_api_get("TaiwanStockShareholding", {
        "data_id": stock_no,
        "start_date": start_date,
    })

    result = None
    if rows:
        try:
            rows_sorted = sorted(rows, key=lambda r: r.get("date", ""))
            if not rows_sorted:
                with _finmind_ext_cache_lock:
                    _finmind_holders_cache[cache_key] = None
                return None

            def _compute_ratio(row):
                """Compute foreign holding ratio (%) from a single row.
                Tries multiple field name variants that FinMind uses.
                """
                # Try pre-computed ratio first
                for key in ("ForeignInvestmentSharesRatio", "foreign_investment_shares_ratio",
                            "PercentageOfShareholding"):
                    v = row.get(key)
                    if v is not None:
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            pass
                # Else compute from shares / issued
                held = None
                issued = None
                for key in ("ForeignInvestmentShares", "foreign_investment_shares"):
                    if key in row:
                        try:
                            held = float(row[key])
                            break
                        except (TypeError, ValueError):
                            pass
                for key in ("NumberOfSharesIssued", "number_of_shares_issued"):
                    if key in row:
                        try:
                            issued = float(row[key])
                            break
                        except (TypeError, ValueError):
                            pass
                if held is not None and issued and issued > 0:
                    return round(held / issued * 100, 4)
                return None

            # Extract latest and prior-period ratios
            latest_row = rows_sorted[-1]
            latest_ratio = _compute_ratio(latest_row)
            latest_date = latest_row.get("date")

            if latest_ratio is None:
                with _finmind_ext_cache_lock:
                    _finmind_holders_cache[cache_key] = None
                return None

            # 4-week (~20 trading day) delta
            delta_4w = None
            if len(rows_sorted) >= 20:
                old_ratio = _compute_ratio(rows_sorted[-20])
                if old_ratio is not None:
                    delta_4w = round(latest_ratio - old_ratio, 3)
            elif len(rows_sorted) >= 2:
                # Fallback to oldest available if we don't have 20 days
                old_ratio = _compute_ratio(rows_sorted[0])
                if old_ratio is not None:
                    delta_4w = round(latest_ratio - old_ratio, 3)

            # Trend classification — threshold 0.2% over 4 weeks is meaningful
            trend = "unknown"
            if delta_4w is not None:
                if delta_4w > 0.2:
                    trend = "accumulating"
                elif delta_4w < -0.2:
                    trend = "reducing"
                else:
                    trend = "stable"

            result = {
                "foreign_holding_date": latest_date,
                "foreign_holding_pct": round(latest_ratio, 2),
                "foreign_holding_4w_delta": delta_4w,
                "foreign_accumulation_trend": trend,
                "foreign_holding_data_available": True,
            }
        except Exception as e:
            logger.warning(f"FinMind foreign shareholding parse error for {stock_no}: {e}")
            result = None

    with _finmind_ext_cache_lock:
        _finmind_holders_cache[cache_key] = result
    return result


def _finmind_lending_fetch(stock_no: str, trade_date):
    """Fetch securities lending (借券) data.

    Returns dict with lending balance and change.
    """
    date_str = trade_date.strftime("%Y-%m-%d")
    cache_key = (stock_no, date_str)
    with _finmind_ext_cache_lock:
        if cache_key in _finmind_lending_cache:
            return _finmind_lending_cache[cache_key]

    start_date = (trade_date - timedelta(days=30)).strftime("%Y-%m-%d")
    rows = _finmind_api_get("TaiwanStockSecuritiesLending", {
        "data_id": stock_no,
        "start_date": start_date,
    })

    result = None
    if rows:
        try:
            rows_sorted = sorted(rows, key=lambda r: r.get("date", ""))
            # Aggregate by date — lending can have multiple rows per day
            by_date: Dict[str, dict] = {}
            for r in rows_sorted:
                d = r.get("date", "")
                if d not in by_date:
                    by_date[d] = {"volume": 0, "value": 0}
                vol = r.get("volume") or r.get("Volume") or 0
                try:
                    by_date[d]["volume"] += int(vol)
                except (TypeError, ValueError):
                    pass

            dates_sorted = sorted(by_date.keys())
            if not dates_sorted:
                with _finmind_ext_cache_lock:
                    _finmind_lending_cache[cache_key] = None
                return None

            latest_date = dates_sorted[-1]
            latest_vol = by_date[latest_date]["volume"]

            # 5-day cumulative change — sum last 5 days vs 5 days before
            vol_5d = sum(by_date[d]["volume"] for d in dates_sorted[-5:])
            vol_5d_prior = sum(by_date[d]["volume"] for d in dates_sorted[-10:-5]) if len(dates_sorted) >= 10 else 0
            vol_5d_change = vol_5d - vol_5d_prior if vol_5d_prior else None

            # Smart short signal: 5d volume >= 2x prior 5d volume and > 100 lots
            smart_short = False
            if vol_5d >= 100 and vol_5d_prior and vol_5d >= vol_5d_prior * 2:
                smart_short = True

            result = {
                "lending_date": latest_date,
                "lending_volume_latest": _safe_int(latest_vol),
                "lending_volume_5d": _safe_int(vol_5d),
                "lending_volume_5d_change": _safe_int(vol_5d_change),
                "flag_smart_short_signal": smart_short,
                "lending_data_available": True,
            }
        except Exception as e:
            logger.warning(f"FinMind lending parse error for {stock_no}: {e}")
            result = None

    with _finmind_ext_cache_lock:
        _finmind_lending_cache[cache_key] = result
    return result


def _finmind_daytrade_fetch(stock_no: str, trade_date, total_volume_map: dict = None):
    """Fetch day-trading (當沖) data from TaiwanStockDayTrading.

    FinMind TaiwanStockDayTrading schema returns ONLY day-trade stats:
      {stock_id, date, BuyAfterSale, Volume, BuyAmount, SellAmount}
    where `Volume` is the day-trade volume (shares turned over intraday).

    To compute the day-trade RATIO we need to divide by total Volume from
    price data. Caller should pass `total_volume_map` = {date_str: total_volume}
    so we can compute ratio without another API call.

    Returns dict with:
      day_trade_ratio_latest          — latest day's DT ratio (%)
      day_trade_ratio_5d_avg          — 5-day moving average
      day_trade_ratio_20d_avg         — 20-day moving average
      day_trade_ratio_percentile_252d — percentile in last 252 days
      flag_day_trade_overheat         — latest > 45% AND 20d avg > 35% (sustained retail frenzy)
      flag_day_trade_divergence       — ratio rising but volume flat/down (散戶在炒、聰明錢離場)
    """
    date_str = trade_date.strftime("%Y-%m-%d")
    cache_key = (stock_no, date_str)
    with _finmind_ext_cache_lock:
        if cache_key in _finmind_daytrade_cache:
            return _finmind_daytrade_cache[cache_key]

    # Need ~300 days for 252-day percentile
    start_date = (trade_date - timedelta(days=380)).strftime("%Y-%m-%d")
    rows = _finmind_api_get("TaiwanStockDayTrading", {
        "data_id": stock_no,
        "start_date": start_date,
    })

    result = None
    if rows and total_volume_map:
        try:
            # Compute ratio per day where we have both DT volume and total volume
            ratios_by_date: Dict[str, float] = {}
            for r in rows:
                d = r.get("date", "")
                dt_vol_raw = r.get("Volume")
                try:
                    dt_vol = float(dt_vol_raw) if dt_vol_raw is not None else 0
                except (TypeError, ValueError):
                    continue
                total_vol = total_volume_map.get(d)
                if total_vol and total_vol > 0 and dt_vol > 0:
                    ratios_by_date[d] = round(dt_vol / total_vol * 100, 2)

            if not ratios_by_date:
                with _finmind_ext_cache_lock:
                    _finmind_daytrade_cache[cache_key] = None
                return None

            dates_sorted = sorted(ratios_by_date.keys())
            ratios_sorted = [ratios_by_date[d] for d in dates_sorted]

            latest = ratios_sorted[-1]
            latest_date = dates_sorted[-1]

            # 5-day MA
            avg_5d = round(sum(ratios_sorted[-5:]) / min(5, len(ratios_sorted)), 2)
            # 20-day MA
            avg_20d = round(sum(ratios_sorted[-20:]) / min(20, len(ratios_sorted)), 2) if len(ratios_sorted) >= 2 else latest

            # 252-day percentile
            window = ratios_sorted[-252:] if len(ratios_sorted) >= 30 else ratios_sorted
            rank = sum(1 for x in window if x <= latest) / len(window) if window else None
            percentile = round(rank, 4) if rank is not None else None

            # Overheat flag: sustained high day-trade activity
            flag_overheat = bool(latest > 45 and avg_20d > 35)

            # Divergence flag: DT ratio rising (last 5d avg vs prior 5d avg) while
            # total volume NOT rising — means the extra turnover is pure retail churn.
            flag_divergence = False
            if len(ratios_sorted) >= 10 and len(dates_sorted) >= 10:
                recent_5_ratio = sum(ratios_sorted[-5:]) / 5
                prior_5_ratio = sum(ratios_sorted[-10:-5]) / 5
                ratio_rising = recent_5_ratio > prior_5_ratio + 3  # ratio climbed at least 3 pp

                # Get corresponding total volumes
                recent_vols = [total_volume_map.get(d, 0) for d in dates_sorted[-5:]]
                prior_vols = [total_volume_map.get(d, 0) for d in dates_sorted[-10:-5]]
                recent_vol_avg = sum(recent_vols) / 5 if recent_vols else 0
                prior_vol_avg = sum(prior_vols) / 5 if prior_vols else 0
                vol_not_rising = recent_vol_avg <= prior_vol_avg * 1.1  # allow 10% noise

                flag_divergence = bool(ratio_rising and vol_not_rising)

            result = {
                "day_trade_date": latest_date,
                "day_trade_ratio_latest": latest,
                "day_trade_ratio_5d_avg": avg_5d,
                "day_trade_ratio_20d_avg": avg_20d,
                "day_trade_ratio_percentile_252d": percentile,
                "flag_day_trade_overheat": flag_overheat,
                "flag_day_trade_divergence": flag_divergence,
                "day_trade_data_available": True,
            }
        except Exception as e:
            logger.warning(f"FinMind daytrade parse error for {stock_no}: {e}")
            result = None

    with _finmind_ext_cache_lock:
        _finmind_daytrade_cache[cache_key] = result
    return result


def _parse_twse_margin_csv(text: str):
    """Parse TWSE MI_MARGN CSV response into a DataFrame.
    Same pattern as _parse_twse_t86_csv — find header row, parse to DataFrame.
    """
    if not text or len(text) < 200:
        return None
    text = text.replace("\r", "").replace("=", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    # Find header row — look for column headers containing 股票代號/代號 + 融資/資
    start = next(
        (
            i
            for i, ln in enumerate(lines)
            if ("代號" in ln or "代碼" in ln) and ("融資" in ln or "資買" in ln or "餘額" in ln)
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
               or lines[j].startswith("\"說明") or lines[j].startswith("\"備註")
        ),
        len(lines),
    )
    try:
        return pd.read_csv(io.StringIO("\n".join(lines[start:end])))
    except Exception:
        return None


def _twse_margin_json(date_yyyymmdd: str, headers: dict):
    """Fetch TWSE margin data for ALL stocks on a given date.

    [FIX] Added full-market cache (like _inst_csv_cache) so that multiple
    stocks on the same date share a single HTTP request. This eliminates
    the rate-limit problem where 7 TWSE stocks = 7 identical requests.

    [FIX] Changed from JSON to CSV format. The JSON endpoint is rate-limited
    more aggressively by TWSE, while the CSV endpoint (same as T86 三大法人)
    works reliably. Added CSV as primary, JSON as fallback.
    """
    sess = _get_session()  # [OPT]

    # [FIX] Check cache first
    cache_key = ("twse_margin", date_yyyymmdd)
    with _margin_cache_lock:
        if cache_key in _margin_csv_cache:
            cached = _margin_csv_cache[cache_key]
            if cached is None:
                return None, "TWSE margin cached no-data"
            return cached, None

    # [FIX] CSV first (reliable, same as T86), JSON as fallback
    urls = [
        (f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?response=csv&date={date_yyyymmdd}&selectType=ALL", "csv"),
        (f"https://www.twse.com.tw/exchangeReport/MI_MARGN?response=csv&date={date_yyyymmdd}&selectType=ALL", "csv"),
        (f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?response=json&date={date_yyyymmdd}&selectType=ALL", "json"),
    ]
    h2 = dict(headers)
    h2["Referer"] = "https://www.twse.com.tw/zh/trading/margin/mi-margn.html"
    last_err = None

    # [FIX] Serialize via semaphore to avoid hammering TWSE
    with _twse_request_semaphore:
        time.sleep(0.35)
        # Double-check cache inside semaphore (another thread may have fetched it)
        with _margin_cache_lock:
            if cache_key in _margin_csv_cache:
                cached = _margin_csv_cache[cache_key]
                if cached is None:
                    return None, "TWSE margin cached no-data"
                return cached, None

        for url, fmt in urls:
            try:
                r = sess.get(url, headers=h2, timeout=15)
                if r.status_code != 200:
                    last_err = f"HTTP {r.status_code}"
                    continue
                if "沒有符合條件的資料" in r.text or "很抱歉" in r.text:
                    with _margin_cache_lock:
                        _margin_csv_cache[cache_key] = None
                    return None, "TWSE margin no data"

                df = None
                if fmt == "csv":
                    df = _parse_twse_margin_csv(r.text)
                else:
                    try:
                        j = r.json()
                        df = pd.DataFrame(j) if isinstance(j, list) else _parse_twse_json_table(j)
                    except Exception:
                        pass

                if df is not None and not df.empty:
                    with _margin_cache_lock:
                        _margin_csv_cache[cache_key] = df
                    return df, None
                last_err = f"{fmt} empty"
            except Exception as e:
                last_err = str(e)

    # All URLs failed — only cache legitimate no-data, not transient errors
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


def _tpex_margin_csv_cached(roc_date: str, headers: dict):
    """Fetch TPEx margin data for ALL stocks on a given date.

    [FIX] Added full-market cache (like TWSE margin) so that multiple
    TPEx stocks on the same date share a single HTTP request.
    """
    sess = _get_session()
    cache_key = ("tpex_margin", roc_date)
    with _margin_cache_lock:
        if cache_key in _margin_csv_cache:
            cached = _margin_csv_cache[cache_key]
            if cached is None:
                return None, "TPEx margin cached no-data"
            return cached, None

    with _twse_request_semaphore:
        time.sleep(0.2)
        # Double-check cache inside semaphore
        with _margin_cache_lock:
            if cache_key in _margin_csv_cache:
                cached = _margin_csv_cache[cache_key]
                if cached is None:
                    return None, "TPEx margin cached no-data"
                return cached, None
        url_csv = (
            "https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php"
            f"?l=zh-tw&d={roc_date}&o=csv&s=0,asc"
        )
        try:
            r = sess.get(url_csv, headers=headers, timeout=15)
            if r.status_code != 200 or len(r.text) < 200:
                return None, f"TPEx margin HTTP {r.status_code}"
            df = _parse_tpex_margin_csv(r.text)
            if df is not None and not df.empty:
                with _margin_cache_lock:
                    _margin_csv_cache[cache_key] = df
                return df, None
            # Legitimate no-data
            with _margin_cache_lock:
                _margin_csv_cache[cache_key] = None
            return None, "TPEx margin empty"
        except Exception as e:
            return None, f"TPEx margin: {e}"


def get_margin_short_data(stock_id: str, trade_date, market_hint: str = None, max_back: int = 10):
    stock_no = _strip_suffix(stock_id)
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "zh-TW,zh;q=0.9"}
    is_two = isinstance(market_hint, str) and market_hint.upper().endswith(".TWO")
    last_error = None

    # [FIX] Try FinMind first — works for both TWSE and TPEx, not blocked by rate limits
    finmind_result = _finmind_margin_fetch(stock_no, trade_date, max_back)
    if finmind_result is not None and finmind_result.get("margin_balance") is not None:
        # Fix the id suffix for TPEx stocks
        if is_two:
            finmind_result["id"] = f"{stock_no}.TWO"
        return finmind_result

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

        # [FIX] TPEx margin — now uses cached helper
        roc_year = d.year - 1911
        roc_date = f"{roc_year:03d}/{d.month:02d}/{d.day:02d}"
        df, err = _tpex_margin_csv_cached(roc_date, headers)
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
        if err:
            last_error = err

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
    rr_ok = bool(rr_val is not None and rr_val >= ENTRY_MIN_RR)  # kept for display, not gating
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

    # [SOFT WARNINGS] These conditions used to be hard vetoes that blocked the trigger.
    # They're now surfaced as warnings for AI to weigh against other context.
    # Rationale: any single rule viewed in isolation can misfire (e.g. strong stock
    # pulling back to MA20 on volume looks like distribution; a breakout running
    # above BB upper for days looks overextended but may continue). Let AI integrate
    # these warnings with fundamentals, chip signals, retail sentiment, and news.
    entry_warnings = []
    if price_down_vol_up and not supertrend_flip_bull:
        # Distribution bar candidate — down close + volume surge, no reversal signal
        entry_warnings.append("price_down_vol_up_distribution")
    if above_bb_upper and volatility_regime in ("high_volatility", "expansion"):
        # Overextension candidate — closed above BB upper in high-vol regime
        entry_warnings.append("above_bb_upper_in_high_vol")
    if not close_gt_open and not supertrend_flip_bull and not kd_golden_cross:
        # Unconfirmed red candle — no reversal signal to rescue it
        entry_warnings.append("red_candle_no_reversal_confirm")

    has_warnings = len(entry_warnings) > 0

    # [P1-B] CONFIRMATION REQUIREMENT — single-event triggers need extra support
    # This one REMAINS a hard gate because it's about signal quality, not risk judgement.
    # bb_squeeze_fire alone is a "candidate", not a trigger. It needs at least one of:
    #   - volume confirmation (vol_surge)
    #   - green candle
    needs_confirmation = (len(today_events) == 1 and today_events[0] == "bb_squeeze_fire")
    has_confirmation = vol_surge or close_gt_open
    single_event_blocked = needs_confirmation and not has_confirmation

    # Final trigger decision
    # [CHANGED] Neither RR nor soft warnings gate the trigger anymore. Only signal
    # quality (has_event + bullish context + confirmation) determines pass/fail.
    # Warnings are shown in a separate field for AI to consider holistically.
    trigger_pass = bool(
        has_event and not_bearish and is_bullish
        and not single_event_blocked
    )

    feat["flag_entry_trigger"] = trigger_pass
    feat["entry_trigger_events"] = today_events if trigger_pass else []

    # [NEW FIELD NAME] entry_warnings replaces entry_trigger_veto
    feat["entry_warnings"] = entry_warnings if entry_warnings else []
    feat["entry_warning_count"] = len(entry_warnings)

    # Backward-compat alias — keep entry_trigger_veto populated so any prompt that
    # still references it won't break. Will be removed in a future cleanup.
    feat["entry_trigger_veto"] = entry_warnings if entry_warnings else []

    if trigger_pass:
        rr_display = f"R:R={rr_val}" if rr_val is not None else "R:R=n/a"
        warn_suffix = f" | warnings: {', '.join(entry_warnings)}" if entry_warnings else ""
        feat["entry_trigger_reason"] = f"Events: {', '.join(today_events)} | {rr_display}{warn_suffix}"
    elif single_event_blocked:
        feat["entry_trigger_reason"] = f"needs_confirmation: {today_events[0]} alone insufficient"
    elif has_event and not not_bearish:
        feat["entry_trigger_reason"] = f"blocked: trend_state={trend_state}"
    elif has_event and not is_bullish:
        feat["entry_trigger_reason"] = "blocked: structure_not_bullish"
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


def _download_yf(ticker: str, period: str, interval: str, max_retries: int = 3):
    """Helper for threaded yf.download — serialised via _yf_lock.

    [FIX] Added retry with exponential backoff to handle Yahoo rate-limiting.
    Yahoo intermittently returns empty JSON (HTTP 200 but no data) or HTTP 429
    when too many requests are made in a short window. This causes yfinance to
    return an empty DataFrame, which downstream code treats as "not found".
    Retrying with increasing delays (2s → 4s → 8s) resolves most transient
    failures without changing any downstream logic.
    """
    for attempt in range(max_retries):
        with _yf_lock:
            try:
                # [FIX] auto_adjust=True and ffill() to avoid missing data/unadjusted price bugs
                df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
                df = clean_yf_columns(_ensure_naive_index(df))
                if not df.empty:
                    df = df.ffill().dropna(subset=["Close"])
                    return df
            except Exception as e:
                logger.warning("yf.download(%s) attempt %d/%d exception: %s",
                               ticker, attempt + 1, max_retries, e)
        # Empty df or exception — wait before retrying (backoff: 2s, 4s, 8s)
        if attempt < max_retries - 1:
            wait = 2 ** (attempt + 1)
            logger.info("yf.download(%s) returned empty, retrying in %ds (%d/%d)",
                        ticker, wait, attempt + 1, max_retries)
            time.sleep(wait)
    logger.warning("yf.download(%s, period=%s, interval=%s) failed after %d retries",
                   ticker, period, interval, max_retries)
    return pd.DataFrame()


def build_ai_features(stock_id: str, as_of_date=None, mode: str = "ai",
                      regulatory_data: dict = None) -> Dict[str, Any]:
    stock_id = stock_id.strip().upper()
    yf_ticker = _resolve_yf_ticker(stock_id)
    stock_no = _strip_suffix(stock_id)

    # -- download daily (sequential — yf.download is NOT thread-safe) --
    df_daily = _download_yf(yf_ticker, "2y", "1d")

    if df_daily.empty and yf_ticker.endswith(".TW"):
        yf_ticker = yf_ticker.replace(".TW", ".TWO")
        df_daily = _download_yf(yf_ticker, "2y", "1d")

    if df_daily.empty:
        logger.warning("%s (%s): all yf.download attempts returned empty — "
                       "likely Yahoo rate-limit, not a bad ticker", stock_id, yf_ticker)
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
    # [FIX] round to 2 decimals to fix yfinance float32 precision artifacts
    # e.g. 69.19999694824219 → 69.2
    c_now = round(float(latest["Close"]), 2)
    o_now = round(float(latest["Open"]), 2)
    h_now = round(float(latest["High"]), 2)
    l_now = round(float(latest["Low"]), 2)
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
                    as_of = margin.get("as_of_date") or margin.get("date")  # FinMind returns 'date', TWSE returns 'as_of_date'
                    # [P0-A FIX] Only mark available if at least one core field is non-null
                    has_real_data = any(v is not None for v in [m_bal, s_bal, m_chg, s_chg, m_usage])
                    return {
                        "margin_balance": m_bal,
                        "margin_change": m_chg,
                        "margin_usage_rate": m_usage,
                        "short_balance": s_bal,
                        "short_change": s_chg,
                        "short_margin_ratio": round(s_bal / m_bal * 100, 2) if m_bal and s_bal and m_bal > 0 else None,
                        "margin_date": as_of,  # [V2] surface data date for staleness checks
                        "margin_data_available": has_real_data,
                    }
                return {"margin_data_available": False, "margin_date": None}
            except Exception:
                return {"margin_data_available": False, "margin_date": None}

        def _fetch_tdcc():
            try:
                tdcc = get_tdcc_distribution(stock_no, weeks_back=2)
                return compute_tdcc_features(tdcc)
            except Exception:
                return {"tdcc_available": False}

        # [NEW] FinMind extended fetches — revenue, EPS, major holders, lending
        def _fetch_revenue():
            try:
                r = _finmind_revenue_fetch(stock_no, latest_overall_ts.date())
                return r if r else {"revenue_data_available": False}
            except Exception:
                return {"revenue_data_available": False}

        def _fetch_eps():
            try:
                r = _finmind_eps_fetch(stock_no, latest_overall_ts.date())
                return r if r else {"eps_data_available": False}
            except Exception:
                return {"eps_data_available": False}

        def _fetch_holders():
            try:
                r = _finmind_holders_fetch(stock_no, latest_overall_ts.date())
                return r if r else {"foreign_holding_data_available": False}
            except Exception:
                return {"foreign_holding_data_available": False}

        def _fetch_lending():
            try:
                r = _finmind_lending_fetch(stock_no, latest_overall_ts.date())
                return r if r else {"lending_data_available": False}
            except Exception:
                return {"lending_data_available": False}

        def _fetch_daytrade():
            try:
                # Build {date_str: total_volume} map from df for ratio computation.
                # df is in scope from build_ai_features' enclosing frame.
                vol_map = {}
                try:
                    for ts, v in df["Volume"].items():
                        ds = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
                        vol_map[ds] = float(v) if v is not None else 0
                except Exception:
                    vol_map = {}
                if not vol_map:
                    return {"day_trade_data_available": False}
                r = _finmind_daytrade_fetch(stock_no, latest_overall_ts.date(), total_volume_map=vol_map)
                return r if r else {"day_trade_data_available": False}
            except Exception:
                return {"day_trade_data_available": False}

        # [OPT] Run all external data fetches in parallel threads
        with ThreadPoolExecutor(max_workers=8) as pool:
            fut_inst = pool.submit(_fetch_institutional)
            fut_margin = pool.submit(_fetch_margin)
            fut_tdcc = pool.submit(_fetch_tdcc)
            fut_revenue = pool.submit(_fetch_revenue)
            fut_eps = pool.submit(_fetch_eps)
            fut_holders = pool.submit(_fetch_holders)
            fut_lending = pool.submit(_fetch_lending)
            fut_daytrade = pool.submit(_fetch_daytrade)

        feat.update(fut_inst.result())
        feat.update(fut_margin.result())
        feat.update(fut_tdcc.result())
        feat.update(fut_revenue.result())
        feat.update(fut_eps.result())
        feat.update(fut_holders.result())
        feat.update(fut_lending.result())
        feat.update(fut_daytrade.result())

    else:
        feat["inst_data_available"] = False
        feat["margin_data_available"] = False
        feat["tdcc_available"] = False
        feat["revenue_data_available"] = False
        feat["eps_data_available"] = False
        feat["foreign_holding_data_available"] = False
        feat["lending_data_available"] = False
        feat["day_trade_data_available"] = False

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

    # Data quality metadata (Item #19)
    # [P0-A] Data quality metadata — now includes external source reliability
    # [V2] Each source now carries its actual data date (no lag_days/stale flags —
    # those are inferred by AI from comparing to analysis_date, since each source
    # has different "normal" lag (revenue ~monthly, EPS ~quarterly, chip daily).
    analysis_date_str = trade_date.isoformat() if trade_date else None
    feat["analysis_date"] = analysis_date_str

    feat["data_quality"] = {
        "price_data_days": len(df),
        "volume_zero_pct": round(float((df["Volume"] == 0).sum()) / len(df) * 100, 1) if len(df) > 0 else None,
        "stale_warning": bool((datetime.now().date() - trade_date).days > 3),
        "analysis_date": analysis_date_str,
    }
    if mode == "ai":
        feat["data_quality"]["inst_data_coverage"] = None
        # [P0-A V2] External source metadata — only date, available, source.
        # AI reads analysis_date and compares with each source's date to determine
        # staleness based on per-source context (monthly vs quarterly vs daily).
        feat["data_quality"]["external_sources"] = {
            "margin": {
                "available": feat.get("margin_data_available", False),
                "date": feat.get("margin_date"),
                "source": "FinMind.TaiwanStockMarginPurchaseShortSale",
            },
            "institutional": {
                "available": feat.get("inst_data_available", False),
                "date": feat.get("inst_date"),
                "source": "TWSE/TPEx.T86",
            },
            "revenue": {
                "available": feat.get("revenue_data_available", False),
                "date": feat.get("revenue_month_label"),
                "source": "FinMind.TaiwanStockMonthRevenue",
            },
            "eps": {
                "available": feat.get("eps_data_available", False),
                "date": feat.get("eps_quarter_label"),
                "source": "FinMind.TaiwanStockFinancialStatements",
            },
            "foreign_holding": {
                "available": feat.get("foreign_holding_data_available", False),
                "date": feat.get("foreign_holding_date"),
                "source": "FinMind.TaiwanStockShareholding",
            },
            "lending": {
                "available": feat.get("lending_data_available", False),
                "date": feat.get("lending_date"),
                "source": "FinMind.TaiwanStockSecuritiesLending",
            },
            "day_trade": {
                "available": feat.get("day_trade_data_available", False),
                "date": feat.get("day_trade_date"),
                "source": "FinMind.TaiwanStockDayTrading",
            },
        }

    # ---- Regulatory list flags (Attention / Disposition) ----
    _reg = regulatory_data or {}
    _attn_set = _reg.get("attention_stocks", set())
    _disp_dict = _reg.get("disposition_stocks", {})
    _is_attention = stock_no in _attn_set
    _is_disposition = stock_no in _disp_dict

    feat["regulatory_attention"] = _is_attention
    feat["regulatory_disposition"] = _is_disposition
    if _is_attention and _is_disposition:
        feat["regulatory_status"] = "BOTH"
    elif _is_attention:
        feat["regulatory_status"] = "ATTENTION"
    elif _is_disposition:
        feat["regulatory_status"] = "DISPOSITION"
    else:
        feat["regulatory_status"] = None

    if _is_disposition and mode == "ai":
        disp_info = _disp_dict[stock_no]
        feat["disposition_detail"] = {
            "measure": disp_info.get("measure", ""),
            "content": disp_info.get("content", ""),
            "remarks": disp_info.get("remarks", ""),
            "period": disp_info.get("period", ""),
            "source": disp_info.get("source", ""),
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

    # Regulatory status badge
    reg_status = feat.get("regulatory_status")
    if reg_status == "BOTH":
        lines.append("  *** [ATTENTION] + [DISPOSITION] ***")
    elif reg_status == "ATTENTION":
        lines.append("  *** [ATTENTION] ***")
    elif reg_status == "DISPOSITION":
        lines.append("  *** [DISPOSITION] ***")

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


def _analyze_one_stock_for_sector(stock_id, as_of_date, mode, regulatory_data=None):
    """Helper for parallel sector analysis."""
    try:
        feat = build_ai_features(stock_id, as_of_date=as_of_date, mode=mode,
                                 regulatory_data=regulatory_data)
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
            "Reg": feat.get("regulatory_status") or "",
            "Score": score,
        }
    except Exception:
        return {"Symbol": stock_id, "Trend": "Exception", "Reg": "", "Score": -1}


def analyze_sector_performance(sector_name: str, as_of_date=None, custom_tickers=None,
                               mode: str = "human", regulatory_data: dict = None) -> str:
    target_list = custom_tickers if custom_tickers else SECTOR_DICT.get(sector_name, [])

    if not target_list:
        return f"No stocks in sector {sector_name}."

    # [FIX] Reduced max_workers from 5 to 2 to lower Yahoo rate-limit risk.
    # yf.download is serialised via _yf_lock anyway, so parallelism only
    # benefits the external data fetches (institutional, margin, etc.).
    results = []
    with ThreadPoolExecutor(max_workers=min(len(target_list), 2)) as pool:
        futures = {
            pool.submit(_analyze_one_stock_for_sector, sid, as_of_date, mode,
                        regulatory_data): sid
            for sid in target_list
        }
        for fut in as_completed(futures):
            results.append(fut.result())

    results.sort(key=lambda x: x.get("Score", 0), reverse=True)

    lines = []
    lines.append(f"Sector Scan：{sector_name}")
    d_str = str(as_of_date) if as_of_date else "Today"
    lines.append(f"Date：{d_str}")
    lines.append("-" * 35)
    lines.append(
        "{:<10} {:<8} {:<12} {:<10} {:<6} {:<8} {:<12} {:<4}".format(
            "Symbol", "Price", "Trend", "MA20Dev", "VolR", "KD", "Regulatory", "Score"))
    lines.append("-" * 35)

    for r in results:
        lines.append(
            "{:<10} {:<8} {:<12} {:<10} {:<6} {:<8} {:<12} {:<4}".format(
                str(r.get("Symbol", "")),
                str(r.get("Close", "")),
                str(r.get("Trend", "")),
                str(r.get("MA20_Dev", "")),
                str(r.get("Vol_R", "")),
                str(r.get("KD", "")),
                str(r.get("Reg", "")),
                str(r.get("Score", "")),
            )
        )

    lines.append("-" * 35)
    lines.append("（Score：uptrend+2，price_up_vol_up+1，KD_golden+1，inst_consensus+2）")

    return "\n".join(lines)


# ===================================================================
# 17. ENTRY POINT
# mode="human" => fast, text only
# mode="ai"    => full JSON
# ===================================================================


def analyze_stock_technical(stock_id: str, as_of_date=None, mode: str = "human",
                            regulatory_data: dict = None) -> dict:
    """
    Main entry point.
    mode="human" => returns {"human_report": str}   (fast, no network calls)
    mode="ai"    => returns {"ai_report": dict}     (full JSON with all data)

    regulatory_data (optional): {
        "attention_stocks": set of stock codes on attention list,
        "disposition_stocks": dict of stock_code -> {
            "measure": str,
            "content": str,
            "remarks": str,
            "period": str,
            "source": str,   # "TWSE" or "TPEx"
        }
    }
    """
    feat = build_ai_features(stock_id, as_of_date, mode=mode,
                             regulatory_data=regulatory_data)

    if mode == "ai":
        return {"ai_report": feat}
    else:
        text = format_text_report(feat)
        return {"human_report": text}


# ===================================================================
# 18. MARKET FEATURES (大盤狀態指標)
# 提供 AI 判讀「現在是什麼盤」所需的客觀數據
# ===================================================================


def compute_market_features(as_of_date=None) -> Dict[str, Any]:
    """
    抓 ^TWII（加權指數）+ ^VIX 計算大盤狀態指標。

    產出可直接給 AI 判市況的客觀數據，包括：
    - 趨勢、動能、波動、量能
    - VIX 等級
    - 大盤 RSI、ADX、MA、SuperTrend、Bollinger

    Returns: dict（可直接 json.dumps 成獨立檔案）
    """
    result = {
        "snapshot_date": str(as_of_date) if as_of_date else datetime.now().strftime("%Y-%m-%d"),
        "fetched_at": datetime.now().isoformat(),
        "data_quality": {"twii_available": False, "vix_available": False, "warnings": []},
    }

    # --------- 1. 下載 TWII 日線 + 週線 ----------
    twii = _download_yf("^TWII", "2y", "1d")
    if twii.empty or len(twii) < 60:
        result["data_quality"]["warnings"].append("TWII 日線資料不足")
        return result

    twii_w = _download_yf("^TWII", "2y", "1wk")

    result["data_quality"]["twii_available"] = True

    # --------- 2. 過濾到 as_of_date 並對齊 ----------
    if as_of_date is not None:
        target_ts = _nearest_trading_ts(twii, as_of_date)
        if target_ts is not None:
            twii = twii.loc[:target_ts]
        if not twii_w.empty:
            target_ts_w = _nearest_trading_ts(twii_w, as_of_date)
            if target_ts_w is not None:
                twii_w = twii_w.loc[:target_ts_w]

    if len(twii) < 60:
        result["data_quality"]["warnings"].append("TWII 過濾後資料不足")
        return result

    close = twii["Close"]
    high = twii["High"]
    low = twii["Low"]
    volume = twii["Volume"]

    # --------- 3. 價格與漲跌幅 ----------
    c_now = float(close.iloc[-1])
    c_prev = float(close.iloc[-2]) if len(close) > 1 else c_now
    daily_change = (c_now / c_prev - 1) * 100 if c_prev > 0 else 0.0

    weekly_change = None
    if len(close) > 6:
        c_5d = float(close.iloc[-6])
        if c_5d > 0:
            weekly_change = (c_now / c_5d - 1) * 100

    result.update({
        "twii_close": round(c_now, 2),
        "twii_open": float(twii["Open"].iloc[-1]),
        "twii_high": float(high.iloc[-1]),
        "twii_low": float(low.iloc[-1]),
        "twii_change_pct": round(daily_change, 2),
        "twii_weekly_change_pct": round(weekly_change, 2) if weekly_change is not None else None,
        "twii_volume": int(volume.iloc[-1]),
    })

    # --------- 4. 移動平均 + 趨勢狀態 ----------
    ma5 = float(close.rolling(5).mean().iloc[-1])
    ma20 = float(close.rolling(20).mean().iloc[-1])
    ma60 = float(close.rolling(60).mean().iloc[-1])

    # ADX
    adx_dict = calculate_adx(twii, 14)
    adx_val = float(adx_dict["adx"].iloc[-1]) if not adx_dict["adx"].empty else 0.0
    plus_di = float(adx_dict["plus_di"].iloc[-1]) if not adx_dict["plus_di"].empty else 0.0
    minus_di = float(adx_dict["minus_di"].iloc[-1]) if not adx_dict["minus_di"].empty else 0.0

    # 用 stock.py 的 classify_trend 一致邏輯
    trend_state, trend_strength = classify_trend(c_now, ma5, ma20, ma60, adx_val)

    result.update({
        "twii_ma5": round(ma5, 2),
        "twii_ma20": round(ma20, 2),
        "twii_ma60": round(ma60, 2),
        "market_trend_state": trend_state,
        "market_trend_strength": trend_strength,
        "market_above_ma20": bool(c_now > ma20),
        "market_above_ma60": bool(c_now > ma60),
        "market_adx": round(adx_val, 2),
        "market_plus_di": round(plus_di, 2),
        "market_minus_di": round(minus_di, 2),
        "market_di_bullish": bool(plus_di > minus_di),
    })

    # --------- 5. 週線趨勢 ----------
    if not twii_w.empty and len(twii_w) >= 20:
        c_w_now = float(twii_w["Close"].iloc[-1])
        wma20 = float(twii_w["Close"].rolling(20).mean().iloc[-1])
        weekly_rsi = float(calculate_rsi(twii_w["Close"], 14).iloc[-1])

        result.update({
            "market_weekly_above_ma20": bool(c_w_now > wma20),
            "market_weekly_trend_state": "uptrend" if c_w_now > wma20 else "downtrend",
            "market_weekly_rsi14": round(weekly_rsi, 2),
        })
    else:
        result["data_quality"]["warnings"].append("TWII 週線資料不足，無 weekly RSI/trend")
        result.update({
            "market_weekly_above_ma20": None,
            "market_weekly_trend_state": None,
            "market_weekly_rsi14": None,
        })

    # --------- 6. RSI / SuperTrend / Bollinger ----------
    rsi_val = float(calculate_rsi(close, 14).iloc[-1])

    st_dict = calculate_supertrend(twii, period=10, multiplier=3.0)
    st_bullish = bool(st_dict["supertrend_bullish"].iloc[-1]) if "supertrend_bullish" in st_dict else False

    bb_dict = calculate_bbands(twii, 20, 2)
    bb_upper = float(bb_dict["bb_upper"].iloc[-1])
    bb_lower = float(bb_dict["bb_lower"].iloc[-1])
    bb_pos = ((c_now - bb_lower) / (bb_upper - bb_lower)) * 100 if bb_upper > bb_lower else 50.0

    result.update({
        "market_rsi14": round(rsi_val, 2),
        "market_supertrend_bullish": st_bullish,
        "market_bb_upper": round(bb_upper, 2),
        "market_bb_lower": round(bb_lower, 2),
        "market_bb_position_pct": round(bb_pos, 2),
    })

    # --------- 7. ATR / 波動 Regime ----------
    atr_series = calculate_atr(twii, 14)
    atr_now = float(atr_series.iloc[-1])
    atr_pct = (atr_now / c_now) * 100 if c_now > 0 else 0.0
    atr_20d_avg = float(atr_series.rolling(20).mean().iloc[-1])

    if atr_now > atr_20d_avg * 1.3:
        vol_regime = "expansion"
    elif atr_now < atr_20d_avg * 0.7:
        vol_regime = "contraction"
    else:
        vol_regime = "normal"

    result.update({
        "market_atr14": round(atr_now, 2),
        "market_atr14_pct": round(atr_pct, 2),
        "market_volatility_regime": vol_regime,
    })

    # --------- 8. 量能 ----------
    vol_5d_avg = float(volume.rolling(5).mean().iloc[-1])
    vol_ratio_5d = float(volume.iloc[-1]) / vol_5d_avg if vol_5d_avg > 0 else 1.0

    result.update({
        "market_vol_ratio_5d": round(vol_ratio_5d, 2),
    })

    # --------- 9. VIX ----------
    try:
        vix = _download_yf("^VIX", "3mo", "1d")
        if not vix.empty and len(vix) >= 2:
            vix_now = float(vix["Close"].iloc[-1])
            vix_prev = float(vix["Close"].iloc[-2])
            vix_change = vix_now - vix_prev

            if vix_now < 18:
                vix_level = "low"          # 低恐慌
            elif vix_now < 25:
                vix_level = "normal"       # 正常
            elif vix_now < 30:
                vix_level = "elevated"     # 偏高警戒
            else:
                vix_level = "panic"        # 恐慌

            result.update({
                "vix_latest": round(vix_now, 2),
                "vix_change": round(vix_change, 2),
                "vix_level": vix_level,
            })
            result["data_quality"]["vix_available"] = True
        else:
            result["data_quality"]["warnings"].append("VIX 資料不足")
            result.update({"vix_latest": None, "vix_change": None, "vix_level": None})
    except Exception as e:
        result["data_quality"]["warnings"].append(f"VIX 下載失敗: {str(e)[:50]}")
        result.update({"vix_latest": None, "vix_change": None, "vix_level": None})

    # --------- 10. 推算市場狀態（給 AI 參考，非必需） ----------
    rsi_d = result.get("market_rsi14", 50)
    rsi_w = result.get("market_weekly_rsi14") or 50
    vix_lvl = result.get("vix_level", "normal")
    daily_chg = result["twii_change_pct"]

    # 簡單分類（AI 仍可基於更多訊號自行判斷）
    suggested_regime = None
    if vix_lvl == "panic" or daily_chg < -2.0:
        suggested_regime = "crisis"           # 黑天鵝
    elif rsi_d > 75 and daily_chg > 1.0:
        suggested_regime = "late_trend_or_blowoff"   # 末段過熱或權值噴發
    elif rsi_d > 75:
        suggested_regime = "late_trend"       # 末段過熱
    elif 50 <= rsi_d <= 75 and adx_val > 25:
        suggested_regime = "trend"            # 趨勢盤
    elif 40 <= rsi_d <= 60 and adx_val < 20:
        suggested_regime = "range"            # 震盪盤
    else:
        suggested_regime = "transition"       # 轉折期

    result["suggested_regime"] = suggested_regime

    # --------- 11. 移除 NaN 與不可序列化值 ----------
    result = _sanitize_numpy(result)

    return result


def save_market_features(features: Dict[str, Any], output_path: str = None) -> str:
    """把大盤 features 存成 JSON 檔案，預設檔名 market_features_YYYY-MM-DD.json"""
    if output_path is None:
        date = features.get("snapshot_date", datetime.now().strftime("%Y-%m-%d"))
        output_path = f"market_features_{date}.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(features, f, ensure_ascii=False, indent=2, default=str)

    print(f"✓ Market features saved: {output_path}")
    return output_path


# ===================================================================
# 19. ENTRY POINT
# ===================================================================

if __name__ == "__main__":
    import sys

    sid = sys.argv[1] if len(sys.argv) > 1 else "2330"
    m = sys.argv[2] if len(sys.argv) > 2 else "human"

    # market 模式 - 只產出大盤 features
    if sid == "market":
        as_of = sys.argv[2] if len(sys.argv) > 2 else None
        features = compute_market_features(as_of_date=as_of)
        save_market_features(features)
        print(json.dumps(features, ensure_ascii=False, indent=2, default=str))
    else:
        result = analyze_stock_technical(sid, mode=m)
        if "human_report" in result:
            print(result["human_report"])
        if "ai_report" in result:
            print(json.dumps(result["ai_report"], ensure_ascii=False, default=str))
