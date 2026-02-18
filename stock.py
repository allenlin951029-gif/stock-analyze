"""
stock_v2.py (整合版)
特色：
1. 完整保留 AI Features (技術/籌碼/型態)
2. 完整保留爬蟲邏輯 (不簡化)
3. 整合族群分析 (Sector Analysis)
4. 相容 App 呼叫介面
"""

import io
import re
import math
import json
import warnings
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

# ═══════════════════════════════════════════════════════════
# 0.  UTILS
# ═══════════════════════════════════════════════════════════

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
        if s in ("", "--", "—", "NaN", "nan", "None"):
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
    return (stock_id.strip().upper()
            .replace(".TW", "").replace(".TWO", "")
            .replace(".KS", "").replace(".T", "").replace(".HK", ""))

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
        parts = str(date_str).split('/')
        if len(parts) == 3:
            year = int(parts[0]) + 1911
            return f"{year}-{parts[1]}-{parts[2]}"
        return str(date_str)
    except Exception:
        return str(date_str)

# ── 統計工具 ──

def percentile_rank(series: pd.Series, window: int = 252) -> Optional[float]:
    """最新值在過去 window 天的百分位 (0.0 ~ 1.0)"""
    s = series.dropna()
    if len(s) < 10:
        return None
    current = s.iloc[-1]
    hist = s.iloc[-window:]
    return round((hist < current).sum() / len(hist), 4)

def zscore_last(series: pd.Series, window: int = 252) -> Optional[float]:
    """最新值的 z-score"""
    s = series.dropna()
    if len(s) < 20:
        return None
    hist = s.iloc[-window:]
    mu, sigma = hist.mean(), hist.std()
    if sigma == 0 or pd.isna(sigma):
        return 0.0
    return round((s.iloc[-1] - mu) / sigma, 4)

def slope_n(series: pd.Series, n: int = 5) -> Optional[float]:
    """最近 n 筆的線性回歸斜率 (per bar)"""
    s = series.dropna().iloc[-n:]
    if len(s) < n:
        return None
    x = np.arange(len(s), dtype=float)
    y = s.values.astype(float)
    if np.all(np.isnan(y)):
        return None
    m, _ = np.polyfit(x, y, 1)
    return round(float(m), 6)

def max_drawdown(series: pd.Series, window: int = 20) -> Optional[float]:
    """近 window 日最大回撤 (負數, 例如 -0.08 = -8%)"""
    s = series.dropna().iloc[-window:]
    if len(s) < 2:
        return None
    peak = s.expanding().max()
    dd = (s - peak) / peak
    return round(float(dd.min()), 4)

# ═══════════════════════════════════════════════════════════
# 1. CANDLE (K棒型態)
# ═══════════════════════════════════════════════════════════

def classify_candle(o, h, l, c) -> str:
    """回傳 K 棒分類字串 (供 flag 用)"""
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
    """偵測動能延續型態：連續紅K、跳空突破等"""
    if df is None or len(df) < n:
        return {"consecutive_bull": 0, "consecutive_bear": 0,
                "gap_up_breakout": False, "gap_down_breakdown": False}
    
    tail = df.tail(n)
    types = [classify_candle(r["Open"], r["High"], r["Low"], r["Close"]) for _, r in tail.iterrows()]

    # 連續紅/黑K
    cons_bull = 0
    for t in reversed(types):
        if t in ("long_bull", "small_bull"): cons_bull += 1
        else: break

    cons_bear = 0
    for t in reversed(types):
        if t in ("long_bear", "small_bear"): cons_bear += 1
        else: break

    # 跳空突破 (最後一根 low > 前一根 high)
    gap_up = False
    gap_down = False
    if len(df) >= 2:
        last, prev = df.iloc[-1], df.iloc[-2]
        if float(last["Low"]) > float(prev["High"]): gap_up = True
        if float(last["High"]) < float(prev["Low"]): gap_down = True

    return {
        "consecutive_bull_bars": int(cons_bull),
        "consecutive_bear_bars": int(cons_bear),
        "gap_up_breakout": bool(gap_up),
        "gap_down_breakdown": bool(gap_down),
    }

# ═══════════════════════════════════════════════════════════
# 2. CORE INDICATORS
# ═══════════════════════════════════════════════════════════

def calculate_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
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
    df2["+DM"] = np.where((df2["UpMove"] > df2["DownMove"]) & (df2["UpMove"] > 0), df2["UpMove"], 0.0)
    df2["-DM"] = np.where((df2["DownMove"] > df2["UpMove"]) & (df2["DownMove"] > 0), df2["DownMove"], 0.0)
    alpha = 1 / period
    df2["TR_s"] = df2["TR"].ewm(alpha=alpha, adjust=False).mean()
    df2["+DM_s"] = df2["+DM"].ewm(alpha=alpha, adjust=False).mean()
    df2["-DM_s"] = df2["-DM"].ewm(alpha=alpha, adjust=False).mean()
    df2["+DI"] = 100 * df2["+DM_s"] / df2["TR_s"].replace(0, np.nan)
    df2["-DI"] = 100 * df2["-DM_s"] / df2["TR_s"].replace(0, np.nan)
    df2["DX"] = 100 * abs(df2["+DI"] - df2["-DI"]) / (df2["+DI"] + df2["-DI"]).replace(0, np.nan)
    df2["ADX"] = df2["DX"].ewm(alpha=alpha, adjust=False).mean()
    return df2["ADX"], df2["+DI"], df2["-DI"]

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h, l, c = df["High"], df["Low"], df["Close"]
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    return tr.rolling(period).mean()

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
    pos = pd.Series(np.where(tp > tp.shift(1), rmf, 0.0), index=df.index).rolling(period).sum()
    neg = pd.Series(np.where(tp < tp.shift(1), rmf, 0.0), index=df.index).rolling(period).sum()
    return 100 - (100 / (1 + pos / neg.abs().replace(0, np.nan)))

def calculate_obv(df: pd.DataFrame) -> pd.Series:
    return (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0):
    df2 = df.copy()
    tr = pd.concat([
        df2["High"] - df2["Low"],
        abs(df2["High"] - df2["Close"].shift(1)),
        abs(df2["Low"] - df2["Close"].shift(1))
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    hl2 = (df2["High"] + df2["Low"]) / 2
    upper_band = (hl2 + multiplier * atr).copy()
    lower_band = (hl2 - multiplier * atr).copy()
    supertrend = pd.Series(np.nan, index=df2.index)
    direction = pd.Series(0, index=df2.index)

    for i in range(1, len(df2)):
        ub_p = upper_band.iloc[i - 1]
        lb_p = lower_band.iloc[i - 1]
        c_p = df2["Close"].iloc[i - 1]
        upper_band.iloc[i] = min(upper_band.iloc[i], ub_p) if c_p <= ub_p else upper_band.iloc[i]
        lower_band.iloc[i] = max(lower_band.iloc[i], lb_p) if c_p >= lb_p else lower_band.iloc[i]
        c_now = df2["Close"].iloc[i]
        st_p = supertrend.iloc[i - 1]
        if pd.isna(st_p) or st_p >= ub_p:
            if c_now > upper_band.iloc[i]:
                supertrend.iloc[i] = lower_band.iloc[i]; direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]; direction.iloc[i] = -1
        else:
            if c_now < lower_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]; direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]; direction.iloc[i] = 1
    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1
    return supertrend, direction

def calculate_avwap(df: pd.DataFrame, anchor_date) -> Optional[pd.Series]:
    mask = df.index >= anchor_date
    if not mask.any(): return None
    sub = df.loc[mask].copy()
    tp = (sub["High"] + sub["Low"] + sub["Close"]) / 3
    return (tp * sub["Volume"]).cumsum() / sub["Volume"].cumsum().replace(0, np.nan)

# ═══════════════════════════════════════════════════════════
# 3. VOLUME PROFILE & SR
# ═══════════════════════════════════════════════════════════

def calculate_volume_profile(df: pd.DataFrame, lookback: int = 60, n_bins: int = 50) -> Dict[str, Any]:
    sub = df.tail(lookback).copy()
    if sub.empty: return {"poc_price": None, "poc_volume_k": None, "price_vs_poc_pct": None}
    lo, hi = float(sub["Low"].min()), float(sub["High"].max())
    if hi <= lo: return {"poc_price": None, "poc_volume_k": None, "price_vs_poc_pct": None}
    
    bins = np.linspace(lo, hi, n_bins + 1)
    vol_by_bin = np.zeros(n_bins)
    for _, row in sub.iterrows():
        r_lo, r_hi, vol = float(row["Low"]), float(row["High"]), float(row["Volume"])
        if r_hi <= r_lo or vol <= 0: continue
        for b in range(n_bins):
            overlap = max(0, min(r_hi, bins[b + 1]) - max(r_lo, bins[b]))
            vol_by_bin[b] += vol * overlap / (r_hi - r_lo)
            
    poc_idx = int(np.argmax(vol_by_bin))
    poc_price = round((bins[poc_idx] + bins[poc_idx + 1]) / 2, 2)
    poc_volume = int(vol_by_bin[poc_idx] / 1000)
    c_now = float(sub["Close"].iloc[-1])
    price_vs_poc = round((c_now - poc_price) / poc_price * 100, 4) if poc_price > 0 else None
    return {"poc_price": poc_price, "poc_volume_k": poc_volume, "price_vs_poc_pct": price_vs_poc}

def detect_short_term_sr(df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
    sub = df.tail(lookback)
    if sub.empty: return {"st_support": None, "st_resistance": None}
    return {"st_support": round(float(sub["Low"].min()), 2), "st_resistance": round(float(sub["High"].max()), 2)}

# ═══════════════════════════════════════════════════════════
# 4. DIVERGENCE & QUALITY
# ═══════════════════════════════════════════════════════════

def detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 60) -> Dict[str, bool]:
    result = {"bearish_divergence": False, "bullish_divergence": False}
    p = price.dropna().iloc[-lookback:]
    ind = indicator.dropna().iloc[-lookback:]
    if len(p) < 20 or len(ind) < 20: return result
    mid = len(p) // 2
    if p.iloc[mid:].max() > p.iloc[:mid].max() and ind.iloc[mid:].max() < ind.iloc[:mid].max():
        result["bearish_divergence"] = True
    if p.iloc[mid:].min() < p.iloc[:mid].min() and ind.iloc[mid:].min() > ind.iloc[:mid].min():
        result["bullish_divergence"] = True
    return result

def calculate_volume_quality(df: pd.DataFrame, period: int = 20) -> Dict[str, Any]:
    sub = df.tail(max(period, 60)).copy()
    vol_pct = percentile_rank(sub["Volume"], window=252)
    tail = sub.tail(period)
    up_mask = tail["Close"] > tail["Close"].shift(1)
    down_mask = tail["Close"] < tail["Close"].shift(1)
    up_vol = float(tail.loc[up_mask, "Volume"].sum()) if up_mask.any() else 0
    down_vol = float(tail.loc[down_mask, "Volume"].sum()) if down_mask.any() else 0
    up_down_ratio = round(up_vol / down_vol, 4) if down_vol > 0 else None
    
    obv = calculate_obv(df)
    return {
        "vol_percentile_252d": vol_pct,
        "up_down_vol_ratio_20d": up_down_ratio,
        "obv_slope_5d": slope_n(obv, 5),
        "obv_percentile_252d": percentile_rank(obv, 252)
    }

# ═══════════════════════════════════════════════════════════
# 5. PATTERNS & GAPS & FIBONACCI
# ═══════════════════════════════════════════════════════════

def calculate_fibonacci_summary(df: pd.DataFrame, lookback: int = 120) -> Dict[str, Any]:
    sub = df.tail(lookback)
    high = float(sub["High"].max())
    low = float(sub["Low"].min())
    diff = high - low
    levels = {
        "fib_0236": round(high - 0.236 * diff, 2),
        "fib_0382": round(high - 0.382 * diff, 2),
        "fib_0500": round(high - 0.500 * diff, 2),
        "fib_0618": round(high - 0.618 * diff, 2),
    }
    c_now = float(sub["Close"].iloc[-1])
    resistances = sorted([v for v in levels.values() if v > c_now])
    supports = sorted([v for v in levels.values() if v <= c_now], reverse=True)
    return {
        "fib_high": high, "fib_low": low,
        "fib_nearest_support_1": supports[0] if len(supports) > 0 else None,
        "fib_nearest_resistance_1": resistances[0] if len(resistances) > 0 else None,
    }

def detect_gaps_summary(df: pd.DataFrame, lookback: int = 30) -> List[Dict]:
    if df is None or df.empty or len(df) < 2: return []
    sub = df.tail(lookback + 1)
    gaps = []
    for i in range(1, len(sub)):
        prev, curr = sub.iloc[i - 1], sub.iloc[i]
        date_str = sub.index[i].strftime("%Y-%m-%d")
        if float(curr["Low"]) > float(prev["High"]):
            future = sub.iloc[i + 1:]
            filled = any(float(r["Low"]) <= float(prev["High"]) for _, r in future.iterrows())
            if not filled: gaps.append({"date": date_str, "type": "up", "upper": float(curr["Low"]), "lower": float(prev["High"])})
        elif float(curr["High"]) < float(prev["Low"]):
            future = sub.iloc[i + 1:]
            filled = any(float(r["High"]) >= float(prev["Low"]) for _, r in future.iterrows())
            if not filled: gaps.append({"date": date_str, "type": "down", "upper": float(prev["Low"]), "lower": float(curr["High"])})
    return gaps[-2:] if len(gaps) > 2 else gaps

# ── Pattern Detection (Full) ──
def _find_pivots(df, left=3, right=3):
    if df is None or df.empty or len(df) < left + right + 5: return [], []
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    idx = df.index
    hp, lp = [], []
    for i in range(left, len(df) - right):
        if np.isfinite(highs[i]) and highs[i] == np.nanmax(highs[i-left:i+right+1]):
            hp.append((i, idx[i], highs[i]))
        if np.isfinite(lows[i]) and lows[i] == np.nanmin(lows[i-left:i+right+1]):
            lp.append((i, idx[i], lows[i]))
    return hp, lp

def _pct_diff(a, b):
    if a == 0 or not np.isfinite(a) or not np.isfinite(b): return np.inf
    return abs(a - b) / abs(a)

def detect_double_top(df, lookback=120, pivot_lr=3, peak_tol=0.018, min_gap=8, max_gap=60, confirm_margin=0.003):
    if df is None or df.empty: return None
    sub = df.tail(lookback).copy()
    hp, _ = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(hp) < 2: return None
    p2 = hp[-1]
    p1 = next((c for c in reversed(hp[:-1]) if min_gap <= p2[0] - c[0] <= max_gap), None)
    if p1 is None or _pct_diff(float(p1[2]), float(p2[2])) > peak_tol: return None
    lo, hi = min(p1[0], p2[0]), max(p1[0], p2[0])
    neckline = float(sub.iloc[lo:hi+1]["Low"].min())
    confirmed = float(sub["Close"].iloc[-1]) < neckline * (1 - confirm_margin)
    return {"pattern": "double_top", "confirmed": confirmed}

def detect_double_bottom(df, lookback=160, pivot_lr=3, trough_tol=0.020, min_gap=8, max_gap=80, confirm_margin=0.003):
    if df is None or df.empty: return None
    sub = df.tail(lookback).copy()
    _, lp = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(lp) < 2: return None
    t2 = lp[-1]
    t1 = next((c for c in reversed(lp[:-1]) if min_gap <= t2[0] - c[0] <= max_gap), None)
    if t1 is None or _pct_diff(float(t1[2]), float(t2[2])) > trough_tol: return None
    lo, hi = min(t1[0], t2[0]), max(t1[0], t2[0])
    neckline = float(sub.iloc[lo:hi+1]["High"].max())
    confirmed = float(sub["Close"].iloc[-1]) > neckline * (1 + confirm_margin)
    return {"pattern": "double_bottom", "confirmed": confirmed}

def detect_head_and_shoulders(df, lookback=180, pivot_lr=3, shoulder_tol=0.025, confirm_margin=0.003):
    if df is None or df.empty: return None
    sub = df.tail(lookback).copy()
    hp, lp = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(hp) < 3 or len(lp) < 2: return None
    ls, head, rs = hp[-3], hp[-2], hp[-1]
    if not (float(head[2]) > float(ls[2]) and float(head[2]) > float(rs[2])): return None
    if _pct_diff(float(ls[2]), float(rs[2])) > shoulder_tol: return None
    mid_lows = [p for p in lp if ls[0] <= p[0] <= rs[0]]
    if len(mid_lows) < 2: return None
    neckline = float(np.mean([p[2] for p in mid_lows]))
    confirmed = float(sub["Close"].iloc[-1]) < neckline * (1 - confirm_margin)
    return {"pattern": "head_and_shoulders_top", "confirmed": confirmed}

def detect_inverse_head_and_shoulders(df, lookback=180, pivot_lr=3, shoulder_tol=0.025, confirm_margin=0.003):
    if df is None or df.empty: return None
    sub = df.tail(lookback).copy()
    hp, lp = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(lp) < 3 or len(hp) < 2: return None
    ls, head, rs = lp[-3], lp[-2], lp[-1]
    if not (float(head[2]) < float(ls[2]) and float(head[2]) < float(rs[2])): return None
    if _pct_diff(float(ls[2]), float(rs[2])) > shoulder_tol: return None
    mid_highs = [p for p in hp if ls[0] <= p[0] <= rs[0]]
    if len(mid_highs) < 2: return None
    neckline = float(np.mean([p[2] for p in mid_highs]))
    confirmed = float(sub["Close"].iloc[-1]) > neckline * (1 + confirm_margin)
    return {"pattern": "inv_head_and_shoulders", "confirmed": confirmed}

def detect_wedge(df, lookback=120, pivot_lr=3, breakout_margin=0.003):
    if df is None or df.empty: return None
    sub = df.tail(lookback).copy()
    hp, lp = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(hp) < 3 or len(lp) < 3: return None
    highs, lows = hp[-3:], lp[-3:]
    xh, yh = np.array([p[0] for p in highs], float), np.array([p[2] for p in highs], float)
    xl, yl = np.array([p[0] for p in lows], float), np.array([p[2] for p in lows], float)
    ah, bh = np.polyfit(xh, yh, 1)
    al, bl = np.polyfit(xl, yl, 1)
    n = len(sub) - 1
    lower_now = al * n + bl
    upper_now = ah * n + bh
    close_now = float(sub["Close"].iloc[-1])
    if ah > 0 and al > 0 and al > ah and close_now < lower_now * (1 - breakout_margin):
        return {"pattern": "rising_wedge", "confirmed": True}
    if ah < 0 and al < 0 and ah < al and close_now > upper_now * (1 + breakout_margin):
        return {"pattern": "falling_wedge", "confirmed": True}
    return None

def detect_patterns(df) -> List[Dict]:
    results = []
    for fn in [detect_double_top, detect_double_bottom,
               detect_head_and_shoulders, detect_inverse_head_and_shoulders,
               detect_wedge]:
        try:
            p = fn(df)
            if p: results.append(p)
        except Exception:
            pass
    return results

# ═══════════════════════════════════════════════════════════
# 6. EXTERNAL DATA (完整爬蟲，不刪減)
# ═══════════════════════════════════════════════════════════

def _parse_twse_t86_csv(text: str):
    text = text.replace("\r", "").replace("=", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    start = next((i for i, ln in enumerate(lines) if "證券代號" in ln and "證券名稱" in ln), None)
    if start is None: return None
    end = next((j for j in range(start + 1, len(lines))
                if lines[j].startswith("說明") or lines[j].startswith("備註")), len(lines))
    try: return pd.read_csv(io.StringIO("\n".join(lines[start:end])))
    except: return None

def _parse_tpex_csv(text: str):
    text = text.replace("\ufeff", "").replace("\r", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    start = next((i for i, ln in enumerate(lines) if "代號" in ln and "名稱" in ln), None)
    if start is None: return None
    end = next((j for j in range(start + 1, len(lines))
                if lines[j].startswith("說明") or lines[j].startswith("備註")), len(lines))
    try: return pd.read_csv(io.StringIO("\n".join(lines[start:end])))
    except: return None

def get_institutional_data(stock_id: str, trade_date, market_hint=None, max_back: int = 10):
    stock_no = _strip_suffix(stock_id)
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "zh-TW,zh;q=0.9"}
    prefer_twse = not (isinstance(market_hint, str) and market_hint.upper().endswith(".TWO"))
    
    def try_twse(d_):
        url = f"https://www.twse.com.tw/fund/T86?response=csv&date={d_.strftime('%Y%m%d')}&selectType=ALLBUT0999"
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200 or len(r.text) < 200: return None
        return _parse_twse_t86_csv(r.text)

    def try_tpex(d_):
        roc_year = d_.year - 1911
        roc_date = f"{roc_year:03d}/{d_.month:02d}/{d_.day:02d}"
        url = f"https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&o=csv&se=EW&t=D&d={roc_date}&s=0,asc"
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200 or len(r.text) < 200: return None
        return _parse_tpex_csv(r.text)

    markets = [("TWSE", try_twse), ("TPEx", try_tpex)]
    if not prefer_twse: markets = [("TPEx", try_tpex), ("TWSE", try_twse)]

    def find_col(colmap, pred):
        return next((orig for nk, orig in colmap.items() if pred(nk)), None)
    def is_foreign(nk):
        if "外資自營商" in nk and "不含外資自營商" not in nk: return False
        return ("外陸資" in nk) or ("外資及陸資" in nk) or ("外資" in nk)

    for back in range(max_back + 1):
        d = trade_date - timedelta(days=back)
        for mkt_name, fn in markets:
            df = fn(d)
            if df is None: continue
            
            cols = list(df.columns)
            colmap = {_norm_col(c): c for c in cols}
            code_col = next((c for c in cols if _norm_col(c) in ("證券代號", "代號")), None)
            if not code_col: continue
            
            row = df[df[code_col].astype(str).str.strip() == stock_no]
            if row.empty: continue
            
            foreign_col = find_col(colmap, lambda nk: is_foreign(nk) and "買賣超" in nk)
            trust_col = find_col(colmap, lambda nk: "投信" in nk and "買賣超" in nk)
            dealer_total = find_col(colmap, lambda nk: "自營商" in nk and "買賣超" in nk and "外資" not in nk and "自行買賣" not in nk and "避險" not in nk)
            dealer_self = find_col(colmap, lambda nk: "自營商" in nk and "自行買賣" in nk and "買賣超" in nk)
            dealer_hedge = find_col(colmap, lambda nk: "自營商" in nk and "避險" in nk and "買賣超" in nk)

            foreign = _safe_int(row.iloc[0][foreign_col]) if foreign_col else None
            trust = _safe_int(row.iloc[0][trust_col]) if trust_col else None
            dealer = None
            if dealer_total: dealer = _safe_int(row.iloc[0][dealer_total])
            elif dealer_self or dealer_hedge:
                a = _safe_int(row.iloc[0][dealer_self]) if dealer_self else 0
                b = _safe_int(row.iloc[0][dealer_hedge]) if dealer_hedge else 0
                dealer = (a or 0) + (b or 0)
                
            yf_suffix = ".TW" if mkt_name == "TWSE" else ".TWO"
            return {"id": f"{stock_no}{yf_suffix}", "date": d.strftime("%Y-%m-%d"),
                    "foreign": foreign, "trust": trust, "dealer": dealer, "error": None}
    
    return {"error": "查無資料", "id": f"{stock_no}.TW"}

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

def compute_institutional_features(chips_multi: List[Dict], price_now: float, price_20d_ago: float) -> Dict[str, Any]:
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
    feat["foreign_slope_20d"] = slope_n(pd.Series(f_vals), len(f_vals))
    
    # 籌碼背離
    price_up_20d = price_now > price_20d_ago
    feat["flag_foreign_divergence"] = bool(price_up_20d and feat["foreign_20d_net"] < 0)
    return feat

# ── Margin (融資融券) ──
def _parse_twse_json_table(obj):
    if not isinstance(obj, dict): return None
    if "fields" in obj and "data" in obj: return pd.DataFrame(obj["data"], columns=obj["fields"])
    if "tables" in obj and isinstance(obj["tables"], list):
        for tbl in obj["tables"]:
            f = tbl.get("fields", [])
            if any("代號" in str(x) for x in f): return pd.DataFrame(tbl["data"], columns=f)
    return None

def _twse_margin_json(date_yyyymmdd: str, headers: dict):
    urls = [
        f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?response=json&date={date_yyyymmdd}&selectType=ALL",
        f"https://www.twse.com.tw/exchangeReport/MI_MARGN?response=json&date={date_yyyymmdd}&selectType=ALL",
    ]
    h2 = headers.copy()
    h2["Referer"] = "https://www.twse.com.tw/zh/page/trading/exchange/MI_MARGN.html"
    for url in urls:
        try:
            r = requests.get(url, headers=h2, timeout=15)
            if r.status_code == 200:
                j = r.json()
                df = pd.DataFrame(j) if isinstance(j, list) else _parse_twse_json_table(j)
                if df is not None and not df.empty: return df, None
        except Exception: pass
    return None, "Error"

def _parse_tpex_margin_csv(text: str):
    if not text: return None
    lines = [ln for ln in text.split("\n") if ln.strip()]
    start = next((i for i, ln in enumerate(lines) if "代號" in ln and "資" in ln), None)
    if start is None: return None
    end = next((j for j in range(start + 1, len(lines)) if lines[j].startswith("*****")), len(lines))
    try: return pd.read_csv(io.StringIO("\n".join(lines[start:end])))
    except: return None

def get_margin_short_data(stock_id: str, trade_date, market_hint: str = None, max_back: int = 10):
    stock_no = _strip_suffix(stock_id)
    headers = {"User-Agent": "Mozilla/5.0"}
    prefer_twse = not (isinstance(market_hint, str) and market_hint.upper().endswith(".TWO"))
    
    def find_col(colmap, contains):
        return next((orig for nk, orig in colmap.items() if all(kw in nk for kw in contains)), None)

    for back in range(max_back + 1):
        d = trade_date - timedelta(days=back)
        if prefer_twse:
            df, _ = _twse_margin_json(d.strftime("%Y%m%d"), headers)
            if df is not None and not df.empty:
                colmap = {_norm_col(c): c for c in df.columns}
                code_col = next((v for k,v in colmap.items() if "代號" in k or "代碼" in k), None)
                if code_col:
                    sub = df[df[code_col].astype(str).str.strip().str.strip('"') == stock_no]
                    if not sub.empty:
                        row = sub.iloc[0]
                        m_bal_c = find_col(colmap, ["融資", "餘額"]) or find_col(colmap, ["資", "餘額"])
                        m_chg_c = find_col(colmap, ["融資", "增減"]) or find_col(colmap, ["資", "增減"])
                        m_rate_c = find_col(colmap, ["融資", "限額"]) or find_col(colmap, ["資", "限額"]) # 用限額算使用率
                        s_bal_c = find_col(colmap, ["融券", "餘額"]) or find_col(colmap, ["券", "餘額"])
                        s_chg_c = find_col(colmap, ["融券", "增減"]) or find_col(colmap, ["券", "增減"])
                        
                        m_bal = _safe_int(row[m_bal_c]) if m_bal_c else None
                        m_lim = _safe_int(row[m_rate_c]) if m_rate_c else None
                        usage = round(m_bal/m_lim*100, 2) if m_bal and m_lim else None
                        
                        return {
                            "margin_balance": m_bal,
                            "margin_change": _safe_int(row[m_chg_c]) if m_chg_c else None,
                            "margin_usage_rate": usage,
                            "short_balance": _safe_int(row[s_bal_c]) if s_bal_c else None,
                            "short_change": _safe_int(row[s_chg_c]) if s_chg_c else None,
                            "error": None
                        }
        else: # TPEx
            url = f"https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php?l=zh-tw&d={d.year-1911}/{d.month:02d}/{d.day:02d}&o=csv&s=0,asc"
            try:
                r = requests.get(url, headers=headers, timeout=15)
                df = _parse_tpex_margin_csv(r.text)
                if df is not None:
                    colmap = {_norm_col(c): c for c in df.columns}
                    code_col = colmap.get("代號")
                    if code_col:
                        sub = df[df[code_col].astype(str).str.strip() == stock_no]
                        if not sub.empty:
                            row = sub.iloc[0]
                            m_bal_c = find_col(colmap, ["資", "餘額"])
                            m_chg_c = find_col(colmap, ["資", "增減"])
                            s_bal_c = find_col(colmap, ["券", "餘額"])
                            s_chg_c = find_col(colmap, ["券", "增減"])
                            return {
                                "margin_balance": _safe_int(row[m_bal_c]) if m_bal_c else None,
                                "margin_change": _safe_int(row[m_chg_c]) if m_chg_c else None,
                                "margin_usage_rate": None,
                                "short_balance": _safe_int(row[s_bal_c]) if s_bal_c else None,
                                "short_change": _safe_int(row[s_chg_c]) if s_chg_c else None,
                                "error": None
                            }
            except: pass
            
    return {"error": "無融資券資料"}

# ── TDCC 集保 ──
def get_tdcc_distribution(stock_no: str, weeks_back: int = 2) -> Dict[str, Any]:
    stock_no = _strip_suffix(stock_no)
    headers = {"User-Agent": "Mozilla/5.0"}
    result = {"error": None, "data": [], "as_of_date": None, "data_lag_days": None}
    try:
        url = f"https://www.tdcc.com.tw/portal/zh/smWeb/QryStockAJ?scaDates=&scaDate=&SqlMethod=StockNo&StockNo={stock_no}&radioStockNo=&StockName=&REession_SCA_150=&clession_SCA_150="
        r = requests.get(url, headers=headers, timeout=15)
        data = r.json()
        if not data: return {"error": "TDCC 無資料", "data": []}
        
        dates = sorted(set(str(d.get("SCA_DATE", "")) for d in data if d.get("SCA_DATE")))
        latest_dates = dates[-weeks_back:] if len(dates) >= weeks_back else dates
        
        for date_str in latest_dates:
            day_data = [d for d in data if str(d.get("SCA_DATE", "")) == date_str]
            total_holders = 0
            total_shares = 0
            retail_shares = 0
            whale_400 = 0
            whale_1000 = 0
            
            for row in day_data:
                holders = _safe_int(row.get("HOLD_NUM")) or 0
                shares = _safe_int(row.get("HOLD_UNIT")) or 0
                total_holders += holders
                total_shares += shares
                
                nums = re.findall(r"[\d,]+", str(row.get("HOLD_COM", "")).replace(",", ""))
                lb = int(nums[0]) if nums else 0
                
                if lb <= 10000: retail_shares += shares
                if lb >= 400000: whale_400 += shares
                if lb >= 1000000: whale_1000 += shares
            
            if total_shares > 0:
                result["data"].append({
                    "date": f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}",
                    "total_holders": total_holders,
                    "retail_pct": round(retail_shares/total_shares*100, 2),
                    "whale_400_pct": round(whale_400/total_shares*100, 2),
                    "whale_1000_pct": round(whale_1000/total_shares*100, 2)
                })
        
        if result["data"]:
            result["as_of_date"] = result["data"][-1]["date"]
            last_dt = datetime.strptime(result["as_of_date"], "%Y-%m-%d").date()
            result["data_lag_days"] = (datetime.now().date() - last_dt).days
            
    except Exception as e:
        result["error"] = str(e)
    return result

def compute_tdcc_features(tdcc_data: Dict) -> Dict[str, Any]:
    feat = {}
    if not tdcc_data.get("data"):
        feat["tdcc_available"] = False
        return feat
    
    feat["tdcc_available"] = True
    recs = tdcc_data["data"]
    lat = recs[-1]
    feat["tdcc_as_of_date"] = tdcc_data.get("as_of_date")
    feat["tdcc_data_lag_days"] = tdcc_data.get("data_lag_days")
    feat["tdcc_total_holders"] = lat.get("total_holders")
    feat["tdcc_retail_pct"] = lat.get("retail_pct")
    feat["tdcc_whale_400_pct"] = lat.get("whale_400_pct")
    
    if len(recs) >= 2:
        prev = recs[-2]
        feat["tdcc_holders_change"] = lat.get("total_holders") - prev.get("total_holders")
        feat["tdcc_retail_pct_change"] = round(lat.get("retail_pct") - prev.get("retail_pct"), 2)
        feat["tdcc_whale_400_pct_change"] = round(lat.get("whale_400_pct") - prev.get("whale_400_pct"), 2)
        
        # 旗標: 大戶增+散戶減
        feat["flag_whale_up_retail_down"] = (feat["tdcc_whale_400_pct_change"] > 0 and feat["tdcc_retail_pct_change"] < 0)
    
    return feat

def get_foreign_holding_ratio(stock_no: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0"}
    urls = [
        f"https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS?response=json&stockNo={stock_no}&queryType=1",
        f"https://www.twse.com.tw/fund/MI_QFIIS?response=json&stockNo={stock_no}&queryType=1",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            j = r.json()
            data = None
            
            # RWD tables
            if isinstance(j.get("tables"), list):
                for tbl in j["tables"]:
                    f = tbl.get("fields", [])
                    if any("比率" in str(x) or "比例" in str(x) for x in f):
                        data = tbl.get("data")
                        break
            
            # Normal data
            if data is None: data = j.get("data")
            
            if data:
                last = data[-1]
                date_str = _parse_roc_date(str(last[0]))
                # Find ratio (usually last col or find by %)
                ratio = None
                for val in reversed(last):
                    raw = str(val).replace("%","").replace(",","").strip()
                    if re.match(r"^-?\d+(\.\d+)?$", raw):
                        v = float(raw)
                        if 0 < v < 100: 
                            ratio = v
                            break
                return {"ratio": ratio, "date": date_str}
        except: pass
    return {"ratio": None}

# ═══════════════════════════════════════════════════════════
# 7. AI FEATURES BUILDER
# ═══════════════════════════════════════════════════════════

def build_ai_features(stock_id: str, as_of_date=None) -> Dict[str, Any]:
    stock_id = stock_id.strip().upper()
    yf_ticker = _resolve_yf_ticker(stock_id)
    stock_no = _strip_suffix(stock_id)

    # 1. Price Data
    df = yf.download(yf_ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
    df = clean_yf_columns(_ensure_naive_index(df))
    if df.empty: return {"error": f"找不到 {stock_id}", "symbol": stock_id}
    
    ts = _nearest_trading_ts(df, as_of_date)
    if not ts: return {"error": "無資料", "symbol": stock_id}
    df = df.loc[:ts].copy()
    if len(df) < 60: return {"error": "資料不足", "symbol": stock_id}

    latest = df.iloc[-1]
    c_now = float(df["Close"].iloc[-1])
    
    # 2. Indicators
    ma5 = float(df["Close"].rolling(5).mean().iloc[-1])
    ma20 = float(df["Close"].rolling(20).mean().iloc[-1])
    ma60 = float(df["Close"].rolling(60).mean().iloc[-1])
    rsi14 = float(calculate_rsi(df["Close"], 14).iloc[-1])
    
    # K, D
    low_min = df["Low"].rolling(9).min()
    high_max = df["High"].rolling(9).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min) * 100
    k_val = rsv.ewm(com=2).mean()
    d_val = k_val.ewm(com=2).mean()
    
    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9).mean()
    osc = dif - dea
    
    # ADX
    adx, pdi, mdi = calculate_adx(df)
    
    # ATR
    atr = calculate_atr(df)
    atr_now = float(atr.iloc[-1])
    
    # BB
    bbw, bb_up, bb_lo = calculate_bbands(df)
    
    # SuperTrend
    st_line, st_dir = calculate_supertrend(df)
    
    # 3. Patterns
    vp = calculate_volume_profile(df)
    sr = detect_short_term_sr(df)
    fib = calculate_fibonacci_summary(df)
    gaps = detect_gaps_summary(df)
    candle_p = detect_momentum_candle_patterns(df)
    chart_p = detect_patterns(df)
    
    # 4. External Data
    # Institutional
    chips = get_institutional_multi_days(stock_id, ts.date(), market_hint=yf_ticker, days=20)
    p_20 = float(df["Close"].iloc[-20]) if len(df)>=20 else c_now
    inst_feat = compute_institutional_features(chips, c_now, p_20)
    
    # Margin
    margin = get_margin_short_data(stock_id, ts.date(), market_hint=yf_ticker)
    
    # TDCC
    tdcc = get_tdcc_distribution(stock_no)
    tdcc_feat = compute_tdcc_features(tdcc)
    
    # Foreign Holding
    fh = get_foreign_holding_ratio(stock_no)
    
    # 5. Assemble
    feat = {
        "symbol": yf_ticker,
        "price_date": str(ts.date()),
        "query_date": str(as_of_date) if as_of_date else str(datetime.now().date()),
        "close": c_now,
        "ma5": round(ma5, 2), "ma20": round(ma20, 2), "ma60": round(ma60, 2),
        "rsi14": round(rsi14, 2),
        "kd_k": round(float(k_val.iloc[-1]), 2),
        "kd_d": round(float(d_val.iloc[-1]), 2),
        "macd_osc": round(float(osc.iloc[-1]), 4),
        "adx": round(float(adx.iloc[-1]), 2),
        "atr14": round(atr_now, 2),
        "bb_width": round(float(bbw.iloc[-1]), 2),
        "supertrend_bullish": bool(st_dir.iloc[-1] == 1),
    }
    
    feat.update(vp)
    feat.update(sr)
    feat.update(fib)
    feat.update(candle_p)
    feat.update(inst_feat)
    feat.update(tdcc_feat)
    
    if not margin.get("error"):
        feat["margin_balance"] = margin.get("margin_balance")
        feat["margin_change"] = margin.get("margin_change")
        feat["margin_usage_rate"] = margin.get("margin_usage_rate")
        feat["short_balance"] = margin.get("short_balance")
        
    feat["foreign_holding_pct"] = fh.get("ratio")
    feat["unfilled_gaps"] = len(gaps)
    feat["chart_patterns"] = [p["pattern"] for p in chart_p]
    
    # Decision
    res = feat.get("fib_nearest_resistance_1") or feat.get("st_resistance")
    sup = feat.get("fib_nearest_support_1") or feat.get("st_support")
    feat["atr_stop_loss"] = round(c_now - 2 * atr_now, 2)
    feat["target_resistance"] = res
    
    return feat

def format_text_report(feat: Dict[str, Any]) -> str:
    if "error" in feat: return f"❌ {feat.get('symbol','?')}: {feat['error']}"
    
    lines = []
    sep = "="*50
    lines.append(sep)
    lines.append(f"📊 {feat['symbol']} 深度分析報告")
    lines.append(f"📅 資料日期: {feat['price_date']}")
    lines.append(sep)
    
    # 1. 價格趨勢
    lines.append("【1. 價格與趨勢】")
    trend = "多頭排列" if feat["ma5"] > feat["ma20"] > feat["ma60"] else "整理/空頭"
    lines.append(f"  現價: {feat['close']}  趨勢: {trend}")
    lines.append(f"  MA5={feat['ma5']}  MA20={feat['ma20']}  MA60={feat['ma60']}")
    lines.append(f"  SuperTrend: {'看多' if feat.get('supertrend_bullish') else '看空'}")
    
    # 2. 動能
    lines.append("【2. 動能指標】")
    lines.append(f"  RSI(14): {feat['rsi14']} ({'偏強' if feat['rsi14']>60 else '偏弱' if feat['rsi14']<40 else '中性'})")
    lines.append(f"  KD(9,3,3): K={feat['kd_k']}, D={feat['kd_d']}")
    lines.append(f"  MACD OSC: {feat['macd_osc']}  ADX: {feat['adx']}")
    
    # 3. 籌碼 (重點)
    lines.append("【3. 籌碼透視】")
    if feat.get("inst_data_available"):
        f_net = feat.get("foreign_5d_net", 0)
        t_net = feat.get("trust_5d_net", 0)
        lines.append(f"  外資近5日: {f_net:+,} 張")
        lines.append(f"  投信近5日: {t_net:+,} 張")
        
        if feat.get("flag_foreign_divergence"):
            lines.append("  ⚠️ 警示：籌碼背離 (價漲但外資累計賣超)")
    else:
        lines.append("  (法人資料暫缺)")
        
    if feat.get("margin_balance"):
        rate = feat.get("margin_usage_rate")
        rate_str = f"{rate}%" if rate else "n/a"
        lines.append(f"  融資餘額: {feat['margin_balance']:,} 張 (使用率 {rate_str})")
        
    if feat.get("tdcc_available"):
        lines.append(f"  集保大戶(>400張): {feat.get('tdcc_whale_400_pct')}%")
        if feat.get("flag_whale_up_retail_down"):
            lines.append("  🔥 籌碼集中訊號 (大戶增散戶減)")
            
    # 4. 型態與支撐壓力
    lines.append("【4. 關鍵價位 & 型態】")
    lines.append(f"  短期壓力: {feat.get('target_resistance', 'n/a')}")
    lines.append(f"  建議停損: {feat.get('atr_stop_loss')} (2倍ATR)")
    if feat.get("unfilled_gaps"):
        lines.append(f"  未回補缺口: {feat['unfilled_gaps']} 個")
    if feat.get("chart_patterns"):
        lines.append(f"  偵測型態: {', '.join(feat['chart_patterns'])}")
        
    lines.append(sep)
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════
# 8. SECTOR ANALYSIS
# ═══════════════════════════════════════════════════════════

SECTOR_DICT = {
    "記憶體族群": ["MU", "WDC", "000660.KS"],
    "被動元件族群": ["VSH", "6981.T", "6976.T"],
    "AI 與伺服器族群": ["NVDA", "SMCI", "TSM", "VRT"],
    "手機與 IC 設計": ["QCOM", "ARM", "1810.HK"],
    "太空組": ["ARKX", "UFO"]
}

def analyze_sector_performance(sector_key: str, as_of_date=None, custom_tickers: list = None):
    tickers = custom_tickers if custom_tickers else SECTOR_DICT.get(sector_key, [])
    if not tickers: return f"❌ 無效族群：{sector_key}"
    if as_of_date is None: as_of_date = datetime.now().date()

    out = [f"📊 {sector_key} 族群行情快篩", f"📅 資料日期: {as_of_date}", "=" * 45]
    total_pct, valid_count, results = 0.0, 0, []

    for t in tickers:
        try:
            resolved = _resolve_yf_ticker(t)
            df = clean_yf_columns(_ensure_naive_index(
                yf.download(resolved, period="1y", interval="1d", progress=False, auto_adjust=False)))
            if df.empty and resolved.endswith(".TW"):
                alt = resolved.replace(".TW", ".TWO")
                df = clean_yf_columns(_ensure_naive_index(
                    yf.download(alt, period="1y", interval="1d", progress=False, auto_adjust=False)))
            
            ts = _nearest_trading_ts(df, as_of_date)
            if ts is None:
                results.append((t, None, None, "無資料")); continue
            loc = df.index.get_loc(ts)
            if loc < 1:
                results.append((t, float(df.iloc[loc]["Close"]), None, "無前日")); continue
            price = float(df.iloc[loc]["Close"])
            prev = float(df.iloc[loc-1]["Close"])
            pct = (price - prev) / prev * 100
            results.append((t, price, pct, None))
            total_pct += pct
            valid_count += 1
        except Exception as e:
            results.append((t, None, None, str(e)))

    avg = total_pct / valid_count if valid_count > 0 else 0.0
    icon = "🔴" if avg > 0 else ("🟢" if avg < 0 else "➖")
    out.append(f"均漲跌: {icon} {avg:+.2f}%")
    out.append("-" * 45)
    for t, p, pct, err in results:
        if err: out.append(f"{t:<10}{'N/A':>10}  {err}")
        else: out.append(f"{t:<10}{p:>10.2f}{pct:>+10.2f}%")
    out.append("-" * 45)
    return "\n".join(out)

# ═══════════════════════════════════════════════════════════
# 9. MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════

def analyze_stock_technical(stock_id: str, as_of_date=None) -> str:
    """App 呼叫的主入口，回傳格式化後的字串報告"""
    feat = build_ai_features(stock_id, as_of_date)
    return format_text_report(feat)



