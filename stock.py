"""
stock_v2.py — 台股技術分析 + AI Features JSON 輸出
重構版：全數值化、標準化、旗標化，供 LLM / ML 模型消費
"""

import io, json, math, re, warnings
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

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
# 1.  CANDLE  (精簡：只留動能型態，刪反轉型)
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
    types = [classify_candle(r["Open"], r["High"], r["Low"], r["Close"])
             for _, r in tail.iterrows()]

    # 連續紅/黑K
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

    # 跳空突破 (最後一根 low > 前一根 high)
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

# ═══════════════════════════════════════════════════════════
# 2.  CORE INDICATORS
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
    """Average True Range"""
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
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
        else:
            if c_now < lower_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
                
    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1
    return supertrend, direction

def calculate_avwap(df: pd.DataFrame, anchor_date) -> Optional[pd.Series]:
    mask = df.index >= anchor_date
    if not mask.any():
        return None
    sub = df.loc[mask].copy()
    tp = (sub["High"] + sub["Low"] + sub["Close"]) / 3
    return (tp * sub["Volume"]).cumsum() / sub["Volume"].cumsum().replace(0, np.nan)

# ═══════════════════════════════════════════════════════════
# 3.  VOLUME PROFILE (POC) + 短期支撐壓力
# ═══════════════════════════════════════════════════════════

def calculate_volume_profile(df: pd.DataFrame, lookback: int = 60, n_bins: int = 50) -> Dict[str, Any]:
    """
    近 lookback 日的成交量分布 → POC (Point of Control)
    回傳 poc_price, poc_volume, price_vs_poc
    """
    sub = df.tail(lookback).copy()
    if sub.empty:
        return {"poc_price": None, "poc_volume_k": None, "price_vs_poc_pct": None}

    lo = float(sub["Low"].min())
    hi = float(sub["High"].max())
    if hi <= lo:
        return {"poc_price": None, "poc_volume_k": None, "price_vs_poc_pct": None}

    bins = np.linspace(lo, hi, n_bins + 1)
    vol_by_bin = np.zeros(n_bins)

    for _, row in sub.iterrows():
        r_lo, r_hi, vol = float(row["Low"]), float(row["High"]), float(row["Volume"])
        if r_hi <= r_lo or vol <= 0:
            continue
        for b in range(n_bins):
            overlap = max(0, min(r_hi, bins[b + 1]) - max(r_lo, bins[b]))
            vol_by_bin[b] += vol * overlap / (r_hi - r_lo)

    poc_idx = int(np.argmax(vol_by_bin))
    poc_price = round((bins[poc_idx] + bins[poc_idx + 1]) / 2, 2)
    poc_volume = int(vol_by_bin[poc_idx] / 1000)  # 張

    c_now = float(sub["Close"].iloc[-1])
    price_vs_poc = round((c_now - poc_price) / poc_price * 100, 4) if poc_price > 0 else None

    return {
        "poc_price": poc_price,
        "poc_volume_k": poc_volume,
        "price_vs_poc_pct": price_vs_poc,
    }

def detect_short_term_sr(df: pd.DataFrame, lookback: int = 20) -> Dict[str, Any]:
    """近 lookback 日明顯高低點 → 短期支撐壓力"""
    sub = df.tail(lookback)
    if sub.empty:
        return {"st_support": None, "st_resistance": None}
    return {
        "st_support": round(float(sub["Low"].min()), 2),
        "st_resistance": round(float(sub["High"].max()), 2),
    }

# ═══════════════════════════════════════════════════════════
# 4.  DIVERGENCE DETECTION (價格 vs 指標背離)
# ═══════════════════════════════════════════════════════════

def detect_divergence(price: pd.Series, indicator: pd.Series, lookback: int = 60, pivot_n: int = 5) -> Dict[str, bool]:
    """
    偵測頂/底背離
    頂背離: 價格創新高，indicator 未創新高
    底背離: 價格創新低，indicator 未創新低
    """
    result = {"bearish_divergence": False, "bullish_divergence": False}
    p = price.dropna().iloc[-lookback:]
    ind = indicator.dropna().iloc[-lookback:]
    if len(p) < 20 or len(ind) < 20:
        return result

    # 簡易 pivot: 分前半/後半比較
    mid = len(p) // 2
    p_first_half_high = p.iloc[:mid].max()
    p_second_half_high = p.iloc[mid:].max()
    ind_first_half_high = ind.iloc[:mid].max()
    ind_second_half_high = ind.iloc[mid:].max()

    p_first_half_low = p.iloc[:mid].min()
    p_second_half_low = p.iloc[mid:].min()
    ind_first_half_low = ind.iloc[:mid].min()
    ind_second_half_low = ind.iloc[mid:].min()

    # 頂背離: price higher high + indicator lower high
    if p_second_half_high > p_first_half_high and ind_second_half_high < ind_first_half_high:
        result["bearish_divergence"] = True

    # 底背離: price lower low + indicator higher low
    if p_second_half_low < p_first_half_low and ind_second_half_low > ind_first_half_low:
        result["bullish_divergence"] = True

    return result

# ═══════════════════════════════════════════════════════════
# 5.  VOLUME QUALITY (量能品質)
# ═══════════════════════════════════════════════════════════

def calculate_volume_quality(df: pd.DataFrame, period: int = 20) -> Dict[str, Any]:
    """
    量分位、上漲量/下跌量比、OBV 斜率/分位
    """
    sub = df.tail(max(period, 60)).copy()
    vol = sub["Volume"]
    close = sub["Close"]

    # 量分位
    vol_pct = percentile_rank(vol, window=252)

    # 上漲量 / 下跌量比 (近 period 日)
    tail = sub.tail(period)
    up_mask = tail["Close"] > tail["Close"].shift(1)
    down_mask = tail["Close"] < tail["Close"].shift(1)
    up_vol = float(tail.loc[up_mask, "Volume"].sum()) if up_mask.any() else 0
    down_vol = float(tail.loc[down_mask, "Volume"].sum()) if down_mask.any() else 0
    up_down_ratio = round(up_vol / down_vol, 4) if down_vol > 0 else None

    # OBV 斜率 & 分位
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

# ═══════════════════════════════════════════════════════════
# 6.  FIBONACCI (摘要化: 最近 2-3 支撐壓力)
# ═══════════════════════════════════════════════════════════

def calculate_fibonacci_summary(df: pd.DataFrame, lookback: int = 120) -> Dict[str, Any]:
    """只回傳離現價最近的 2-3 個支撐/壓力位"""
    sub = df.tail(lookback)
    high = float(sub["High"].max())
    low = float(sub["Low"].min())
    diff = high - low
    c_now = float(sub["Close"].iloc[-1])

    levels = {
        "fib_0236": round(high - 0.236 * diff, 2),
        "fib_0382": round(high - 0.382 * diff, 2),
        "fib_0500": round(high - 0.500 * diff, 2),
        "fib_0618": round(high - 0.618 * diff, 2),
    }

    # 分成壓力(>現價) 和支撐(<現價)
    resistances = sorted([v for v in levels.values() if v > c_now])
    supports = sorted([v for v in levels.values() if v <= c_now], reverse=True)

    return {
        "fib_high": high,
        "fib_low": low,
        "fib_nearest_support_1": supports[0] if len(supports) > 0 else None,
        "fib_nearest_support_2": supports[1] if len(supports) > 1 else None,
        "fib_nearest_resistance_1": resistances[0] if len(resistances) > 0 else None,
        "fib_nearest_resistance_2": resistances[1] if len(resistances) > 1 else None,
    }

# ═══════════════════════════════════════════════════════════
# 7.  GAPS (摘要化: 最近 1-2 未回補缺口)
# ═══════════════════════════════════════════════════════════

def detect_gaps_summary(df: pd.DataFrame, lookback: int = 30) -> List[Dict]:
    """只回傳未回補的缺口 (最近 2 個)"""
    if df is None or df.empty or len(df) < 2:
        return []
    sub = df.tail(lookback + 1)
    gaps = []
    for i in range(1, len(sub)):
        prev, curr = sub.iloc[i - 1], sub.iloc[i]
        date_str = sub.index[i].strftime("%Y-%m-%d")
        if float(curr["Low"]) > float(prev["High"]):
            future = sub.iloc[i + 1:]
            filled = any(float(r["Low"]) <= float(prev["High"]) for _, r in future.iterrows())
            if not filled:
                gaps.append({"date": date_str, "type": "up",
                             "lower": float(prev["High"]), "upper": float(curr["Low"])})
        elif float(curr["High"]) < float(prev["Low"]):
            future = sub.iloc[i + 1:]
            filled = any(float(r["High"]) >= float(prev["Low"]) for _, r in future.iterrows())
            if not filled:
                gaps.append({"date": date_str, "type": "down",
                             "lower": float(curr["High"]), "upper": float(prev["Low"])})
    # 只取最近 2 個
    return gaps[-2:] if len(gaps) > 2 else gaps

# ═══════════════════════════════════════════════════════════
# 8.  PATTERN DETECTION (保留反轉型態偵測函數，但不輸出 K 棒反轉型)
# ═══════════════════════════════════════════════════════════

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

def detect_double_top(df, lookback=120, pivot_lr=3, peak_tol=0.018, min_gap=8, max_gap=60, confirm_margin=0.003):
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
    neckline = float(sub.iloc[lo:hi + 1]["Low"].min())
    confirmed = float(sub["Close"].iloc[-1]) < neckline * (1 - confirm_margin)
    height = max(float(p1[2]), float(p2[2])) - neckline
    return {"pattern": "double_top", "confirmed": confirmed,
            "neckline": neckline, "target": round(neckline - height, 2)}

def detect_double_bottom(df, lookback=160, pivot_lr=3, trough_tol=0.020, min_gap=8, max_gap=80, confirm_margin=0.003):
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
    neckline = float(sub.iloc[lo:hi + 1]["High"].max())
    confirmed = float(sub["Close"].iloc[-1]) > neckline * (1 + confirm_margin)
    height = neckline - min(float(t1[2]), float(t2[2]))
    return {"pattern": "double_bottom", "confirmed": confirmed,
            "neckline": neckline, "target": round(neckline + height, 2)}

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
    confirmed = float(sub["Close"].iloc[-1]) < neckline * (1 - confirm_margin)
    height = float(head[2]) - neckline
    return {"pattern": "head_and_shoulders_top", "confirmed": confirmed,
            "neckline": round(neckline, 2), "target": round(neckline - height, 2)}

def detect_inverse_head_and_shoulders(df, lookback=180, pivot_lr=3, shoulder_tol=0.025, confirm_margin=0.003):
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
    confirmed = float(sub["Close"].iloc[-1]) > neckline * (1 + confirm_margin)
    height = neckline - float(head[2])
    return {"pattern": "inv_head_and_shoulders", "confirmed": confirmed,
            "neckline": round(neckline, 2), "target": round(neckline + height, 2)}

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
        confirmed = close_now < lower_now * (1 - breakout_margin)
        return {"pattern": "rising_wedge", "confirmed": confirmed}
    if ah < 0 and al < 0 and ah < al:
        confirmed = close_now > upper_now * (1 + breakout_margin)
        return {"pattern": "falling_wedge", "confirmed": confirmed}
    return None

def detect_patterns(df) -> List[Dict]:
    results = []
    for fn in [detect_double_top, detect_double_bottom,
               detect_head_and_shoulders, detect_inverse_head_and_shoulders,
               detect_wedge]:
        try:
            p = fn(df)
            if p:
                results.append(p)
        except Exception:
            pass
    return results

# ═══════════════════════════════════════════════════════════
# 9.  INSTITUTIONAL DATA (三大法人) — 改看 20 日
# ═══════════════════════════════════════════════════════════

def _parse_twse_t86_csv(text: str):
    text = text.replace("\r", "").replace("=", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    start = next((i for i, ln in enumerate(lines) if "證券代號" in ln and "證券名稱" in ln), None)
    if start is None:
        return None
    end = next((j for j in range(start + 1, len(lines))
                if lines[j].startswith("說明") or lines[j].startswith("備註")), len(lines))
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
    end = next((j for j in range(start + 1, len(lines))
                if lines[j].startswith("說明") or lines[j].startswith("備註")), len(lines))
    try:
        return pd.read_csv(io.StringIO("\n".join(lines[start:end])))
    except Exception:
        return None

def get_institutional_data(stock_id: str, trade_date, market_hint=None, max_back: int = 10):
    stock_no = _strip_suffix(stock_id)
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "zh-TW,zh;q=0.9"}
    prefer_twse = not (isinstance(market_hint, str) and market_hint.upper().endswith(".TWO"))
    last_error = None

    def try_twse(d_):
        url = f"https://www.twse.com.tw/fund/T86?response=csv&date={d_.strftime('%Y%m%d')}&selectType=ALLBUT0999"
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200 or len(r.text) < 200:
            return None, f"TWSE HTTP {r.status_code}"
        if "沒有符合條件的資料" in r.text or "很抱歉" in r.text:
            return None, "TWSE 無資料(休市)"
        df = _parse_twse_t86_csv(r.text)
        return (df, None) if df is not None else (None, "TWSE 解析失敗")

    def try_tpex(d_):
        roc_year = d_.year - 1911
        roc_date = f"{roc_year:03d}/{d_.month:02d}/{d_.day:02d}"
        url = (
            "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"
            f"?l=zh-tw&o=csv&se=EW&t=D&d={roc_date}&s=0,asc"
        )
        h2 = {**headers, "Referer": "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge.php"}
        r = requests.get(url, headers=h2, timeout=15)
        if r.status_code != 200 or len(r.text) < 200:
            return None, f"TPEx HTTP {r.status_code}"
        if "沒有符合條件的資料" in r.text or "很抱歉" in r.text:
            return None, "TPEx 無資料(休市)"
        df = _parse_tpex_csv(r.text)
        return (df, None) if df is not None else (None, "TPEx 解析失敗")

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
                last_error = err; continue
            cols = list(df.columns)
            colmap = {_norm_col(c): c for c in cols}
            code_col = next((c for c in cols if _norm_col(c) in ("證券代號", "代號")), None)
            if code_col is None:
                last_error = f"{mkt_name} 找不到代號欄"; continue
            row = df[df[code_col].astype(str).str.strip() == stock_no]
            if row.empty:
                last_error = f"{mkt_name} 找不到 {stock_no}"; continue

            foreign_col = find_col(colmap, lambda nk: is_foreign(nk) and "買賣超" in nk)
            trust_col = find_col(colmap, lambda nk: "投信" in nk and "買賣超" in nk)
            dealer_total = find_col(colmap, lambda nk: "自營商" in nk and "買賣超" in nk
                                    and "外資" not in nk and "自行買賣" not in nk and "避險" not in nk)
            dealer_self = find_col(colmap, lambda nk: "自營商" in nk and "自行買賣" in nk
                                   and "買賣超" in nk and "外資" not in nk)
            dealer_hedge = find_col(colmap, lambda nk: "自營商" in nk and "避險" in nk
                                    and "買賣超" in nk and "外資" not in nk)

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
                buy_c = find_col(colmap, lambda nk: is_foreign(nk)
                                 and ("買進" in nk or "買入" in nk) and "買賣超" not in nk)
                sell_c = find_col(colmap, lambda nk: is_foreign(nk)
                                  and "賣出" in nk and "買賣超" not in nk)
                bv = _safe_int(row.iloc[0][buy_c]) if buy_c else None
                sv = _safe_int(row.iloc[0][sell_c]) if sell_c else None
                if bv is not None and sv is not None:
                    foreign = bv - sv

            yf_suffix = ".TW" if mkt_name == "TWSE" else ".TWO"
            return {"id": f"{stock_no}{yf_suffix}", "date": d.strftime("%Y-%m-%d"),
                    "foreign": foreign, "trust": trust, "dealer": dealer, "error": None}

    return {"error": last_error or "未知錯誤", "id": f"{stock_no}.TW"}

def get_institutional_multi_days(stock_id: str, end_date, market_hint=None, days=20):
    """抓近 N 個交易日三大法人，回傳 list（最舊在前）"""
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
    """從多日法人資料算出 AI features"""
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

    # 5 日 & 20 日累計
    feat["foreign_5d_net"] = sum(f_vals[-5:])
    feat["foreign_20d_net"] = sum(f_vals)
    feat["trust_5d_net"] = sum(t_vals[-5:])
    feat["trust_20d_net"] = sum(t_vals)
    feat["dealer_5d_net"] = sum(d_vals[-5:])
    feat["dealer_20d_net"] = sum(d_vals)

    # 斜率 (20 日)
    feat["foreign_slope_20d"] = slope_n(pd.Series(f_vals), len(f_vals))
    feat["trust_slope_20d"] = slope_n(pd.Series(t_vals), len(t_vals))

    # 連續天數
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
        return count * direction  # 正=連買, 負=連賣
    feat["foreign_consecutive_days"] = _consec(f_vals)
    feat["trust_consecutive_days"] = _consec(t_vals)

    # 籌碼背離旗標
    price_up_20d = price_now > price_20d_ago
    feat["flag_foreign_divergence"] = bool(price_up_20d and feat["foreign_20d_net"] < 0)
    feat["flag_inst_consensus_buy"] = bool(feat["foreign_20d_net"] > 0 and feat["trust_20d_net"] > 0)
    feat["flag_inst_consensus_sell"] = bool(feat["foreign_20d_net"] < 0 and feat["trust_20d_net"] < 0)

    return feat

# ── 外資持股比率 ──

def get_foreign_holding_ratio(stock_no: str) -> dict:
    headers = {"User-Agent": "Mozilla/5.0"}
    urls = [
        f"https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS?response=json&stockNo={stock_no}&queryType=1",
        f"https://www.twse.com.tw/fund/MI_QFIIS?response=json&stockNo={stock_no}&queryType=1",
    ]
    last_err = None
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"; continue
            j = r.json()
            fields, data = None, None
            if isinstance(j.get("tables"), list):
                for tbl in j["tables"]:
                    f = tbl.get("fields", [])
                    d = tbl.get("data", [])
                    if f and d:
                        joined = "".join(str(x) for x in f)
                        if any(kw in joined for kw in ("持股比", "比率", "比例")):
                            fields, data = f, d
                            break
            if any("外資" in str(x) for x in f) and fields is None:
                fields, data = f, d
            if fields is None and j.get("fields") and j.get("data"):
                fields, data = j["fields"], j["data"]
            if fields is None and isinstance(j.get("data"), list) and j["data"]:
                data = j["data"]
            if not data:
                last_err = "無資料"; continue
            last = data[-1]
            if not last:
                last_err = "空資料行"; continue
            date_str = _parse_roc_date(str(last[0]))
            ratio = None
            if fields:
                norm_fields = [_norm_col(f) for f in fields]
                for i, nf in enumerate(norm_fields):
                    if any(kw in nf for kw in ("持股比", "比率", "比例", "Percentage")):
                        if i < len(last):
                            raw = str(last[i]).replace("%", "").replace(",", "").strip()
                            if raw and re.match(r"^-?\d+(\.\d+)?$", raw):
                                ratio = float(raw)
                            break
            if ratio is not None:
                return {"ratio": ratio, "date": date_str, "error": None}
            else:
                last_err = "找不到持股比率欄"; continue
        except Exception as e:
            last_err = str(e); continue
    return {"ratio": None, "date": None, "error": last_err or "未知錯誤"}

# ═══════════════════════════════════════════════════════════
# 10. MARGIN TRADING (融資融券) — 修正上市股支援
# ═══════════════════════════════════════════════════════════

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
    urls = [
        f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?response=json&date={date_yyyymmdd}&selectType=ALL",
        f"https://www.twse.com.tw/exchangeReport/MI_MARGN?response=json&date={date_yyyymmdd}&selectType=ALL",
    ]
    h2 = headers.copy()
    h2["Referer"] = "https://www.twse.com.tw/zh/page/trading/exchange/MI_MARGN.html"
    last_err = None
    for url in urls:
        try:
            r = requests.get(url, headers=h2, timeout=15)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"; continue
            j = r.json()
            df = pd.DataFrame(j) if isinstance(j, list) else _parse_twse_json_table(j)
            if df is not None and not df.empty:
                return df, None
            last_err = "空資料/格式不符"
        except Exception as e:
            last_err = str(e)
    return None, last_err or "取資料失敗"

def _parse_tpex_margin_csv(text: str):
    if not text:
        return None
    text = text.replace("\ufeff", "").replace("\r", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    start = next((i for i, ln in enumerate(lines)
                  if "代號" in ln and "名稱" in ln and ("資" in ln or "融資" in ln)), None)
    if start is None:
        return None
    end = next((j for j in range(start + 1, len(lines))
                if lines[j].startswith("*****") or lines[j].startswith("說明")
                or lines[j].startswith("備註")), len(lines))
    try:
        return pd.read_csv(io.StringIO("\n".join(lines[start:end])))
    except Exception:
        return None

def get_margin_short_data(stock_id: str, trade_date, market_hint: str = None, max_back: int = 10):
    """修正版：上市/上櫃都嘗試"""
    stock_no = _strip_suffix(stock_id)
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "zh-TW,zh;q=0.9"}
    # 改為：兩個市場都嘗試，不再只看 prefer
    is_two = isinstance(market_hint, str) and market_hint.upper().endswith(".TWO")
    last_error = None

    def find_col(colmap, contains):
        return next((orig for nk, orig in colmap.items()
                     if all(kw in nk for kw in contains)), None)

    for back in range(max_back + 1):
        d = trade_date - timedelta(days=back)

        # ── 嘗試 TWSE ──
        if not is_two:
            df, err = _twse_margin_json(d.strftime("%Y%m%d"), headers=headers)
            if df is not None and not df.empty:
                colmap = {_norm_col(c): c for c in df.columns}
                code_col = next((v for k, v in colmap.items()
                                 if k in ("股票代號", "證券代號", "證券代碼", "Code", "代號")), None)
                if not code_col:
                    code_col = next((v for k, v in colmap.items()
                                     if "代號" in k or "代碼" in k), None)
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
                        return {"id": f"{stock_no}.TW", "date": d.strftime("%Y-%m-%d"),
                                "margin_balance": m_bal, "margin_change": m_chg,
                                "margin_limit": m_lim, "margin_usage_rate": usage,
                                "short_balance": s_bal, "short_change": s_chg, "error": None}
                    else:
                        last_error = f"TWSE margin 找不到 {stock_no}"
                else:
                    last_error = "TWSE margin 找不到代號欄"
            else:
                last_error = err

        # ── 嘗試 TPEx ──
        roc_year = d.year - 1911
        roc_date = f"{roc_year:03d}/{d.month:02d}/{d.day:02d}"
        url_csv = (
            "https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php"
            f"?l=zh-tw&d={roc_date}&o=csv&s=0,asc"
        )
        try:
            r = requests.get(url_csv, headers=headers, timeout=15)
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
                        return {"id": f"{stock_no}.TWO", "date": d.strftime("%Y-%m-%d"),
                                "margin_balance": _safe_int(row[m_bal_c]) if m_bal_c else None,
                                "margin_change": _safe_int(row[m_chg_c]) if m_chg_c else None,
                                "margin_limit": None, "margin_usage_rate": None,
                                "short_balance": _safe_int(row[s_bal_c]) if s_bal_c else None,
                                "short_change": _safe_int(row[s_chg_c]) if s_chg_c else None,
                                "error": None}
        except Exception as e:
            last_error = f"TPEx margin: {e}"

    return {"error": last_error or "未知錯誤"}

# ═══════════════════════════════════════════════════════════
# 11. 集保/大戶持股 (TDCC)
# ═══════════════════════════════════════════════════════════

def get_tdcc_distribution(stock_no: str, weeks_back: int = 2) -> Dict[str, Any]:
    """
    從集保中心抓股權分散表 → 大戶/散戶持股比例 + 股東戶數
    回傳最近 weeks_back 週資料
    ⚠️ 這是週資料，有 data lag，必須標註 as_of_date
    """
    stock_no = _strip_suffix(stock_no)
    headers = {"User-Agent": "Mozilla/5.0"}
    result = {"error": None, "data": [], "as_of_date": None, "data_lag_days": None}

    try:
        # TDCC OpenData API
        url = (
            "https://www.tdcc.com.tw/portal/zh/smWeb/QryStockAJ?"
            f"scaDates=&scaDate=&SqlMethod=StockNo&StockNo={stock_no}"
            "&radioStockNo=&StockName=&REession_SCA_150=&clession_SCA_150="
        )
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200:
            result["error"] = f"TDCC HTTP {r.status_code}"
            return result

        # 嘗試 JSON 解析
        try:
            data = r.json()
        except Exception:
            result["error"] = "TDCC JSON 解析失敗"
            return result

        if not data:
            result["error"] = "TDCC 無資料"
            return result

        # 收集所有日期
        dates = sorted(set(str(d.get("SCA_DATE", "")) for d in data if d.get("SCA_DATE")))
        if not dates:
            result["error"] = "TDCC 無日期資料"
            return result

        latest_dates = dates[-weeks_back:] if len(dates) >= weeks_back else dates

        for date_str in latest_dates:
            day_data = [d for d in data if str(d.get("SCA_DATE", "")) == date_str]
            if not day_data:
                continue

            total_holders = 0
            total_shares = 0
            retail_shares = 0   # <=10張 (<=10,000股)
            whale_400_shares = 0  # >=400張 (>=400,000股)
            whale_1000_shares = 0  # >=1000張 (>=1,000,000股)

            for row in day_data:
                hold_com = str(row.get("HOLD_COM", ""))
                holders = _safe_int(row.get("HOLD_NUM")) or 0
                shares = _safe_int(row.get("HOLD_UNIT")) or 0
                total_holders += holders
                total_shares += shares

                # 判斷級距
                nums = re.findall(r"[\d,]+", hold_com.replace(",", ""))
                if nums:
                    try:
                        lower_bound = int(nums[0])
                    except Exception:
                        lower_bound = 0

                    if lower_bound <= 10000:  # 10張以下
                        retail_shares += shares
                    if lower_bound >= 400000:  # 400張以上
                        whale_400_shares += shares
                    if lower_bound >= 1000000:  # 1000張以上
                        whale_1000_shares += shares

            retail_pct = round(retail_shares / total_shares * 100, 2) if total_shares > 0 else None
            whale_400_pct = round(whale_400_shares / total_shares * 100, 2) if total_shares > 0 else None
            whale_1000_pct = round(whale_1000_shares / total_shares * 100, 2) if total_shares > 0 else None

            # 日期格式化
            try:
                dt = datetime.strptime(date_str, "%Y%m%d")
                date_fmt = dt.strftime("%Y-%m-%d")
            except Exception:
                date_fmt = date_str

            result["data"].append({
                "date": date_fmt,
                "total_holders": total_holders,
                "retail_pct": retail_pct,
                "whale_400_pct": whale_400_pct,
                "whale_1000_pct": whale_1000_pct,
            })

        if result["data"]:
            result["as_of_date"] = result["data"][-1]["date"]
            # 計算 data_lag
            try:
                last_data_date = datetime.strptime(result["as_of_date"], "%Y-%m-%d").date()
                result["data_lag_days"] = (datetime.now().date() - last_data_date).days
            except Exception:
                result["data_lag_days"] = None

    except Exception as e:
        result["error"] = str(e)

    return result

def compute_tdcc_features(tdcc_data: Dict) -> Dict[str, Any]:
    """從 TDCC 資料計算 AI features"""
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

    # 週變動
    if len(records) >= 2:
        prev = records[-2]
        feat["tdcc_holders_change"] = (latest.get("total_holders") or 0) - (prev.get("total_holders") or 0)
        feat["tdcc_retail_pct_change"] = round(
            (latest.get("retail_pct") or 0) - (prev.get("retail_pct") or 0), 2)
        feat["tdcc_whale_400_pct_change"] = round(
            (latest.get("whale_400_pct") or 0) - (prev.get("whale_400_pct") or 0), 2)
        feat["tdcc_whale_1000_pct_change"] = round(
            (latest.get("whale_1000_pct") or 0) - (prev.get("whale_1000_pct") or 0), 2)

        # 旗標: 大戶增+散戶減 = 籌碼集中
        feat["flag_whale_up_retail_down"] = bool(
            (feat.get("tdcc_whale_400_pct_change") or 0) > 0 and
            (feat.get("tdcc_retail_pct_change") or 0) < 0
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

# ═══════════════════════════════════════════════════════════
# 12. RELATIVE STRENGTH
# ═══════════════════════════════════════════════════════════

def calc_relative_strength(stock_df, benchmark_ticker="0050.TW", period=20):
    try:
        bench = clean_yf_columns(_ensure_naive_index(
            yf.download(benchmark_ticker, period="3mo", progress=False, auto_adjust=False)))
        if bench.empty or len(stock_df) < period:
            return None
        s_ret = (float(stock_df["Close"].iloc[-1]) / float(stock_df["Close"].iloc[-period]) - 1) * 100
        b_ret = (float(bench["Close"].iloc[-1]) / float(bench["Close"].iloc[-period]) - 1) * 100
        return {"stock_ret_20d": round(s_ret, 2), "bench_ret_20d": round(b_ret, 2),
                "rs_20d": round(s_ret - b_ret, 2)}
    except Exception:
        return None

# ═══════════════════════════════════════════════════════════
# 13. DECISION FIELDS (入場/停損/目標/R:R)
# ═══════════════════════════════════════════════════════════

def compute_decision_fields(c_now: float, atr_now: float,
                            resistance: Optional[float],
                            support: Optional[float],
                            supertrend_dir: int) -> Dict[str, Any]:
    """
    ATR-based 停損/目標 + R:R
    """
    feat: Dict[str, Any] = {}

    if atr_now is None or atr_now <= 0 or not np.isfinite(atr_now):
        return {"decision_available": False}

    feat["decision_available"] = True

    # 停損: 2x ATR below current
    stop_loss = round(c_now - 2 * atr_now, 2)
    feat["atr_stop_loss"] = stop_loss
    feat["atr_stop_loss_pct"] = round((stop_loss - c_now) / c_now * 100, 2)

    # 目標: 最近壓力
    if resistance and resistance > c_now:
        feat["target_resistance"] = resistance
        feat["target_distance_pct"] = round((resistance - c_now) / c_now * 100, 2)
        risk = c_now - stop_loss
        reward = resistance - c_now
        feat["risk_reward_ratio"] = round(reward / risk, 2) if risk > 0 else None
    else:
        feat["target_resistance"] = None
        feat["target_distance_pct"] = None
        feat["risk_reward_ratio"] = None

    # 支撐距離
    if support and support < c_now:
        feat["nearest_support"] = support
        feat["support_distance_pct"] = round((support - c_now) / c_now * 100, 2)
    else:
        feat["nearest_support"] = None
        feat["support_distance_pct"] = None

    # 入場觸發 (簡易規則)
    feat["flag_entry_trigger"] = bool(
        supertrend_dir == 1  # SuperTrend 多頭確認
    )

    return feat

# ═══════════════════════════════════════════════════════════
# 14.  MAIN: BUILD AI FEATURES JSON
# ═══════════════════════════════════════════════════════════

def build_ai_features(stock_id: str, as_of_date=None) -> Dict[str, Any]:
    """
    核心函數：產出完整 AI Features JSON
    全數值化、標準化、旗標化
    """
    stock_id = stock_id.strip().upper()
    yf_ticker = _resolve_yf_ticker(stock_id)
    stock_no = _strip_suffix(stock_id)

    # ── 下載日線 ──
    df_daily = yf.download(yf_ticker, period="2y", interval="1d",
                           progress=False, auto_adjust=False)
    df_daily = clean_yf_columns(_ensure_naive_index(df_daily))

    if df_daily.empty and yf_ticker.endswith(".TW"):
        yf_ticker = yf_ticker.replace(".TW", ".TWO")
        df_daily = yf.download(yf_ticker, period="2y", interval="1d",
                               progress=False, auto_adjust=False)
        df_daily = clean_yf_columns(_ensure_naive_index(df_daily))

    if df_daily.empty:
        return {"error": f"找不到 {stock_id}", "symbol": stock_id}

    chosen_ts = _nearest_trading_ts(df_daily, as_of_date)
    if chosen_ts is None:
        return {"error": "選定日期前無資料", "symbol": stock_id}

    df = df_daily.loc[:chosen_ts].copy()
    if len(df) < 60:
        return {"error": "資料不足 60 天", "symbol": stock_id}

    latest = df.iloc[-1]
    trade_date = chosen_ts.date()
    query_date = as_of_date if as_of_date else datetime.now().date()

    # ── 下載週線 ──
    df_weekly = clean_yf_columns(_ensure_naive_index(
        yf.download(yf_ticker, period="2y", interval="1wk",
                    progress=False, auto_adjust=False)))
    df_weekly_upto = (df_weekly.loc[:chosen_ts]
                      if df_weekly is not None and not df_weekly.empty else None)

    # ── 基本價量 ──
    close = df["Close"]
    c_now = float(close.iloc[-1])
    o_now = float(latest["Open"])
    h_now = float(latest["High"])
    l_now = float(latest["Low"])
    vol_now = float(latest["Volume"])

    feat: Dict[str, Any] = {
        "symbol": yf_ticker,
        "price_date": str(trade_date),
        "query_date": str(query_date),
    }

    feat["close"] = c_now
    feat["open"] = o_now
    feat["high"] = h_now
    feat["low"] = l_now
    feat["volume"] = int(vol_now)
    feat["volume_k"] = int(vol_now / 1000)

    # ── 均線 + 乖離 + 乖離百分位 ──
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    ma5_now = float(ma5.iloc[-1])
    ma20_now = float(ma20.iloc[-1])
    ma60_now = float(ma60.iloc[-1])

    feat["ma5"] = round(ma5_now, 2)
    feat["ma20"] = round(ma20_now, 2)
    feat["ma60"] = round(ma60_now, 2)

    # 乖離率
    ma20_dev = (c_now - ma20_now) / ma20_now * 100
    ma60_dev = (c_now - ma60_now) / ma60_now * 100
    feat["ma20_dev_pct"] = round(ma20_dev, 4)
    feat["ma60_dev_pct"] = round(ma60_dev, 4)

    # 乖離率歷史百分位
    ma20_dev_series = (close - ma20) / ma20 * 100
    ma60_dev_series = (close - ma60) / ma60 * 100
    feat["ma20_dev_percentile_252d"] = percentile_rank(ma20_dev_series.dropna(), 252)
    feat["ma60_dev_percentile_252d"] = percentile_rank(ma60_dev_series.dropna(), 252)

    # 均線方向 (斜率)
    feat["ma5_slope_5d"] = slope_n(ma5, 5)
    feat["ma20_slope_5d"] = slope_n(ma20, 5)

    # ── 趨勢狀態 ──
    if c_now > ma20_now > ma60_now and ma5_now > ma20_now:
        trend_state = "uptrend"
    elif c_now < ma20_now < ma60_now and ma5_now < ma20_now:
        trend_state = "downtrend"
    elif c_now > ma20_now and ma20_now < ma60_now:
        trend_state = "bottom_bounce"
    elif c_now < ma20_now and ma20_now > ma60_now:
        trend_state = "top_pullback"
    else:
        trend_state = "consolidation"
    feat["trend_state"] = trend_state

    # 52 週位置
    high_52 = float(df.tail(252)["High"].max())
    low_52 = float(df.tail(252)["Low"].min())
    feat["pos_52w_pct"] = round((c_now - low_52) / (high_52 - low_52) * 100, 2) if high_52 != low_52 else 50.0

    # 週線 vs 20 週均
    if df_weekly_upto is not None and not df_weekly_upto.empty and len(df_weekly_upto) >= 20:
        wma20 = float(df_weekly_upto["Close"].rolling(20).mean().iloc[-1])
        feat["weekly_above_ma20"] = bool(float(df_weekly_upto["Close"].iloc[-1]) > wma20)
    else:
        feat["weekly_above_ma20"] = None

    # ── 相對強度 ──
    rs_data = calc_relative_strength(df, benchmark_ticker="0050.TW", period=20)
    if rs_data:
        feat["rs_vs_bench_20d"] = rs_data["rs_20d"]
        feat["stock_ret_20d"] = rs_data["stock_ret_20d"]
        feat["bench_ret_20d"] = rs_data["bench_ret_20d"]
    else:
        feat["rs_vs_bench_20d"] = None

    # ── 量價分析 ──
    avg_vol_5 = float(df["Volume"].tail(5).mean())
    feat["vol_ratio_5d"] = round(vol_now / avg_vol_5, 4) if avg_vol_5 > 0 else None
    feat["flag_price_up_vol_up"] = bool(c_now > float(df["Close"].iloc[-2]) and vol_now > avg_vol_5)
    feat["flag_price_up_vol_down"] = bool(c_now > float(df["Close"].iloc[-2]) and vol_now < avg_vol_5)
    feat["flag_price_down_vol_up"] = bool(c_now < float(df["Close"].iloc[-2]) and vol_now > avg_vol_5)

    # 量能品質
    vq = calculate_volume_quality(df)
    feat.update(vq)

    # ── RSI (只保留 14) ──
    rsi14 = calculate_rsi(close, 14)
    rsi14_now = float(rsi14.iloc[-1])
    feat["rsi14"] = round(rsi14_now, 2)
    feat["rsi14_percentile_252d"] = percentile_rank(rsi14, 252)
    feat["rsi14_slope_5d"] = slope_n(rsi14, 5)

    # ── KD ──
    low_min = df["Low"].rolling(9).min()
    high_max = df["High"].rolling(9).max()
    rsv = (close - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    k_val = rsv.ewm(com=2).mean()
    d_val = k_val.ewm(com=2).mean()
    feat["kd_k"] = round(float(k_val.iloc[-1]), 2)
    feat["kd_d"] = round(float(d_val.iloc[-1]), 2)
    feat["flag_kd_golden_cross"] = bool(
        len(k_val) >= 2 and len(d_val) >= 2 and
        k_val.iloc[-2] < d_val.iloc[-2] and k_val.iloc[-1] > d_val.iloc[-1]
    )
    feat["flag_kd_death_cross"] = bool(
        len(k_val) >= 2 and len(d_val) >= 2 and
        k_val.iloc[-2] > d_val.iloc[-2] and k_val.iloc[-1] < d_val.iloc[-1]
    )

    # ── MACD ──
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
        len(dif) >= 2 and len(dea) >= 2 and
        dif.iloc[-2] < dea.iloc[-2] and dif.iloc[-1] > dea.iloc[-1]
    )
    feat["flag_macd_death_cross"] = bool(
        len(dif) >= 2 and len(dea) >= 2 and
        dif.iloc[-2] > dea.iloc[-2] and dif.iloc[-1] < dea.iloc[-1]
    )

    # ── ADX ──
    adx, pdi, mdi = calculate_adx(df)
    feat["adx"] = round(float(adx.iloc[-1]), 2)
    feat["plus_di"] = round(float(pdi.iloc[-1]), 2)
    feat["minus_di"] = round(float(mdi.iloc[-1]), 2)
    feat["flag_strong_trend"] = bool(float(adx.iloc[-1]) > 25)
    feat["flag_di_bullish"] = bool(float(pdi.iloc[-1]) > float(mdi.iloc[-1]))

    # ── ATR (新增) ──
    atr_series = calculate_atr(df, 14)
    atr_now = float(atr_series.iloc[-1])
    feat["atr14"] = round(atr_now, 2)
    feat["atr14_pct"] = round(atr_now / c_now * 100, 4)
    feat["atr14_percentile_252d"] = percentile_rank(atr_series, 252)

    # ── 波動率 (新增) ──
    returns_20d = close.pct_change().tail(20)
    feat["volatility_20d"] = round(float(returns_20d.std()) * 100, 4) if len(returns_20d) >= 10 else None
    feat["max_drawdown_20d"] = max_drawdown(close, 20)
    feat["max_drawdown_60d"] = max_drawdown(close, 60)

    # ── Bollinger Bands ──
    bbw, bb_upper, bb_lower = calculate_bbands(df)
    bb_upper_now = float(bb_upper.iloc[-1])
    bb_lower_now = float(bb_lower.iloc[-1])
    bb_range = bb_upper_now - bb_lower_now
    feat["bb_width_pct"] = round(float(bbw.iloc[-1]), 4)
    feat["bb_position_pct"] = round((c_now - bb_lower_now) / bb_range * 100, 2) if bb_range > 0 else 50.0
    feat["bb_width_percentile_252d"] = percentile_rank(bbw, 252)
    feat["flag_bb_squeeze"] = bool(feat["bb_width_percentile_252d"] is not None
                                   and feat["bb_width_percentile_252d"] < 0.15)
    feat["flag_above_bb_upper"] = bool(c_now > bb_upper_now)

    # ── SuperTrend (降級為確認用) ──
    try:
        st_line, st_dir = calculate_supertrend(df)
        st_val = float(st_line.iloc[-1])
        st_direction = int(st_dir.iloc[-1])
        feat["supertrend_bullish"] = bool(st_direction == 1)
        feat["supertrend_distance_pct"] = round((c_now - st_val) / st_val * 100, 4)
    except Exception:
        feat["supertrend_bullish"] = None
        feat["supertrend_distance_pct"] = None
        st_direction = 0

    # ── MFI ──
    mfi_s = calculate_mfi(df)
    feat["mfi14"] = round(float(mfi_s.iloc[-1]), 2)

    # ── AVWAP ──
    avwap = calculate_avwap(df, f"{chosen_ts.year}-01-01")
    if avwap is not None and not avwap.empty and np.isfinite(float(avwap.iloc[-1])):
        avwap_now = float(avwap.iloc[-1])
        feat["avwap_ytd"] = round(avwap_now, 2)
        feat["avwap_dev_pct"] = round((c_now - avwap_now) / avwap_now * 100, 4)
    else:
        feat["avwap_ytd"] = None
        feat["avwap_dev_pct"] = None

    # ── Volume Profile / POC ──
    vp = calculate_volume_profile(df, lookback=60)
    feat.update(vp)

    # ── 短期支撐壓力 ──
    sr = detect_short_term_sr(df, lookback=20)
    feat.update(sr)

    # ── 斐波那契 (摘要) ──
    fib = calculate_fibonacci_summary(df)
    feat.update(fib)

    # ── 缺口 (摘要：最近未回補) ──
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

    # ── 背離偵測 ──
    div_rsi = detect_divergence(close, rsi14, lookback=60)
    feat["flag_bearish_divergence_rsi"] = div_rsi["bearish_divergence"]
    feat["flag_bullish_divergence_rsi"] = div_rsi["bullish_divergence"]

    div_macd = detect_divergence(close, osc, lookback=60)
    feat["flag_bearish_divergence_macd"] = div_macd["bearish_divergence"]
    feat["flag_bullish_divergence_macd"] = div_macd["bullish_divergence"]

    # ── K 線動能型態 ──
    candle_patterns = detect_momentum_candle_patterns(df, n=5)
    feat.update(candle_patterns)

    # ── 圖形型態 ──
    chart_patterns = detect_patterns(df)
    feat["chart_pattern_count"] = len(chart_patterns)
    if chart_patterns:
        feat["chart_pattern_latest"] = chart_patterns[-1].get("pattern")
        feat["chart_pattern_confirmed"] = chart_patterns[-1].get("confirmed", False)
    else:
        feat["chart_pattern_latest"] = None
        feat["chart_pattern_confirmed"] = False

    # ═══════════════════════════════════
    #  外部資料 (網路請求)
    # ═══════════════════════════════════
    latest_overall_ts = df_daily.index[-1]

    # ── 三大法人 (改 20 日) ──
    try:
        chips_multi = get_institutional_multi_days(
            stock_id, trade_date, market_hint=yf_ticker, days=20)
        price_20d_ago = float(df["Close"].iloc[-20]) if len(df) >= 20 else c_now
        inst_feat = compute_institutional_features(chips_multi, c_now, price_20d_ago)
        feat.update(inst_feat)
    except Exception as e:
        feat["inst_data_available"] = False
        feat["inst_error"] = str(e)

    # ── 外資持股比率 ──
    try:
        if yf_ticker.endswith(".TW"):
            fh = get_foreign_holding_ratio(stock_no)
            feat["foreign_holding_pct"] = fh.get("ratio")
        else:
            feat["foreign_holding_pct"] = None
    except Exception:
        feat["foreign_holding_pct"] = None

    # ── 融資融券 ──
    try:
        margin = get_margin_short_data(
            stock_id, trade_date=latest_overall_ts.date(),
            market_hint=yf_ticker, max_back=7)
        if isinstance(margin, dict) and margin.get("error") is None:
            feat["margin_balance"] = margin.get("margin_balance")
            feat["margin_change"] = margin.get("margin_change")
            feat["margin_usage_rate"] = margin.get("margin_usage_rate")
            feat["short_balance"] = margin.get("short_balance")
            feat["short_change"] = margin.get("short_change")
            m_bal = margin.get("margin_balance")
            s_bal = margin.get("short_balance")
            feat["short_margin_ratio"] = (
                round(s_bal / m_bal * 100, 2)
                if m_bal and s_bal and m_bal > 0 else None
            )
            feat["margin_data_available"] = True
        else:
            feat["margin_data_available"] = False
    except Exception:
        feat["margin_data_available"] = False

    # ── 集保/大戶 ──
    try:
        tdcc = get_tdcc_distribution(stock_no, weeks_back=2)
        tdcc_feat = compute_tdcc_features(tdcc)
        feat.update(tdcc_feat)
    except Exception:
        feat["tdcc_available"] = False

    # ── 決策欄位 ──
    resistance = feat.get("fib_nearest_resistance_1") or feat.get("st_resistance")
    support = feat.get("fib_nearest_support_1") or feat.get("st_support")
    decision = compute_decision_fields(c_now, atr_now, resistance, support, st_direction)
    feat.update(decision)

    return feat

# ═══════════════════════════════════════════════════════════
# 15.  TEXT REPORT (精簡版，人類可讀)
# ═══════════════════════════════════════════════════════════

def format_text_report(feat: Dict[str, Any]) -> str:
    """從 AI Features JSON 產出精簡文字報告"""
    if "error" in feat and feat["error"]:
        return f"❌ {feat.get('symbol', '?')}: {feat['error']}"

    lines = []
    SEP = "=" * 56

    lines.append(SEP)
    lines.append(f"  {feat['symbol']}  技術分析摘要")
    lines.append(f"  資料日: {feat['price_date']}  查詢日: {feat['query_date']}")
    lines.append(SEP)

    # 價格 & 趨勢
    lines.append(f"  收盤: {feat['close']}  趨勢: {feat['trend_state']}")
    lines.append(f"  MA20乖離: {feat['ma20_dev_pct']:+.2f}% (pctl={feat.get('ma20_dev_percentile_252d')})")
    lines.append(f"  MA60乖離: {feat['ma60_dev_pct']:+.2f}% (pctl={feat.get('ma60_dev_percentile_252d')})")
    lines.append(f"  52週位置: {feat['pos_52w_pct']:.1f}%")

    # 動能
    lines.append(f"  RSI14: {feat['rsi14']}  KD: {feat['kd_k']}/{feat['kd_d']}")
    lines.append(f"  MACD OSC: {feat['macd_osc']:.4f}  ADX: {feat['adx']}")
    lines.append(f"  ATR14: {feat['atr14']} ({feat['atr14_pct']:.2f}%)")

    # 量能
    lines.append(f"  量比: {feat.get('vol_ratio_5d')}  OBV斜率5d: {feat.get('obv_slope_5d')}")

    # 旗標
    flags = [k for k, v in feat.items() if k.startswith("flag_") and v is True]
    if flags:
        lines.append(f"  🚩 Flags: {', '.join(f.replace('flag_', '') for f in flags)}")

    # 決策
    if feat.get("decision_available"):
        lines.append(f"  停損: {feat.get('atr_stop_loss')} ({feat.get('atr_stop_loss_pct')}%)")
        lines.append(f"  目標: {feat.get('target_resistance')}  R:R={feat.get('risk_reward_ratio')}")

    lines.append(SEP)
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════
# 16.  SECTOR ANALYSIS (補足 app.py 所需功能)
# ═══════════════════════════════════════════════════════════

SECTOR_DICT = {
    "半導體權值": ["2330.TW", "2454.TW", "3711.TW", "3034.TW", "2303.TW"],
    "AI 伺服器": ["2382.TW", "3231.TW", "6669.TW", "2356.TW", "2376.TW"],
    "IC 設計高價": ["3661.TW", "3443.TW", "3529.TW", "3035.TW", "4966.TW"],
    "金融保險": ["2881.TW", "2882.TW", "2886.TW", "2891.TW", "5880.TW"],
    "航運貨櫃": ["2603.TW", "2609.TW", "2615.TW", "2637.TW", "2618.TW"],
    "重電綠能": ["1513.TW", "1519.TW", "1503.TW", "1514.TW", "6806.TW"],
    "散熱模組": ["3017.TW", "3324.TW", "3338.TW", "6230.TW", "2421.TW"],
}

def analyze_sector_performance(sector_name: str, as_of_date=None, custom_tickers=None) -> str:
    """
    族群漲跌快篩：讀取族群內股票，產出表格摘要
    """
    target_list = custom_tickers if custom_tickers else SECTOR_DICT.get(sector_name, [])
    
    if not target_list:
        return f"❌ 族群 {sector_name} 無股票清單。"

    results = []
    
    # 批次抓取並建立摘要
    for stock_id in target_list:
        try:
            # 使用既有的 build_ai_features 取得數據
            feat = build_ai_features(stock_id, as_of_date=as_of_date)
            
            if feat.get("error"):
                results.append({
                    "Symbol": _strip_suffix(stock_id), 
                    "Close": "-", "Trend": "Error", "Vol_R": "-", "KD": "-/-", "Score": 0
                })
                continue

            # 簡單評分邏輯 (趨勢多頭+2, 價漲量增+1, KD金叉+1, 法人同買+2)
            score = 0
            if feat.get("trend_state") == "uptrend": score += 2
            if feat.get("flag_price_up_vol_up"): score += 1
            if feat.get("flag_kd_golden_cross"): score += 1
            if feat.get("flag_inst_consensus_buy"): score += 2
            
            summary = {
                "Symbol": _strip_suffix(feat.get("symbol", stock_id)),
                "Close": feat.get("close", "-"),
                "Trend": feat.get("trend_state", "-"),
                "MA20_Dev": f"{feat.get('ma20_dev_pct', 0):+.2f}%",
                "Vol_R": feat.get("vol_ratio_5d", "-"),
                "KD": f"{feat.get('kd_k',0):.0f}/{feat.get('kd_d',0):.0f}",
                "Score": score
            }
            results.append(summary)

        except Exception as e:
            results.append({"Symbol": stock_id, "Trend": "Exception", "Score": -1})

    # 排序：分數高到低
    results.sort(key=lambda x: x.get("Score", 0), reverse=True)

    # 轉為文字表格輸出
    lines = []
    lines.append(f"📊 族群快篩：{sector_name}")
    d_str = str(as_of_date) if as_of_date else "Today"
    lines.append(f"日期: {d_str}")
    lines.append("-" * 65)
    # Formatted Header
    lines.append(f"{'代號':<10} {'價格':<8} {'趨勢':<12} {'MA20乖離':<10} {'量比':<6} {'KD':<8} {'分數':<4}")
    lines.append("-" * 65)

    for r in results:
        sym = str(r.get("Symbol", ""))
        pri = str(r.get("Close", ""))
        tre = str(r.get("Trend", ""))
        dev = str(r.get("MA20_Dev", ""))
        vol = str(r.get("Vol_R", ""))
        kd = str(r.get("KD", ""))
        sco = str(r.get("Score", ""))
        
        lines.append(f"{sym:<10} {pri:<8} {tre:<12} {dev:<10} {vol:<6} {kd:<8} {sco:<4}")
    
    lines.append("-" * 65)
    lines.append("(分數說明: 多頭+2, 價漲量增+1, KD金叉+1, 土洋合買+2)")
    
    return "\n".join(lines)

# ═══════════════════════════════════════════════════════════
# 17.  CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════

def analyze_stock_technical(stock_id: str, as_of_date=None) -> dict:
    """
    修改版主入口：回傳 Dictionary 包含兩種報告格式
    """
    # 1. 計算所有數值與指標 (AI Data)
    feat = build_ai_features(stock_id, as_of_date)
    
    # 2. 產生文字報告 (Human Data)
    text = format_text_report(feat)
    
    # 3. 回傳包含兩者的字典
    return {
        "human_report": text,
        "ai_report": feat
    }

if __name__ == "__main__":
    import sys
    sid = sys.argv[1] if len(sys.argv) > 1 else "2330"
    result = analyze_stock_technical(sid)
    print(result["human_report"])
    print("\n=== AI JSON Data ===")
    print(json.dumps(result["ai_report"], default=str))
