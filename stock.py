import io
import re
from datetime import datetime, timedelta, date as dt_date

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Optional: helps some SSL envs
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass


# ---------------------------
# Helpers
# ---------------------------
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
        if s in ("", "--", "—", "NaN", "nan", "None", "N/A", "n/a"):
            return None
        return int(float(s))
    except Exception:
        return None


def _resolve_yf_ticker(stock_id: str) -> str:
    s = stock_id.strip().upper()
    if s.endswith(".TW") or s.endswith(".TWO"):
        return s
    return f"{s}.TW"  # default .TW, fallback to .TWO later


def _parse_as_of(as_of_date):
    if as_of_date is None:
        return None
    if isinstance(as_of_date, dt_date):
        return as_of_date
    if isinstance(as_of_date, str):
        try:
            return datetime.strptime(as_of_date.strip(), "%Y-%m-%d").date()
        except Exception:
            return None
    return None


def _to_lots(v_shares_or_shareslike):
    v = _safe_int(v_shares_or_shareslike)
    if v is None:
        return None
    return int(round(v / 1000))


def _fmt_net_lots(lots):
    if lots is None:
        return "n/a"
    if lots > 0:
        return f"🔴 買超 {lots:,} 張"
    if lots < 0:
        return f"🟢 賣超 {abs(lots):,} 張"
    return "➖ 無變動"


def _fmt_delta(today_lots, prev_lots):
    if today_lots is None or prev_lots is None:
        return ""
    d = today_lots - prev_lots
    if d == 0:
        return "（較前一交易日 0 張）"
    sign = "+" if d > 0 else ""
    return f"（較前一交易日 {sign}{d:,} 張）"


# ---------------------------
# Candle (emoji + pattern)
# ---------------------------
def get_k_status(open_p, close_p, high_p=None, low_p=None):
    try:
        o = float(open_p)
        c = float(close_p)
        if high_p is not None and low_p is not None:
            h = float(high_p)
            l = float(low_p)
            rng = h - l
            if rng > 0 and abs(c - o) / rng <= 0.10:
                return "➖ 十字"
        return "🔴 紅棒" if c > o else ("🟢 綠棒" if c < o else "➖ 十字")
    except Exception:
        return "➖"


def describe_candle(open_p, high_p, low_p, close_p):
    o, h, l, c = float(open_p), float(high_p), float(low_p), float(close_p)
    rng = h - l
    if rng <= 0:
        return "一字線"

    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l

    body_r = body / rng
    upper_r = upper / rng
    lower_r = lower / rng
    bullish = c > o

    if body_r <= 0.10:
        if upper_r >= 0.65 and lower_r <= 0.15:
            return "墓碑十字"
        if lower_r >= 0.65 and upper_r <= 0.15:
            return "蜻蜓十字"
        return "十字星"

    if body_r >= 0.60:
        return "長紅K" if bullish else "長黑K"

    if lower_r >= 0.60 and body_r <= 0.30 and upper_r <= 0.20:
        return "錘頭線" if bullish else "吊人線"

    if upper_r >= 0.60 and body_r <= 0.30 and lower_r <= 0.20:
        return "倒錘線" if bullish else "流星線"

    base = "小紅K" if bullish else "小黑K"
    if upper_r >= 0.35 and lower_r >= 0.35:
        return base + "（上下影明顯）"
    if upper_r >= 0.45:
        return base + "（上影偏長）"
    if lower_r >= 0.45:
        return base + "（下影偏長）"
    return base


# ---------------------------
# Institutional: TWSE / TPEx CSV parsing
# ---------------------------
def _parse_twse_t86_csv(text):
    text = text.replace("\r", "").replace("=", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    start = None
    for i, ln in enumerate(lines):
        if ("證券代號" in ln) and ("證券名稱" in ln):
            start = i
            break
    if start is None:
        return None

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("說明") or lines[j].startswith("備註"):
            end = j
            break

    csv_text = "\n".join(lines[start:end])
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return None


def _parse_tpex_csv(text):
    text = text.replace("\ufeff", "").replace("\r", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    start = None
    for i, ln in enumerate(lines):
        if ("代號" in ln) and ("名稱" in ln):
            start = i
            break
    if start is None:
        return None

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("說明") or lines[j].startswith("備註"):
            end = j
            break

    csv_text = "\n".join(lines[start:end])
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return None


def get_institutional_data(stock_id, trade_date, market_hint=None):
    """
    回傳 foreign/trust/dealer 皆為「股」（shares）
    顯示時再 /1000 轉「張」
    外資抓不到買賣超 → 用(買進-賣出)算
    """
    stock_no = stock_id.strip().upper().replace(".TW", "").replace(".TWO", "")

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    prefer_twse = True
    if isinstance(market_hint, str) and market_hint.upper().endswith(".TWO"):
        prefer_twse = False

    last_error = None

    def try_twse(d_):
        url = f"https://www.twse.com.tw/fund/T86?response=csv&date={d_.strftime('%Y%m%d')}&selectType=ALLBUT0999"
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200 or len(r.text) < 200:
            return None, f"TWSE HTTP {r.status_code}"
        if "沒有符合條件的資料" in r.text or "很抱歉" in r.text:
            return None, "TWSE 無資料(可能休市)"
        df = _parse_twse_t86_csv(r.text)
        return df, None if df is not None else "TWSE 解析失敗"

    def try_tpex(d_):
        roc_year = d_.year - 1911
        roc_date = f"{roc_year:03d}/{d_.month:02d}/{d_.day:02d}"
        url = (
            "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"
            f"?l=zh-tw&o=csv&se=EW&t=D&d={roc_date}&s=0,asc"
        )
        headers2 = dict(headers)
        headers2["Referer"] = "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge.php"
        r = requests.get(url, headers=headers2, timeout=15)
        if r.status_code != 200 or len(r.text) < 200:
            return None, f"TPEx HTTP {r.status_code}"
        if "沒有符合條件的資料" in r.text or "很抱歉" in r.text:
            return None, "TPEx 無資料(可能休市)"
        df = _parse_tpex_csv(r.text)
        return df, None if df is not None else "TPEx 解析失敗"

    markets = [("TWSE", try_twse), ("TPEx", try_tpex)]
    if not prefer_twse:
        markets = [("TPEx", try_tpex), ("TWSE", try_twse)]

    def norm(s: str) -> str:
        s = str(s).replace("\ufeff", "")
        s = s.replace("（", "(").replace("）", ")")
        s = re.sub(r"\s+", "", s)
        return s

    def build_colmap(cols):
        return {norm(c): c for c in cols}

    def find_first(colmap, predicate):
        for nk, orig in colmap.items():
            if predicate(nk):
                return orig
        return None

    def is_foreign_block(nk: str) -> bool:
        return ("外陸資" in nk) or ("外資及陸資" in nk) or ("外資" in nk)

    def is_foreign_usable(nk: str) -> bool:
        if "外資自營商" in nk and "不含外資自營商" not in nk:
            return False
        return is_foreign_block(nk)

    for back in range(0, 10):
        d = trade_date - timedelta(days=back)

        for mkt_name, fn in markets:
            df, err = fn(d)
            if df is None:
                last_error = err
                continue

            cols = list(df.columns)
            colmap = build_colmap(cols)

            code_col = None
            for c in cols:
                if norm(c) in ("證券代號", "代號"):
                    code_col = c
                    break
            if code_col is None:
                last_error = f"{mkt_name} 欄位找不到代號欄"
                continue

            row = df[df[code_col].astype(str).str.strip() == stock_no]
            if row.empty:
                last_error = f"{mkt_name} 當日資料找不到 {stock_no}"
                continue

            foreign_col = find_first(colmap, lambda nk: is_foreign_usable(nk) and ("買賣超" in nk))
            trust_col = find_first(colmap, lambda nk: ("投信" in nk) and ("買賣超" in nk)) \
                        or find_first(colmap, lambda nk: "投信" in nk)

            dealer_total_col = find_first(
                colmap,
                lambda nk: ("自營商" in nk)
                and ("買賣超" in nk)
                and ("外資" not in nk)
                and ("自行買賣" not in nk)
                and ("避險" not in nk),
            )
            dealer_self_col = find_first(colmap, lambda nk: ("自營商" in nk) and ("自行買賣" in nk) and ("買賣超" in nk) and ("外資" not in nk))
            dealer_hedge_col = find_first(colmap, lambda nk: ("自營商" in nk) and ("避險" in nk) and ("買賣超" in nk) and ("外資" not in nk))

            foreign = _safe_int(row.iloc[0][foreign_col]) if foreign_col else None
            trust = _safe_int(row.iloc[0][trust_col]) if trust_col else None

            dealer = None
            if dealer_total_col:
                dealer = _safe_int(row.iloc[0][dealer_total_col])
            elif dealer_self_col or dealer_hedge_col:
                a = _safe_int(row.iloc[0][dealer_self_col]) if dealer_self_col else 0
                b = _safe_int(row.iloc[0][dealer_hedge_col]) if dealer_hedge_col else 0
                dealer = (a or 0) + (b or 0)

            if foreign is None:
                buy_col = find_first(
                    colmap,
                    lambda nk: is_foreign_usable(nk)
                    and (("買進" in nk) or ("買入" in nk))
                    and ("買賣超" not in nk),
                )
                sell_col = find_first(
                    colmap,
                    lambda nk: is_foreign_usable(nk)
                    and ("賣出" in nk)
                    and ("買賣超" not in nk),
                )
                buy_v = _safe_int(row.iloc[0][buy_col]) if buy_col else None
                sell_v = _safe_int(row.iloc[0][sell_col]) if sell_col else None
                if buy_v is not None and sell_v is not None:
                    foreign = buy_v - sell_v

            yf_suffix = ".TW" if mkt_name == "TWSE" else ".TWO"
            return {
                "id": f"{stock_no}{yf_suffix}",
                "date": d.strftime("%Y-%m-%d"),
                "foreign": foreign,  # 股
                "trust": trust,      # 股
                "dealer": dealer,    # 股
                "error": None,
            }

    return {"error": last_error or "未知錯誤", "id": f"{stock_no}.TW"}


# ---------------------------
# Margin trading (融資融券)
# ---------------------------
def _parse_twse_margn_csv(text: str):
    """
    TWSE MI_MARGN CSV：有時候會有 =、\r、說明列；用「證券代號」當表頭起點
    """
    text = text.replace("\r", "").replace("=", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    start = None
    for i, ln in enumerate(lines):
        if ("證券代號" in ln) and ("證券名稱" in ln):
            start = i
            break
    if start is None:
        return None

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("說明") or lines[j].startswith("備註"):
            end = j
            break

    csv_text = "\n".join(lines[start:end])
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return None


def get_margin_data(stock_id, trade_date, market_hint=None):
    """
    回傳:
      finance_balance / short_balance：股(share) -> 顯示時 /1000 張
    """
    stock_no = stock_id.strip().upper().replace(".TW", "").replace(".TWO", "")

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    prefer_twse = True
    if isinstance(market_hint, str) and market_hint.upper().endswith(".TWO"):
        prefer_twse = False

    last_error = None

    def norm(s: str) -> str:
        s = str(s).replace("\ufeff", "")
        s = s.replace("（", "(").replace("）", ")")
        s = re.sub(r"\s+", "", s)
        return s

    def find_col(df, want_keys):
        mp = {norm(c): c for c in df.columns}
        for k in want_keys:
            for nk, orig in mp.items():
                if k in nk:
                    return orig
        return None

    def try_twse(d_):
        # MI_MARGN：融資融券餘額
        url = f"https://www.twse.com.tw/exchangeReport/MI_MARGN?response=csv&date={d_.strftime('%Y%m%d')}&selectType=ALL"
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200 or len(r.text) < 200:
            return None, f"TWSE HTTP {r.status_code}"
        if "沒有符合條件的資料" in r.text or "很抱歉" in r.text:
            return None, "TWSE 無資料(可能休市)"
        df = _parse_twse_margn_csv(r.text)
        return df, None if df is not None else "TWSE 解析失敗"

    def try_tpex(d_):
        roc_year = d_.year - 1911
        roc_date = f"{roc_year:03d}/{d_.month:02d}/{d_.day:02d}"
        url = (
            "https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php"
            f"?l=zh-tw&o=csv&t=D&d={roc_date}&s=0,asc"
        )
        headers2 = dict(headers)
        headers2["Referer"] = "https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal.php"
        r = requests.get(url, headers=headers2, timeout=15)
        if r.status_code != 200 or len(r.text) < 200:
            return None, f"TPEx HTTP {r.status_code}"
        if "沒有符合條件的資料" in r.text or "很抱歉" in r.text:
            return None, "TPEx 無資料(可能休市)"
        df = _parse_tpex_csv(r.text)
        return df, None if df is not None else "TPEx 解析失敗"

    markets = [("TWSE", try_twse), ("TPEx", try_tpex)]
    if not prefer_twse:
        markets = [("TPEx", try_tpex), ("TWSE", try_twse)]

    for back in range(0, 10):
        d = trade_date - timedelta(days=back)
        for mkt_name, fn in markets:
            df, err = fn(d)
            if df is None:
                last_error = err
                continue

            # 代號欄
            code_col = None
            for c in df.columns:
                if norm(c) in ("證券代號", "代號"):
                    code_col = c
                    break
            if code_col is None:
                last_error = f"{mkt_name} 欄位找不到代號欄"
                continue

            row = df[df[code_col].astype(str).str.strip() == stock_no]
            if row.empty:
                last_error = f"{mkt_name} 當日資料找不到 {stock_no}"
                continue

            # 欄位名稱不同站點會不太一樣：用關鍵字找
            # 目標：融資餘額、融券餘額
            fin_col = find_col(df, ["融資餘額", "融資餘額(股)", "融資餘額股數", "融資餘額股"])
            sh_col = find_col(df, ["融券餘額", "融券餘額(股)", "融券餘額股數", "融券餘額股"])

            fin = _safe_int(row.iloc[0][fin_col]) if fin_col else None
            sh = _safe_int(row.iloc[0][sh_col]) if sh_col else None

            yf_suffix = ".TW" if mkt_name == "TWSE" else ".TWO"
            return {
                "id": f"{stock_no}{yf_suffix}",
                "date": d.strftime("%Y-%m-%d"),
                "finance_balance": fin,  # 股
                "short_balance": sh,     # 股
                "error": None,
            }

    return {"error": last_error or "未知錯誤", "id": f"{stock_no}.TW"}


# ---------------------------
# Indicators
# ---------------------------
def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_adx(df, period=14):
    df = df.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)

    df["UpMove"] = df["High"] - df["High"].shift(1)
    df["DownMove"] = df["Low"].shift(1) - df["Low"]
    df["+DM"] = np.where((df["UpMove"] > df["DownMove"]) & (df["UpMove"] > 0), df["UpMove"], 0.0)
    df["-DM"] = np.where((df["DownMove"] > df["UpMove"]) & (df["DownMove"] > 0), df["DownMove"], 0.0)

    alpha = 1 / period
    df["TR_s"] = df["TR"].ewm(alpha=alpha, adjust=False).mean()
    df["+DM_s"] = df["+DM"].ewm(alpha=alpha, adjust=False).mean()
    df["-DM_s"] = df["-DM"].ewm(alpha=alpha, adjust=False).mean()

    df["+DI"] = 100 * (df["+DM_s"] / df["TR_s"].replace(0, np.nan))
    df["-DI"] = 100 * (df["-DM_s"] / df["TR_s"].replace(0, np.nan))

    df["DX"] = 100 * abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"]).replace(0, np.nan)
    df["ADX"] = df["DX"].ewm(alpha=alpha, adjust=False).mean()
    return df["ADX"], df["+DI"], df["-DI"]


def calculate_bbw(df, period=20, std_dev=2):
    ma = df["Close"].rolling(window=period).mean()
    std = df["Close"].rolling(window=period).std()
    upper = ma + (std * std_dev)
    lower = ma - (std * std_dev)
    bbw = (upper - lower) / ma * 100
    return bbw, upper, lower


def calculate_mfi(df, period=14):
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    raw_money_flow = typical_price * df["Volume"]

    pos = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0.0)
    neg = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0.0)

    pos_sum = pd.Series(pos, index=df.index).rolling(period).sum()
    neg_sum = pd.Series(neg, index=df.index).rolling(period).sum().abs()

    ratio = pos_sum / neg_sum.replace(0, np.nan)
    return 100 - (100 / (1 + ratio))


def calculate_avwap(df, anchor_date):
    mask = df.index >= anchor_date
    if not mask.any():
        return None
    sub = df.loc[mask].copy()
    typical = (sub["High"] + sub["Low"] + sub["Close"]) / 3
    cum_pv = (typical * sub["Volume"]).cumsum()
    cum_v = sub["Volume"].cumsum().replace(0, np.nan)
    return cum_pv / cum_v


# ---------------------------
# Pattern detection (no chart)
# ---------------------------
def _find_pivots(df: pd.DataFrame, left: int = 3, right: int = 3):
    if df is None or df.empty or len(df) < (left + right + 5):
        return [], []

    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    idx = df.index

    hp, lp = [], []
    n = len(df)

    for i in range(left, n - right):
        h_win = highs[i - left : i + right + 1]
        l_win = lows[i - left : i + right + 1]

        if np.isfinite(highs[i]) and highs[i] == np.nanmax(h_win):
            if not hp or hp[-1][0] != i - 1:
                hp.append((i, idx[i], highs[i]))

        if np.isfinite(lows[i]) and lows[i] == np.nanmin(l_win):
            if not lp or lp[-1][0] != i - 1:
                lp.append((i, idx[i], lows[i]))

    return hp, lp


def _between(df, a_pos: int, b_pos: int) -> pd.DataFrame:
    lo = min(a_pos, b_pos)
    hi = max(a_pos, b_pos)
    return df.iloc[lo : hi + 1]


def _pct_diff(a: float, b: float) -> float:
    if a == 0 or not np.isfinite(a) or not np.isfinite(b):
        return np.inf
    return abs(a - b) / abs(a)


def detect_double_top(
    df: pd.DataFrame,
    lookback: int = 120,
    pivot_lr: int = 3,
    peak_tol: float = 0.018,
    min_gap: int = 8,
    max_gap: int = 60,
    confirm_margin: float = 0.003,
):
    if df is None or df.empty:
        return None

    sub = df.tail(lookback).copy()
    hp, _ = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(hp) < 2:
        return None

    p2 = hp[-1]
    p1 = None
    for cand in reversed(hp[:-1]):
        gap = p2[0] - cand[0]
        if min_gap <= gap <= max_gap:
            p1 = cand
            break
    if p1 is None:
        return None

    peak1 = float(p1[2])
    peak2 = float(p2[2])
    if _pct_diff(peak1, peak2) > peak_tol:
        return None

    mid = _between(sub, p1[0], p2[0])
    neckline = float(mid["Low"].min())

    close_now = float(sub["Close"].iloc[-1])
    status = "形成中"
    confirmed = False
    if close_now < neckline * (1 - confirm_margin):
        status = "已確認（跌破頸線）"
        confirmed = True

    height = max(peak1, peak2) - neckline
    target = neckline - height

    return {
        "pattern": "M頭（Double Top）",
        "status": status,
        "confirmed": confirmed,
        "peak1_date": str(p1[1].date()),
        "peak2_date": str(p2[1].date()),
        "peak1": peak1,
        "peak2": peak2,
        "neckline": neckline,
        "target": target,
    }


def detect_double_bottom(
    df: pd.DataFrame,
    lookback: int = 160,
    pivot_lr: int = 3,
    trough_tol: float = 0.020,
    min_gap: int = 8,
    max_gap: int = 80,
    confirm_margin: float = 0.003,
):
    if df is None or df.empty:
        return None

    sub = df.tail(lookback).copy()
    _, lp = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(lp) < 2:
        return None

    t2 = lp[-1]
    t1 = None
    for cand in reversed(lp[:-1]):
        gap = t2[0] - cand[0]
        if min_gap <= gap <= max_gap:
            t1 = cand
            break
    if t1 is None:
        return None

    trough1 = float(t1[2])
    trough2 = float(t2[2])
    if _pct_diff(trough1, trough2) > trough_tol:
        return None

    mid = _between(sub, t1[0], t2[0])
    neckline = float(mid["High"].max())

    close_now = float(sub["Close"].iloc[-1])
    status = "形成中"
    confirmed = False
    if close_now > neckline * (1 + confirm_margin):
        status = "已確認（突破頸線）"
        confirmed = True

    height = neckline - min(trough1, trough2)
    target = neckline + height

    return {
        "pattern": "W底（Double Bottom）",
        "status": status,
        "confirmed": confirmed,
        "trough1_date": str(t1[1].date()),
        "trough2_date": str(t2[1].date()),
        "trough1": trough1,
        "trough2": trough2,
        "neckline": neckline,
        "target": target,
    }


def detect_sym_triangle(
    df: pd.DataFrame,
    lookback: int = 180,
    pivot_lr: int = 3,
    need_points: int = 3,
    tighten_ratio: float = 0.65,
    breakout_margin: float = 0.003,
):
    if df is None or df.empty:
        return None

    sub = df.tail(lookback).copy()
    hp, lp = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(hp) < need_points or len(lp) < need_points:
        return None

    highs = hp[-need_points:]
    lows = lp[-need_points:]

    highs_prices = [x[2] for x in highs]
    lows_prices = [x[2] for x in lows]

    if not (highs_prices[0] > highs_prices[1] > highs_prices[2]):
        return None
    if not (lows_prices[0] < lows_prices[1] < lows_prices[2]):
        return None

    xh = np.array([p[0] for p in highs], dtype=float)
    yh = np.array([p[2] for p in highs], dtype=float)
    xl = np.array([p[0] for p in lows], dtype=float)
    yl = np.array([p[2] for p in lows], dtype=float)

    ah, bh = np.polyfit(xh, yh, 1)
    al, bl = np.polyfit(xl, yl, 1)

    if not (ah < 0 and al > 0):
        return None

    early = sub.iloc[: max(30, len(sub) // 3)]
    late = sub.iloc[-max(30, len(sub) // 3) :]
    early_range = float(early["High"].max() - early["Low"].min())
    late_range = float(late["High"].max() - late["Low"].min())

    if early_range <= 0 or late_range / early_range > tighten_ratio:
        return None

    last_pos = len(sub) - 1
    upper_now = ah * last_pos + bh
    lower_now = al * last_pos + bl
    close_now = float(sub["Close"].iloc[-1])

    status = "形成中"
    direction = "未突破"
    if close_now > upper_now * (1 + breakout_margin):
        status = "已確認（向上突破）"
        direction = "向上"
    elif close_now < lower_now * (1 - breakout_margin):
        status = "已確認（向下跌破）"
        direction = "向下"

    return {
        "pattern": "收斂三角（Sym Triangle）",
        "status": status,
        "direction": direction,
        "upper_now": float(upper_now),
        "lower_now": float(lower_now),
        "high_pivots": [(str(p[1].date()), float(p[2])) for p in highs],
        "low_pivots": [(str(p[1].date()), float(p[2])) for p in lows],
        "range_shrink": float(late_range / early_range),
    }


def detect_patterns(df: pd.DataFrame):
    res = []
    m = detect_double_top(df)
    if m:
        res.append(m)
    w = detect_double_bottom(df)
    if w:
        res.append(w)
    t = detect_sym_triangle(df)
    if t:
        res.append(t)
    return res


# ---------------------------
# Main analysis (returns string for Streamlit)
# ---------------------------
def analyze_stock_technical(stock_id: str, as_of_date=None) -> str:
    stock_id = stock_id.strip().upper()
    if not stock_id:
        return "請輸入股票代號"

    req_date = _parse_as_of(as_of_date)

    yf_ticker = _resolve_yf_ticker(stock_id)
    out = []
    out.append(f"🔄 正在分析 {stock_id} ... (連線 Yahoo Finance)")

    df_daily = yf.download(yf_ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
    df_daily = clean_yf_columns(df_daily)

    if df_daily.empty and yf_ticker.endswith(".TW"):
        alt = yf_ticker.replace(".TW", ".TWO")
        out.append(f"⚠️ .TW 無資料，改試上櫃代碼: {alt}")
        yf_ticker = alt
        df_daily = yf.download(yf_ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
        df_daily = clean_yf_columns(df_daily)

    if df_daily.empty:
        return "\n".join(out + [f"❌ 找不到股票代號 {stock_id} (Yahoo Finance 無數據)"])

    # 日期系統：截到 <= 你選的日期（最近可用交易日）
    if req_date is not None:
        df_daily = df_daily[df_daily.index.date <= req_date]
        if df_daily.empty:
            return "\n".join(out + [f"❌ 你選的基準日 {req_date} 太早，1y 區間內無可用交易日資料"])

    # 週/月資料
    df_weekly = yf.download(yf_ticker, period="1y", interval="1wk", progress=False, auto_adjust=False)
    df_weekly = clean_yf_columns(df_weekly)
    if req_date is not None and df_weekly is not None and not df_weekly.empty:
        df_weekly = df_weekly[df_weekly.index.date <= req_date]

    df_monthly = yf.download(yf_ticker, period="2y", interval="1mo", progress=False, auto_adjust=False)
    df_monthly = clean_yf_columns(df_monthly)
    if req_date is not None and df_monthly is not None and not df_monthly.empty:
        df_monthly = df_monthly[df_monthly.index.date <= req_date]

    latest_daily = df_daily.iloc[-1]
    prev_daily = df_daily.iloc[-2] if len(df_daily) >= 2 else latest_daily

    latest_trade_date = df_daily.index[-1].date()
    prev_trade_date = df_daily.index[-2].date() if len(df_daily) >= 2 else None

    # 三大法人：當日 + 前一交易日（若能取到）
    chips_today = get_institutional_data(stock_id, trade_date=latest_trade_date, market_hint=yf_ticker)
    chips_prev = None
    if prev_trade_date is not None:
        chips_prev = get_institutional_data(stock_id, trade_date=prev_trade_date, market_hint=yf_ticker)
        if (
            isinstance(chips_prev, dict)
            and isinstance(chips_today, dict)
            and chips_prev.get("error") is None
            and chips_today.get("error") is None
            and chips_prev.get("date") == chips_today.get("date")
        ):
            chips_prev = None

    # 融資融券：最新（同 trade_date），可選擇抓前一交易日做比較
    margin_today = get_margin_data(stock_id, trade_date=latest_trade_date, market_hint=yf_ticker)
    margin_prev = None
    if prev_trade_date is not None:
        margin_prev = get_margin_data(stock_id, trade_date=prev_trade_date, market_hint=yf_ticker)
        if (
            isinstance(margin_prev, dict)
            and isinstance(margin_today, dict)
            and margin_prev.get("error") is None
            and margin_today.get("error") is None
            and margin_prev.get("date") == margin_today.get("date")
        ):
            margin_prev = None

    out.append("")
    out.append("=" * 50)
    out.append(f"📊 {yf_ticker} 專業版數據表 (含三大法人)")
    out.append(f"📅 資料日期: {df_daily.index[-1].strftime('%Y-%m-%d')}")
    if req_date is not None:
        out.append(f"🗓️ 查詢基準日: {req_date.strftime('%Y-%m-%d')}（實際落在最近交易日）")
    out.append("=" * 50)

    # 近 5 日
    out.append("📋 近 5 日交易紀錄:")
    out.append(f"{'日期':<12} {'收盤':<12} {'漲跌':<12} {'K棒'}")
    out.append("-" * 50)
    recent_5 = df_daily.tail(5)
    for idx, row in recent_5.iterrows():
        date_str = idx.strftime("%m-%d")
        close_p = float(row["Close"])
        open_p = float(row["Open"])
        high_p = float(row["High"])
        low_p = float(row["Low"])
        loc = df_daily.index.get_loc(idx)
        change = close_p - float(df_daily.iloc[loc - 1]["Close"]) if loc > 0 else 0.0
        out.append(f"{date_str:<12} {close_p:<12.2f} {change:<+12.2f} {get_k_status(open_p, close_p, high_p, low_p)}")
    out.append("-" * 50)

    # 當日K棒型態
    out.append(f"🕯 當日K棒型態: {describe_candle(latest_daily['Open'], latest_daily['High'], latest_daily['Low'], latest_daily['Close'])}")
    out.append("-" * 50)

    # 三大法人 + 前一交易日差
    out.append("💰 籌碼面 (三大法人):")
    if isinstance(chips_today, dict) and chips_today.get("error") is None:
        prev_ok = isinstance(chips_prev, dict) and (chips_prev.get("error") is None)

        f_today = _to_lots(chips_today.get("foreign"))
        t_today = _to_lots(chips_today.get("trust"))
        d_today = _to_lots(chips_today.get("dealer"))

        f_prev = _to_lots(chips_prev.get("foreign")) if prev_ok else None
        t_prev = _to_lots(chips_prev.get("trust")) if prev_ok else None
        d_prev = _to_lots(chips_prev.get("dealer")) if prev_ok else None

        out.append(f"• 外資 : {_fmt_net_lots(f_today)}{_fmt_delta(f_today, f_prev)}")
        out.append(f"• 投信 : {_fmt_net_lots(t_today)}{_fmt_delta(t_today, t_prev)}")
        out.append(f"• 自營商: {_fmt_net_lots(d_today)}{_fmt_delta(d_today, d_prev)}")

        if prev_ok:
            out.append(f"  (日期: {chips_today.get('date')}；比較基準: {chips_prev.get('date')}；市場代碼推定: {chips_today.get('id')})")
        else:
            out.append(f"  (日期: {chips_today.get('date')}；市場代碼推定: {chips_today.get('id')}；前一交易日比較: n/a)")
    else:
        out.append(f"⚠️ 無法抓取三大法人數據 ({chips_today.get('error') if isinstance(chips_today, dict) else '未知錯誤'})")
    out.append("-" * 50)

    # 融資融券（最新）
    out.append("💸 融資融券（散戶槓桿指標）:")
    if isinstance(margin_today, dict) and margin_today.get("error") is None:
        prev_ok = isinstance(margin_prev, dict) and (margin_prev.get("error") is None)

        fin_today = _to_lots(margin_today.get("finance_balance"))
        sh_today = _to_lots(margin_today.get("short_balance"))

        fin_prev = _to_lots(margin_prev.get("finance_balance")) if prev_ok else None
        sh_prev = _to_lots(margin_prev.get("short_balance")) if prev_ok else None

        if fin_today is None:
            out.append("• 融資餘額: n/a")
        else:
            out.append(f"• 融資餘額: {fin_today:,} 張{_fmt_delta(fin_today, fin_prev)}")

        if sh_today is None:
            out.append("• 融券餘額: n/a")
        else:
            out.append(f"• 融券餘額: {sh_today:,} 張{_fmt_delta(sh_today, sh_prev)}")

        if prev_ok:
            out.append(f"  (日期: {margin_today.get('date')}；比較基準: {margin_prev.get('date')}；市場代碼推定: {margin_today.get('id')})")
        else:
            out.append(f"  (日期: {margin_today.get('date')}；市場代碼推定: {margin_today.get('id')}；前一交易日比較: n/a)")
    else:
        out.append("⚠️ 無法抓取融資融券 (抓取失敗/休市/資料源未回應)")
    out.append("-" * 50)

    # 長線趨勢
    out.append("📈 長線趨勢:")
    if df_weekly is not None and (not df_weekly.empty) and "Close" in df_weekly.columns:
        ma20_week = df_weekly["Close"].rolling(20).mean().iloc[-1]
        out.append(f"• [週 K] 收: {float(df_weekly.iloc[-1]['Close']):.2f} | 20週均價: {float(ma20_week):.2f}")
    if df_monthly is not None and (not df_monthly.empty) and "Close" in df_monthly.columns:
        out.append(f"• [月 K] 收: {float(df_monthly.iloc[-1]['Close']):.2f}")
    out.append("-" * 50)

    # 基礎指標
    out.append("🔍 基礎指標:")
    ma5 = df_daily["Close"].rolling(5).mean().iloc[-1]
    ma20 = df_daily["Close"].rolling(20).mean().iloc[-1]
    ma60 = df_daily["Close"].rolling(60).mean().iloc[-1]
    out.append(f"• 均線: MA5={float(ma5):.2f}, MA20={float(ma20):.2f}, MA60={float(ma60):.2f}")

    # Volume fallback（當日 volume=0/NaN -> 用前一交易日）
    vol_today = float(latest_daily["Volume"]) if "Volume" in latest_daily else float("nan")
    vol_yesterday = float(prev_daily["Volume"]) if "Volume" in prev_daily else float("nan")

    use_prev_vol = (pd.isna(vol_today) or vol_today == 0) and len(df_daily) >= 2
    if use_prev_vol:
        vol_used = vol_yesterday
        vol_prev_used = float(df_daily.iloc[-3]["Volume"]) if len(df_daily) >= 3 else vol_yesterday
        vol_note = f" (改用前一交易日量 {df_daily.index[-2].strftime('%Y-%m-%d')})"
    else:
        vol_used = vol_today
        vol_prev_used = vol_yesterday
        vol_note = ""

    vol_in_lots = int(vol_used / 1000) if not pd.isna(vol_used) else 0
    vol_diff = int((vol_used - vol_prev_used) / 1000) if (not pd.isna(vol_used) and not pd.isna(vol_prev_used)) else 0
    vol_status = "量增" if (not pd.isna(vol_used) and not pd.isna(vol_prev_used) and vol_used > vol_prev_used) else "量縮"
    out.append(f"• 成交量: {vol_in_lots:,} 張 ({vol_status}, 較昨 {vol_diff:+,} 張){vol_note}")

    rsi6 = calculate_rsi(df_daily["Close"], 6).iloc[-1]

    low_min = df_daily["Low"].rolling(9).min()
    high_max = df_daily["High"].rolling(9).max()
    rsv = (df_daily["Close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()

    ema12 = df_daily["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df_daily["Close"].ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    osc = dif - dea

    out.append(f"• RSI(6): {float(rsi6):.2f}")
    out.append(f"• KD(9,3,3): K={float(k.iloc[-1]):.2f}, D={float(d.iloc[-1]):.2f}")
    out.append(f"• MACD: OSC={float(osc.iloc[-1]):.2f}")
    out.append("-" * 50)

    # 進階指標
    out.append("🚀 進階指標 (趨勢/波動/量能):")
    adx, pdi, mdi = calculate_adx(df_daily)
    out.append(f"• ADX(14): {float(adx.iloc[-1]):.2f} | +DI: {float(pdi.iloc[-1]):.2f}, -DI: {float(mdi.iloc[-1]):.2f}")

    bbw, upper, lower = calculate_bbw(df_daily)
    out.append(f"• BBW: {float(bbw.iloc[-1]):.2f}% | 上軌: {float(upper.iloc[-1]):.2f}, 下軌: {float(lower.iloc[-1]):.2f}")

    mfi = calculate_mfi(df_daily)
    out.append(f"• MFI(14): {float(mfi.iloc[-1]):.2f} (資金流向)")

    y = (req_date.year if req_date is not None else datetime.now().year)
    avwap = calculate_avwap(df_daily, f"{y}-01-01")
    if avwap is not None and not avwap.empty and np.isfinite(float(avwap.iloc[-1])):
        dist = (float(latest_daily["Close"]) - float(avwap.iloc[-1])) / float(avwap.iloc[-1]) * 100
        out.append(f"• AVWAP (YTD): {float(avwap.iloc[-1]):.2f} (乖離: {dist:+.2f}%)")
    else:
        out.append("• AVWAP (YTD): 資料不足")
    out.append("-" * 50)

    # 🧠 型態辨識
    out.append("🧠 型態辨識（M頭 / W底 / 收斂三角）:")
    try:
        patterns = detect_patterns(df_daily)
        if not patterns:
            out.append("• 未偵測到明確型態（或資料不足/型態不符合條件）")
        else:
            for p in patterns:
                if p["pattern"].startswith("M頭"):
                    out.append(
                        f"• {p['pattern']}：{p['status']} | 頸線 {p['neckline']:.2f} | 目標 {p['target']:.2f} "
                        f"(頭1 {p['peak1_date']} {p['peak1']:.2f}, 頭2 {p['peak2_date']} {p['peak2']:.2f})"
                    )
                elif p["pattern"].startswith("W底"):
                    out.append(
                        f"• {p['pattern']}：{p['status']} | 頸線 {p['neckline']:.2f} | 目標 {p['target']:.2f} "
                        f"(底1 {p['trough1_date']} {p['trough1']:.2f}, 底2 {p['trough2_date']} {p['trough2']:.2f})"
                    )
                else:
                    out.append(
                        f"• {p['pattern']}：{p['status']} | 上緣 {p['upper_now']:.2f} / 下緣 {p['lower_now']:.2f} "
                        f"| 收斂比 {p['range_shrink']:.2f} | 方向 {p['direction']}"
                    )
    except Exception:
        out.append("• 型態辨識：計算失敗（資料不足或格式異常）")

    out.append("-" * 50)
    out.append("=" * 50)
    return "\n".join(out)
