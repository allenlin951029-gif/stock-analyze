import io
import re
import time
from datetime import datetime, timedelta, date as date_type

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass

# ---------------------------
# Tiny TTL cache (no streamlit needed)
# ---------------------------
_TTL_CACHE = {}
def _cache_get(key):
    v = _TTL_CACHE.get(key)
    if not v:
        return None
    exp, data = v
    if time.time() > exp:
        _TTL_CACHE.pop(key, None)
        return None
    return data

def _cache_set(key, data, ttl_sec=30):
    _TTL_CACHE[key] = (time.time() + ttl_sec, data)

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
        if s in ("", "--", "—", "NaN", "nan", "None", "null"):
            return None
        return int(float(s))
    except Exception:
        return None

def _resolve_yf_ticker(stock_id: str) -> str:
    s = stock_id.strip().upper()
    if s.endswith(".TW") or s.endswith(".TWO"):
        return s
    return f"{s}.TW"

def _to_roc_date(d: datetime | date_type) -> str:
    if isinstance(d, datetime):
        d = d.date()
    roc_year = d.year - 1911
    return f"{roc_year:03d}/{d.month:02d}/{d.day:02d}"

def _decode_bytes(b: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp950", "big5"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("latin1", errors="ignore")

def _norm_col(s: str) -> str:
    s = str(s).replace("\ufeff", "")
    s = s.replace("（", "(").replace("）", ")")
    s = re.sub(r"\s+", "", s)
    return s

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
# yfinance fetch (cached) + intraday compose "today" candle
# ---------------------------
def _yf_download_cached(ticker: str, period: str, interval: str, ttl=30) -> pd.DataFrame:
    key = ("yf", ticker, period, interval)
    df = _cache_get(key)
    if df is not None:
        return df.copy()
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    df = clean_yf_columns(df)
    _cache_set(key, df.copy(), ttl_sec=ttl)
    return df

def _compose_today_from_intraday(ticker: str, target_date: date_type) -> dict | None:
    """
    用 intraday (5m) 拼出 target_date 的當日 OHLCV。
    回傳 dict: Open/High/Low/Close/Volume
    """
    df_i = _yf_download_cached(ticker, period="1d", interval="5m", ttl=5)
    if df_i is None or df_i.empty:
        return None

    # yfinance intraday index is tz-aware sometimes; use .date()
    sub = df_i[df_i.index.date == target_date]
    if sub.empty:
        return None

    o = float(sub["Open"].iloc[0])
    h = float(sub["High"].max())
    l = float(sub["Low"].min())
    c = float(sub["Close"].iloc[-1])
    v = float(sub["Volume"].sum()) if "Volume" in sub.columns else 0.0
    return {"Open": o, "High": h, "Low": l, "Close": c, "Volume": v}

# ---------------------------
# TWSE/TPEx 三大法人（你原本那套，略：保持不動，避免引入新 bug）
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
    stock_no = stock_id.strip().upper().replace(".TW", "").replace(".TWO", "")

    # cache by day (法人一天更新一次，不用每 5 秒抓)
    key = ("chips", stock_no, str(trade_date), str(market_hint))
    cached = _cache_get(key)
    if cached is not None:
        return dict(cached)

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
            return None, "TWSE 無資料(可能休市/尚未公布)"
        df = _parse_twse_t86_csv(r.text)
        return df, None if df is not None else "TWSE 解析失敗"

    def try_tpex(d_):
        roc_date = _to_roc_date(d_)
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
            return None, "TPEx 無資料(可能休市/尚未公布)"
        df = _parse_tpex_csv(r.text)
        return df, None if df is not None else "TPEx 解析失敗"

    markets = [("TWSE", try_twse), ("TPEx", try_tpex)]
    if not prefer_twse:
        markets = [("TPEx", try_tpex), ("TWSE", try_twse)]

    def build_colmap(cols):
        mp = {}
        for c in cols:
            mp[_norm_col(c)] = c
        return mp

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
                if _norm_col(c) in ("證券代號", "代號"):
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
            dealer_self_col = find_first(
                colmap,
                lambda nk: ("自營商" in nk) and ("自行買賣" in nk) and ("買賣超" in nk) and ("外資" not in nk)
            )
            dealer_hedge_col = find_first(
                colmap,
                lambda nk: ("自營商" in nk) and ("避險" in nk) and ("買賣超" in nk) and ("外資" not in nk)
            )

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
                    lambda nk: is_foreign_usable(nk) and (("買進" in nk) or ("買入" in nk)) and ("買賣超" not in nk),
                )
                sell_col = find_first(
                    colmap,
                    lambda nk: is_foreign_usable(nk) and ("賣出" in nk) and ("買賣超" not in nk),
                )
                buy_v = _safe_int(row.iloc[0][buy_col]) if buy_col else None
                sell_v = _safe_int(row.iloc[0][sell_col]) if sell_col else None
                if buy_v is not None and sell_v is not None:
                    foreign = buy_v - sell_v

            yf_suffix = ".TW" if mkt_name == "TWSE" else ".TWO"
            res = {
                "id": f"{stock_no}{yf_suffix}",
                "date": d.strftime("%Y-%m-%d"),
                "foreign": foreign,
                "trust": trust,
                "dealer": dealer,
                "error": None,
            }
            _cache_set(key, dict(res), ttl_sec=3600)
            return res

    res = {"error": last_error or "未知錯誤", "id": f"{stock_no}.TW"}
    _cache_set(key, dict(res), ttl_sec=120)
    return res

# ---------------------------
# Margin (融資融券)：只做 TWSE 可靠版；TPEx 先回 n/a（避免假數字）
# ---------------------------
def _twse_margn_df(d: datetime | date_type) -> pd.DataFrame | None:
    if isinstance(d, datetime):
        d = d.date()

    key = ("twse_margn", d.strftime("%Y%m%d"))
    cached = _cache_get(key)
    if cached is not None:
        return cached.copy()

    url = "https://www.twse.com.tw/exchangeReport/MI_MARGN"
    params = {"response": "json", "date": d.strftime("%Y%m%d"), "selectType": "ALL"}
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.7"}
    r = requests.get(url, params=params, headers=headers, timeout=20)
    if r.status_code != 200:
        return None
    js = r.json()

    fields = js.get("fields")
    data = js.get("data")
    if not fields or not data:
        return None

    df = pd.DataFrame(data, columns=fields)
    _cache_set(key, df.copy(), ttl_sec=3600)
    return df

def get_margin_data(stock_id: str, trade_date: date_type, market_hint: str | None = None):
    """
    盤中/當日可能未公布 → 會自動往回找最近一次可用日
    這裡避免 TPEx 亂抓造成「兩天一樣」：若 .TWO 就回 n/a
    """
    stock_no = stock_id.strip().upper().replace(".TW", "").replace(".TWO", "")
    if isinstance(market_hint, str) and market_hint.upper().endswith(".TWO"):
        return {"error": "TPEx 融資融券資料源未穩定，避免假數據先回 n/a", "id": f"{stock_no}.TWO"}

    def pick_row(df: pd.DataFrame) -> pd.Series | None:
        if df is None or df.empty:
            return None
        cols = list(df.columns)
        colmap = {_norm_col(c): c for c in cols}
        code_col = colmap.get("股票代號") or colmap.get("證券代號")
        if not code_col:
            return None
        sub = df[df[code_col].astype(str).str.strip() == stock_no]
        return sub.iloc[0] if not sub.empty else None

    def get_cols(df: pd.DataFrame):
        cols = list(df.columns)
        colmap = {_norm_col(c): c for c in cols}
        # TWSE 常見：融資餘額/融券餘額（單位股），這裡只抓「餘額」欄
        margin_col = colmap.get("融資餘額") or colmap.get("資餘額") or colmap.get("融資餘額(股)") or colmap.get("資餘額(股)")
        short_col  = colmap.get("融券餘額") or colmap.get("券餘額") or colmap.get("融券餘額(股)") or colmap.get("券餘額(股)")
        return margin_col, short_col

    def nearest(d0: date_type):
        for back in range(0, 14):
            d = d0 - timedelta(days=back)
            df = _twse_margn_df(d)
            row = pick_row(df) if df is not None else None
            if row is None:
                continue
            mcol, scol = get_cols(df)
            if not mcol or not scol:
                continue
            m = _safe_int(row[mcol])
            s = _safe_int(row[scol])
            if m is None and s is None:
                continue
            return d, m, s
        return None

    def prev(d_found: date_type):
        for back in range(1, 14):
            d = d_found - timedelta(days=back)
            df = _twse_margn_df(d)
            row = pick_row(df) if df is not None else None
            if row is None:
                continue
            mcol, scol = get_cols(df)
            if not mcol or not scol:
                continue
            m = _safe_int(row[mcol])
            s = _safe_int(row[scol])
            if m is None and s is None:
                continue
            return d, m, s
        return None

    got = nearest(trade_date)
    if not got:
        return {"error": "抓取失敗/休市/尚未公布", "id": f"{stock_no}.TW"}

    d_found, m_bal_sh, s_bal_sh = got
    prev_got = prev(d_found)

    m_prev = prev_got[1] if prev_got else None
    s_prev = prev_got[2] if prev_got else None

    # convert shares -> lots(張)
    mb = int(round(m_bal_sh / 1000)) if m_bal_sh is not None else None
    sb = int(round(s_bal_sh / 1000)) if s_bal_sh is not None else None
    dm = int(round((m_bal_sh - m_prev) / 1000)) if (m_bal_sh is not None and m_prev is not None) else None
    ds = int(round((s_bal_sh - s_prev) / 1000)) if (s_bal_sh is not None and s_prev is not None) else None

    return {
        "id": f"{stock_no}.TW",
        "date": d_found.strftime("%Y-%m-%d"),
        "market": "TWSE",
        "margin_balance_lot": mb,
        "short_balance_lot": sb,
        "delta_margin_lot": dm,
        "delta_short_lot": ds,
        "error": None,
    }

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
# Patterns (keep your original; omitted here for brevity)
# 你原本的 detect_* 可以直接貼回來（不影響本次「當日/加速」核心）
# ---------------------------
def detect_patterns(df: pd.DataFrame):
    return []  # 你要保留型態辨識就把原本那段整段貼回來

# ---------------------------
# Main analysis
# ---------------------------
def analyze_stock_technical(stock_id: str, as_of: date_type | None = None) -> str:
    stock_id = stock_id.strip().upper()
    if not stock_id:
        return "請輸入股票代號"
    if as_of is None:
        as_of = datetime.now().date()

    yf_ticker = _resolve_yf_ticker(stock_id)

    out = []
    out.append(f"🔄 正在分析 {stock_id} ... (連線 Yahoo Finance)")
    out.append(f"📅 目標日期: {as_of.strftime('%Y-%m-%d')}")

    # 只抓日K一次（快取 30 秒）
    df_daily = _yf_download_cached(yf_ticker, period="2y", interval="1d", ttl=30)
    if df_daily.empty and yf_ticker.endswith(".TW"):
        alt = yf_ticker.replace(".TW", ".TWO")
        out.append(f"⚠️ .TW 無資料，改試上櫃代碼: {alt}")
        yf_ticker = alt
        df_daily = _yf_download_cached(yf_ticker, period="2y", interval="1d", ttl=30)

    if df_daily.empty:
        return "\n".join(out + [f"❌ 找不到股票代號 {stock_id} (Yahoo Finance 無數據)"])

    df_daily = df_daily[df_daily.index.date <= as_of].copy()
    if df_daily.empty:
        return "\n".join(out + [f"❌ {as_of.strftime('%Y-%m-%d')} 之前無交易資料"])

    # ✅ 盤中補「今天」：如果 as_of 是今天，但日K沒有今天，就用 intraday 拼一根
    if as_of == datetime.now().date() and (df_daily.index.date[-1] != as_of):
        today_bar = _compose_today_from_intraday(yf_ticker, as_of)
        if today_bar:
            out.append("⚡ 盤中模式：用 5m 即時資料拼出『今日暫時K』")
            new_idx = pd.Timestamp(as_of)
            df_daily.loc[new_idx, ["Open", "High", "Low", "Close", "Volume"]] = [
                today_bar["Open"], today_bar["High"], today_bar["Low"], today_bar["Close"], today_bar["Volume"]
            ]
            df_daily = df_daily.sort_index()

    latest_daily = df_daily.iloc[-1]
    prev_daily = df_daily.iloc[-2] if len(df_daily) >= 2 else latest_daily
    latest_trade_date = df_daily.index[-1].date()

    chips = get_institutional_data(stock_id, trade_date=latest_trade_date, market_hint=yf_ticker)
    margin = get_margin_data(stock_id, trade_date=latest_trade_date, market_hint=yf_ticker)

    out.append("")
    out.append("=" * 50)
    out.append(f"📊 {yf_ticker} 專業版數據表")
    out.append(f"📅 資料日期: {df_daily.index[-1].strftime('%Y-%m-%d')}")
    out.append("=" * 50)

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

    out.append(f"🕯 當日K棒型態: {describe_candle(latest_daily['Open'], latest_daily['High'], latest_daily['Low'], latest_daily['Close'])}")
    out.append("-" * 50)

    out.append("💰 籌碼面 (三大法人):")
    if isinstance(chips, dict) and chips.get("error") is None:
        def fmt_to_lots(v_shares):
            v = _safe_int(v_shares)
            if v is None:
                return "n/a（今日可能尚未公布）"
            lots = int(round(v / 1000))
            if lots > 0:
                return f"🔴 買超 {lots:,} 張"
            if lots < 0:
                return f"🟢 賣超 {abs(lots):,} 張"
            return "➖ 無變動"
        out.append(f"• 外資 : {fmt_to_lots(chips.get('foreign'))}")
        out.append(f"• 投信 : {fmt_to_lots(chips.get('trust'))}")
        out.append(f"• 自營商: {fmt_to_lots(chips.get('dealer'))}")
        out.append(f"  (日期: {chips.get('date')}, 代碼推定: {chips.get('id')})")
    else:
        out.append(f"⚠️ 無法抓取三大法人（{chips.get('error') if isinstance(chips, dict) else '未知錯誤'}）")
    out.append("-" * 50)

    out.append("💸 融資融券（散戶槓桿指標）:")
    if isinstance(margin, dict) and margin.get("error") is None:
        def fmt_delta(x):
            if x is None:
                return "（較前日 n/a）"
            if x > 0:
                return f"（較前日 +{x:,} 張，增加）"
            if x < 0:
                return f"（較前日 {x:,} 張，減少）"
            return "（較前日 0 張，持平）"
        out.append(f"• 融資餘額: {margin.get('margin_balance_lot'):,} 張 {fmt_delta(margin.get('delta_margin_lot'))}")
        out.append(f"• 融券餘額: {margin.get('short_balance_lot'):,} 張 {fmt_delta(margin.get('delta_short_lot'))}")
        out.append(f"  (日期: {margin.get('date')}, 市場: {margin.get('market')})")
    else:
        out.append(f"⚠️ 無法抓取融資融券（{margin.get('error') if isinstance(margin, dict) else '未知錯誤'}）")
    out.append("-" * 50)

    # indicators（省略部分，保留你原本的即可）
    ma5 = df_daily["Close"].rolling(5).mean().iloc[-1]
    ma20 = df_daily["Close"].rolling(20).mean().iloc[-1]
    out.append("🔍 基礎指標:")
    out.append(f"• 均線: MA5={float(ma5):.2f}, MA20={float(ma20):.2f}")

    # volume fallback
    vol_today = float(latest_daily.get("Volume", 0.0))
    vol_yesterday = float(prev_daily.get("Volume", 0.0))
    use_prev = (np.isnan(vol_today) or vol_today == 0) and len(df_daily) >= 2
    v_used = vol_yesterday if use_prev else vol_today
    v_prev = float(df_daily.iloc[-3].get("Volume", vol_yesterday)) if (use_prev and len(df_daily) >= 3) else vol_yesterday
    v_note = f" (改用昨日量 {df_daily.index[-2].strftime('%Y-%m-%d')})" if use_prev else ""
    out.append(f"• 成交量: {int(v_used/1000):,} 張（較昨 {int((v_used-v_prev)/1000):+,} 張）{v_note}")

    out.append("=" * 50)
    return "\n".join(out)
