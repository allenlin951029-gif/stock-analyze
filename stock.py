# stock.py
# 1) 三大法人單位：官方回來是「股」→ 顯示改成「張」（股/1000）
# 2) 外資抓不到時：改用「外資買進 - 外資賣出」計算淨買賣（fallback）
# 3) 自營商避免誤抓「外資自營商」
# 4) 當日VOL=0/NaN → 改用昨日VOL（量增量縮用昨日 vs 前日）
# 5) 額外：當日K棒型態

import io
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# optional (if you have truststore in requirements)
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass


# ---------------------------
# Basic helpers
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
        if s in ("", "--", "—", "NaN", "nan", "None"):
            return None
        return int(float(s))
    except Exception:
        return None


def _resolve_yf_ticker(stock_id: str) -> str:
    s = stock_id.strip().upper()
    if s.endswith(".TW") or s.endswith(".TWO"):
        return s
    return f"{s}.TW"


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
            # 若實體極小，視為十字
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
def _clean_header_list(cols):
    """移除 CSV 標頭中的換行符號與空白，方便比對"""
    return [str(c).replace("\n", "").replace("\r", "").strip() for c in cols]

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
        df = pd.read_csv(io.StringIO(csv_text))
        df.columns = _clean_header_list(df.columns) # 清理標頭
        return df
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
        df = pd.read_csv(io.StringIO(csv_text))
        df.columns = _clean_header_list(df.columns) # 清理標頭
        return df
    except Exception:
        return None


def get_institutional_data(stock_id, trade_date, market_hint=None):
    """
    回傳 foreign/trust/dealer 皆為「股」（shares）
    顯示時再 /1000 轉「張」
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

    # 往前找 10 天（避開週末/休市）
    for back in range(0, 10):
        d = trade_date - timedelta(days=back)

        for mkt_name, fn in markets:
            df, err = fn(d)
            if df is None:
                last_error = err
                continue

            cols = list(df.columns)
            
            # 尋找代號欄
            code_col = next((c for c in cols if c in ("證券代號", "代號")), None)
            if code_col is None:
                last_error = f"{mkt_name} 欄位找不到代號欄"
                continue

            # 轉字串並去空白比對
            row = df[df[code_col].astype(str).str.strip() == stock_no]
            if row.empty:
                last_error = f"{mkt_name} 當日資料找不到 {stock_no}"
                continue
            
            # ==========================================
            # 核心邏輯修正：精準抓取欄位
            # ==========================================
            def get_val(col_name):
                if col_name and col_name in row.columns:
                    return _safe_int(row.iloc[0][col_name])
                return None

            def find_col(keywords, must_not_have=None):
                """在 cols 中尋找符合所有關鍵字的欄位"""
                for c in cols:
                    if all(k in c for k in keywords):
                        if must_not_have and any(bad in c for bad in must_not_have):
                            continue
                        return c
                return None

            # --- 1. 外資 Foreign ---
            # 優先級 1: 完整名稱 (TWSE: "外陸資買賣超股數(不含外資自營商)", TPEx: "外資及陸資買賣超股數(不含外資自營商)")
            # 我們搜尋 "買賣超" 且包含 "(不含外資自營商)"，這是最準確的
            foreign_col = find_col(["買賣超", "(不含外資自營商)"], must_not_have=["買進", "賣出"])
            
            # 優先級 2: 若找不到，找 "外陸資" 或 "外資" + "買賣超"，但必須 *不* 以 "外資自營商" 開頭
            if not foreign_col:
                # 這裡的邏輯是：欄位名稱可以是 "外陸資買賣超..." 但不能是 "外資自營商買賣超..."
                # 注意：find_col 的 must_not_have 如果放 "外資自營商"，會把正確的 "(不含外資自營商)" 給殺掉
                # 所以我們手動遍歷
                for c in cols:
                    if ("外陸資" in c or "外資" in c) and "買賣超" in c:
                        if c.startswith("外資自營商"): # 關鍵修正：只排除以「外資自營商」開頭的
                            continue
                        foreign_col = c
                        break
            
            foreign = get_val(foreign_col)

            # --- 2. 投信 Trust ---
            trust_col = find_col(["投信", "買賣超"])
            trust = get_val(trust_col)

            # --- 3. 自營商 Dealer ---
            # 邏輯：總合欄位優先 -> 否則自行買賣+避險
            # 排除外資相關、排除權證相關(如果有)
            dealer_total_col = find_col(["自營商", "買賣超"], must_not_have=["外資", "自行", "避險"])
            
            dealer = None
            if dealer_total_col:
                dealer = get_val(dealer_total_col)
            else:
                dealer_self = get_val(find_col(["自營商", "自行", "買賣超"])) or 0
                dealer_hedge = get_val(find_col(["自營商", "避險", "買賣超"])) or 0
                dealer = dealer_self + dealer_hedge

            # --- 4. 外資 Fallback (買進 - 賣出) ---
            if foreign is None:
                # 找買進欄位 (包含 "不含外資自營商" 或 不以 "外資自營商" 開頭)
                buy_col = find_col(["買進", "(不含外資自營商)"])
                if not buy_col:
                    for c in cols:
                        if ("外陸資" in c or "外資" in c) and "買進" in c and not c.startswith("外資自營商"):
                            buy_col = c
                            break
                
                # 找賣出欄位
                sell_col = find_col(["賣出", "(不含外資自營商)"])
                if not sell_col:
                    for c in cols:
                        if ("外陸資" in c or "外資" in c) and "賣出" in c and not c.startswith("外資自營商"):
                            sell_col = c
                            break
                            
                b_val = get_val(buy_col)
                s_val = get_val(sell_col)
                
                if b_val is not None and s_val is not None:
                    foreign = b_val - s_val

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
# Indicators
# ---------------------------
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


def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# ---------------------------
# Main analysis (returns string for Streamlit)
# ---------------------------
def analyze_stock_technical(stock_id: str) -> str:
    stock_id = stock_id.strip().upper()
    if not stock_id:
        return "請輸入股票代號"

    yf_ticker = _resolve_yf_ticker(stock_id)
    out = []
    out.append(f"🔄 正在分析 {stock_id} ... (連線 Yahoo Finance)")

    # Daily
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

    # Weekly/Monthly
    df_weekly = yf.download(yf_ticker, period="1y", interval="1wk", progress=False, auto_adjust=False)
    df_weekly = clean_yf_columns(df_weekly)
    df_monthly = yf.download(yf_ticker, period="2y", interval="1mo", progress=False, auto_adjust=False)
    df_monthly = clean_yf_columns(df_monthly)

    latest_daily = df_daily.iloc[-1]
    prev_daily = df_daily.iloc[-2] if len(df_daily) >= 2 else latest_daily
    latest_trade_date = df_daily.index[-1].date()

    # Institutional (shares)
    chips = get_institutional_data(stock_id, trade_date=latest_trade_date, market_hint=yf_ticker)

    out.append("")
    out.append("=" * 50)
    out.append(f"📊 {yf_ticker} 專業版數據表 (含三大法人)")
    out.append(f"📅 資料日期: {df_daily.index[-1].strftime('%Y-%m-%d')}")
    out.append("=" * 50)

    # Last 5 days
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

    # Candle pattern
    try:
        out.append(f"🕯 當日K棒型態: {describe_candle(latest_daily['Open'], latest_daily['High'], latest_daily['Low'], latest_daily['Close'])}")
        out.append("-" * 50)
    except Exception:
        pass

    # Chips formatting: shares -> lots
    out.append("💰 籌碼面 (三大法人):")
    if chips and chips.get("error") is None:
        def fmt_to_lots(v_shares):
            v = _safe_int(v_shares)
            if v is None:
                return "n/a"
            lots = int(round(v / 1000))
            if lots > 0:
                return f"🔴 買超 {lots:,} 張"
            if lots < 0:
                return f"🟢 賣超 {abs(lots):,} 張"
            return "➖ 無變動"

        out.append(f"• 外資 : {fmt_to_lots(chips.get('foreign'))}")
        out.append(f"• 投信 : {fmt_to_lots(chips.get('trust'))}")
        out.append(f"• 自營商: {fmt_to_lots(chips.get('dealer'))}")
        out.append(f"  (日期: {chips.get('date')}, 市場代碼推定: {chips.get('id')})")
    else:
        out.append(f"⚠️ 無法抓取三大法人數據 ({chips.get('error') if isinstance(chips, dict) else '未知錯誤'})")
    out.append("-" * 50)

    # Long trend
    out.append("📈 長線趨勢:")
    if df_weekly is not None and (not df_weekly.empty) and "Close" in df_weekly.columns:
        ma20_week = df_weekly["Close"].rolling(20).mean().iloc[-1]
        out.append(f"• [週 K] 收: {float(df_weekly.iloc[-1]['Close']):.2f} | 20週均價: {float(ma20_week):.2f}")
    if df_monthly is not None and (not df_monthly.empty) and "Close" in df_monthly.columns:
        out.append(f"• [月 K] 收: {float(df_monthly.iloc[-1]['Close']):.2f}")
    out.append("-" * 50)

    # Base indicators
    out.append("🔍 基礎指標:")
    ma5 = df_daily["Close"].rolling(5).mean().iloc[-1]
    ma20 = df_daily["Close"].rolling(20).mean().iloc[-1]
    ma60 = df_daily["Close"].rolling(60).mean().iloc[-1]
    out.append(f"• 均線: MA5={float(ma5):.2f}, MA20={float(ma20):.2f}, MA60={float(ma60):.2f}")

    # Volume fallback (lots)
    vol_today = float(latest_daily["Volume"]) if "Volume" in latest_daily else float("nan")
    vol_yesterday = float(prev_daily["Volume"]) if "Volume" in prev_daily else float("nan")

    use_prev_vol = (pd.isna(vol_today) or vol_today == 0) and len(df_daily) >= 2
    if use_prev_vol:
        vol_used = vol_yesterday
        vol_prev_used = float(df_daily.iloc[-3]["Volume"]) if len(df_daily) >= 3 else vol_yesterday
        vol_note = f" (改用昨日量 {df_daily.index[-2].strftime('%Y-%m-%d')})"
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

    # Advanced
    out.append("🚀 進階指標 (趨勢/波動/量能):")
    adx, pdi, mdi = calculate_adx(df_daily)
    out.append(f"• ADX(14): {float(adx.iloc[-1]):.2f} | +DI: {float(pdi.iloc[-1]):.2f}, -DI: {float(mdi.iloc[-1]):.2f}")

    bbw, upper, lower = calculate_bbw(df_daily)
    out.append(f"• BBW: {float(bbw.iloc[-1]):.2f}% | 上軌: {float(upper.iloc[-1]):.2f}, 下軌: {float(lower.iloc[-1]):.2f}")

    mfi = calculate_mfi(df_daily)
    out.append(f"• MFI(14): {float(mfi.iloc[-1]):.2f} (資金流向)")

    current_year = datetime.now().year
    avwap = calculate_avwap(df_daily, f"{current_year}-01-01")
    if avwap is not None and not avwap.empty and not np.isnan(float(avwap.iloc[-1])):
        dist = (float(latest_daily["Close"]) - float(avwap.iloc[-1])) / float(avwap.iloc[-1]) * 100
        out.append(f"• AVWAP (YTD): {float(avwap.iloc[-1]):.2f} (乖離: {dist:+.2f}%)")
    else:
        out.append("• AVWAP (YTD): 資料不足")

    out.append("=" * 50)
    return "\n".join(out)

import pandas as pd
import numpy as np
import yfinance as yf

def get_chart_data(stock_id: str, period: str = "6mo"):
    """
    回傳 (yf_ticker, df)
    df 欄位包含: Open High Low Close Volume MA5 MA20 MA60 BB_MID BB_UP BB_DN RSI14 K D
    """
    stock_id = stock_id.strip().upper()
    if not stock_id:
        return None, None

    yf_ticker = _resolve_yf_ticker(stock_id)
    df = yf.download(yf_ticker, period=period, interval="1d", progress=False, auto_adjust=False)
    df = clean_yf_columns(df)

    # .TW 沒資料就試 .TWO
    if (df is None or df.empty) and yf_ticker.endswith(".TW"):
        yf_ticker = yf_ticker.replace(".TW", ".TWO")
        df = yf.download(yf_ticker, period=period, interval="1d", progress=False, auto_adjust=False)
        df = clean_yf_columns(df)

    if df is None or df.empty:
        return yf_ticker, None

    df = df.copy()
    # --- MA ---
    df["MA5"] = df["Close"].rolling(5).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA60"] = df["Close"].rolling(60).mean()

    # --- Bollinger (20,2) ---
    mid = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_MID"] = mid
    df["BB_UP"] = mid + 2 * std
    df["BB_DN"] = mid - 2 * std

    # --- RSI(14) ---
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    # --- KD(9,3,3) ---
    low_min = df["Low"].rolling(9).min()
    high_max = df["High"].rolling(9).max()
    rsv = (df["Close"] - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    df["K"] = rsv.ewm(com=2).mean()
    df["D"] = df["K"].ewm(com=2).mean()

    return yf_ticker, df
