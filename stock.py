# -*- coding: utf-8 -*-
import sys
import subprocess
import importlib
import random
import time
import io
from datetime import datetime, timedelta

# --- 0. 自動安裝套件功能 (Auto-Install) ---
def install_and_import(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"📦 偵測到缺少 {package_name}，正在為您自動安裝中...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} 安裝成功！")
        except Exception as e:
            print(f"❌ 安裝失敗: {e}")
            sys.exit(1)

# --- 1. 正式匯入套件 ---
import truststore
truststore.inject_into_ssl()

import yfinance as yf
import pandas as pd
import numpy as np
import requests


# --- 數據清理與計算函式 ---
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


def get_k_status(open_p, close_p):
    if close_p > open_p:
        return "🔴 紅棒"
    elif close_p < open_p:
        return "🟢 綠棒"
    else:
        return "➖ 十字"


# --- 新增：K棒型態（當天） ---
def describe_candle(open_p, high_p, low_p, close_p):
    """
    回傳較具體的K棒型態（簡化常用型態）
    - 長紅/長黑
    - 十字星/蜻蜓十字/墓碑十字
    - 錘頭/吊人
    - 倒錘/流星
    - 小紅/小黑 + 影線特徵
    """
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

    # 十字系（本體極小）
    if body_r <= 0.10:
        if upper_r >= 0.65 and lower_r <= 0.15:
            return "墓碑十字"
        if lower_r >= 0.65 and upper_r <= 0.15:
            return "蜻蜓十字"
        return "十字星"

    # 長實體
    if body_r >= 0.60:
        return "長紅K" if bullish else "長黑K"

    # 下影長：錘頭/吊人
    if lower_r >= 0.60 and body_r <= 0.30 and upper_r <= 0.20:
        return "錘頭線" if bullish else "吊人線"

    # 上影長：倒錘/流星
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


# --- 官方來源抓三大法人（取代 Yahoo 爬蟲）---
def _safe_int(x):
    try:
        s = str(x).strip().replace(",", "")
        if s in ("", "--", "—", "NaN", "nan", "None"):
            return None
        return int(float(s))
    except Exception:
        return None


def _find_net_col(cols, keyword, exclude=None):
    """
    找同時含 keyword 與 買賣超 的欄位；可排除關鍵字（避免誤抓「外資自營商」）
    """
    exclude = exclude or []
    cands = []
    for c in cols:
        s = str(c)
        if (keyword in s) and ("買賣超" in s) and not any(e in s for e in exclude):
            cands.append(c)
    return cands[0] if cands else None


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


def get_institutional_data(stock_id, trade_date=None, market_hint=None):
    """
    用 TWSE / TPEx 官方 CSV 取三大法人買賣超（較 Yahoo 股市頁面穩定很多）
    trade_date：建議傳入「最後交易日」(date)，避免假日/休市無資料
    market_hint：傳入 'xxxx.TW' 或 'xxxx.TWO'，用來決定先試哪個市場
    """
    stock_no = stock_id.strip().upper().replace(".TW", "").replace(".TWO", "")
    if trade_date is None:
        trade_date = datetime.now().date()

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
        r = requests.get(url, headers=headers, timeout=10)
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
        r = requests.get(url, headers=headers2, timeout=10)
        if r.status_code != 200 or len(r.text) < 200:
            return None, f"TPEx HTTP {r.status_code}"
        if "沒有符合條件的資料" in r.text or "很抱歉" in r.text:
            return None, "TPEx 無資料(可能休市)"
        df = _parse_tpex_csv(r.text)
        return df, None if df is not None else "TPEx 解析失敗"

    markets = [("TWSE", try_twse), ("TPEx", try_tpex)]
    if not prefer_twse:
        markets = [("TPEx", try_tpex), ("TWSE", try_twse)]

    # 往前找 10 天，避免週末/休市
    for back in range(0, 10):
        d = trade_date - timedelta(days=back)

        for mkt_name, fn in markets:
            df, err = fn(d)
            if df is None:
                last_error = err
                continue

            # 找代號欄
            code_col = None
            for c in df.columns:
                if str(c).strip() in ("證券代號", "代號"):
                    code_col = c
                    break
            if code_col is None:
                last_error = f"{mkt_name} 欄位找不到代號欄"
                continue

            row = df[df[code_col].astype(str).str.strip() == stock_no]
            if row.empty:
                last_error = f"{mkt_name} 當日資料找不到 {stock_no}"
                continue

            cols = list(df.columns)

            # 外資：排除「外資自營商」相關欄位，避免被自營商誤抓
            foreign_col = (
                _find_net_col(cols, "外陸資", exclude=["外資自營商"])
                or _find_net_col(cols, "外資及陸資", exclude=["外資自營商"])
                or _find_net_col(cols, "外資", exclude=["外資自營商"])
                or next((c for c in cols if ("外資" in str(c)) and ("買賣超" in str(c)) and ("外資自營商" not in str(c))), None)
            )

            trust_col = (
                _find_net_col(cols, "投信")
                or next((c for c in cols if ("投信" in str(c)) and ("買賣超" in str(c))), None)
                or next((c for c in cols if "投信" in str(c)), None)
            )

            # 自營商：務必排除「外資自營商」與任何含外資字樣，避免抓錯欄
            dealer_total_col = (
                _find_net_col(cols, "自營商", exclude=["外資", "外資自營商", "自行買賣", "避險"])
                or next((c for c in cols
                         if ("自營商" in str(c)) and ("買賣超" in str(c))
                         and ("外資" not in str(c)) and ("外資自營商" not in str(c))
                         and ("自行買賣" not in str(c)) and ("避險" not in str(c))), None)
            )

            # 若沒有總欄，嘗試：自行買賣 + 避險
            dealer_self_col = next((c for c in cols if ("自營商" in str(c)) and ("自行買賣" in str(c)) and ("買賣超" in str(c)) and ("外資" not in str(c))), None)
            dealer_hedge_col = next((c for c in cols if ("自營商" in str(c)) and ("避險" in str(c)) and ("買賣超" in str(c)) and ("外資" not in str(c))), None)

            foreign = _safe_int(row.iloc[0][foreign_col]) if foreign_col else None
            trust = _safe_int(row.iloc[0][trust_col]) if trust_col else None

            dealer = None
            if dealer_total_col:
                dealer = _safe_int(row.iloc[0][dealer_total_col])
            elif dealer_self_col or dealer_hedge_col:
                a = _safe_int(row.iloc[0][dealer_self_col]) if dealer_self_col else 0
                b = _safe_int(row.iloc[0][dealer_hedge_col]) if dealer_hedge_col else 0
                dealer = (a or 0) + (b or 0)

            yf_suffix = ".TW" if mkt_name == "TWSE" else ".TWO"
            return {
                "id": f"{stock_no}{yf_suffix}",
                "date": d.strftime("%Y-%m-%d"),
                "foreign": foreign if foreign is not None else (row.iloc[0][foreign_col] if foreign_col else None),
                "trust": trust if trust is not None else (row.iloc[0][trust_col] if trust_col else None),
                "dealer": dealer if dealer is not None else (row.iloc[0][dealer_total_col] if dealer_total_col else None),
                "error": None,
            }

    return {"error": last_error or "未知錯誤", "id": f"{stock_no}.TW"}


# --- 指標計算函式 ---
def calculate_adx(df, period=14):
    df = df.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["UpMove"] = df["High"] - df["High"].shift(1)
    df["DownMove"] = df["Low"].shift(1) - df["Low"]
    df["+DM"] = np.where((df["UpMove"] > df["DownMove"]) & (df["UpMove"] > 0), df["UpMove"], 0)
    df["-DM"] = np.where((df["DownMove"] > df["UpMove"]) & (df["DownMove"] > 0), df["DownMove"], 0)
    alpha = 1 / period
    df["TR_s"] = df["TR"].ewm(alpha=alpha, adjust=False).mean()
    df["+DM_s"] = df["+DM"].ewm(alpha=alpha, adjust=False).mean()
    df["-DM_s"] = df["-DM"].ewm(alpha=alpha, adjust=False).mean()
    df["+DI"] = 100 * (df["+DM_s"] / df["TR_s"])
    df["-DI"] = 100 * (df["-DM_s"] / df["TR_s"])
    df["DX"] = 100 * abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"])
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
    positive_flow = np.where(typical_price > typical_price.shift(1), raw_money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1), raw_money_flow, 0)
    positive_mf = pd.Series(positive_flow, index=df.index).rolling(window=period).sum()
    negative_mf = pd.Series(negative_flow, index=df.index).rolling(window=period).sum()
    mfi = 100 - (100 / (1 + (positive_mf / negative_mf)))
    return mfi


def calculate_avwap(df, anchor_date):
    mask = df.index >= anchor_date
    if not mask.any():
        return None
    df_anchored = df.loc[mask].copy()
    typical_price = (df_anchored["High"] + df_anchored["Low"] + df_anchored["Close"]) / 3
    cum_vol_price = (typical_price * df_anchored["Volume"]).cumsum()
    cum_vol = df_anchored["Volume"].cumsum()
    return cum_vol_price / cum_vol


def _resolve_yf_ticker(stock_id: str) -> str:
    s = stock_id.strip().upper()
    if s.endswith(".TW") or s.endswith(".TWO"):
        return s
    return f"{s}.TW"


def analyze_stock_technical(stock_id):
    stock_id = stock_id.strip().upper()
    if not stock_id:
        return

    yf_ticker = _resolve_yf_ticker(stock_id)
    print(f"\n🔄 正在分析 {stock_id} ... (連線 Yahoo Finance)")

    # 1) 先抓 yfinance（先定出最後交易日）
    try:
        df_daily = yf.download(yf_ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
        df_daily = clean_yf_columns(df_daily)

        # 若 .TW 抓不到，改試 .TWO
        if df_daily.empty and yf_ticker.endswith(".TW"):
            alt = yf_ticker.replace(".TW", ".TWO")
            print(f"⚠️ .TW 無資料，改試上櫃代碼: {alt}")
            yf_ticker = alt
            df_daily = yf.download(yf_ticker, period="1y", interval="1d", progress=False, auto_adjust=False)
            df_daily = clean_yf_columns(df_daily)

        if df_daily.empty:
            print(f"❌ 找不到股票代號 {stock_id} (Yahoo Finance 無數據)")
            return

        df_weekly = yf.download(yf_ticker, period="1y", interval="1wk", progress=False, auto_adjust=False)
        df_weekly = clean_yf_columns(df_weekly)

        df_monthly = yf.download(yf_ticker, period="2y", interval="1mo", progress=False, auto_adjust=False)
        df_monthly = clean_yf_columns(df_monthly)

    except Exception as e:
        print(f"❌ yfinance 發生錯誤: {e}")
        return

    if "Close" not in df_daily.columns:
        print("❌ 資料格式錯誤：缺少 Close")
        return

    latest_daily = df_daily.iloc[-1]
    prev_daily = df_daily.iloc[-2] if len(df_daily) >= 2 else latest_daily
    latest_weekly = df_weekly.iloc[-1] if not df_weekly.empty else None
    latest_monthly = df_monthly.iloc[-1] if not df_monthly.empty else None

    # 2) 用「最後交易日」去抓三大法人（官方 CSV）
    latest_trade_date = df_daily.index[-1].date()
    chips_data = get_institutional_data(stock_id, trade_date=latest_trade_date, market_hint=yf_ticker)

    # --- 輸出報告 ---
    print("\n" + "=" * 50)
    print(f"📊 {yf_ticker} 專業版數據表 (含三大法人)")
    print(f"📅 資料日期: {df_daily.index[-1].strftime('%Y-%m-%d')}")
    print("=" * 50)

    # Part A: 近期走勢
    print("📋 近 5 日交易紀錄:")
    recent_5 = df_daily.tail(5)
    print(f"{'日期':<12} {'收盤':<10} {'漲跌':<10} {'K棒'}")
    print("-" * 50)
    for idx, row in recent_5.iterrows():
        date_str = idx.strftime("%m-%d")
        close_p = float(row["Close"])
        open_p = float(row["Open"])
        loc = df_daily.index.get_loc(idx)
        change = close_p - float(df_daily.iloc[loc - 1]["Close"]) if loc > 0 else 0.0
        print(f"{date_str:<12} {close_p:<10.2f} {change:<+10.2f} {get_k_status(open_p, close_p)}")
    print("-" * 50)

    # ✅ 新增：當天K棒型態
    try:
        candle_type = describe_candle(
            latest_daily["Open"], latest_daily["High"], latest_daily["Low"], latest_daily["Close"]
        )
        print(f"🕯 當日K棒型態: {candle_type}")
        print("-" * 50)
    except Exception:
        # 不讓型態計算影響主流程
        pass

    # Part B: 籌碼面
    print("💰 籌碼面 (三大法人):")
    if chips_data and chips_data.get("error") is None:
        def format_chip(val):
            v = _safe_int(val)
            if v is None:
                return str(val)
            if v > 0:
                return f"🔴 買超 {v:,} 張"
            if v < 0:
                return f"🟢 賣超 {abs(v):,} 張"
            return "➖ 無變動"

        print(f"• 外資 : {format_chip(chips_data.get('foreign'))}")
        print(f"• 投信 : {format_chip(chips_data.get('trust'))}")
        print(f"• 自營商: {format_chip(chips_data.get('dealer'))}")
        print(f"  (日期: {chips_data.get('date')}, 市場代碼推定: {chips_data.get('id')})")
    else:
        err_reason = chips_data.get("error") if isinstance(chips_data, dict) else "未知錯誤"
        print(f"⚠️ 無法抓取三大法人數據 ({err_reason})")
    print("-" * 50)

    # Part C: 長線趨勢
    print("📈 長線趨勢:")
    if latest_weekly is not None and "Close" in df_weekly.columns:
        ma20_week = df_weekly["Close"].rolling(20).mean().iloc[-1]
        print(f"• [週 K] 收: {float(latest_weekly['Close']):.2f} | 20週均價: {float(ma20_week):.2f}")
    if latest_monthly is not None and "Close" in df_monthly.columns:
        print(f"• [月 K] 收: {float(latest_monthly['Close']):.2f}")
    print("-" * 50)

    # Part D: 基礎指標
    print("🔍 基礎指標:")
    ma5 = df_daily["Close"].rolling(5).mean().iloc[-1]
    ma20 = df_daily["Close"].rolling(20).mean().iloc[-1]
    ma60 = df_daily["Close"].rolling(60).mean().iloc[-1]
    print(f"• 均線: MA5={float(ma5):.2f}, MA20={float(ma20):.2f}, MA60={float(ma60):.2f}")

    # ✅ 修正：若當天VOL抓不到(0/NaN) → 改抓昨天VOL
    vol_today = float(latest_daily["Volume"]) if "Volume" in latest_daily else float("nan")
    vol_yesterday = float(prev_daily["Volume"]) if "Volume" in prev_daily else float("nan")

    use_prev_vol = (pd.isna(vol_today) or vol_today == 0) and len(df_daily) >= 2
    if use_prev_vol:
        vol_used = vol_yesterday
        # 用「昨天 vs 前天」算量增量縮
        if len(df_daily) >= 3:
            vol_prev_used = float(df_daily.iloc[-3]["Volume"])
        else:
            vol_prev_used = vol_yesterday
        vol_note = f" (改用昨日量 {df_daily.index[-2].strftime('%Y-%m-%d')})"
    else:
        vol_used = vol_today
        vol_prev_used = vol_yesterday
        vol_note = ""

    vol_in_lots = int(vol_used / 1000) if not pd.isna(vol_used) else 0
    vol_diff = int((vol_used - vol_prev_used) / 1000) if (not pd.isna(vol_used) and not pd.isna(vol_prev_used)) else 0
    vol_status = "量增" if (not pd.isna(vol_used) and not pd.isna(vol_prev_used) and vol_used > vol_prev_used) else "量縮"
    print(f"• 成交量: {vol_in_lots} 張 ({vol_status}, 較昨 {vol_diff:+} 張){vol_note}")

    def calculate_rsi(series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    rsi6 = calculate_rsi(df_daily["Close"], 6).iloc[-1]

    low_min = df_daily["Low"].rolling(9).min()
    high_max = df_daily["High"].rolling(9).max()
    rsv = (df_daily["Close"] - low_min) / (high_max - low_min) * 100
    df_daily["K"] = rsv.ewm(com=2).mean()
    df_daily["D"] = df_daily["K"].ewm(com=2).mean()
    k_val = df_daily["K"].iloc[-1]
    d_val = df_daily["D"].iloc[-1]

    ema12 = df_daily["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df_daily["Close"].ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    macd = dif.ewm(span=9, adjust=False).mean()
    osc = dif - macd

    print(f"• RSI(6): {float(rsi6):.2f}")
    print(f"• KD(9,3,3): K={float(k_val):.2f}, D={float(d_val):.2f}")
    print(f"• MACD: OSC={float(osc.iloc[-1]):.2f}")
    print("-" * 50)

    # Part E: 進階指標
    print("🚀 進階指標 (趨勢/波動/量能):")
    adx, pdi, mdi = calculate_adx(df_daily)
    print(f"• ADX(14): {float(adx.iloc[-1]):.2f} | +DI: {float(pdi.iloc[-1]):.2f}, -DI: {float(mdi.iloc[-1]):.2f}")

    bbw, upper, lower = calculate_bbw(df_daily)
    print(f"• BBW: {float(bbw.iloc[-1]):.2f}% | 上軌: {float(upper.iloc[-1]):.2f}, 下軌: {float(lower.iloc[-1]):.2f}")

    mfi = calculate_mfi(df_daily)
    print(f"• MFI(14): {float(mfi.iloc[-1]):.2f} (資金流向)")

    current_year = datetime.now().year
    avwap = calculate_avwap(df_daily, f"{current_year}-01-01")
    if avwap is not None and not avwap.empty:
        dist = (float(latest_daily["Close"]) - float(avwap.iloc[-1])) / float(avwap.iloc[-1]) * 100
        print(f"• AVWAP (YTD): {float(avwap.iloc[-1]):.2f} (乖離: {dist:+.2f}%)")
    else:
        print("• AVWAP: 資料不足")

    print("=" * 50)


if __name__ == "__main__":
    print("\n🚀 股票分析器 (官方三大法人版) 已啟動！")
    while True:
        user_input = input("\n請輸入股票代號 (輸入 q 離開): ").strip()
        if user_input.lower() == "q":
            break
        if not user_input:
            continue
        analyze_stock_technical(user_input)
