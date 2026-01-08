import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
from datetime import datetime, timedelta

# --- 設定網頁標題與排版 (必須在第一行) ---
st.set_page_config(page_title="台股籌碼分析儀表板", page_icon="📈", layout="wide")

st.title("📱 台股 AI 戰情室 (官方籌碼版)")
st.caption("數據來源：Yahoo Finance + 證交所(TWSE) + 櫃買中心(TPEx)")


# --- 核心邏輯函式 ---

def clean_yf_columns(df):
    if df is None or df.empty: return df
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
        return "🔴"
    elif close_p < open_p:
        return "🟢"
    else:
        return "➖"


def _safe_int(x):
    try:
        s = str(x).strip().replace(",", "")
        if s in ("", "--", "—", "NaN", "nan", "None"): return None
        return int(float(s))
    except:
        return None


def _find_net_col(cols, keyword):
    cands = [c for c in cols if (keyword in str(c)) and ("買賣超" in str(c))]
    return cands[0] if cands else None


def _parse_twse_t86_csv(text):
    text = text.replace("\r", "").replace("=", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    start = next((i for i, ln in enumerate(lines) if ("證券代號" in ln) and ("證券名稱" in ln)), None)
    if start is None: return None

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("說明") or lines[j].startswith("備註"):
            end = j
            break
    csv_text = "\n".join(lines[start:end])
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except:
        return None


def _parse_tpex_csv(text):
    text = text.replace("\ufeff", "").replace("\r", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    start = next((i for i, ln in enumerate(lines) if ("代號" in ln) and ("名稱" in ln)), None)
    if start is None: return None

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("說明") or lines[j].startswith("備註"):
            end = j
            break
    csv_text = "\n".join(lines[start:end])
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except:
        return None


# 加入快取 (Cache)，避免每次按按鈕都重新抓證交所，提升手機體驗速度
@st.cache_data(ttl=3600)
def get_institutional_data(stock_id, trade_date=None, market_hint=None):
    stock_no = stock_id.strip().upper().replace(".TW", "").replace(".TWO", "")
    if trade_date is None: trade_date = datetime.now().date()

    headers = {"User-Agent": "Mozilla/5.0"}
    prefer_twse = True
    if isinstance(market_hint, str) and market_hint.upper().endswith(".TWO"): prefer_twse = False

    def try_twse(d_):
        url = f"https://www.twse.com.tw/fund/T86?response=csv&date={d_.strftime('%Y%m%d')}&selectType=ALLBUT0999"
        try:
            r = requests.get(url, headers=headers, timeout=10)
            if r.status_code != 200 or len(r.text) < 200: return None, "HTTP Err"
            df = _parse_twse_t86_csv(r.text)
            return df, None
        except:
            return None, "Req Err"

    def try_tpex(d_):
        roc_year = d_.year - 1911
        roc_date = f"{roc_year:03d}/{d_.month:02d}/{d_.day:02d}"
        url = f"https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php?l=zh-tw&o=csv&se=EW&t=D&d={roc_date}&s=0,asc"
        headers2 = dict(headers)
        headers2["Referer"] = "https://www.tpex.org.tw/"
        try:
            r = requests.get(url, headers=headers2, timeout=10)
            if r.status_code != 200 or len(r.text) < 200: return None, "HTTP Err"
            df = _parse_tpex_csv(r.text)
            return df, None
        except:
            return None, "Req Err"

    markets = [("TWSE", try_twse), ("TPEx", try_tpex)]
    if not prefer_twse: markets = [("TPEx", try_tpex), ("TWSE", try_twse)]

    for back in range(0, 5):  # 往前找 5 天即可
        d = trade_date - timedelta(days=back)
        for mkt_name, fn in markets:
            df, err = fn(d)
            if df is None: continue

            code_col = next((c for c in df.columns if str(c).strip() in ("證券代號", "代號")), None)
            if code_col is None: continue

            row = df[df[code_col].astype(str).str.strip() == stock_no]
            if row.empty: continue

            cols = list(df.columns)
            foreign_col = _find_net_col(cols, "外資") or next((c for c in cols if "外資" in str(c)), None)
            trust_col = _find_net_col(cols, "投信") or next((c for c in cols if "投信" in str(c)), None)
            dealer_col = _find_net_col(cols, "自營商") or next(
                (c for c in cols if ("自營商" in str(c)) and ("買賣超" in str(c))), None)

            foreign = _safe_int(row.iloc[0][foreign_col]) if foreign_col else None
            trust = _safe_int(row.iloc[0][trust_col]) if trust_col else None
            dealer = _safe_int(row.iloc[0][dealer_col]) if dealer_col else None

            yf_suffix = ".TW" if mkt_name == "TWSE" else ".TWO"
            return {
                "id": f"{stock_no}{yf_suffix}",
                "date": d.strftime("%Y-%m-%d"),
                "foreign": foreign,
                "trust": trust,
                "dealer": dealer
            }
    return None


def calculate_indicators(df):
    # MA
    df['MA5'] = df['Close'].rolling(5).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA60'] = df['Close'].rolling(60).mean()

    # RSI (6日)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(6).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(6).mean()
    rs = gain / loss
    df['RSI6'] = 100 - (100 / (1 + rs))

    # KD
    low_min = df['Low'].rolling(9).min()
    high_max = df['High'].rolling(9).max()
    rsv = (df['Close'] - low_min) / (high_max - low_min) * 100
    df['K'] = rsv.ewm(com=2).mean()
    df['D'] = df['K'].ewm(com=2).mean()

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    macd = dif.ewm(span=9, adjust=False).mean()
    df['OSC'] = dif - macd

    return df


# --- 手機版介面優化 ---

# 輸入區塊
col1, col2 = st.columns([3, 1])
with col1:
    stock_input = st.text_input("輸入代號 (如 2330, 0050)", value="2330")
with col2:
    st.write("")
    st.write("")
    run_btn = st.button("🔍 分析", type="primary")

if run_btn and stock_input:
    stock_id = stock_input.strip().upper()

    # 預設後綴
    yf_ticker = f"{stock_id}.TW" if not (stock_id.endswith(".TW") or stock_id.endswith(".TWO")) else stock_id

    with st.spinner("正在連線 Yahoo 與 證交所資料庫..."):
        try:
            # 1. 抓取 K 線
            df = yf.download(yf_ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)
            df = clean_yf_columns(df)

            # 自動切換上櫃
            if df.empty and ".TW" in yf_ticker:
                yf_ticker = yf_ticker.replace(".TW", ".TWO")
                df = yf.download(yf_ticker, period="6mo", interval="1d", progress=False, auto_adjust=False)
                df = clean_yf_columns(df)

            if df.empty:
                st.error("❌ 找不到此股票")
                st.stop()

            # 計算指標
            df = calculate_indicators(df)
            latest = df.iloc[-1]
            prev = df.iloc[-2]

            # 2. 抓取籌碼 (帶入最後交易日)
            last_date = df.index[-1].date()
            chips = get_institutional_data(stock_id, last_date, market_hint=yf_ticker)

        except Exception as e:
            st.error(f"錯誤: {e}")
            st.stop()

    # --- 數據呈現 ---

    # 1. 股價大卡片
    change = latest['Close'] - prev['Close']
    pct = (change / prev['Close']) * 100
    color = "inverse" if change > 0 else "normal"  # Streamlit metric 自動處理顏色

    st.metric(
        label=f"{stock_id} 收盤價",
        value=f"{latest['Close']:.2f}",
        delta=f"{change:+.2f} ({pct:+.2f}%)"
    )

    # 2. 籌碼面 (重點顯示)
    st.markdown("### 💰 三大法人 (單位: 張)")
    if chips:
        c1, c2, c3 = st.columns(3)


        def show_chip(col, title, val):
            if val is None:
                col.metric(title, "-")
            else:
                val_lot = int(val / 1000)  # 換算張
                col.metric(title, f"{val_lot:,}", delta=None)  # delta可以用來顯示買賣超顏色，但metric會自動紅綠，這裡手動比較好


        # 為了強制 紅=買, 綠=賣，我們用 markdown 模擬
        def chip_html(title, val):
            if val is None: return f"**{title}**: N/A"
            val_lot = int(val / 1000)
            color = "#ff4b4b" if val_lot > 0 else "#09ab3b"  # Streamlit 紅綠
            arrow = "🔺" if val_lot > 0 else "🔻"
            if val_lot == 0: return f"**{title}**: 持平"
            return f"**{title}**<br><span style='color:{color}; font-size: 1.2rem; font-weight:bold'>{arrow} {abs(val_lot):,}</span>"


        c1.markdown(chip_html("外資", chips['foreign']), unsafe_allow_html=True)
        c2.markdown(chip_html("投信", chips['trust']), unsafe_allow_html=True)
        c3.markdown(chip_html("自營商", chips['dealer']), unsafe_allow_html=True)
        st.caption(f"籌碼日期: {chips['date']}")
    else:
        st.warning("⚠️ 查無法人資料")

    st.divider()

    # 3. 技術指標儀表板 (手機版面)
    st.markdown("### 🔍 技術指標診斷")

    col_k, col_rsi = st.columns(2)

    # KD
    k_v, d_v = latest['K'], latest['D']
    kd_msg = "黃金交叉 ↗️" if k_v > d_v else "死亡交叉 ↘️"
    if k_v > 80: kd_msg = "🔥 高檔鈍化"
    col_k.info(f"**KD(9,3,3)**\n\nK: {k_v:.1f} | D: {d_v:.1f}\n\n{kd_msg}")

    # RSI
    rsi_v = latest['RSI6']
    rsi_msg = "正常"
    if rsi_v > 80:
        rsi_msg = "🔥 過熱警戒"
    elif rsi_v < 20:
        rsi_msg = "❄️ 超賣區"
    col_rsi.info(f"**RSI(6)**\n\n數值: {rsi_v:.1f}\n\n{rsi_msg}")

    # MACD & 均線
    ma5_v = latest['MA5']
    osc_v = latest['OSC']
    price = latest['Close']
    ma_status = "站上 MA5 (短多)" if price > ma5_v else "跌破 MA5 (轉弱)"
    macd_status = "多頭擴大" if osc_v > 0 and osc_v > df.iloc[-2]['OSC'] else ("多頭縮小" if osc_v > 0 else "空頭")

    st.success(f"**趨勢判讀**: {ma_status} | MACD {macd_status}")

    # 4. K線圖
    st.markdown("### 📈 近 3 個月走勢")
    st.line_chart(df['Close'].tail(90))

    # 5. 近五日數據表
    with st.expander("查看近 5 日詳細數據"):
        recent = df.tail(5)[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        recent['Volume'] = (recent['Volume'] / 1000).astype(int)  # 換算張
        recent = recent.sort_index(ascending=False)  # 最新在上面
        st.dataframe(recent.style.format("{:.2f}"))