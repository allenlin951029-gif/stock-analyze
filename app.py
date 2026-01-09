import json
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_cookies_manager import EncryptedCookieManager

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from stock import analyze_stock_technical, get_chart_data

st.set_page_config(page_title="Stock Analyze", layout="wide")
st.title("Stock Analyze")

# -------------------------
# Cookies (persistent)
# -------------------------
cookies = EncryptedCookieManager(
    prefix="stock_analyze_",
    password=st.secrets.get("COOKIE_PASSWORD", "dev_password_change_me_32chars_min_____"),
)

if not cookies.ready():
    st.stop()

HIST_KEY = "history"
CUR_KEY = "current_id"

def load_history_from_cookie():
    raw = cookies.get(HIST_KEY)
    if not raw:
        return []
    try:
        v = json.loads(raw)
        if isinstance(v, list):
            return [str(x).strip().upper() for x in v if str(x).strip()]
    except Exception:
        return []
    return []

def load_current_from_cookie():
    v = cookies.get(CUR_KEY)
    return (v or "0050").strip().upper()

# ✅ 同一個 rerun 只存一次，避免 DuplicateElementKey
st.session_state["_cookie_saved_this_run"] = False

def commit_cookies_once():
    if not st.session_state.get("_cookie_saved_this_run", False):
        cookies.save()
        st.session_state["_cookie_saved_this_run"] = True

def save_history_to_cookie(history_list):
    cookies[HIST_KEY] = json.dumps(history_list, ensure_ascii=False)
    commit_cookies_once()

def save_current_to_cookie(sid):
    cookies[CUR_KEY] = sid.strip().upper()
    commit_cookies_once()

# -------------------------
# Session state init
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = load_history_from_cookie()
if "current_id" not in st.session_state:
    st.session_state.current_id = load_current_from_cookie()
if "report" not in st.session_state:
    st.session_state.report = ""

# -------------------------
# Helpers
# -------------------------
def push_history(stock_id: str):
    sid = stock_id.strip().upper()
    if not sid:
        return
    st.session_state.history = [x for x in st.session_state.history if x != sid]
    st.session_state.history.insert(0, sid)
    st.session_state.history = st.session_state.history[:5]
    save_history_to_cookie(st.session_state.history)

def run_analysis(stock_id: str, write_history: bool):
    sid = stock_id.strip().upper()
    if not sid:
        st.session_state.report = "請輸入股票代號"
        return

    st.session_state.current_id = sid
    save_current_to_cookie(sid)

    if write_history:
        push_history(sid)

    with st.spinner(f"正在分析 {sid} ..."):
        st.session_state.report = analyze_stock_technical(sid)

# -------------------------
# Chart (cache to reduce API hits)
# -------------------------
@st.cache_data(ttl=8, show_spinner=False)  # 5秒刷新下，至少不要每次都重打 yfinance
def cached_chart(stock_id: str, period: str):
    return get_chart_data(stock_id, period=period)

def plot_kline(ticker: str, df):
    df = df.dropna(subset=["Open","High","Low","Close"]).copy()
    if df.empty:
        return None

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.60, 0.18, 0.22],
        specs=[[{"type":"candlestick"}],[{"type":"bar"}],[{"type":"scatter"}]],
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
            name="K線",
        ),
        row=1, col=1
    )

    # MA
    fig.add_trace(go.Scatter(x=df.index, y=df["MA5"],  name="MA5",  mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["MA60"], name="MA60", mode="lines"), row=1, col=1)

    # Bollinger
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_UP"],  name="BB上軌", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_MID"], name="BB中軌", mode="lines"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["BB_DN"],  name="BB下軌", mode="lines"), row=1, col=1)

    # Volume
    vol = df["Volume"].fillna(0)
    fig.add_trace(go.Bar(x=df.index, y=vol, name="成交量"), row=2, col=1)

    # KD + RSI
    fig.add_trace(go.Scatter(x=df.index, y=df["K"], name="K", mode="lines"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["D"], name="D", mode="lines"), row=3, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df["RSI14"], name="RSI14", mode="lines"), row=3, col=1)

    fig.update_layout(
        title=f"{ticker} K線圖",
        xaxis_rangeslider_visible=False,
        height=800,
        legend_orientation="h",
    )
    return fig

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.subheader("設定")
    auto = st.toggle("即時更新（每 5 秒刷新）", value=False)
    if auto:
        st_autorefresh(interval=5000, key="autorefresh_5s")

    st.divider()
    show_chart = st.toggle("顯示視覺化 K 線圖", value=True)
    period = st.selectbox("圖表期間", ["1mo","3mo","6mo","1y","2y"], index=2)

    st.divider()
    st.subheader("搜尋歷史（前 5 筆，會保存）")

    if st.session_state.history:
        picked = st.selectbox("點選快速回查", st.session_state.history, index=0)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("回查這筆", use_container_width=True):
                run_analysis(picked, write_history=False)
        with col_b:
            if st.button("清除歷史", use_container_width=True):
                st.session_state.history = []
                save_history_to_cookie([])
    else:
        st.caption("尚無歷史紀錄")

# -------------------------
# Main UI
# -------------------------
col1, col2 = st.columns([2, 1])
with col1:
    stock_id = st.text_input(
        "輸入股票代號（例：0050 / 2330 / 2330.TW / 6223.TWO）",
        value=st.session_state.current_id,
    )
with col2:
    st.write("")
    st.write("")
    search = st.button("開始分析", use_container_width=True)

if search:
    run_analysis(stock_id, write_history=True)

# Auto refresh rerun (do not spam history)
if auto and st.session_state.current_id:
    run_analysis(st.session_state.current_id, write_history=False)

# -------------------------
# Output
# -------------------------
st.divider()

if show_chart and st.session_state.current_id:
    ticker, df = cached_chart(st.session_state.current_id, period)
    if df is None or df.empty:
        st.warning("圖表資料抓取失敗（可能代號不支援或資料來源暫時無回應）")
    else:
        fig = plot_kline(ticker, df)
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="kline_chart")

st.subheader("文字報表")
if st.session_state.report and str(st.session_state.report).strip():
    st.code(st.session_state.report, language="text")
else:
    st.info("尚未分析。請輸入代號後按「開始分析」。")
