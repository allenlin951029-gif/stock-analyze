import json
from datetime import date

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_cookies_manager import EncryptedCookieManager

from stock import analyze_stock_technical

st.set_page_config(page_title="Stock Analyze", layout="wide")
st.title("Stock Analyze")

# -------------------------
# Cookies
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
# Session init
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = load_history_from_cookie()
if "current_id" not in st.session_state:
    st.session_state.current_id = load_current_from_cookie()
if "report" not in st.session_state:
    st.session_state.report = ""
if "as_of_date" not in st.session_state:
    st.session_state.as_of_date = date.today()

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

def run_analysis(stock_id: str, as_of_date, write_history: bool):
    sid = stock_id.strip().upper()
    if not sid:
        st.session_state.report = "請輸入股票代號"
        return

    st.session_state.current_id = sid
    save_current_to_cookie(sid)

    if write_history:
        push_history(sid)

    with st.spinner(f"正在分析 {sid}（基準日 {as_of_date}）..."):
        st.session_state.report = analyze_stock_technical(sid, as_of_date=as_of_date)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.subheader("設定")
    auto = st.toggle("即時更新（每 5 秒刷新）", value=False)
    if auto:
        st_autorefresh(interval=5000, key="autorefresh_5s")

    st.divider()
    st.subheader("搜尋歷史（前 5 筆，會保存）")

    if st.session_state.history:
        picked = st.selectbox("點選快速回查", st.session_state.history, index=0)
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("回查這筆", use_container_width=True):
                run_analysis(picked, st.session_state.as_of_date, write_history=False)
        with col_b:
            if st.button("清除歷史", use_container_width=True):
                st.session_state.history = []
                save_history_to_cookie([])
    else:
        st.caption("尚無歷史紀錄")

# -------------------------
# Main
# -------------------------
col1, col2 = st.columns([2, 1])

with col1:
    stock_id = st.text_input(
        "輸入股票代號（例：0050 / 2330 / 2330.TW / 6223.TWO）",
        value=st.session_state.current_id,
    )

with col2:
    st.session_state.as_of_date = st.date_input(
        "基準日（預設今天）",
        value=st.session_state.as_of_date,
    )
    search = st.button("開始分析", use_container_width=True)

if search:
    run_analysis(stock_id, st.session_state.as_of_date, write_history=True)

# Auto refresh: do not spam history
if auto and st.session_state.current_id:
    run_analysis(st.session_state.current_id, st.session_state.as_of_date, write_history=False)

st.divider()
if st.session_state.report and str(st.session_state.report).strip():
    st.code(st.session_state.report, language="text")
else:
    st.info("尚未分析。請輸入代號後按「開始分析」。")
