import streamlit as st
from streamlit_autorefresh import st_autorefresh

from stock import analyze_stock_technical

st.set_page_config(page_title="Stock Analyze", layout="wide")
st.title("Stock Analyze")

# -------------------------
# Session state init
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []  # 最近 5 筆
if "current_id" not in st.session_state:
    st.session_state.current_id = "0050"
if "last_manual_query" not in st.session_state:
    st.session_state.last_manual_query = None  # 避免自動刷新一直寫入歷史
if "report" not in st.session_state:
    st.session_state.report = ""


# -------------------------
# Helpers
# -------------------------
def push_history(stock_id: str):
    sid = stock_id.strip().upper()
    if not sid:
        return
    # 去重：若已存在先移除，再插到最前
    st.session_state.history = [x for x in st.session_state.history if x != sid]
    st.session_state.history.insert(0, sid)
    st.session_state.history = st.session_state.history[:5]


def run_analysis(stock_id: str, write_history: bool):
    sid = stock_id.strip().upper()
    if not sid:
        st.session_state.report = "請輸入股票代號"
        return

    # 只在「手動查詢」才寫歷史；自動刷新不寫，避免 2 秒洗版
    if write_history:
        push_history(sid)
        st.session_state.last_manual_query = sid

    with st.spinner(f"正在分析 {sid} ..."):
        st.session_state.report = analyze_stock_technical(sid)


# -------------------------
# Sidebar: controls
# -------------------------
with st.sidebar:
    st.subheader("設定")

    auto = st.toggle("即時更新（每 5 秒刷新）", value=False)

    # Auto refresh trigger
    if auto:
        st_autorefresh(interval=5000, key="autorefresh_5s")

    st.divider()
    st.subheader("搜尋歷史（前 5 筆）")

    if st.session_state.history:
        picked = st.selectbox(
            "點選快速回查",
            st.session_state.history,
            index=0,
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("回查這筆", use_container_width=True):
                st.session_state.current_id = picked
                run_analysis(picked, write_history=False)  # 回查不重複寫入
        with col_b:
            if st.button("清除歷史", use_container_width=True):
                st.session_state.history = []
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
    st.write("")  # spacing
    st.write("")  # spacing
    search = st.button("開始分析", use_container_width=True)

# 手動查詢
if search:
    st.session_state.current_id = stock_id.strip().upper()
    run_analysis(st.session_state.current_id, write_history=True)

# 自動刷新：若開啟且已有 current_id，就每次 rerun 重新分析
# 但不寫歷史（避免每 2 秒重複塞入）
if auto and st.session_state.current_id:
    run_analysis(st.session_state.current_id, write_history=False)

# Output
st.divider()
if st.session_state.report and str(st.session_state.report).strip():
    st.code(st.session_state.report, language="text")
else:
    st.info("尚未分析。請輸入代號後按「開始分析」。")

