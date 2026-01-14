import json
import io
from contextlib import redirect_stdout
from datetime import datetime

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_cookies_manager import EncryptedCookieManager

# 假設 stock.py 在同一目錄下，且未更動
from stock import analyze_stock_technical

st.set_page_config(page_title="Stock Analyze", layout="wide")
st.title("Stock Analyze (翻頁紀錄版)")

# -------------------------
# Cookies (保留原本儲存 代號歷史 的功能)
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

# [新增] 用來儲存完整分析結果的列表 (List of dict)
if "results_archive" not in st.session_state:
    st.session_state.results_archive = []

# [新增] 目前觀看的頁籤索引
if "view_index" not in st.session_state:
    st.session_state.view_index = 0

if "_last_debug" not in st.session_state:
    st.session_state._last_debug = ""
if "last_tick" not in st.session_state:
    st.session_state.last_tick = 0
if "as_of_date" not in st.session_state:
    st.session_state.as_of_date = datetime.now().date()

# -------------------------
# Helpers
# -------------------------
def push_history_cookie(stock_id: str):
    """只更新 Cookie 中的代號列表，不涉及報告內容"""
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
        # 如果是空的，不做任何存檔動作，直接返回
        return

    st.session_state.current_id = sid
    save_current_to_cookie(sid)

    if write_history:
        push_history_cookie(sid)

    # 執行分析
    buf = io.StringIO()
    ret = None
    final_report = ""
    
    try:
        with redirect_stdout(buf):
            ret = analyze_stock_technical(sid, as_of_date=as_of_date)

        stdout_text = buf.getvalue()
        ret_len = len(ret) if isinstance(ret, str) else -1

        st.session_state._last_debug = (
            f"ret_type={type(ret).__name__}, ret_len={ret_len}, "
            f"ret_is_none={ret is None}, stdout_len={len(stdout_text)}"
        )

        if isinstance(ret, str) and ret.strip():
            final_report = ret
        elif stdout_text.strip():
            final_report = stdout_text
        elif isinstance(ret, str):
            final_report = "（函式回傳字串但只有空白）"
        else:
            final_report = "（函式沒有有效輸出）"

    except Exception as e:
        final_report = f"⚠️ 分析失敗：{type(e).__name__}: {e}"
        st.session_state._last_debug = f"exception={type(e).__name__}"

    # [新增] 將結果存入 Archive
    record = {
        "id": sid,
        "date": str(as_of_date),
        "content": final_report,
        "created_at": datetime.now().strftime("%H:%M:%S")
    }
    
    # 加入列表尾端
    st.session_state.results_archive.append(record)
    
    # 限制最多 10 筆 (先進先出)
    if len(st.session_state.results_archive) > 10:
        st.session_state.results_archive.pop(0)
    
    # 自動跳轉到最新一頁 (最後一筆)
    st.session_state.view_index = len(st.session_state.results_archive) - 1


# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.subheader("設定")
    auto = st.toggle("即時更新（每 10 秒刷新）", value=False, key="auto_refresh")
    tick = 0
    if auto:
        tick = st_autorefresh(interval=10_000, key="autorefresh_10s")
        st.caption(f"autorefresh tick = {tick}")
    st.caption(f"debug: {st.session_state._last_debug}")

    st.divider()
    st.subheader("搜尋歷史（前 5 筆）")

    if st.session_state.history:
        picked = st.selectbox("點選回查", st.session_state.history, index=0, key="history_pick")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("回查這筆", use_container_width=True, key="history_run"):
                with st.spinner(f"正在分析 {picked} ..."):
                    run_analysis(picked, st.session_state.as_of_date, write_history=False)
        with col_b:
            if st.button("清除歷史", use_container_width=True, key="history_clear"):
                st.session_state.history = []
                save_history_to_cookie([])
    else:
        st.caption("尚無歷史紀錄")
    
    st.divider()
    st.info("💡 下方主畫面可左右翻頁查看最近 10 次的分析結果。")

# -------------------------
# Main
# -------------------------
col1, col_mid, col2 = st.columns([2.0, 1.1, 1.0])

with col1:
    stock_id = st.text_input(
        "輸入股票代號（例：0050 / 2330 / 2330.TW / 6223.TWO）",
        value=st.session_state.current_id,
        key="stock_id_input",
    )

with col_mid:
    as_of = st.date_input(
        "資料日期（預設今天）",
        value=st.session_state.as_of_date,
        key="as_of_date_input",
    )
    st.session_state.as_of_date = as_of

with col2:
    st.write("")
    st.write("")
    search = st.button("開始分析", use_container_width=True, key="run_btn")

# 1) 手動分析（會寫歷史）
if search:
    with st.spinner(f"正在分析 {stock_id.strip().upper()} ..."):
        run_analysis(stock_id, st.session_state.as_of_date, write_history=True)
        # 分析完需要 rerun 讓下方的 view_index 更新生效
        st.rerun()

# 2) 自動刷新：只在 tick 有變化時跑一次（不寫歷史）
if auto:
    if tick != st.session_state.last_tick:
        st.session_state.last_tick = tick
        with st.spinner(f"自動更新中：{st.session_state.current_id} ..."):
            run_analysis(st.session_state.current_id, st.session_state.as_of_date, write_history=False)
            st.rerun()

st.divider()

# -------------------------
# Pagination Display (翻頁顯示)
# -------------------------
archive_len = len(st.session_state.results_archive)

if archive_len > 0:
    # 確保索引不越界
    if st.session_state.view_index < 0:
        st.session_state.view_index = 0
    if st.session_state.view_index >= archive_len:
        st.session_state.view_index = archive_len - 1
    
    current_idx = st.session_state.view_index
    record = st.session_state.results_archive[current_idx]
    
    # 建立控制列：上一頁 | 資訊 | 下一頁
    nav_c1, nav_c2, nav_c3 = st.columns([1, 2, 1])
    
    with nav_c1:
        # 第一頁時禁用上一頁
        if st.button("⬅️ 上一頁", disabled=(current_idx == 0), use_container_width=True):
            st.session_state.view_index -= 1
            st.rerun()
            
    with nav_c3:
        # 最後一頁時禁用下一頁
        if st.button("下一頁 ➡️", disabled=(current_idx == archive_len - 1), use_container_width=True):
            st.session_state.view_index += 1
            st.rerun()

    with nav_c2:
        st.markdown(
            f"""
            <div style="text-align: center; background-color: #262730; padding: 10px; border-radius: 5px; border: 1px solid #464b5c;">
                <span style="font-size: 1.2em; font-weight: bold; color: #ffffff;">
                    {record['id']}
                </span>
                <br>
                <span style="color: #cccccc; font-size: 0.9em;">
                    資料日期: {record['date']} | 分析時間: {record['created_at']}
                </span>
                <br>
                <span style="font-size: 0.8em; color: #aaaaaa;">
                    (第 {current_idx + 1} / {archive_len} 筆紀錄)
                </span>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # 顯示報告內容
    st.code(record['content'], language="text")

else:
    st.info("尚未分析或目前沒有紀錄。請輸入代號並按「開始分析」。")
