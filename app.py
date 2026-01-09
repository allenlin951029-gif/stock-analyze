# -*- coding: utf-8 -*-
import json
import io
from contextlib import redirect_stdout
from datetime import datetime

import streamlit as st

# ---- 套件檢查（避免白屏）----
missing = []
for pkg, imp in [
    ("yfinance", "yfinance"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("requests", "requests"),
    ("streamlit-autorefresh", "streamlit_autorefresh"),
    ("streamlit-cookies-manager", "streamlit_cookies_manager"),
]:
    try:
        __import__(imp)
    except Exception:
        missing.append(pkg)

if missing:
    st.set_page_config(page_title="專業股票分析儀表板", page_icon="📈", layout="wide")
    st.error(f"🚫 缺少套件：{', '.join(missing)}")
    st.stop()

from streamlit_autorefresh import st_autorefresh
from streamlit_cookies_manager import EncryptedCookieManager

import stock
import fundamental

st.set_page_config(page_title="專業股票分析儀表板", page_icon="📈", layout="wide")

# ----------------------------
# Cookie（保留前 5 筆查詢）
# ----------------------------
COOKIE_PASSWORD = st.secrets.get(
    "COOKIE_PASSWORD",
    "dev_only_change_me_to_a_random_string_32_chars_min!!!"
)

cookies = EncryptedCookieManager(prefix="stock_analyze", password=COOKIE_PASSWORD)
if not cookies.ready():
    st.stop()

def load_history_from_cookie():
    raw = cookies.get("history")
    if not raw:
        return []
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return [str(x) for x in data][:5]
    except Exception:
        pass
    return []

def save_history_to_cookie(history_list):
    # 避免同一次 rerun 多次 save 造成 component key 重複
    if st.session_state.get("_history_cookie_saved_in_run"):
        return
    cookies["history"] = json.dumps(history_list[:5], ensure_ascii=False)
    cookies.save()
    st.session_state["_history_cookie_saved_in_run"] = True

# ----------------------------
# Session State
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = load_history_from_cookie()

if "last_sid" not in st.session_state:
    st.session_state.last_sid = ""

if "last_mode" not in st.session_state:
    st.session_state.last_mode = "技術面"

if "last_result_tech" not in st.session_state:
    st.session_state.last_result_tech = ""

if "last_result_fund" not in st.session_state:
    st.session_state.last_result_fund = ""

st.session_state["_history_cookie_saved_in_run"] = False

def push_history(x: str):
    x = (x or "").strip().upper()
    if not x:
        return
    h = [i for i in st.session_state.history if i != x]
    h.insert(0, x)
    st.session_state.history = h[:5]
    save_history_to_cookie(st.session_state.history)

# ----------------------------
# UI：標題與上方操作列（按鈕在右邊）
# ----------------------------
st.title("📈 專業股票分析儀表板")
st.caption(f"最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

top = st.container()
with top:
    c1, c2, c3, c4 = st.columns([2.2, 1.0, 1.2, 0.8], vertical_alignment="bottom")

    with c1:
        sid = st.text_input(
            "股票代號（台股：2330 / 2330.TW；美股：AAPL）",
            value=st.session_state.last_sid or "2330",
            key="sid_input_main"
        ).strip().upper()

    with c2:
        mode = st.selectbox(
            "分析模式",
            ["技術面", "基本面", "全部"],
            index=["技術面", "基本面", "全部"].index(st.session_state.last_mode),
            key="mode_select_main",
        )

    with c3:
        auto_refresh = st.checkbox("即時更新（每 5 秒刷新）", value=False, key="auto_refresh_main")

    with c4:
        run_btn = st.button("分析", type="primary", use_container_width=True, key="run_btn_main")

# ----------------------------
# Sidebar：歷史紀錄
# ----------------------------
st.sidebar.subheader("🕘 最近查詢（前 5 筆）")
for i, hsid in enumerate(st.session_state.history):
    if st.sidebar.button(hsid, key=f"history_btn_{i}"):
        st.session_state.sid_input_main = hsid
        st.session_state.last_sid = hsid
        st.rerun()

# ----------------------------
# 自動刷新
# ----------------------------
if auto_refresh:
    st_autorefresh(interval=5000, key="autorefresh_5s")

# ----------------------------
# 執行分析（關鍵：捕捉例外，避免技術面 crash）
# ----------------------------
def run_capture_safe(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    try:
        with redirect_stdout(buf):
            fn(*args, **kwargs)
        txt = buf.getvalue()
        return txt if txt.strip() else "（技術面沒有輸出：可能抓取失敗或代號無資料）"
    except Exception as e:
        return f"⚠️ 技術面分析失敗：{type(e).__name__}: {e}"

def run_analysis(sid_: str, mode_: str):
    sid_ = (sid_ or "").strip().upper()
    if not sid_:
        return

    st.session_state.last_sid = sid_
    st.session_state.last_mode = mode_
    push_history(sid_)

    if mode_ in ("技術面", "全部"):
        st.session_state.last_result_tech = run_capture_safe(stock.analyze_stock_technical, sid_)

    if mode_ in ("基本面", "全部"):
        try:
            st.session_state.last_result_fund = fundamental.analyze_fundamental(sid_)
        except Exception as e:
            st.session_state.last_result_fund = f"⚠️ 基本面分析失敗：{type(e).__name__}: {e}"

# 觸發條件：按鈕 or 自動刷新
if run_btn and sid:
    run_analysis(sid, mode)
elif auto_refresh and st.session_state.last_sid:
    run_analysis(st.session_state.last_sid, st.session_state.last_mode)

# ----------------------------
# 顯示結果
# ----------------------------
mode_now = st.session_state.last_mode

if mode_now == "技術面":
    st.subheader("📊 技術面")
    st.code(st.session_state.last_result_tech or "尚未分析。", language="text")

elif mode_now == "基本面":
    st.subheader("🏢 基本面")
    st.code(st.session_state.last_result_fund or "尚未分析。", language="text")

else:
    t1, t2 = st.tabs(["📊 技術面", "🏢 基本面"])
    with t1:
        st.code(st.session_state.last_result_tech or "尚未分析。", language="text")
    with t2:
        st.code(st.session_state.last_result_fund or "尚未分析。", language="text")
