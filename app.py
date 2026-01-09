# -*- coding: utf-8 -*-
import json
import io
from contextlib import redirect_stdout
from datetime import datetime

import streamlit as st

# ----------------------------
# 0) 依賴檢查（避免部署後直接白屏）
# ----------------------------
missing = []
for pkg, imp in [
    ("yfinance", "yfinance"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("requests", "requests"),
    ("truststore", "truststore"),
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
    st.info(
        "請確認 requirements.txt 已包含上述套件，並在 Streamlit Cloud 重新部署。\n"
        "若你已更新 requirements.txt 但仍缺套件，通常是快取卡住：Delete app → New app → Deploy。"
    )
    st.stop()

from streamlit_autorefresh import st_autorefresh
from streamlit_cookies_manager import EncryptedCookieManager

import stock
import fundamental  # ✅ 新增：基本面模組

st.set_page_config(page_title="專業股票分析儀表板", page_icon="📈", layout="wide")

# ----------------------------
# 1) Cookie（跨頁面/跨重開保留近 5 筆）
# ----------------------------
# 建議放在 Streamlit Cloud → App settings → Secrets：
# COOKIE_PASSWORD = "一段至少 32 字的隨機長字串"
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
# 2) Session State 初始化
# ----------------------------
if "history" not in st.session_state:
    st.session_state.history = load_history_from_cookie()

if "last_sid" not in st.session_state:
    st.session_state.last_sid = ""

if "last_result_tech" not in st.session_state:
    st.session_state.last_result_tech = ""

if "last_result_fund" not in st.session_state:
    st.session_state.last_result_fund = ""

if "last_mode" not in st.session_state:
    st.session_state.last_mode = "技術面"

# 每次 rerun 先清一次「已保存」旗標
st.session_state["_history_cookie_saved_in_run"] = False

# ----------------------------
# 3) Sidebar：輸入、歷史、模式、刷新
# ----------------------------
st.sidebar.title("🔍 參數")

mode = st.sidebar.radio(
    "分析模式",
    ["技術面", "基本面", "全部"],
    index=["技術面", "基本面", "全部"].index(st.session_state.last_mode),
    key="mode_radio"
)

sid = st.sidebar.text_input(
    "股票代號（台股：2330 / 2330.TW；美股：AAPL）",
    value=st.session_state.last_sid or "2330",
    key="sid_input"
).strip().upper()

auto_refresh = st.sidebar.checkbox("即時更新（每 5 秒刷新）", value=False, key="auto_refresh")
st.sidebar.caption("提示：開著即時更新＝每 5 秒會重新抓一次資料（慢很正常）。")

st.sidebar.markdown("---")
st.sidebar.subheader("🕘 最近查詢（前 5 筆）")

def push_history(x: str):
    x = (x or "").strip().upper()
    if not x:
        return
    h = [i for i in st.session_state.history if i != x]
    h.insert(0, x)
    st.session_state.history = h[:5]
    save_history_to_cookie(st.session_state.history)

# 歷史按鈕（點了會帶入 sid，但不自動跑，除非你開 auto_refresh）
for i, hsid in enumerate(st.session_state.history):
    if st.sidebar.button(hsid, key=f"history_btn_{i}"):
        sid = hsid
        st.session_state.last_sid = sid

st.sidebar.markdown("---")
run_btn = st.sidebar.button("▶️ 分析", type="primary", key="run_btn")

# ----------------------------
# 4) 自動刷新
# ----------------------------
if auto_refresh and sid:
    st_autorefresh(interval=5000, key="autorefresh_5s")

# ----------------------------
# 5) 執行分析（技術面 + 基本面）
# ----------------------------
def run_capture(fn, *args, **kwargs) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        fn(*args, **kwargs)
    return buf.getvalue()

def run_analysis(sid_: str, mode_: str):
    sid_ = (sid_ or "").strip().upper()
    if not sid_:
        return

    st.session_state.last_sid = sid_
    st.session_state.last_mode = mode_
    push_history(sid_)

    # 技術面：沿用你 stock.py 的 print 輸出（用 capture）
    if mode_ in ("技術面", "全部"):
        tech = run_capture(stock.analyze_stock_technical, sid_)
        st.session_state.last_result_tech = tech

    # 基本面：fundamental.py 回傳字串
    if mode_ in ("基本面", "全部"):
        fund = fundamental.analyze_fundamental(sid_)
        st.session_state.last_result_fund = fund

# 觸發條件：按鈕 / 自動刷新（以最後一次查詢為主）
if run_btn and sid:
    run_analysis(sid, mode)
elif auto_refresh and st.session_state.last_sid:
    run_analysis(st.session_state.last_sid, st.session_state.last_mode)

# ----------------------------
# 6) 主畫面：依模式顯示 tab（切換更直覺）
# ----------------------------
st.title("📈 專業股票分析儀表板")
st.caption(f"最後更新：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if st.session_state.last_mode == "技術面":
    st.subheader("📊 技術面")
    if st.session_state.last_result_tech:
        st.code(st.session_state.last_result_tech, language="text")
    else:
        st.info("尚未分析。請在左側輸入代號後按「分析」。")

elif st.session_state.last_mode == "基本面":
    st.subheader("🏢 基本面")
    if st.session_state.last_result_fund:
        st.code(st.session_state.last_result_fund, language="text")
    else:
        st.info("尚未分析。請在左側輸入代號後按「分析」。")

else:  # 全部
    tab1, tab2 = st.tabs(["📊 技術面", "🏢 基本面"])

    with tab1:
        if st.session_state.last_result_tech:
            st.code(st.session_state.last_result_tech, language="text")
        else:
            st.info("尚未分析。請在左側輸入代號後按「分析」。")

    with tab2:
        if st.session_state.last_result_fund:
            st.code(st.session_state.last_result_fund, language="text")
        else:
            st.info("尚未分析。請在左側輸入代號後按「分析」。")
