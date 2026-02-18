import json
import io
import os
from contextlib import redirect_stdout
from datetime import datetime

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_cookies_manager import EncryptedCookieManager
from google.oauth2 import service_account
from google.cloud import firestore

# [span_0](start_span)確保 stock.py 位於同一目錄下[span_0](end_span)
from stock import (
    analyze_stock_technical,
    analyze_sector_performance,
    SECTOR_DICT,
    build_ai_features,
    format_text_report,
)

st.set_page_config(page_title="Stock Analyze", layout="wide")
st.title("Stock Analyze")

# -------------------------
# Firestore Configuration
# -------------------------
FS_COLLECTION = "stock_app_data"
FS_DOCUMENT = "config"

@st.cache_resource
def get_db():
    if "firebase" in st.secrets:
        try:
            # strict=False 允許 JSON 內的控制字元
            key_dict = json.loads(st.secrets["firebase"]["text_key"], strict=False)
            creds = service_account.Credentials.from_service_account_info(key_dict)
            db = firestore.Client(credentials=creds, project=key_dict["project_id"])
            return db
        except Exception as e:
            st.error(f"Firebase error: {e}")
            return None
    return None

def load_sectors_from_db():
    db = get_db()
    if db:
        try:
            doc_ref = db.collection(FS_COLLECTION).document(FS_DOCUMENT)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict().get("custom_sectors", {})
            return {}
        except Exception as e:
            st.warning(f"DB read failed: {e}")
            return {}
    return st.session_state.get("_temp_local_sectors", {})

def save_sectors_to_db(data):
    db = get_db()
    if db:
        try:
            doc_ref = db.collection(FS_COLLECTION).document(FS_DOCUMENT)
            doc_ref.set({"custom_sectors": data}, merge=True)
        except Exception as e:
            st.error(f"DB write failed: {e}")
    else:
        st.session_state["_temp_local_sectors"] = data
        st.warning("No Firebase. Data is temporary.")

# -------------------------
# Cookies Configuration
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
# Session Initialization
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = load_history_from_cookie()
if "current_id" not in st.session_state:
    st.session_state.current_id = load_current_from_cookie()
if "custom_sectors" not in st.session_state:
    st.session_state.custom_sectors = load_sectors_from_db()
if "results_archive" not in st.session_state:
    st.session_state.results_archive = []
if "view_index" not in st.session_state:
    st.session_state.view_index = 0
if "_last_debug" not in st.session_state:
    st.session_state._last_debug = ""
if "last_tick" not in st.session_state:
    st.session_state.last_tick = 0
if "as_of_date" not in st.session_state:
    st.session_state.as_of_date = datetime.now().date()
if "sector_as_of_date" not in st.session_state:
    st.session_state.sector_as_of_date = datetime.now().date()
if "report_mode" not in st.session_state:
    st.session_state.report_mode = "human"

# -------------------------
# Helper Functions
# -------------------------
def is_ai_mode():
    return st.session_state.report_mode == "ai"

def push_history_cookie(stock_id):
    sid = stock_id.strip().upper()
    if not sid:
        return
    st.session_state.history = [x for x in st.session_state.history if x != sid]
    st.session_state.history.insert(0, sid)
    st.session_state.history = st.session_state.history[:5]
    save_history_to_cookie(st.session_state.history)

def save_to_archive(display_title, display_date, content):
    record = {
        "id": display_title,
        "date": str(display_date),
        "content": content,
        "created_at": datetime.now().strftime("%H:%M:%S"),
        # 紀錄當時生成的模式，方便後續除錯或顯示
        "generated_mode": st.session_state.report_mode
    }
    st.session_state.results_archive.append(record)
    if len(st.session_state.results_archive) > 10:
        st.session_state.results_archive.pop(0)
    st.session_state.view_index = len(st.session_state.results_archive) - 1

def run_analysis(stock_id, as_of_date, write_history):
    """
    執行個股分析的核心函式。
    根據 session_state.report_mode 決定跑 'human' (快) 還是 'ai' (慢)。
    """
    sid = stock_id.strip().upper()
    if not sid:
        return
    
    # 1. 更新 Cookie 與 Session
    st.session_state.current_id = sid
    save_current_to_cookie(sid)
    if write_history:
        push_history_cookie(sid)

    # 2. 執行分析
    final_result = {}
    current_mode = st.session_state.report_mode  # 'human' or 'ai'

    try:
        # [span_1](start_span)呼叫 stock.py，傳入對應的 mode[span_1](end_span)
        # mode="human" -> 只回傳文字，速度快 (Quick Screen)
        # mode="ai"    -> 回傳完整 JSON，速度慢 (Deep Dive)
        raw_result = analyze_stock_technical(sid, as_of_date=as_of_date, mode=current_mode)
        
        if current_mode == "human":
            # Quick Screen 模式：省時間，不產生 AI 數據
            final_result["human_report"] = raw_result.get("human_report", "No report generated.")
            final_result["ai_report"] = None 
        else:
            # Deep Dive 模式：取得完整數據，並順便產生文字報告
            feat_data = raw_result.get("ai_report", {})
            final_result["ai_report"] = feat_data
            
            # 手動產生文字報告，這樣使用者切換回 Quick Screen View 時也能看到內容
            if feat_data:
                final_result["human_report"] = format_text_report(feat_data)
            else:
                final_result["human_report"] = "Deep Dive Analysis failed (no data)."

    except Exception as e:
        err_msg = f"Error: {type(e).__name__}: {e}"
        final_result = {
            "human_report": err_msg,
            "ai_report": {"error": str(e)},
        }
        st.session_state._last_debug = f"exception={type(e).__name__}"

    # 3. 存檔
    save_to_archive(sid, as_of_date, final_result)

def run_sector_analysis(sector_name, as_of_date, custom_list=None):
    """
    類股快速掃描 (通常用 Quick Screen 模式顯示列表)
    """
    final_report = ""
    try:
        # [span_2](start_span)Sector Scan 固定輸出文字列表 (Human mode)[span_2](end_span)
        final_report = analyze_sector_performance(
            sector_name, as_of_date=as_of_date, custom_tickers=custom_list, mode="human"
        )
    except Exception as e:
        final_report = f"Sector analysis failed: {e}"
    save_to_archive(f"Quick: {sector_name}", as_of_date, final_report)

def run_full_sector_report(sector_name, as_of_date, custom_list=None):
    """
    類股完整報告 (根據當前模式跑迴圈)
    """
    target_list = custom_list if custom_list else SECTOR_DICT.get(sector_name, [])
    if not target_list:
        save_to_archive(f"Full: {sector_name}", as_of_date, "No stocks.")
        return

    mode = st.session_state.report_mode # 'human' or 'ai'

    if mode == "ai": # Deep Dive
        all_reports = {}
        for stock in target_list:
            try:
                # 強制跑 AI 模式收集數據
                res = analyze_stock_technical(stock, as_of_date=as_of_date, mode="ai")
                if isinstance(res, dict):
                    all_reports[stock] = res.get("ai_report", {})
                else:
                    all_reports[stock] = {"error": "unexpected format"}
            except Exception as e:
                all_reports[stock] = {"error": str(e)}

        combined = {
            "human_report": f"Sector [{sector_name}] Deep Dive report ({len(target_list)} stocks)\nDate: {as_of_date}",
            "ai_report": {
                "sector": sector_name,
                "date": str(as_of_date),
                "stocks": all_reports,
            },
        }
        save_to_archive(f"Full: {sector_name}", as_of_date, combined)
    else:
        # Quick Screen (Human Mode)
        full_content = []
        full_content.append(f"Sector [{sector_name}] Quick Screen Report")
        full_content.append(f"Date: {as_of_date}")
        full_content.append(f"Stocks: {', '.join(target_list)}")
        full_content.append("=" * 60)
        full_content.append("")

        for stock in target_list:
            try:
                # 跑 Human 模式 (快)
                res = analyze_stock_technical(stock, as_of_date=as_of_date, mode="human")
                if isinstance(res, dict):
                    full_content.append(res.get("human_report", str(res)))
                else:
                    full_content.append(str(res))
                full_content.append("")
                full_content.append("=" * 60)
                full_content.append("")
            except Exception as e:
                full_content.append(f"FAIL {stock}: {e}")
                full_content.append("-" * 60)

        combined_report = "\n".join(full_content)
        # 這裡因為是純文字合併，結構稍微不同，為了統一 UI 顯示，我們包裝一下
        final_struct = {
            "human_report": combined_report,
            "ai_report": None
        }
        save_to_archive(f"Full: {sector_name}", as_of_date, final_struct)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.subheader("Settings")
    st.markdown("---")
    st.markdown("#### Report Mode")
    
    # 邏輯映射：Quick Screen -> Human mode, Deep Dive -> AI mode
    idx = 0 if st.session_state.report_mode == "human" else 1
    mode_choice = st.radio(
        "Select output mode:",
        ["Quick Screen", "Deep Dive"],
        index=idx,
        key="report_mode_radio",
        help="Quick Screen: Fast, text summary.\nDeep Dive: Slow, full JSON with all indicators.",
    )
    # 更新 session state
    st.session_state.report_mode = "human" if mode_choice == "Quick Screen" else "ai"

    if is_ai_mode():
        st.info("Deep Dive mode: 完整數據分析 (較慢，含籌碼/營收/JSON)。")
    else:
        st.success("Quick Screen mode: 快速文字摘要 (較快，僅技術面)。")

    st.markdown("---")
    auto = st.toggle("Auto-refresh (10s)", value=False, key="auto_refresh")
    tick = 0
    if auto:
        tick = st_autorefresh(interval=10_000, key="autorefresh_10s")
        st.caption(f"tick = {tick}")

    st.divider()
    st.subheader("Search History")
    if st.session_state.history:
        picked = st.selectbox("Select", st.session_state.history, index=0, key="history_pick")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Recall", use_container_width=True, key="history_run"):
                with st.spinner(f"Analyzing {picked} ..."):
                    run_analysis(picked, st.session_state.as_of_date, write_history=False)
                    st.rerun()
        with col_b:
            if st.button("Clear", use_container_width=True, key="history_clear"):
                st.session_state.history = []
                save_history_to_cookie([])
                st.rerun()
    else:
        st.caption("No history yet")

    st.divider()
    if "firebase" in st.secrets:
        st.success("Cloud DB connected")
    else:
        st.warning("No cloud DB")
    
    st.info("Use pagination below to browse results.")

# -------------------------
# Main Content
# -------------------------
tab1, tab2, tab3 = st.tabs(["Stock Analysis", "Sector Analysis", "Custom Sectors"])

# --- Tab 1: Stock Analysis ---
with tab1:
    col1, col_mid, col2 = st.columns([2.0, 1.1, 1.0])
    with col1:
        stock_id = st.text_input(
            "Stock ID (e.g. 2330 / 0050 / 6531.TW)",
            value=st.session_state.current_id,
            key="stock_id_input",
        )
    with col_mid:
        as_of = st.date_input("Date", value=st.session_state.as_of_date, key="as_of_date_input")
        st.session_state.as_of_date = as_of
    with col2:
        st.write("")
        st.write("")
        btn_label = "Deep Dive" if is_ai_mode() else "Quick Screen"
        search = st.button(btn_label, use_container_width=True, key="run_btn")

    if search:
        mode_str = "Deep Dive" if is_ai_mode() else "Quick Screen"
        with st.spinner(f"Analyzing {stock_id.strip().upper()} ({mode_str}) ..."):
            run_analysis(stock_id, st.session_state.as_of_date, write_history=True)
            st.rerun()

# --- Tab 2: Sector Analysis ---
with tab2:
    source_type = st.radio("Source:", ["Built-in", "Custom"], horizontal=True)
    c1, c2 = st.columns([2, 1])

    selected_sector = None
    target_list = []

    with c1:
        if source_type == "Built-in":
            opts = list(SECTOR_DICT.keys())
            selected_sector = st.selectbox("Sector", opts, key="sector_select_builtin")
            if selected_sector:
                target_list = SECTOR_DICT[selected_sector]
        else:
            custom_opts = list(st.session_state.custom_sectors.keys())
            if not custom_opts:
                st.warning("No custom sectors yet.")
            else:
                selected_sector = st.selectbox("Custom sector", custom_opts, key="sector_select_custom")
                if selected_sector:
                    target_list = st.session_state.custom_sectors[selected_sector]

    with c2:
        sector_date = st.date_input(
            "Date", value=st.session_state.sector_as_of_date, key="sector_date"
        )
        st.session_state.sector_as_of_date = sector_date

    if selected_sector:
        if target_list:
            st.markdown(f"**Stocks**: `{', '.join(target_list)}`")
        else:
            st.markdown("**Stocks**: (none)")

        if is_ai_mode():
            st.caption("Deep Dive mode - full report outputs JSON")
        else:
            st.caption("Quick Screen mode - full report outputs text summary")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Quick Scan", use_container_width=True):
                with st.spinner(f"Scanning {selected_sector} ..."):
                    clist = target_list if source_type == "Custom" else None
                    run_sector_analysis(selected_sector, sector_date, custom_list=clist)
                    st.rerun()
        with b2:
            if st.button("Full Report", use_container_width=True):
                mode_str = "Deep Dive" if is_ai_mode() else "Quick Screen"
                with st.spinner(f"Generating {selected_sector} full report ({mode_str}) ..."):
                    clist = target_list if source_type == "Custom" else None
                    run_full_sector_report(selected_sector, sector_date, custom_list=clist)
                    st.rerun()

# --- Tab 3: Custom Sectors ---
with tab3:
    st.header("Custom Sector Management")
    col_mgmt_1, col_mgmt_2 = st.columns(2)

    with col_mgmt_1:
        with st.container(border=True):
            st.subheader("Add Sector")
            new_group = st.text_input("Sector name")
            if st.button("Create"):
                if not new_group.strip():
                    st.error("Name cannot be empty")
                elif new_group in st.session_state.custom_sectors:
                    st.error("Name already exists")
                else:
                    st.session_state.custom_sectors[new_group] = []
                    save_sectors_to_db(st.session_state.custom_sectors)
                    st.success(f"Created {new_group}")
                    st.rerun()

    with col_mgmt_2:
        with st.container(border=True):
            st.subheader("Edit Sector")
            if not st.session_state.custom_sectors:
                st.info("No data")
            else:
                edit_group = st.selectbox(
                    "Select sector",
                    list(st.session_state.custom_sectors.keys()),
                    key="mgmt_select",
                )
                current_list = st.session_state.custom_sectors[edit_group]

                c_add1, c_add2 = st.columns([3, 1])
                with c_add1:
                    stock_to_add = st.text_input("Stock ID", key="mgmt_add_input")
                with c_add2:
                    st.write("")
                    st.write("")
                    if st.button("Add"):
                        val = stock_to_add.strip().upper()
                        if val and val not in current_list:
                            current_list.append(val)
                            save_sectors_to_db(st.session_state.custom_sectors)
                            st.success(f"Added {val}")
                            st.rerun()

                st.divider()
                if not current_list:
                    st.caption("(empty)")
                else:
                    for s in current_list:
                        cr1, cr2 = st.columns([4, 1])
                        with cr1:
                            st.text(f"  {s}")
                        with cr2:
                            if st.button("Remove", key=f"del_{edit_group}_{s}"):
                                current_list.remove(s)
                                save_sectors_to_db(st.session_state.custom_sectors)
                                st.rerun()

                st.divider()
                if st.button("Delete this sector", type="primary"):
                    del st.session_state.custom_sectors[edit_group]
                    save_sectors_to_db(st.session_state.custom_sectors)
                    st.rerun()

# -------------------------
# Auto Refresh Logic
# -------------------------
if auto and tick != st.session_state.last_tick:
    st.session_state.last_tick = tick
    with st.spinner(f"Auto-refreshing {st.session_state.current_id} ..."):
        run_analysis(st.session_state.current_id, st.session_state.as_of_date, write_history=False)
        st.rerun()

st.divider()

# -------------------------
# Pagination & Result Display
# -------------------------
archive_len = len(st.session_state.results_archive)

if archive_len > 0:
    if st.session_state.view_index < 0:
        st.session_state.view_index = 0
    if st.session_state.view_index >= archive_len:
        st.session_state.view_index = archive_len - 1

    current_idx = st.session_state.view_index
    record = st.session_state.results_archive[current_idx]

    # 顯示頂部的狀態 Bar (更新標籤)
    mode_badge = "Deep Dive" if is_ai_mode() else "Quick Screen"
    badge_color = "#1a73e8" if is_ai_mode() else "#2e7d32"
    
    st.markdown(
        f'<div style="text-align:center;background:#262730;padding:10px;border-radius:5px;'
        f'border:1px solid #464b5c;margin-bottom:10px;">'
        f'<span style="font-size:1.2em;font-weight:bold;color:#fff;">{record["id"]}</span>'
        f'<span style="color:#ccc;font-size:0.9em;margin-left:10px;">({record["date"]})</span>'
        f'<span style="background:{badge_color};color:#fff;padding:2px 8px;border-radius:10px;'
        f'font-size:0.75em;margin-left:8px;">{mode_badge} View</span>'
        f'<br><span style="font-size:0.8em;color:#aaa;">'
        f"{current_idx + 1} / {archive_len} ({record['created_at']})</span></div>",
        unsafe_allow_html=True,
    )

    # 導覽按鈕 + 模式切換按鈕
    c_prev, c_switch, c_next = st.columns([1, 1, 1])
    with c_prev:
        if st.button("Prev", disabled=(current_idx == 0), use_container_width=True):
            st.session_state.view_index -= 1
            st.rerun()
            
    with c_switch:
        # 切換按鈕 (更新標籤)
        switch_label = "Switch to Quick Screen" if is_ai_mode() else "Switch to Deep Dive"
        if st.button(switch_label, use_container_width=True, key="inline_mode_switch"):
            if is_ai_mode():
                st.session_state.report_mode = "human"
            else:
                st.session_state.report_mode = "ai"
            st.rerun()
            
    with c_next:
        if st.button("Next", disabled=(current_idx == archive_len - 1), use_container_width=True):
            st.session_state.view_index += 1
            st.rerun()

    # 內容顯示邏輯
    content = record["content"]
    st.write("---")

    # 檢查內容是否為我們定義的標準格式 (包含 human_report 與 ai_report)
    if isinstance(content, dict) and "human_report" in content and "ai_report" in content:
        
        if is_ai_mode():
            st.markdown("### Deep Dive Data")
            ai_data = content["ai_report"]
            
            # 檢查 Deep Dive (AI) 資料是否存在
            if ai_data is None:
                st.warning("⚠️ 此筆紀錄為 Quick Screen 模式生成 (快速)，無 Deep Dive 詳細數據。")
                st.info("若需查看詳細籌碼/指標數據，請將左側模式切換為 'Deep Dive' 並重新按 Analyze。")
            else:
                # 正常顯示 AI 數據
                if isinstance(ai_data, dict) and "stocks" in ai_data:
                    # 這是 Sector Report
                    st.markdown(f"**Sector**: {ai_data.get('sector', '?')} | **Date**: {ai_data.get('date', '?')}")
                    st.markdown(f"**{len(ai_data['stocks'])} stocks**")

                    for stock_key, stock_data in ai_data["stocks"].items():
                        with st.expander(stock_key, expanded=False):
                            st.json(stock_data)

                    json_str = json.dumps(ai_data, indent=2, default=str, ensure_ascii=False)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"sector_{ai_data.get('sector', 'unknown')}_{ai_data.get('date', '')}.json",
                        mime="application/json",
                        key=f"dl_sector_json_{current_idx}",
                    )
                else:
                    # 這是 Single Stock Report
                    st.json(ai_data)
                    json_str = json.dumps(ai_data, indent=2, default=str, ensure_ascii=False)
                    st.download_button(
                        label="Download JSON",
                        data=json_str,
                        file_name=f"{record['id']}_{record['date']}_deep_dive.json",
                        mime="application/json",
                        key=f"dl_json_{current_idx}",
                    )
        else:
            # Human Mode 顯示
            st.markdown("### Quick Screen Report")
            st.code(content["human_report"], language="text")
    else:
        # 非標準格式 (例如舊資料或錯誤訊息)
        st.code(str(content), language="text")
else:
    st.info("No results yet. Use the tabs above to run an analysis.")
