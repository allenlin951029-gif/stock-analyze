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

# 確保 stock.py 裡有這些函式與變數
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
# Firestore
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
# Helpers
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
    }
    st.session_state.results_archive.append(record)
    if len(st.session_state.results_archive) > 10:
        st.session_state.results_archive.pop(0)
    st.session_state.view_index = len(st.session_state.results_archive) - 1

def run_analysis(stock_id, as_of_date, write_history):
    # 1. 處理輸入與 Cookie
    sid = stock_id.strip().upper()
    if not sid:
        return
    st.session_state.current_id = sid
    save_current_to_cookie(sid)
    if write_history:
        push_history_cookie(sid)

    # 2. 執行分析 (縮排修正重點)
    final_result = None
    try:
        final_result = analyze_stock_technical(sid, as_of_date=as_of_date)
    except Exception as e:
        err_msg = "Error: {}: {}".format(type(e).__name__, e)
        final_result = {
            "human_report": err_msg,
            "ai_report": {"error": str(e)},
        }
        st.session_state._last_debug = "exception={}".format(type(e).__name__)

    # 3. 存檔 (縮排修正重點)
    save_to_archive(sid, as_of_date, final_result)

def run_sector_analysis(sector_name, as_of_date, custom_list=None):
    final_report = ""
    try:
        final_report = analyze_sector_performance(
            sector_name, as_of_date=as_of_date, custom_tickers=custom_list
        )
    except Exception as e:
        final_report = "Sector analysis failed: {}".format(e)
    save_to_archive("Quick: {}".format(sector_name), as_of_date, final_report)

def run_full_sector_report(sector_name, as_of_date, custom_list=None):
    target_list = custom_list if custom_list else SECTOR_DICT.get(sector_name, [])
    if not target_list:
        save_to_archive("Full: {}".format(sector_name), as_of_date, "No stocks.")
        return

    # 縮排修正重點：if/else 必須在函式內
    if is_ai_mode():
        all_reports = {}
        for stock in target_list:
            try:
                res = analyze_stock_technical(stock, as_of_date=as_of_date)
                if isinstance(res, dict):
                    all_reports[stock] = res.get("ai_report", {})
                else:
                    all_reports[stock] = {"error": "unexpected format"}
            except Exception as e:
                all_reports[stock] = {"error": str(e)}

        combined = {
            "human_report": "Sector [{}] AI report ({} stocks)\nDate: {}".format(
                sector_name, len(target_list), as_of_date
            ),
            "ai_report": {
                "sector": sector_name,
                "date": str(as_of_date),
                "stocks": all_reports,
            },
        }
        save_to_archive("Full: {}".format(sector_name), as_of_date, combined)
    else:
        full_content = []
        full_content.append("Sector [{}] Full Report".format(sector_name))
        full_content.append("Date: {}".format(as_of_date))
        full_content.append("Stocks: {}".format(", ".join(target_list)))
        full_content.append("=" * 60)
        full_content.append("")

        for stock in target_list:
            try:
                res = analyze_stock_technical(stock, as_of_date=as_of_date)
                if isinstance(res, dict):
                    full_content.append(res.get("human_report", str(res)))
                else:
                    full_content.append(str(res))
                full_content.append("")
                full_content.append("=" * 60)
                full_content.append("")
            except Exception as e:
                full_content.append("FAIL {}: {}".format(stock, e))
                full_content.append("-" * 60)

        combined_report = "\n".join(full_content)
        save_to_archive("Full: {}".format(sector_name), as_of_date, combined_report)

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.subheader("Settings")

    st.markdown("---")
    st.markdown("#### Report Mode")
    
    idx = 0 if st.session_state.report_mode == "human" else 1
    mode_choice = st.radio(
        "Select output mode:",
        ["Human (readable)", "AI (JSON data)"],
        index=idx,
        key="report_mode_radio",
        help="Human: concise text summary.\nAI: full JSON with all indicators.",
    )
    st.session_state.report_mode = "human" if mode_choice == "Human (readable)" else "ai"

    if is_ai_mode():
        st.info("AI mode: full JSON output with all numeric indicators and flags, downloadable.")
    else:
        st.success("Human mode: concise text summary with trend, indicators, and key flags.")

    st.markdown("---")
    auto = st.toggle("Auto-refresh (10s)", value=False, key="auto_refresh")
    tick = 0
    if auto:
        tick = st_autorefresh(interval=10_000, key="autorefresh_10s")
        st.caption("tick = {}".format(tick))

    st.divider()
    st.subheader("Search History")
    if st.session_state.history:
        picked = st.selectbox("Select", st.session_state.history, index=0, key="history_pick")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Recall", use_container_width=True, key="history_run"):
                with st.spinner("Analyzing {} ...".format(picked)):
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

# --- Tab 1 ---
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
        btn_label = "AI Analyze" if is_ai_mode() else "Analyze"
        search = st.button(btn_label, use_container_width=True, key="run_btn")

    if search:
        mode_str = "AI" if is_ai_mode() else "Human"
        with st.spinner("Analyzing {} ({}) ...".format(stock_id.strip().upper(), mode_str)):
            run_analysis(stock_id, st.session_state.as_of_date, write_history=True)
            st.rerun()

# --- Tab 2 ---
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
            st.markdown("**Stocks**: `{}`".format(", ".join(target_list)))
        else:
            st.markdown("**Stocks**: (none)")

        if is_ai_mode():
            st.caption("AI mode - full report outputs JSON")
        else:
            st.caption("Human mode - full report outputs text summary")

        b1, b2 = st.columns(2)
        with b1:
            if st.button("Quick Scan", use_container_width=True):
                with st.spinner("Scanning {} ...".format(selected_sector)):
                    clist = target_list if source_type == "Custom" else None
                    run_sector_analysis(selected_sector, sector_date, custom_list=clist)
                    st.rerun()
        with b2:
            if st.button("Full Report", use_container_width=True):
                mode_str = "AI" if is_ai_mode() else "Human"
                with st.spinner("Generating {} full report ({}) ...".format(selected_sector, mode_str)):
                    clist = target_list if source_type == "Custom" else None
                    run_full_sector_report(selected_sector, sector_date, custom_list=clist)
                    st.rerun()

# --- Tab 3 ---
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
                    st.success("Created {}".format(new_group))
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
                            st.success("Added {}".format(val))
                            st.rerun()

                st.divider()
                if not current_list:
                    st.caption("(empty)")
                else:
                    for s in current_list:
                        cr1, cr2 = st.columns([4, 1])
                        with cr1:
                            st.text("  {}".format(s))
                        with cr2:
                            if st.button("Remove", key="del_{}_{}".format(edit_group, s)):
                                current_list.remove(s)
                                save_sectors_to_db(st.session_state.custom_sectors)
                                st.rerun()

                st.divider()
                if st.button("Delete this sector", type="primary"):
                    del st.session_state.custom_sectors[edit_group]
                    save_sectors_to_db(st.session_state.custom_sectors)
                    st.rerun()

# -------------------------
# Auto Refresh
# -------------------------
if auto and tick != st.session_state.last_tick:
    st.session_state.last_tick = tick
    with st.spinner("Auto-refreshing {} ...".format(st.session_state.current_id)):
        run_analysis(st.session_state.current_id, st.session_state.as_of_date, write_history=False)
        st.rerun()

st.divider()

# -------------------------
# Pagination Display
# -------------------------
archive_len = len(st.session_state.results_archive)

if archive_len > 0:
    if st.session_state.view_index < 0:
        st.session_state.view_index = 0
    if st.session_state.view_index >= archive_len:
        st.session_state.view_index = archive_len - 1

    current_idx = st.session_state.view_index
    record = st.session_state.results_archive[current_idx]

    mode_badge = "AI" if is_ai_mode() else "Human"
    badge_color = "#1a73e8" if is_ai_mode() else "#2e7d32"

    st.markdown(
        '<div style="text-align:center;background:#262730;padding:10px;border-radius:5px;'
        'border:1px solid #464b5c;margin-bottom:10px;">'
        '<span style="font-size:1.2em;font-weight:bold;color:#fff;">{}</span>'
        '<span style="color:#ccc;font-size:0.9em;margin-left:10px;">({})</span>'
        '<span style="background:{};color:#fff;padding:2px 8px;border-radius:10px;'
        'font-size:0.75em;margin-left:8px;">{}</span>'
        '<br><span style="font-size:0.8em;color:#aaa;">'
        "{} / {} ({})</span></div>".format(
            record["id"],
            record["date"],
            badge_color,
            mode_badge,
            current_idx + 1,
            archive_len,
            record["created_at"],
        ),
        unsafe_allow_html=True,
    )

    # Pagination + inline mode switch
    c_prev, c_switch, c_next = st.columns([1, 1, 1])
    with c_prev:
        if st.button("Prev", disabled=(current_idx == 0), use_container_width=True):
            st.session_state.view_index -= 1
            st.rerun()
    with c_switch:
        content = record["content"]
        if isinstance(content, dict) and "human_report" in content and "ai_report" in content:
            switch_label = "Switch to Human" if is_ai_mode() else "Switch to AI"
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

    # Content display
    content = record["content"]
    st.write("---")

    if isinstance(content, dict) and "human_report" in content and "ai_report" in content:
        if is_ai_mode():
            st.markdown("### AI Features JSON Data")
            ai_data = content["ai_report"]

            if isinstance(ai_data, dict) and "stocks" in ai_data:
                st.markdown("**Sector**: {} | **Date**: {}".format(
                    ai_data.get("sector", "?"), ai_data.get("date", "?")
                ))
                st.markdown("**{} stocks**".format(len(ai_data["stocks"])))

                for stock_key, stock_data in ai_data["stocks"].items():
                    with st.expander(stock_key, expanded=False):
                        st.json(stock_data)

                json_str = json.dumps(ai_data, indent=2, default=str, ensure_ascii=False)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="sector_{}_{}.json".format(
                        ai_data.get("sector", "unknown"), ai_data.get("date", "")
                    ),
                    mime="application/json",
                    key="dl_sector_json_{}".format(current_idx),
                )
            else:
                st.json(ai_data)
                json_str = json.dumps(ai_data, indent=2, default=str, ensure_ascii=False)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="{}_{}_ai.json".format(record["id"], record["date"]),
                    mime="application/json",
                    key="dl_json_{}".format(current_idx),
                )
        else:
            st.markdown("### Human Report")
            st.code(content["human_report"], language="text")
    else:
        st.code(str(content), language="text")
else:
    st.info("No results yet. Use the tabs above to run an analysis.")
