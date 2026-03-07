import json
import io
import os
import csv
from contextlib import redirect_stdout
from datetime import datetime

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_cookies_manager import EncryptedCookieManager
from google.oauth2 import service_account
from google.cloud import firestore

# 確保 stock.py 位於同一目錄下
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


# --- 獨立儲存四種類別，避免互相覆蓋，並加入 metadata 讓手機端也看得到上傳紀錄 ---
def load_regulatory_from_db():
    db = get_db()
    data = {
        "twse_attention": [],
        "tpex_attention": [],
        "twse_disposition": {},
        "tpex_disposition": {},
        "metadata": {}
    }
    if db:
        try:
            doc_ref = db.collection(FS_COLLECTION).document("regulatory_data")
            doc = doc_ref.get()
            if doc.exists:
                db_data = doc.to_dict()
                data["twse_attention"] = db_data.get("twse_attention", [])
                data["tpex_attention"] = db_data.get("tpex_attention", [])
                data["twse_disposition"] = db_data.get("twse_disposition", {})
                data["tpex_disposition"] = db_data.get("tpex_disposition", {})
                data["metadata"] = db_data.get("metadata", {})
        except Exception as e:
            st.warning(f"DB read regulatory failed: {e}")
    return data


def save_regulatory_to_db(data):
    db = get_db()
    if db:
        try:
            doc_ref = db.collection(FS_COLLECTION).document("regulatory_data")
            doc_ref.set(data, merge=True)
        except Exception as e:
            st.error(f"DB write regulatory failed: {e}")


def get_combined_regulatory_data():
    """將 DB 中分開儲存的 4 份名單，融合成 stock.py 需要的 2 個集合"""
    reg = st.session_state.regulatory_data
    attention_stocks = set(reg.get("twse_attention", [])) | set(reg.get("tpex_attention", []))
    disposition_stocks = {**reg.get("twse_disposition", {}), **reg.get("tpex_disposition", {})}
    return {
        "attention_stocks": attention_stocks,
        "disposition_stocks": disposition_stocks
    }


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
if "regulatory_data" not in st.session_state:
    # App 啟動時立刻從資料庫抓取最新資料
    st.session_state.regulatory_data = load_regulatory_from_db()


# -------------------------
# CSV Parsing for Regulatory Lists
# -------------------------
def _detect_encoding(raw_bytes):
    """Try common TW encodings."""
    for enc in ("utf-8-sig", "utf-8", "big5", "cp950"):
        try:
            raw_bytes.decode(enc)
            return enc
        except (UnicodeDecodeError, ValueError):
            continue
    return "big5"


def _find_header_row(rows):
    """Return (header_row_index, headers) by looking for '證券代號'."""
    for i, row in enumerate(rows):
        for cell in row:
            if "證券代號" in str(cell):
                return i, row
    return None, None


def _parse_regulatory_csv(uploaded_file, list_type):
    """
    Parse a regulatory CSV into structured data.
    list_type: "attention" or "disposition"
    Returns: (stock_codes_set, disposition_details_dict, count, error_msg)
    """
    try:
        raw = uploaded_file.read()
        enc = _detect_encoding(raw)
        text = raw.decode(enc, errors="replace")
        reader = csv.reader(io.StringIO(text))
        rows = list(reader)

        hdr_idx, headers = _find_header_row(rows)
        if hdr_idx is None:
            return set(), {}, 0, "無法找到欄位標頭 (需包含「證券代號」)"

        # Build column index mapping
        col_map = {}
        for ci, h in enumerate(headers):
            h_clean = h.strip()
            col_map[h_clean] = ci

        code_col = col_map.get("證券代號")
        if code_col is None:
            return set(), {}, 0, "找不到「證券代號」欄位"

        codes = set()
        details = {}

        for row in rows[hdr_idx + 1:]:
            if len(row) <= code_col:
                continue
            stock_code = row[code_col].strip()
            if not stock_code or not any(c.isdigit() for c in stock_code):
                continue

            codes.add(stock_code)

            if list_type == "disposition":
                # Extract disposition detail fields
                measure = ""
                content = ""
                remarks = ""
                period = ""

                for key in ("處置措施", "處置原因"):
                    if key in col_map and len(row) > col_map[key]:
                        measure = row[col_map[key]].strip()
                        if measure:
                            break

                if "處置內容" in col_map and len(row) > col_map["處置內容"]:
                    content = row[col_map["處置內容"]].strip()

                for key in ("備註",):
                    if key in col_map and len(row) > col_map[key]:
                        remarks = row[col_map[key]].strip()

                for key in ("處置起迄時間", "處置起訖時間"):
                    if key in col_map and len(row) > col_map[key]:
                        period = row[col_map[key]].strip()
                        if period:
                            break

                if stock_code not in details:
                    details[stock_code] = {
                        "measure": measure,
                        "content": content,
                        "remarks": remarks,
                        "period": period,
                    }

        return codes, details, len(codes), None
    except Exception as e:
        return set(), {}, 0, f"解析錯誤: {e}"


def process_and_save_upload(file, category, list_type, source_label):
    """處理單一檔案上傳，覆寫該類別並寫入資料庫，保留其他類別的資料"""
    codes, details, count, err = _parse_regulatory_csv(file, list_type)
    if err:
        st.error(err)
        return

    reg = st.session_state.regulatory_data
    
    # 🌟 新增保護機制：預防舊的 session_state 缺少 metadata 導致 KeyError
    if "metadata" not in reg:
        reg["metadata"] = {}

    if list_type == "attention":
        reg[category] = list(codes)
    else:
        for c, d in details.items():
            d["source"] = source_label
        reg[category] = details

    # 將 metadata 存檔，方便手機端讀取顯示
    reg["metadata"][category] = {
        "filename": file.name,
        "count": count,
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    st.session_state.regulatory_data = reg
    save_regulatory_to_db(reg)
    st.success(f"Successfully updated cloud DB with {file.name}!")
    st.rerun()


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
        "generated_mode": st.session_state.report_mode
    }
    st.session_state.results_archive.append(record)
    if len(st.session_state.results_archive) > 10:
        st.session_state.results_archive.pop(0)
    st.session_state.view_index = len(st.session_state.results_archive) - 1


def run_analysis(stock_id, as_of_date, write_history):
    sid = stock_id.strip().upper()
    if not sid:
        return

    st.session_state.current_id = sid
    save_current_to_cookie(sid)
    if write_history:
        push_history_cookie(sid)

    final_result = {}
    current_mode = st.session_state.report_mode

    try:
        raw_result = analyze_stock_technical(
            sid, as_of_date=as_of_date, mode=current_mode,
            regulatory_data=get_combined_regulatory_data()
        )

        if current_mode == "human":
            final_result["human_report"] = raw_result.get("human_report", "No report generated.")
            final_result["ai_report"] = None
        else:
            feat_data = raw_result.get("ai_report", {})
            final_result["ai_report"] = feat_data
            if feat_data:
                final_result["human_report"] = format_text_report(feat_data)
            else:
                final_result["human_report"] = "Deep Dive Analysis failed (no data)."

    except Exception as e:
        err_msg = f"Error: {type(e).__name__}: {e}"
        final_result = {"human_report": err_msg, "ai_report": {"error": str(e)}}
        st.session_state._last_debug = f"exception={type(e).__name__}"

    save_to_archive(sid, as_of_date, final_result)


def run_sector_analysis(sector_name, as_of_date, custom_list=None):
    final_report = ""
    try:
        final_report = analyze_sector_performance(
            sector_name, as_of_date=as_of_date, custom_tickers=custom_list, mode="human",
            regulatory_data=get_combined_regulatory_data()
        )
    except Exception as e:
        final_report = f"Sector analysis failed: {e}"
    save_to_archive(f"Quick: {sector_name}", as_of_date, final_report)


def run_full_sector_report(sector_name, as_of_date, custom_list=None):
    target_list = custom_list if custom_list else SECTOR_DICT.get(sector_name, [])
    if not target_list:
        save_to_archive(f"Full: {sector_name}", as_of_date, "No stocks.")
        return

    mode = st.session_state.report_mode

    if mode == "ai":
        all_reports = {}
        for stock in target_list:
            try:
                res = analyze_stock_technical(stock, as_of_date=as_of_date, mode="ai",
                                              regulatory_data=get_combined_regulatory_data())
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
        full_content = []
        full_content.append(f"Sector [{sector_name}] Quick Screen Report")
        full_content.append(f"Date: {as_of_date}")
        full_content.append(f"Stocks: {', '.join(target_list)}")
        full_content.append("=" * 60)
        full_content.append("")

        for stock in target_list:
            try:
                res = analyze_stock_technical(stock, as_of_date=as_of_date, mode="human",
                                              regulatory_data=get_combined_regulatory_data())
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
        final_struct = {"human_report": combined_report, "ai_report": None}
        save_to_archive(f"Full: {sector_name}", as_of_date, final_struct)


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
        ["Quick Screen", "Deep Dive"],
        index=idx,
        key="report_mode_radio",
        help="Quick Screen: Fast, text summary.\nDeep Dive: Slow, full JSON with all indicators.",
    )
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

    # Sidebar: Regulatory list status (Combined for display)
    comb_reg = get_combined_regulatory_data()
    _reg_attn = len(comb_reg["attention_stocks"])
    _reg_disp = len(comb_reg["disposition_stocks"])
    
    if _reg_attn or _reg_disp:
        st.divider()
        st.subheader("Regulatory Lists")
        st.caption(f"Attention: {_reg_attn} | Disposition: {_reg_disp}")
    else:
        st.divider()
        st.caption("No regulatory lists loaded")

    st.divider()
    st.subheader("🤖 AI 操盤手分析指令")
    st.caption("💡 點擊右上方按鈕一鍵複製，連同下載的 JSON 貼給 AI")

    ai_prompt = """我上傳了最新的台股快篩數據，我的交易風格是「專注於高盈虧比、跟隨法人籌碼的短波段交易」。請幫我執行嚴格的汰弱留強。並且只要分析我上傳的檔案。

【第一階段：系統性過濾 (請在內心執行，不需全部列出)】
的標的分為「🎯 優先重壓區」、「⚠️ 觀望/防守區」、「🚨 必須停損區」。

【第二階段：深度分析輸出 (請針對名單中的重點股票回答)】
請針對各區塊的代表性股票，用以下 5 大維度給出簡潔、拳拳到肉的分析：
1. 趨勢與型態：目前趨勢狀態？是否剛完成什麼底部/突破型態？
2. 籌碼與動能：法人 (特別是外資) 是否在偷偷吃貨？MACD/RSI 有無關鍵轉折或背離？
3. 風險報酬 (R:R)：現在進場的盈虧比是多少？潛在獲利與風險是否對等？
4. 極限防守線：若要進場/續抱，具體的 ATR 停損價位設在多少？會不會容易被洗盤 (Whipsaw)？
5. 操盤手決策：給出明確的明日行動指令（如：開盤直接買、掛回踩價低接、立刻市價停利）。

【排版要求】
語氣要專業、果斷、像操盤手對老闆的精準匯報。請多用列點，並將股票代號、停損價位、盈虧比等關鍵字加粗。"""
    st.code(ai_prompt, language="text")

# -------------------------
# Main Content
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Stock Analysis", "Sector Analysis", "Custom Sectors", "Regulatory Lists"])

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

# --- Tab 4: Regulatory Lists ---
with tab4:
    st.header("Regulatory Lists (Attention / Disposition)")
    st.caption("檔案上傳後將儲存至雲端，手機端不需重新上傳。")

    comb = get_combined_regulatory_data()
    c_attn = len(comb["attention_stocks"])
    c_disp = len(comb["disposition_stocks"])
    c_both = len(comb["attention_stocks"] & set(comb["disposition_stocks"].keys()))

    if c_attn or c_disp:
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("總計注意股", c_attn)
        sc2.metric("總計處置股", c_disp)
        sc3.metric("兩者皆是", c_both)
    else:
        st.info("目前雲端無處置股/注意股資料，請上傳 CSV。")

    st.markdown("---")

    reg_db = st.session_state.regulatory_data
    meta = reg_db.get("metadata", {})

    def display_metadata(category_key):
        if category_key in meta:
            m = meta[category_key]
            st.success(f"✅ 雲端已載入：**{m['count']}** 檔\n\n📄 檔案：{m['filename']}\n\n⏱️ 更新：{m['updated_at']}")
        else:
            st.info("☁️ 雲端尚無此資料")

    st.subheader("Attention List (注意股)")
    att_c1, att_c2 = st.columns(2)

    with att_c1:
        with st.container(border=True):
            st.markdown("**TWSE Attention (上市)**")
            display_metadata("twse_attention")
            twse_att_file = st.file_uploader("Upload TWSE Attention CSV", type=["csv"], key="up_twse_att")
            if twse_att_file:
                fid = f"{twse_att_file.name}_{twse_att_file.size}"
                if st.session_state.get("_last_twse_att") != fid:
                    st.session_state["_last_twse_att"] = fid
                    process_and_save_upload(twse_att_file, "twse_attention", "attention", "TWSE")

    with att_c2:
        with st.container(border=True):
            st.markdown("**TPEx Attention (上櫃)**")
            display_metadata("tpex_attention")
            tpex_att_file = st.file_uploader("Upload TPEx Attention CSV", type=["csv"], key="up_tpex_att")
            if tpex_att_file:
                fid = f"{tpex_att_file.name}_{tpex_att_file.size}"
                if st.session_state.get("_last_tpex_att") != fid:
                    st.session_state["_last_tpex_att"] = fid
                    process_and_save_upload(tpex_att_file, "tpex_attention", "attention", "TPEx")

    st.markdown("---")
    st.subheader("Disposition List (處置股)")
    disp_c1, disp_c2 = st.columns(2)

    with disp_c1:
        with st.container(border=True):
            st.markdown("**TWSE Disposition (上市)**")
            display_metadata("twse_disposition")
            twse_disp_file = st.file_uploader("Upload TWSE Disposition CSV", type=["csv"], key="up_twse_disp")
            if twse_disp_file:
                fid = f"{twse_disp_file.name}_{twse_disp_file.size}"
                if st.session_state.get("_last_twse_disp") != fid:
                    st.session_state["_last_twse_disp"] = fid
                    process_and_save_upload(twse_disp_file, "twse_disposition", "disposition", "TWSE")

    with disp_c2:
        with st.container(border=True):
            st.markdown("**TPEx Disposition (上櫃)**")
            display_metadata("tpex_disposition")
            tpex_disp_file = st.file_uploader("Upload TPEx Disposition CSV", type=["csv"], key="up_tpex_disp")
            if tpex_disp_file:
                fid = f"{tpex_disp_file.name}_{tpex_disp_file.size}"
                if st.session_state.get("_last_tpex_disp") != fid:
                    st.session_state["_last_tpex_disp"] = fid
                    process_and_save_upload(tpex_disp_file, "tpex_disposition", "disposition", "TPEx")

    st.markdown("---")

    if c_attn or c_disp:
        with st.expander("查看已載入的股票代號", expanded=False):
            if c_attn:
                st.markdown(f"**Attention ({c_attn})**: {', '.join(sorted(comb['attention_stocks']))}")
            if c_disp:
                st.markdown(f"**Disposition ({c_disp})**: {', '.join(sorted(comb['disposition_stocks'].keys()))}")

        if st.button("🗑️ 清空所有雲端處置/注意股資料", type="primary"):
            empty_data = {
                "twse_attention": [], "tpex_attention": [],
                "twse_disposition": {}, "tpex_disposition": {}, "metadata": {}
            }
            st.session_state.regulatory_data = empty_data
            save_regulatory_to_db(empty_data)
            # 清除所有快取 flag
            for k in ["_last_twse_att", "_last_tpex_att", "_last_twse_disp", "_last_tpex_disp"]:
                st.session_state.pop(k, None)
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

    c_prev, c_switch, c_next = st.columns([1, 1, 1])
    with c_prev:
        if st.button("Prev", disabled=(current_idx == 0), use_container_width=True):
            st.session_state.view_index -= 1
            st.rerun()

    with c_switch:
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

    content = record["content"]
    st.write("---")

    if isinstance(content, dict) and "human_report" in content and "ai_report" in content:
        if is_ai_mode():
            st.markdown("### Deep Dive Data")
            ai_data = content["ai_report"]

            if ai_data is None:
                st.warning("⚠️ 此筆紀錄為 Quick Screen 模式生成 (快速)，無 Deep Dive 詳細數據。")
                st.info("若需查看詳細籌碼/指標數據，請將左側模式切換為 'Deep Dive' 並重新按 Analyze。")
            else:
                if isinstance(ai_data, dict) and "stocks" in ai_data:
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
            st.markdown("### Quick Screen Report")
            st.code(content["human_report"], language="text")
    else:
        st.code(str(content), language="text")
else:
    st.info("No results yet. Use the tabs above to run an analysis.")
