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


def load_holdings_from_db():
    db = get_db()
    if db:
        try:
            doc_ref = db.collection(FS_COLLECTION).document(FS_DOCUMENT)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict().get("holdings", [])
            return []
        except Exception:
            return []
    return st.session_state.get("_temp_local_holdings", [])


def save_holdings_to_db(data):
    db = get_db()
    if db:
        try:
            doc_ref = db.collection(FS_COLLECTION).document(FS_DOCUMENT)
            doc_ref.set({"holdings": data}, merge=True)
        except Exception as e:
            st.error(f"DB write (holdings) failed: {e}")
    else:
        st.session_state["_temp_local_holdings"] = data


def load_regulatory_from_db():
    """Load attention & disposition lists from Firestore."""
    db = get_db()
    if db:
        try:
            doc_ref = db.collection(FS_COLLECTION).document(FS_DOCUMENT)
            doc = doc_ref.get()
            if doc.exists:
                d = doc.to_dict()
                attn_list = d.get("regulatory_attention", [])
                disp_dict = d.get("regulatory_disposition", {})
                return {
                    "attention_stocks": set(attn_list),
                    "disposition_stocks": disp_dict,
                }
        except Exception:
            pass
    return st.session_state.get("_temp_local_regulatory", {
        "attention_stocks": set(),
        "disposition_stocks": {},
    })


def save_regulatory_to_db(reg_data):
    """Save attention & disposition lists to Firestore."""
    db = get_db()
    attn = list(reg_data.get("attention_stocks", set()))
    disp = reg_data.get("disposition_stocks", {})
    if db:
        try:
            doc_ref = db.collection(FS_COLLECTION).document(FS_DOCUMENT)
            doc_ref.set({
                "regulatory_attention": attn,
                "regulatory_disposition": disp,
            }, merge=True)
        except Exception as e:
            st.error(f"DB write (regulatory) failed: {e}")
    else:
        st.session_state["_temp_local_regulatory"] = reg_data


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
if "holdings" not in st.session_state:
    st.session_state.holdings = load_holdings_from_db()
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
    st.session_state.regulatory_data = load_regulatory_from_db()
if "regulatory_upload_info" not in st.session_state:
    st.session_state.regulatory_upload_info = {
        "twse_attention": None,
        "tpex_attention": None,
        "twse_disposition": None,
        "tpex_disposition": None,
    }


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

                # TWSE format: 處置措施, 處置內容, 備註, 處置起迄時間
                # TPEx format: 處置原因, 處置內容, 處置起訖時間
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

                # Keep only the latest entry per stock code
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


def rebuild_regulatory_data():
    """Rebuild the merged regulatory_data from all uploads."""
    info = st.session_state.regulatory_upload_info
    all_attention = set()
    all_disposition = {}

    for key in ("twse_attention", "tpex_attention"):
        data = info.get(key)
        if data and "codes" in data:
            all_attention.update(data["codes"])

    for key, source_label in (("twse_disposition", "TWSE"), ("tpex_disposition", "TPEx")):
        data = info.get(key)
        if data and "details" in data:
            for code, detail in data["details"].items():
                detail_with_source = dict(detail, source=source_label)
                if code not in all_disposition:
                    all_disposition[code] = detail_with_source

    st.session_state.regulatory_data = {
        "attention_stocks": all_attention,
        "disposition_stocks": all_disposition,
    }
    save_regulatory_to_db(st.session_state.regulatory_data)


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
        raw_result = analyze_stock_technical(
            sid, as_of_date=as_of_date, mode=current_mode,
            regulatory_data=st.session_state.get("regulatory_data")
        )

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
            sector_name, as_of_date=as_of_date, custom_tickers=custom_list, mode="human",
            regulatory_data=st.session_state.get("regulatory_data")
        )
    except Exception as e:
        final_report = f"Sector analysis failed: {e}"
    save_to_archive(f"Quick: {sector_name}", as_of_date, final_report)


def _get_holdings_lookup():
    """Build a lookup dict: stock_id -> {avg_price, shares} from holdings."""
    lookup = {}
    for h in st.session_state.get("holdings", []):
        sid = h.get("stock_id", "").strip().upper()
        if sid:
            lookup[sid] = {
                "avg_price": h.get("avg_price", 0),
                "shares": h.get("shares", 0),
            }
    return lookup


def _inject_holdings_info(report_data, stock_id, holdings_lookup):
    """Inject holdings info (avg_price, shares, return_pct) into a report dict."""
    sid = stock_id.strip().upper()
    h = holdings_lookup.get(sid)
    if not h or not isinstance(report_data, dict):
        return report_data

    avg_price = h["avg_price"]
    shares = h["shares"]
    close = report_data.get("close")

    report_data["holding_avg_price"] = avg_price
    report_data["holding_shares"] = shares
    report_data["holding_cost"] = round(avg_price * shares, 2)

    if close and avg_price > 0:
        ret_pct = round((close - avg_price) / avg_price * 100, 2)
        report_data["holding_return_pct"] = ret_pct
        report_data["holding_unrealized_pnl"] = round((close - avg_price) * shares, 2)
    else:
        report_data["holding_return_pct"] = None
        report_data["holding_unrealized_pnl"] = None

    return report_data


def _holdings_text_block(stock_id, close, holdings_lookup):
    """Generate a text block for holdings info to append to human reports."""
    sid = stock_id.strip().upper()
    h = holdings_lookup.get(sid)
    if not h:
        return ""

    avg_price = h["avg_price"]
    shares = h["shares"]
    cost = avg_price * shares
    lines = []
    lines.append(f"  ── Holdings Info ──")
    lines.append(f"  Avg Price：{avg_price:.2f}  |  Shares：{shares}")
    lines.append(f"  Cost：{cost:,.0f}")
    if close and avg_price > 0:
        ret_pct = (close - avg_price) / avg_price * 100
        pnl = (close - avg_price) * shares
        emoji = "📈" if ret_pct >= 0 else "📉"
        lines.append(f"  {emoji} Return：{ret_pct:+.2f}%  |  P&L：{pnl:+,.0f}")
    return "\n".join(lines)


def run_full_sector_report(sector_name, as_of_date, custom_list=None):
    """
    類股完整報告 (根據當前模式跑迴圈)
    """
    target_list = custom_list if custom_list else SECTOR_DICT.get(sector_name, [])
    if not target_list:
        save_to_archive(f"Full: {sector_name}", as_of_date, "No stocks.")
        return

    mode = st.session_state.report_mode  # 'human' or 'ai'
    holdings_lookup = _get_holdings_lookup()

    if mode == "ai":  # Deep Dive
        all_reports = {}
        for stock in target_list:
            try:
                # 強制跑 AI 模式收集數據
                res = analyze_stock_technical(stock, as_of_date=as_of_date, mode="ai",
                                              regulatory_data=st.session_state.get("regulatory_data"))
                if isinstance(res, dict):
                    stock_data = res.get("ai_report", {})
                    # Inject holdings info if this stock is in holdings
                    stock_data = _inject_holdings_info(stock_data, stock, holdings_lookup)
                    all_reports[stock] = stock_data
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
                res = analyze_stock_technical(stock, as_of_date=as_of_date, mode="human",
                                              regulatory_data=st.session_state.get("regulatory_data"))
                if isinstance(res, dict):
                    report_text = res.get("human_report", str(res))
                    full_content.append(report_text)
                    # Append holdings info if available
                    # Extract close price from report text
                    close_val = None
                    for line in report_text.split("\n"):
                        if "Close" in line and "：" in line:
                            try:
                                close_str = line.split("Close：")[1].split()[0]
                                close_val = float(close_str)
                            except (IndexError, ValueError):
                                pass
                    h_block = _holdings_text_block(stock, close_val, holdings_lookup)
                    if h_block:
                        full_content.append(h_block)
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

    # Regulatory list status
    _reg_attn = len(st.session_state.regulatory_data.get("attention_stocks", set()))
    _reg_disp = len(st.session_state.regulatory_data.get("disposition_stocks", {}))
    if _reg_attn or _reg_disp:
        st.divider()
        st.subheader("Regulatory Lists")
        st.caption(f"Attention: {_reg_attn} | Disposition: {_reg_disp}")
    else:
        st.divider()
        st.caption("No regulatory lists loaded")

    # ==========================================
    # AI 操盤手提示詞複製區（可折疊）
    # ==========================================
    st.divider()
    st.subheader("🤖 AI 提示詞")
    st.caption("展開複製，連同 JSON 貼給 AI")

    with st.expander("📋 提示詞 A：潛力股篩選", expanded=False):
        prompt_a = r"""你是一位台股技術分析專家，擅長多時間框架趨勢分析與量價結構判讀。
我將上傳一份 JSON 格式的候選股技術面資料，請依照以下框架嚴格篩選出具潛力的標的。

═══ 分析框架（按優先級排序）═══
「趨勢第一、賠率第二、法人籌碼做加減分」
【第一層：趨勢結構 — 決定能不能看】
- trend_state（uptrend / uptrend_pullback 才合格）
- mtf_alignment（aligned_bull 最佳，partial_bull 次之）
- daily_weekly_trend_agree = true
- weekly_trend_state = "uptrend"，weekly_above_ma20 = true
- ma5 > ma20 > ma60（均線多頭排列）
- ma5_slope_5d > 0 且 ma20_slope_5d > 0

【第二層：動能確認 — 決定值不值得追】
- RSI14：40–70 健康區間，>70 過熱
- KD：K > D 且 K < 80，flag_kd_golden_cross = true 加分
- MACD：macd_osc > 0 或 macd_osc_slope_5d > 0
- flag_macd_golden_cross = true 為強進攻信號
- ADX > 20 且 flag_di_bullish = true

【第三層：量價結構 — 是否有資金撐腰】
- flag_price_up_vol_up = true（價量齊揚）
- vol_ratio_5d > 1.0，up_down_vol_ratio_20d > 1.0
- obv_slope_5d > 0 且 obv_slope_20d > 0
⚠️ 反向審計：價格上漲但 vol_ratio_5d < 0.8 或 obv_slope_20d < 0 → 標註「量能不足」

【第四層：籌碼面 — 修正項，非決定項】
- foreign_20d_net > 0 或 trust_20d_net > 0
- flag_inst_consensus_buy = true 最佳
- flag_foreign_divergence = true → 標註背離

【第五層：風險評估】
- risk_reward_ratio ≥ 1.5 合格，≥ 2.0 優秀
- flag_poor_risk_reward = true 直接排除
- beta_60d > 2.0 標註「高 Beta」

═══ 輸出 ═══
每檔：5星評級、趨勢/動能/量價/籌碼/風險摘要、綜合判斷、風險提醒。最後排序總表。

═══ 規則 ═══
1. 交叉驗證至少 3 維度
2. 每個看多附反面檢查
3. supertrend_bullish=false 必須標註
4. entry_trigger_veto 不為空逐條列出

═══ 資料 ═══
【貼上 sector_候選名單 JSON】"""
        st.code(prompt_a, language="text")

    with st.expander("🔥 提示詞 B：進攻名單與買點（需開網頁搜尋）", expanded=False):
        prompt_b = r"""你是一位台股短中線交易策略師，風格「灌溉鮮花、砍掉雜草」。
⚠️ 請先開啟「網頁搜尋」功能。

═══ 第零步：網頁搜尋 ═══
「趨勢第一、賠率第二、法人籌碼做加減分」
【搜尋 1：市場與恐慌指標】
"VIX index today"、"S&P 500 this week"、"台股 大盤 本週"
→ VIX>25 警戒，VIX>40「CTA大逃殺買點」

【搜尋 2：地緣政治】
"geopolitical risk 2026"、"台海 最新"、"oil price today"

【搜尋 3：川普社群發文 🇺🇸】
"Trump Truth Social latest"、"Trump tariff latest"、"川普 關稅 最新"
"Trump statement market reaction"
→ 是否提及關稅/制裁/中國/台灣/科技業/Fed？
→ 市場已反應程度？對候選標的有無衝擊？
⚠️ 口頭威脅 vs 正式行政命令要區分

【搜尋 4：個股消息（Bottom-up）】
每檔搜："[代號] 最新消息"、"[公司] 供應鏈 拉貨"、"[產業] 趨勢 2026"

【搜尋 5：逆風觀點】
"[公司] 利空 風險"、"[產業] 泡沫 過熱"
⚠️ 忽略社群農場文，優先法說會/財報/供應鏈數據

═══ 搜尋結果彙整（先輸出）═══
📡 市場環境（VIX/美股/台股/地緣/油價）
🇺🇸 川普動態（發文摘要/議題/反應程度/性質/對候選影響）
📰 每檔消息面（驅動力🟢🟡🔴/重大消息/供應鏈/利多出盡/逆風觀點）

═══ 選股邏輯 ═══
1. 只收鮮花：uptrend + aligned_bull，pos_52w_pct>70
2. 需求驅動優先：🟢需求>🟡成本改善>🔴成本轉嫁
3. 基本面+技術面：有EPS支撐的突破更安心
4. 捕捉市場犯錯：大跌但基本面無虞→「錯殺」
5. 利多出盡：利多但ret≤0→「overhyped」
6. 川普風險折價：正式政策衝擊→部位打5折或暫緩
7. 禁止在downtrend攤平

【買點】AVWAP→POC→Fib→均線→缺口
【部位】VIX>30打7折；川普衝擊再折

═══ 輸出 ═══
Tier1/Tier2/排除，每檔含：進場區間、停損、目標、RR、驅動力、川普風險、最大隱憂
資金配置（現金15-20%，高風險25%+）

═══ 規則 ═══
1. RR<1.5禁入 2. 全不合格就「建議觀望」
3. 川普推文恐慌≠基本面崩壞 4. 消息標來源不編造

═══ 前一輪結果 ═══
【貼上提示詞A輸出】
═══ 技術面資料 ═══
【貼上 JSON】"""
        st.code(prompt_b, language="text")

    with st.expander("🌸 提示詞 C：持股賣/抱決策（需開網頁搜尋）", expanded=False):
        prompt_c = r"""你是一位台股投資組合風控專家，核心原則「灌溉鮮花、砍掉雜草」。
⚠️ 請先開啟「網頁搜尋」功能。

═══ 第零步：網頁搜尋 ═══
「趨勢第一、賠率第二、法人籌碼做加減分」
【搜尋 1：系統性風險】
"VIX index today"、"台股 本週"、"Fed 利率"、"美中關係"、"oil price"
→ 🟢低/🟡中/🔴高

【搜尋 2：川普發文 🇺🇸】
"Trump Truth Social latest"、"Trump tariff 2026"、"川普 關稅 最新"
"Trump China Taiwan semiconductor"
→ 48hr內新發文？對持股產業衝擊？口頭vs行政命令？
→ 川普推文恐慌 ≠ 基本面崩壞

【搜尋 3：每檔持股消息】
"[代號] 最新消息"、"[公司] 法說會 財報"、"[公司] 供應鏈"

【搜尋 4：逆風觀點】
"[公司] 風險 利空"、"[產業] 衰退 泡沫"
→ 找不到看空觀點 = 過度擁擠

═══ 搜尋結果（先輸出）═══
📡 系統性風險（VIX/等級/美股/台股/地緣）
🇺🇸 川普動態（發文/性質/影響持股/歷史模式/應對建議）
📰 每檔消息面

═══ 分析框架 ═══
【鮮花】uptrend+aligned_bull+supertrend_bullish+跑贏大盤+價量齊揚+RR≥1.5+法人買超
【雜草】downtrend+aligned_bear+supertrend翻空+頂背離+弱於大盤+RR<1.0+法人賣超+放量殺盤

═══ 消息面交叉驗證 ═══
- 利多不漲⚠️：利多+ret≤0+量縮 → overhyped
- 利空不跌✅：利空+trend仍uptrend → 籌碼穩定
- 川普衝擊：推文但技術未破→觀察；技術已破+推文加壓→減碼
- 擁擠風險：beta>1.5+high_vol+無逆風觀點 → 連鎖停損
- 戰略彈性：供應鏈實際拉貨支撐→可短轉長；weekly也轉空→出場

═══ 輸出 ═══
每檔：🌸鮮花/⚠️邊緣/🪓雜草
含趨勢、鮮花vs雜草計分、風險、籌碼、消息面、川普影響
決策表：基本/轉強/惡化/系統性風險/川普升級 各情境
反向審計：續抱最大風險？賣出可能錯過？
持股總表（含消息面+川普風險欄位）
整體組合建議（鮮花雜草比/川普曝險度/現金部位/集中度）

═══ 規則 ═══
1. 全面轉空不續抱 2. 續抱必附停損
3. 雜草>50%→「組合需大幅調整」
4. VIX>40+CTA殺盤→不恐慌賣基本面好的（阿呆谷）
5. 但擁擠高槓桿股仍先砍部分（風控成本）
6. 川普口頭威脅≠實質政策 7. 農場文不算依據

═══ 持股資料 ═══
【貼上 sector_持股 JSON】"""
        st.code(prompt_c, language="text")
# -------------------------
# Main Content
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Stock Analysis", "Sector Analysis", "Custom Sectors", "Regulatory Lists", "Holdings"])

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
                sector_keys = list(st.session_state.custom_sectors.keys())
                # Guard: if widget remembers a deleted key, reset it
                if st.session_state.get("mgmt_select") not in sector_keys:
                    st.session_state["mgmt_select"] = sector_keys[0] if sector_keys else None

                edit_group = st.selectbox(
                    "Select sector",
                    sector_keys,
                    key="mgmt_select",
                )

                if edit_group and edit_group in st.session_state.custom_sectors:
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
                        # Force widget to forget the deleted value
                        st.session_state["mgmt_select"] = None
                        st.rerun()

# --- Tab 4: Regulatory Lists ---
with tab4:
    st.header("Regulatory Lists (Attention / Disposition)")
    st.caption(
        "Upload CSV files downloaded from TWSE or TPEx. "
        "Stocks on these lists will be flagged during analysis."
    )

    # --- Current status summary ---
    reg = st.session_state.regulatory_data
    attn_count = len(reg.get("attention_stocks", set()))
    disp_count = len(reg.get("disposition_stocks", {}))
    both_count = len(reg.get("attention_stocks", set()) & set(reg.get("disposition_stocks", {}).keys()))

    if attn_count or disp_count:
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Attention", attn_count)
        sc2.metric("Disposition", disp_count)
        sc3.metric("Both", both_count)
    else:
        st.info("No regulatory lists loaded. Upload CSVs below.")

    st.markdown("---")

    # Helper: check if an uploaded file is the same one we already processed
    def _is_new_file(uploaded_file, info_key):
        """Return True only if this file hasn't been processed yet."""
        if uploaded_file is None:
            return False
        existing = st.session_state.regulatory_upload_info.get(info_key)
        if existing is None:
            return True
        # Compare name + size to detect a genuinely new upload
        return (uploaded_file.name != existing.get("filename")
                or uploaded_file.size != existing.get("filesize"))

    # --- Attention uploads ---
    st.subheader("Attention List (注意股)")
    att_c1, att_c2 = st.columns(2)

    with att_c1:
        with st.container(border=True):
            st.markdown("**TWSE Attention**")
            twse_att_file = st.file_uploader(
                "Upload TWSE Attention CSV", type=["csv"],
                key="upload_twse_attention"
            )
            info_ta = st.session_state.regulatory_upload_info.get("twse_attention")
            if info_ta:
                st.success(f"Loaded: {info_ta['count']} stocks  ({info_ta['filename']})")
            if _is_new_file(twse_att_file, "twse_attention"):
                codes, _, count, err = _parse_regulatory_csv(twse_att_file, "attention")
                if err:
                    st.error(err)
                else:
                    st.session_state.regulatory_upload_info["twse_attention"] = {
                        "codes": codes, "count": count,
                        "filename": twse_att_file.name,
                        "filesize": twse_att_file.size,
                    }
                    rebuild_regulatory_data()
                    st.rerun()

    with att_c2:
        with st.container(border=True):
            st.markdown("**TPEx Attention**")
            tpex_att_file = st.file_uploader(
                "Upload TPEx Attention CSV", type=["csv"],
                key="upload_tpex_attention"
            )
            info_pa = st.session_state.regulatory_upload_info.get("tpex_attention")
            if info_pa:
                st.success(f"Loaded: {info_pa['count']} stocks  ({info_pa['filename']})")
            if _is_new_file(tpex_att_file, "tpex_attention"):
                codes, _, count, err = _parse_regulatory_csv(tpex_att_file, "attention")
                if err:
                    st.error(err)
                else:
                    st.session_state.regulatory_upload_info["tpex_attention"] = {
                        "codes": codes, "count": count,
                        "filename": tpex_att_file.name,
                        "filesize": tpex_att_file.size,
                    }
                    rebuild_regulatory_data()
                    st.rerun()

    st.markdown("---")

    # --- Disposition uploads ---
    st.subheader("Disposition List (處置股)")
    disp_c1, disp_c2 = st.columns(2)

    with disp_c1:
        with st.container(border=True):
            st.markdown("**TWSE Disposition**")
            twse_disp_file = st.file_uploader(
                "Upload TWSE Disposition CSV", type=["csv"],
                key="upload_twse_disposition"
            )
            info_td = st.session_state.regulatory_upload_info.get("twse_disposition")
            if info_td:
                st.success(f"Loaded: {info_td['count']} stocks  ({info_td['filename']})")
            if _is_new_file(twse_disp_file, "twse_disposition"):
                codes, details, count, err = _parse_regulatory_csv(twse_disp_file, "disposition")
                if err:
                    st.error(err)
                else:
                    st.session_state.regulatory_upload_info["twse_disposition"] = {
                        "codes": codes, "details": details, "count": count,
                        "filename": twse_disp_file.name,
                        "filesize": twse_disp_file.size,
                    }
                    rebuild_regulatory_data()
                    st.rerun()

    with disp_c2:
        with st.container(border=True):
            st.markdown("**TPEx Disposition**")
            tpex_disp_file = st.file_uploader(
                "Upload TPEx Disposition CSV", type=["csv"],
                key="upload_tpex_disposition"
            )
            info_pd = st.session_state.regulatory_upload_info.get("tpex_disposition")
            if info_pd:
                st.success(f"Loaded: {info_pd['count']} stocks  ({info_pd['filename']})")
            if _is_new_file(tpex_disp_file, "tpex_disposition"):
                codes, details, count, err = _parse_regulatory_csv(tpex_disp_file, "disposition")
                if err:
                    st.error(err)
                else:
                    st.session_state.regulatory_upload_info["tpex_disposition"] = {
                        "codes": codes, "details": details, "count": count,
                        "filename": tpex_disp_file.name,
                        "filesize": tpex_disp_file.size,
                    }
                    rebuild_regulatory_data()
                    st.rerun()

    st.markdown("---")

    # --- Loaded stock codes detail ---
    if attn_count or disp_count:
        with st.expander("View loaded stock codes", expanded=False):
            if attn_count:
                attn_sorted = sorted(reg["attention_stocks"])
                st.markdown(f"**Attention ({attn_count})**: {', '.join(attn_sorted)}")
            if disp_count:
                disp_sorted = sorted(reg["disposition_stocks"].keys())
                st.markdown(f"**Disposition ({disp_count})**: {', '.join(disp_sorted)}")

        # Clear all button
        if st.button("Clear all regulatory data", type="primary"):
            st.session_state.regulatory_data = {
                "attention_stocks": set(),
                "disposition_stocks": {},
            }
            st.session_state.regulatory_upload_info = {
                "twse_attention": None,
                "tpex_attention": None,
                "twse_disposition": None,
                "tpex_disposition": None,
            }
            save_regulatory_to_db(st.session_state.regulatory_data)
            st.rerun()

# --- Tab 5: Holdings ---
with tab5:
    st.header("📊 Holdings Management")
    st.caption("管理持股清單（股票代碼、成交均價、股數），方便搭配分析器使用")

    col_h1, col_h2 = st.columns([1, 1])

    with col_h1:
        with st.container(border=True):
            st.subheader("新增持股")
            h_c1, h_c2, h_c3 = st.columns(3)
            with h_c1:
                h_stock = st.text_input("股票代碼", key="h_stock_input", placeholder="例：2330")
            with h_c2:
                h_avg_price = st.number_input("成交均價", min_value=0.0, step=0.1, format="%.2f", key="h_avg_price")
            with h_c3:
                h_shares = st.number_input("股數（股）", min_value=0, step=1, key="h_shares")

            if st.button("➕ 新增", key="h_add_btn", use_container_width=True):
                h_val = h_stock.strip().upper()
                if not h_val:
                    st.error("請輸入股票代碼")
                elif h_avg_price <= 0:
                    st.error("成交均價須 > 0")
                elif h_shares <= 0:
                    st.error("股數須 > 0")
                else:
                    existing = [h["stock_id"] for h in st.session_state.holdings]
                    if h_val in existing:
                        st.error(f"{h_val} 已存在，請先刪除再重新新增")
                    else:
                        st.session_state.holdings.append({
                            "stock_id": h_val,
                            "avg_price": h_avg_price,
                            "shares": h_shares,
                        })
                        save_holdings_to_db(st.session_state.holdings)
                        st.success(f"已新增 {h_val}")
                        st.rerun()

    with col_h2:
        with st.container(border=True):
            st.subheader("快速操作")
            if st.session_state.holdings:
                holdings_tickers = [h["stock_id"] for h in st.session_state.holdings]
                st.markdown(f"**持股清單**：`{', '.join(holdings_tickers)}`")
                if st.button("🔍 建立「持股」族群", key="h_run_sector", use_container_width=True):
                    st.session_state.custom_sectors["持股"] = holdings_tickers
                    save_sectors_to_db(st.session_state.custom_sectors)
                    st.success("已建立「持股」族群，請切到 Sector Analysis → Custom → 持股 執行分析")
                    st.rerun()
            else:
                st.info("尚無持股資料")

    st.divider()
    if st.session_state.holdings:
        st.subheader("目前持股")
        hdr1, hdr2, hdr3, hdr4, hdr5 = st.columns([2, 2, 2, 2, 1])
        hdr1.markdown("**股票代碼**")
        hdr2.markdown("**成交均價**")
        hdr3.markdown("**股數（股）**")
        hdr4.markdown("**成本金額**")
        hdr5.markdown("**操作**")

        total_cost = 0.0
        indices_to_delete = []
        for i, h in enumerate(st.session_state.holdings):
            cost = h["avg_price"] * h["shares"]
            total_cost += cost
            r1, r2, r3, r4, r5 = st.columns([2, 2, 2, 2, 1])
            r1.text(h["stock_id"])
            r2.text(f"${h['avg_price']:,.2f}")
            r3.text(f"{h['shares']}")
            r4.text(f"${cost:,.0f}")
            with r5:
                if st.button("🗑️", key=f"h_del_{i}_{h['stock_id']}"):
                    indices_to_delete.append(i)

        if indices_to_delete:
            for idx in sorted(indices_to_delete, reverse=True):
                st.session_state.holdings.pop(idx)
            save_holdings_to_db(st.session_state.holdings)
            st.rerun()

        st.divider()
        st.markdown(f"**持股總成本**：`${total_cost:,.0f}`　|　**持股檔數**：`{len(st.session_state.holdings)}`")

        if st.button("🗑️ 清空全部持股", type="primary", key="h_clear_all"):
            st.session_state.holdings = []
            save_holdings_to_db([])
            st.rerun()
    else:
        st.info("尚無持股資料，請在上方新增。")

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
