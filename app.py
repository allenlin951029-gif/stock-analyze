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
    compute_market_features,
)

st.set_page_config(page_title="Stock Analyze", layout="wide")
st.title("Stock Analyze")

# [FIX] FinMind token — 融資融券資料來源（TWSE MI_MARGN 在雲端被擋，改用 FinMind API）
try:
    if "FINMIND_TOKEN" in st.secrets:
        os.environ["FINMIND_TOKEN"] = st.secrets["FINMIND_TOKEN"]
except Exception:
    pass

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
            # 【修復】使用 update 完全覆蓋欄位，避免 merge=True 導致被刪除的字典 Key 殘留
            doc_ref.update({"custom_sectors": data})
        except Exception:
            # 若文件不存在，則改用 set 建立
            doc_ref.set({"custom_sectors": data}, merge=True)
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
            doc_ref.update({"holdings": data})
        except Exception:
            doc_ref.set({"holdings": data}, merge=True)
    else:
        st.session_state["_temp_local_holdings"] = data


def load_attack_list_from_db():
    """
    載入進攻名單（依日期分組）
    結構: { "2026-04-30": [ {stock_id, stock_name, ai_judgments: [...]}, ... ] }
    """
    db = get_db()
    if db:
        try:
            doc_ref = db.collection(FS_COLLECTION).document(FS_DOCUMENT)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict().get("attack_list", {})
            return {}
        except Exception:
            return {}
    return st.session_state.get("_temp_local_attack_list", {})


def save_attack_list_to_db(data):
    db = get_db()
    if db:
        try:
            doc_ref = db.collection(FS_COLLECTION).document(FS_DOCUMENT)
            doc_ref.update({"attack_list": data})
        except Exception:
            doc_ref.set({"attack_list": data}, merge=True)
    else:
        st.session_state["_temp_local_attack_list"] = data


def load_regulatory_from_db():
    """
    【修復】同時載入 解析後的清單 與 上傳檔案資訊，避免休眠後狀態重置
    """
    db = get_db()
    res_data = {"attention_stocks": set(), "disposition_stocks": {}}
    res_info = {
        "twse_attention": None,
        "tpex_attention": None,
        "twse_disposition": None,
        "tpex_disposition": None,
    }

    if db:
        try:
            doc_ref = db.collection(FS_COLLECTION).document(FS_DOCUMENT)
            doc = doc_ref.get()
            if doc.exists:
                d = doc.to_dict()
                attn = d.get("regulatory_attention", [])
                disp = d.get("regulatory_disposition", {})
                res_data["attention_stocks"] = set(attn) if isinstance(attn, list) else set()
                res_data["disposition_stocks"] = disp if isinstance(disp, dict) else {}

                # 載入上傳資訊，並將裡面的 codes list 轉回 set
                db_info = d.get("regulatory_upload_info", {})
                if db_info:
                    for k in res_info.keys():
                        if db_info.get(k):
                            item = db_info[k]
                            if "codes" in item and isinstance(item["codes"], list):
                                item["codes"] = set(item["codes"])
                            res_info[k] = item
        except Exception:
            pass

    return res_data, res_info


def save_regulatory_to_db():
    """
    【修復】將上傳檔案的紀錄 (regulatory_upload_info) 一併存入資料庫
    """
    db = get_db()
    if not db:
        return
    reg = st.session_state.get("regulatory_data", {})
    info = st.session_state.get("regulatory_upload_info", {})

    # 準備可 JSON 序列化的 info (將 set 轉為 list)
    safe_info = {}
    for k, v in info.items():
        if v is None:
            safe_info[k] = None
        else:
            safe_dict = dict(v)
            if "codes" in safe_dict and isinstance(safe_dict["codes"], set):
                safe_dict["codes"] = list(safe_dict["codes"])
            safe_info[k] = safe_dict

    try:
        doc_ref = db.collection(FS_COLLECTION).document(FS_DOCUMENT)
        doc_ref.update({
            "regulatory_attention": sorted(reg.get("attention_stocks", set())),
            "regulatory_disposition": reg.get("disposition_stocks", {}),
            "regulatory_upload_info": safe_info
        })
    except Exception:
        try:
            doc_ref.set({
                "regulatory_attention": sorted(reg.get("attention_stocks", set())),
                "regulatory_disposition": reg.get("disposition_stocks", {}),
                "regulatory_upload_info": safe_info
            }, merge=True)
        except Exception as e:
            st.error(f"DB write (regulatory) failed: {e}")


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

# 【修復】解構載入資料與上傳資訊
if "regulatory_data" not in st.session_state or "regulatory_upload_info" not in st.session_state:
    r_data, r_info = load_regulatory_from_db()
    st.session_state.regulatory_data = r_data
    st.session_state.regulatory_upload_info = r_info

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
if "prompt_m_ai_result" not in st.session_state:
    st.session_state.prompt_m_ai_result = ""
if "attack_list" not in st.session_state:
    st.session_state.attack_list = load_attack_list_from_db()


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
    save_regulatory_to_db()


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
            regulatory_data=st.session_state.get("regulatory_data")
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
        final_result = {
            "human_report": err_msg,
            "ai_report": {"error": str(e)},
        }
        st.session_state._last_debug = f"exception={type(e).__name__}"

    save_to_archive(sid, as_of_date, final_result)


def run_sector_analysis(sector_name, as_of_date, custom_list=None):
    final_report = ""
    try:
        final_report = analyze_sector_performance(
            sector_name, as_of_date=as_of_date, custom_tickers=custom_list, mode="human",
            regulatory_data=st.session_state.get("regulatory_data")
        )
    except Exception as e:
        final_report = f"Sector analysis failed: {e}"
    save_to_archive(f"Quick: {sector_name}", as_of_date, final_report)


def _get_holdings_map():
    m = {}
    for h in st.session_state.get("holdings", []):
        sid = h.get("stock_id", "").strip().upper()
        if sid:
            m[sid] = {"avg_price": h.get("avg_price", 0), "shares": h.get("shares", 0)}
    return m


def run_full_sector_report(sector_name, as_of_date, custom_list=None):
    target_list = custom_list if custom_list else SECTOR_DICT.get(sector_name, [])
    if not target_list:
        save_to_archive(f"Full: {sector_name}", as_of_date, "No stocks.")
        return

    mode = st.session_state.report_mode
    h_map = _get_holdings_map()

    if mode == "ai":
        all_reports = {}
        for stock in target_list:
            try:
                res = analyze_stock_technical(stock, as_of_date=as_of_date, mode="ai",
                                              regulatory_data=st.session_state.get("regulatory_data"))
                if isinstance(res, dict):
                    feat = res.get("ai_report", {})
                    hinfo = h_map.get(stock.strip().upper())
                    if hinfo and isinstance(feat, dict):
                        avg = hinfo["avg_price"]
                        shares = hinfo["shares"]
                        feat["holding_avg_price"] = avg
                        feat["holding_shares"] = shares
                        feat["holding_cost"] = round(avg * shares, 2)
                        c = feat.get("close")
                        if c and avg > 0:
                            feat["holding_return_pct"] = round((c - avg) / avg * 100, 2)
                            feat["holding_unrealized_pnl"] = round((c - avg) * shares, 2)
                    all_reports[stock] = feat
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
                                              regulatory_data=st.session_state.get("regulatory_data"))
                if isinstance(res, dict):
                    txt = res.get("human_report", str(res))
                    full_content.append(txt)
                    hinfo = h_map.get(stock.strip().upper())
                    if hinfo:
                        avg = hinfo["avg_price"]
                        shares = hinfo["shares"]
                        close_val = None
                        for line in txt.split("\n"):
                            if "Close：" in line:
                                try:
                                    close_val = float(line.split("Close：")[1].split()[0])
                                except (IndexError, ValueError):
                                    pass
                        h_lines = [f"  ── Holdings ──",
                                   f"  Avg：{avg:.2f}  Shares：{shares}  Cost：{avg * shares:,.0f}"]
                        if close_val and avg > 0:
                            ret = (close_val - avg) / avg * 100
                            pnl = (close_val - avg) * shares
                            emoji = "📈" if ret >= 0 else "📉"
                            h_lines.append(f"  {emoji} Return：{ret:+.2f}%  P&L：{pnl:+,.0f}")
                        full_content.append("\n".join(h_lines))
                else:
                    full_content.append(str(res))
                full_content.append("")
                full_content.append("=" * 60)
                full_content.append("")
            except Exception as e:
                full_content.append(f"FAIL {stock}: {e}")
                full_content.append("-" * 60)

        combined_report = "\n".join(full_content)
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

    with st.expander("🌐 提示詞 M：市況判讀（先跑這個）", expanded=False):
        prompt_m = r"""# 🌐 提示詞 M：市況判讀 + 題材熱度地圖

你是台股市場狀態分析師。
**單一任務**：產出兩個區段：
1. **🤖 AI 區段**：客觀市況快照 + 激進度量表（給 A/B/C 引用）
2. **👤 用戶區段**：題材熱度地圖（**僅給用戶看，不可貼給 A/B/C**）

**核心理念**：避免 AI 因池子熱度標籤產生錨定偏誤。

---

## 步驟 1：讀取大盤客觀數據

從 market_features JSON 讀取：
- TWII 收盤、日 RSI、週 RSI、ADX、VIX
- 趨勢、波動 regime、BB 位置、量比
- suggested_regime

## 步驟 2：合併所有池子計算個股代理訊號

合併所有 sector JSON 個股（去重），計算：
- 樣本數、趨勢一致度、RSI 中位、過熱比例
- 量增/量縮比例、外資累積/減碼比例
- 多頭/空頭背離比例、RS 正比例、平均 ATR%

## 步驟 3：題材熱度地圖（僅用戶區段使用）

對**每一個池子**獨立計算：

| 指標 | 計算 |
|---|---|
| 池子樣本數 | 該池子個股數 |
| 池子 RSI 中位 | 該池所有 rsi14 中位數 |
| 池子外資累積比 | accumulating 個股 / 池子總數 |
| 池子外資減碼比 | reducing 個股 / 池子總數 |
| 池子趨勢一致度 | strong_uptrend+uptrend+strong_bottom_bounce 比例 |
| 池子多頭背離比 | bullish_divergence 任一 true 比例 |
| 池子平均 RS 20D | rs_vs_bench_20d 平均 |
| 池子量增比 | vol_ratio_5d>1.0 比例 |

### 池子熱度評分（0-100）

```
熱度分 = (RSI中位 × 0.20)
       + (外資累積比 × 0.25)
       + (趨勢一致度 × 0.20)
       - (外資減碼比 × 0.20)
       + (RS平均正規化 × 0.10)
       + (量增比 × 0.15)
       - (空頭背離比 × 0.10)
正規化到 0-100
```

### 池子分類

| 熱度分 | 標籤 |
|---|---|
| ≥ 70 | 🔥🔥 火熱（資金主軸）|
| 55-69 | 🔥 偏熱（題材有戲）|
| 40-54 | 🌊 中性（無明確方向）|
| 25-39 | 🌫️ 偏冷（資金觀望/退潮）|
| < 25 | ❄️ 冷清（資金撤退）|

## 步驟 4：激進度量表（0-100）

### 4 個子分數（各 0-100）

#### A. 趨勢面分（市場結構是否健康）
- TWII strong_uptrend + 週 uptrend + ST bullish = 90+
- 趨勢盤但 weekly RSI 過熱 = 70-80
- 震盪盤 / 轉折期 = 40-60
- 跌破週 MA20 = 20-40
- weekly downtrend = 0-20

#### B. 過熱程度分（越高越過熱）
- 日 RSI<60 + 週 RSI<70 + BB位置<70% = 0-30（無過熱）
- 日 RSI 60-75 或 BB 70-85% = 30-60（中性）
- 日 RSI>75 或 週 RSI>78 或 BB>85% = 60-85（過熱警戒）
- 三項全中或合併池過熱比例>50% = 85-100（極度過熱）

#### C. 系統風險分（越高越危險）
- VIX<18 + 無突發事件 = 0-20
- VIX 18-25 + 川普口頭威脅 = 20-50
- VIX>25 或單日 -2% = 50-80
- VIX>30 或 daily<-3% 或行政命令 = 80-100

#### D. 廣度健康分（越高越健康）
- 上漲家數>下跌家數 + RS 正比例>60% = 70-100
- 上漲=下跌 + RS 正 40-60% = 50-70
- 下跌>上漲 但 RS 正>40% = 30-50（權值噴發）
- 下跌遠多 + RS 正<30% = 0-30（廣度極差）

### 綜合激進度

```
激進度 = (趨勢面 × 0.30)
       + ((100 - 過熱程度) × 0.25)
       + ((100 - 系統風險) × 0.25)
       + (廣度健康 × 0.20)
範圍：0-100
```

### 激進度區間 → 行動指引

| 激進度 | 標籤 | 大致取向 |
|---|---|---|
| ≥ 80 | 🟢 全力進攻 | Tier1 寬鬆、加碼鮮花 |
| 65-79 | 🟢 積極進攻 | 標準 Tier1、續抱鮮花 |
| 50-64 | 🟡 中性平衡 | 收緊標準、減弱勢 |
| 35-49 | 🟠 偏向防禦 | Tier1 提高門檻、減 20% |
| 20-34 | 🔴 強防禦 | 暫停新進場、減 50% |
| < 20 | 🚨 危機應對 | 核心 ETF 不動、其餘出清 |

## 步驟 5：建議現金比

```
建議現金比 = 100 - 激進度 × 0.7
```

---

## 輸出格式（雙區段）

### 🤖 第一區段：AI 區段（複製貼到 A/B/C）

```
═══════ 市況快照 {YYYY-MM-DD}（給 AI 用）═══════

🌐 市場狀態：{狀態}（系統建議：{suggested_regime}）

📊 大盤客觀：
   TWII {n} ({±%}%) / 日RSI {n} / 週RSI {n} / VIX {n}{level}
   趨勢 {trend_state}日/{weekly}週 / ST{狀態} / BB{n}% / ADX{n}

📉 合併池訊號（{n}檔，{n}個池子）：
   趨勢一致 {n}% | RSI中位 {n} | 過熱 {n}% | 量增 {n}%
   外資累積 {n}% | 外資減 {n}% | 多頭背離 {n}% | 空頭背離 {n}%

🌡️ 激進度量表：{n}/100 → {🟢全力/🟢積極/🟡中性/🟠防禦/🔴強防/🚨危機}
   ├─ 趨勢面：{n}/100（{文字評語}）
   ├─ 過熱程度：{n}/100（{文字評語}）
   ├─ 系統風險：{n}/100（{文字評語}）
   └─ 廣度健康：{n}/100（{文字評語}）
   💰 建議現金比：{n}%

🎯 推斷：{≤30字 整合性脈絡}
⚠️ 警示：{1-2 條風險}

═══════ 以上請複製貼到 Prompt A/B/C 開頭 ═══════
```

### 👤 第二區段：用戶區段（僅供你判斷，不可貼給 AI）

```
─────────────────────────────────────
👤 以下為「用戶儀表板」— 僅供你判斷
   ⚠️ 不可貼到 A/B/C，避免 AI 產生熱度錨定偏誤
─────────────────────────────────────

🔥 題材熱度地圖：

   🔥🔥 火熱（資金主軸）：
     1. {池子}（熱度{n} / RSI{n} / 外資+{n}% / 趨勢一致{n}%）

   🔥 偏熱：
     2. {池子}（熱度{n} / RSI{n} / 外資+{n}% / 趨勢一致{n}%）

   🌊 中性：
     3. {池子}（熱度{n} / RSI{n} / 外資+{n}% / 趨勢一致{n}%）

   🌫️ 偏冷：
     4. {池子}（熱度{n} / RSI{n} / 外資-{n}% / 趨勢一致{n}%）

   ❄️ 冷清（資金撤退）：
     5. {池子}（熱度{n} / RSI{n} / 外資-{n}% / 趨勢一致{n}%）

📌 用戶觀察重點：
   - 領漲池：{名稱}（熱度{n}）
   - 領跌池：{名稱}（熱度{n}）
   - 分散度：{大/中/小} → {全面多頭/題材分化/全面回檔}

─────────────────────────────────────
（用戶區段結束，以上不要貼給 A/B/C）
─────────────────────────────────────
```

---

## 輸出規則

1. **嚴格輸出兩個區段**，AI 區段在前、用戶區段在後
2. **AI 區段不可包含任何「池子名稱 + 熱度」資訊**
3. 用戶區段獨立用「─────」分隔線標示
4. 樣本 <10 檔 → AI 區段「合併池訊號」改寫「樣本不足」
5. 池子<2 個 → 用戶區段「題材熱度地圖」改寫「單池無熱度比較」
6. 池子內個股<3 檔 → 該池子標「樣本不足」
7. 不評論個股、不給投資建議

## 資料

【貼上 market_features JSON（從 Tab 6 下載）】
【貼上 sector 池子 1 JSON】
【貼上 sector 池子 2 JSON】
...（可多池）"""
        st.code(prompt_m, language="text")
        st.caption("💡 跑完 M 後，把「🤖 AI 區段」貼到 Tab 6 最下方的「Prompt M 結果」框，A/B/C 會自動帶入。")

    with st.expander("📋 提示詞 A：候選股海選評分（動能右側專用）", expanded=False):
        prompt_a = r"""# 📋 提示詞 A：候選股海選評分（動能右側專用）

你是台股**動能右側交易**分析師。
**單一任務**：對每個池子的候選股逐檔評分,依市況加權排序。

**核心信條**：
- ✅ 順勢延續、突破追進、量價齊揚
- ❌ 反向摸底、跌深反彈、左側逢低

**規則**：
- **層內計分**：嚴格依規則執行,AI 不能改分數
- **層間判讀**：AI 在「💭」欄位解讀脈絡
- 每檔 ≤ 12 行

---

## 第零步：填入市況(從 Prompt M 結果複製整段 AI 區段)

```
═══════ 市況快照 {YYYY-MM-DD}(給 AI 用)═══════
[整段貼上]
═══════ 以上請複製貼到 Prompt A/B/C 開頭 ═══════
```

---

## 計分規則(動能右側專用,AI 嚴格執行)

### L1 趨勢(30 分)

trend_state 基礎分(15 分)：
| 狀態 | 分數 | 動能交易者觀點 |
|---|---|---|
| strong_uptrend | 15 | 完美追進 |
| uptrend | 13 | 趨勢延續 |
| uptrend_pullback | 10 | 強勢拉回(模式 B 機會)|
| **strong_bottom_bounce** | **8** | 反轉初期,需確認(降權重)|
| top_pullback | 5 | 觀察 |
| consolidation | 6 | 待突破(模式 C 機會)|
| downtrend / downtrend_bounce | 0 | **動能交易者禁區** |
| strong_downtrend | 0 | 排除 |

搭配加分(15 分)：
- mtf_alignment=aligned_bull → +5;mixed_bull_bias → +2
- weekly_trend_state=uptrend → +5
- supertrend_bullish=true → +3
- **flag_supertrend_flip_bull=true → +2(剛剛轉多,最理想)**

### L2 動能(20 分,動能交易者調整 RSI 曲線)

**RSI(8 分)動能右側調整**：
| RSI | 分數 | 動能交易者觀點 |
|---|---|---|
| **65-80** | **8** | 強勢延續區(最愛)|
| 50-65 | 6 | 多頭中段 |
| 80-85 | 5 | 過熱但仍多頭 |
| 40-50 | 3 | 動能不足 |
| 85+ | 3 | 過熱警戒 |
| < 40 | 0 | 動能交易者不碰 |

**MACD(6 分)**：
- osc>0 且 slope_5d>0 → 6(動能強化)
- osc>0 且 slope_5d≤0 → 3(動能轉弱)
- osc<0 → 0

**ADX(6 分)**：
- ADX>30 + di_bullish → 6(強趨勢)
- 25-30 + di_bullish → 4
- 20-25 + di_bullish → 2
- 其他 → 0

### L3 量價(25 分,動能交易者最重要) ⭐ 提權重

- **flag_price_up_vol_up=true → +8**(量價齊揚是動能核心)
- vol_ratio_5d>1.5 → +5;>1.2 → +3;>1.0 → +1
- OBV slope_5d 和 slope_20d 雙正 → +6;任一正 → +3
- pos_52w_pct>90 → +6;>70 → +4

### L4 籌碼(15 分)

- foreign_accumulation_trend=accumulating → +6;stable → +2
- flag_inst_consensus_buy → +5;flag_inst_consensus_sell → -5
- foreign_holding_4w_delta>0 → +2
- foreign_20d_net>0 → +2

### L5 基本面(10 分,OR 邏輯,動能交易者不主導)

- revenue_momentum=accelerating → +3;stable → +1;decelerating → 0
- eps_trend_4q=improving → +3;mixed → +1;deteriorating → 0
- revenue_mom_latest>30 → +2
- revenue_yoy_latest>10 → +2

### 動能右側加分(最多 +10) ⭐ 取代「反轉加分」

- **flag_supertrend_flip_bull=true → +3**(剛轉多,最理想進場點)
- **rs_vs_bench_20d > 5 → +2**(顯著強過大盤)
- **ma20_dev_pct 介於 0~5% → +2**(高於 MA20 但不過熱)
- **flag_bb_squeeze_fire=true → +3**(壓縮後爆發訊號)
- 加分上限 +10

### L6 散戶情緒扣分(最低 -8,調降)

動能股噴出時散戶湧入正常,不應重扣：

- flag_day_trade_overheat = -2(從 -3 調降)
- **flag_day_trade_divergence = -4**(散戶買主力倒貨,仍重扣)
- flag_smart_short_signal = -2(從 -3 調降)

### 警示扣分(最低 -10)

- price_down_vol_up_distribution = -5(疑似主力出貨,重扣)
- red_candle_no_reversal_confirm = -2
- above_bb_upper_in_high_vol：
  - 強勢突破(pos_52w>90 + flag_price_up_vol_up=true) = **+1**(動能突破,反而加分)
  - 否則 = -2

### 動能右側硬排除(強制 ★☆☆☆☆)

1. trend_state ∈ (strong_downtrend, downtrend) → **禁區**
2. supertrend_bullish=false → **直接排除**(動能交易者不買破線)
3. weekly_trend_state=downtrend → **直接排除**(週線多頭是底線)
4. flag_day_trade_divergence=true 且 foreign_accumulation_trend=reducing → 排除
5. **rs_vs_bench_20d < -3 → 排除**(跑輸大盤太多,動能已失)
6. **ma20_dev_pct < -3 → 排除**(已破短期趨勢)

### 總分計算

```
總分 = L1 + L2 + L3 + L4 + L5 + 動能加分 + L6 + 警示
範圍：-18 至 110
```

### 評級對照

- **★★★★★** ≥ 85(動能完美風暴)
- **★★★★☆** 70-84
- **★★★☆☆** 55-69
- **★★☆☆☆** 40-54
- **★☆☆☆☆** < 40

### 主導 edge 判定(動能右側調整)

- **L1+L2+L3 三層平均≥20 → S1 趨勢延續(動能核心)**
- 動能加分≥6 + L1≥18 → **S1+ 強動能延續**(特別標記)
- L1≥10 + bb_squeeze_fire → S3 突破式震盪
- ETF 或 beta<1 + L4≥10 → S4 避險防禦
- L4≥12 主導其他層偏弱 → S5 籌碼跟隨
- ⚠️ **動能右側交易者基本不做 S2 反轉捕捉**(左側買法)

### 排序分(動能右側專用加權)

| 激進度 | 偏好 edge | 加成 |
|---|---|---|
| ≥ 80 全力 | S1+ / S1 為絕對主力 | S1+ ×1.30、S1 ×1.20、其他 ×1.0 |
| 65-79 積極 | S1+ / S1 為主 | S1+ ×1.20、S1 ×1.15、其他 ×1.0 |
| 50-64 中性 | S1 + S5 平衡 | S1 ×1.10、S5 ×1.05、其他 ×1.0 |
| 35-49 防禦 | S5 / S4 為主 | S5/S4 ×1.15、S1 ×0.9 |
| <35 強防 | S4 為主 | S4 ×1.20、其他 ×0.85 |

```
排序分 = 個股總分 × 主導 edge 加成
```

### 資料時效

- 月/季資料落後 10-90 天 = 正常
- 日級資料落後 >2 天 → 對應層 ×0.7
- 當沖落後 >5 天 → L6 三 flag 視為 false
- 外資/借券落後 >5 天 → L4 ×0.7

### 個股輸出格式(每檔 12 行)

```
【{代號} {symbol}】★★★★☆ {n}分 (排序{n})
L1 {trend_state}/週{weekly}/ST{狀態}/flip{Y/N} = {n}/30
L2 RSI{n}/MACD{狀態}/ADX{n} = {n}/20
L3 vol{n}/OBV{狀態}/pos{n}%/PUVU{Y/N} = {n}/25
L4 外{foreign_20d}/投{trust_20d}/外資比{n}%{trend} = {n}/15
L5 營{yoy}%{momentum}/EPS{yoy}%{trend}/MoM{n}% = {n}/10
動能加分:ST_flip/RS+/MA20dev/BB_fire = {n}/10
L6 當沖/分歧/smart_short = {n}
警示: {entry_warnings} = {n}
🔍層間判讀:{≤30字}
💡建議:{Tier1/Tier2/觀察/排除} | 主導 edge:{S1+/S1/S3/S4/S5}
🔥動能特徵:{順勢/拉回/突破/盤整待噴/反轉初期 — 標一個}
```

### 層間判讀模式(必擇一,動能右側視角)

1. **強勢延續**:L1+L2+L3 全強、量價齊揚 → 「動能完美、可追」
2. **拉回機會**:trend_state=uptrend_pullback + ST 仍多 → 「強勢拉回,等回 MA5 接」
3. **突破在即**:bb_squeeze + 量增 → 「壓縮待噴、量出即追」
4. **強勢過熱**:L1+L2+L3 全強、RSI>80 → 「順勢但部位收緊」
5. **層間矛盾**:總分高但 RS 轉負 → 「分數虛高、跑輸大盤」
6. **反轉初期**(謹慎):strong_bottom_bounce + ST_flip → 「左側風險,待確認 5 日」

**AI 不能反駁分數**,只能解釋脈絡。

---

## 輸出總表

### 跨池子總表(合併池排序前 10 名)

| 代號 | 池子 | 評級 | 總分 | 排序分 | 主導edge | 動能特徵 | 層間判讀 | 建議 |
|---|---|---|---|---|---|---|---|---|

### 各池子內部排序(每個池子獨立一張)

#### 池子 X：{名稱}({n}檔)

| 代號 | 評級 | 總分 | 排序分 | 主導edge | L1 | L2 | L3 | L4 | L5 | 動能加分 | 扣分 | 動能特徵 | 建議 |

---

## 輸出規則

1. 不重複輸出市況(已在 Prompt M 完成)
2. 個股每檔 ≤ 12 行
3. 排序依**排序分降序**
4. 不寫總結段
5. 樣本>30 檔時,跨池子總表前 10、各池子內前 5

## 資料

【貼上 Prompt M 的 🤖 AI 區段】
【貼上 sector 池子 1 JSON】
【貼上 sector 池子 2 JSON】
...
"""
        # 動態組合：若有 M 結果，自動插入到 prompt 開頭
        m_result = st.session_state.get("prompt_m_ai_result", "").strip()
        if m_result:
            st.success("✅ 已自動帶入 Prompt M 結果")
            final_prompt_a = (
                "## 🌐 今日市況（來自 Prompt M）\n\n"
                f"{m_result}\n\n"
                "---\n\n"
                f"{prompt_a}"
            )
        else:
            st.warning("⚠️ 尚未在 Tab 6 貼上 Prompt M 結果，建議先跑 M 再回來複製")
            final_prompt_a = prompt_a
        st.code(final_prompt_a, language="text")

    with st.expander("🔥 提示詞 B v3.1：動能右側進攻名單（需開網頁搜尋）", expanded=False):
        prompt_b = r"""# 🔥 提示詞 B v3.1：動能右側進攻名單｜實戰最終版

你是台股「動能右側交易策略師」。
風格:「灌溉鮮花、砍掉雜草,只追強勢、不撿便宜」。

任務:
從 Prompt M 市況快照 + sector JSON 原始資料 + 最新網路資訊中,獨立判斷今日可進攻標的。
**禁止參考 Prompt A 的分數**。

每檔輸出精簡:
- Tier1-A:每檔 ≤ 7 行(含動能評分 + 監控訊號)
- Tier1-B:每檔 ≤ 6 行
- Tier2-B:每檔 ≤ 5 行
- Tier2-A:每檔 ≤ 3 行
- 軟排除/硬排除:一行帶過

---

# 一、核心信條

## ✅ 進場原則
- 突破前高 → 可追,須有量能、消息催化、隔日不跌回突破點
- 拉回不破 MA5 / AVWAP → 可接
- 帶量站穩盤整突破點 → 可進
- **強勢股不一定回 MA5,只等 MA5 會錯過主升段**
- 右側交易追的是「有續航力的強」,不是看到漲就追

## ❌ 絕對不做
- 跌深反彈、破線股、下降趨勢摸底
- 用 Fib 38.2/50/61.8 當逆勢買點
- 用 POC 當跌深買點
- 在 ma20_dev_pct < -3% 時進場

---

# 二、必做:網路搜尋(精簡版)

## 必搜(8 個關鍵字)

**市場層級**:
- `VIX latest`
- `費半 本週`
- `Trump latest tariff Taiwan`

**個股層級(每檔)**:
- `{代號} 最新消息`
- `{代號} 利空`
- `{代號} 法說`

**選搜(時間夠才做)**:
- `Fed latest`
- `{產業} 庫存 / 需求`

## 搜尋彙整輸出

```
📡 市場補充:VIX{n}/{方向}|費半{強/弱}|地緣{🟢🟡🔴}
🇺🇸 政策:Trump{Y/N}/{議題}/影響{標的}
📰 個股:
  {代號}:{🟢需求/🟡成本/🔴轉嫁}/{要點}/逆風{有無}
```

---

# 三、第零步:填入市況

```
═══════ 市況快照 {YYYY-MM-DD}(給 AI 用)═══════
[整段貼上]
═══════ 以上請複製貼到 Prompt A/B/C 開頭 ═══════
```

---

# 四、第一步:必填市況套用聲明(缺則拒絕分析)

```
【市況套用聲明】
今日激進度:{n}/100 → {🟢全力/🟢積極/🟡中性/🟠防禦/🔴強防/🚨危機}
建議現金比:{n}%

我在這次分析中會做的調整:
├─ Tier1-A 標準:{依激進度自填}
├─ Tier1-B 標準:{依市況自填}
├─ Tier2-B 試單條件:{依市況自填}
├─ 部位修正係數:×{依激進度自填}
└─ 大盤約束:{1-2句具體說明}

範例:「大盤週 RSI 80,屬 late_trend;不無腦追高,但個股 RS 強、ADX 強、放量續強,允許 1-2% Tier2-B 試單。」
```

---

# 五、Tier 分類(六層)

## 5-1. Tier1-A:正式進攻(我相信它,敢正常部位 → 4-5%)

**全部符合**才能列 Tier1-A:

```
□ trend_state ∈ (strong_uptrend / uptrend)
□ supertrend_bullish = true
□ weekly_trend_state = uptrend
□ rs_vs_bench_20d > 0
□ pos_52w_pct > 60
□ MA5 > MA20
□ flag_inst_consensus_buy = true 或 foreign_accumulation_trend = accumulating
□ ATR% < 5
□ RR ≥ 1(嚴格,Tier1-A 紀律)
□ 無 RSI/MACD 明顯背離
□ 最新消息無明確利空
```

差異點:**Tier1-A = 我相信它,RR 好、未過熱、有催化、可正常部位**

## 5-2. Tier1-B:高動能小部位(我懷疑它但機會大,小試 → 1-2%)

```
□ trend_state = strong_uptrend
□ weekly_trend_state = uptrend
□ supertrend_bullish = true
□ rs_vs_bench_20d > +10
□ ADX > 35
□ MA5 > MA20
□ vol_ratio_5d > 1.2 或 法人買超明確
□ 收盤接近當日高點 或 突破前高未跌回
□ 無爆量長上影
□ 無跌破 MA5
□ 最新消息無明確利空
```

差異點:**Tier1-B = 動能訊號極強,但 RR<1 或 RSI 75-85 或 BB>100,所以小部位**

部位:2-3%(若 RSI>85 或 ATR>5,部位再 ×0.7)

## 5-3. Tier2-A:強但未給進場點(觀察,0%)

任一情境:
- 強勢股距 MA5/AVWAP 太遠,進場區距現價 >3%
- strong_uptrend 但 RSI/BB 偏熱,尚未續強確認
- 量價強,但隔日承接未知
- RR 偏低,但趨勢未破壞
- 法人分歧,但價格結構強
- 最新消息中性,缺催化

部位:0%,只觀察

## 5-4. Tier2-B:高動能續強試單(0.5-1.5%)

當市況 ∈ (trend / late_trend / late_trend_or_blowoff / transition)且激進度 ≥ 50:

```
□ trend_state = strong_uptrend
□ weekly_trend_state = uptrend
□ supertrend_bullish = true
□ rs_vs_bench_20d > +10
□ ADX > 35
□ MA5 > MA20
□ vol_ratio_5d > 1.2 或 法人買超明確
□ 收盤接近當日高點 或 突破前高未跌回
□ 無爆量長上影、無跌破 MA5、無跌回突破點
□ 最新消息無明確利空
```

部位:0.5-1.5%(訊號極強可到 2%)

## 5-5. 軟排除(可翻案)

只觸發以下才列軟排除觀察:
- RSI > 80
- BB position > 100
- RR < 0.8
- RSI/MACD 背離
- 當沖偏熱
- 位置過高

T+1~T+3 出現「放量突破 + 收盤創高 + 不跌回突破點 + 無利空」→ 升 Tier2-B

## 5-6. 硬排除(不可翻案)

任一觸發 → 直接排除:
- trend_state = downtrend / strong_downtrend
- supertrend_bullish = false 且 weekly_trend ≠ uptrend
- ma20_dev_pct < -3
- 跌破 MA20 且未快速站回
- 明確基本面利空
- 法人同步大賣 + 價跌量增
- 注意股/處置股 + 爆量長上影 + 當沖過熱
- 最新消息直接衝擊基本面

---

# 六、過熱處理規則

## RSI
| RSI | 處理 |
|---|---|
| 65-75 | 健康強勢區,可正常 |
| 75-85 | 「高動能區」,Tier1-B/Tier2-B 小部位 |
| >85 | 高風險,部位×0.7;同時長上影/爆量不漲 → 末端陷阱 |

## BB position
| BB pos | 處理 |
|---|---|
| >100 | 不直接排除;若 RS 強+ADX 強+量增 → 「噴出延伸候選」 |
| >120 | 高風險;同時爆量長上影/跌回突破點/法人轉賣 → 末端陷阱 |

## 末端陷阱(以下任 3 條同時出現)
- RSI > 85
- BB position > 120
- 爆量長上影
- 隔日跌破前一日低點
- 跌回突破點
- 法人同步賣超
- 當沖比異常升高但價格不漲
- 價跌量增
- 最新消息明確利空

## 背離處理
RSI/MACD 背離 + 以下任 2 條 → 末端陷阱:
- 爆量長上影、收盤跌破前低、跌破 MA5、跌回突破點、法人賣超、當沖過熱不漲、價跌量增

只有背離但價格收高、RS 強、ADX 強、量能健康 → 不排除,降部位即可

---

# 七、進場模式

## 模式 A:突破追進

適用:剛突破前高/盤整、SuperTrend 翻多、族群同步啟動
- 進場:突破點 ~ 突破點 +1.5%
- 條件:量 ≥ 5日均量 1.2 倍
- 停損:跌破突破點 或 -1ATR

## 模式 B:拉回不破 MA5/AVWAP

適用:強勢上升中拉回測 MA5
- 進場:MA5 ~ AVWAP
- 條件:站回 MA5 + 收紅 K
- 停損:跌破 MA20 或 -1.5ATR

## 模式 C:站穩關鍵位

適用:盤整收斂、bb_squeeze 待噴
- 進場:盤整上緣突破點
- 條件:量 ≥ 1.5×五日均量
- 停損:盤整下緣 -0.5ATR

## 模式 D:高動能續強試單(actionable)⭐ v3.1 改寫

**今日判斷條件**(全部符合才考慮):
- trend_state = strong_uptrend
- weekly_trend_state = uptrend
- supertrend_bullish = true
- RS20 > +10
- ADX > 35
- 今日收盤接近高點(收盤 ≥ 當日高 -0.5ATR)
- 今日無爆量長上影
- 今日無跌破 MA5
- 最新消息無利空

**明日進場觸發**(三選一,盤中執行):

```
觸發 1(積極):明日開盤 9:00-9:30 不跌破今日低點
  → 開盤價 +0.5% 內進場(限價單)

觸發 2(順勢):明日盤中突破今日高點
  → 突破價當下進場(市價或突破單)

觸發 3(穩健):明日收盤前確認站穩今日均價
  → 13:00 後若仍在今日均價之上,可進場
```

**停損(進場後)**:
- 跌破今日(進場日前一日)低點
- 或跌破 MA5 -0.3ATR
- 取較嚴格者

**部位**:0.5-2%,不可重倉

## 禁用模式(左側買法)

❌ Fib 38.2/50/61.8 拉回(逆勢逢低)
❌ POC 成交密集區(跌深買)
❌ ma20_dev_pct < -3% 時進場

---

# 八、特例規則

## 8-1. RS 弱勢救贖

`rs_vs_bench_20d < -3` 但同時:
- RS 5D/10D 明顯轉強
- 放量突破
- 族群同步轉強
- weekly = uptrend
- supertrend = true
- 最新消息有催化

→ 可列 Tier2-A 觀察(不直接硬排除)

## 8-2. 創高股 RR 修正

創 52W 高 / 突破前高 / 上方無壓力時:

❌ 不用舊壓力當唯一目標
✅ 改用 ATR extension 計算
✅ 或用「隔日是否站穩突破點」判斷續航力

但若同時有:RR<0.8、背離、爆量長上影、法人轉賣、當沖過熱跌前低、明確利空 → 仍不可進攻

## 8-3. 權值龍頭股特例

大型權值/高流動性龍頭,即使 RR<0.8:
- trend_state = strong_uptrend
- weekly = uptrend
- supertrend = true
- vol_ratio_5d > 1.5
- MA5 > MA20
- 收盤接近高點/創波段高
- 無利空

→ 列 Tier2-B 備用,部位 0.5-1%
停損:跌回突破點 / 跌破 MA5 / -1ATR

---

# 九、Tier 標準依激進度動態

| 激進度 | Tier1-A | Tier1-B | Tier2-B | 偏好模式 |
|---|---|---|---|---|
| ≥80 全力 | 全條 + 模式 | 積極小部位 | 積極試單 | A / D |
| 65-79 積極 | 全條 + RR≥1 | RS強+ADX強 | 可試單 | A / B / D |
| 50-64 中性 | 全條 + ATR<5%+RR合理 | 小部位高動能 | 0.5-1.5% 試單 | B / D |
| 35-49 防禦 | 全條 + RSI<70 | 原則不開 | 不開除非極強 | C |
| 20-34 強防 | 暫停 | 不開 | 不開 | 守持股 |
| <20 危機 | 全停 | 不開 | 不開 | 守 ETF |

**中性盤(50-64)特別規則**:
- 不可無腦追高
- 不可一律排除 RSI 75-85、BB>100
- 符合 Tier1-B / Tier2-B 條件 → 允許小部位試單
- RR<0.8 + 背離 → 不得列 Tier1-A

---

# 十、部位計算

```
部位 = 基礎部位 × 激進度修正 × 個股修正
```

**基礎部位**:
- Tier1-A:3-5%
- Tier1-B:2-3%
- Tier2-B:0.5-1.5%
- 核心 ETF:15-25%

**激進度修正**:×1.0 / ×0.85 / ×0.7 / ×0.55 / ×0.4(對應全力/積極/中性/防禦/強防)

**個股修正**:
- pos_52w_pct > 95:×0.85
- RSI 75-85:不扣,標記高動能區
- RSI > 85:×0.7
- ATR > 5%:×0.85
- 處置股:×0.5
- flag_smart_short_signal:×0.85
- RR < 0.8:不得唯一 Tier1-A
- RR < 0.8 + 背離:降 Tier2-A/B 或軟排除
- 爆量長上影:×0.5 或排除
- 最新消息明確利空:排除或降階

---

# 十一、單日新增曝險上限

| 市況 | 上限 |
|---|---|
| transition | 5-8% |
| trend | 8-12% |
| late_trend | 8-12%,單檔不重倉 |
| blowoff | 5-8%,只試單 |
| defensive | 0-5% |
| crisis | 0% |

---

# 十二、輸出格式 ⭐ v3.1 加動能評分 + 監控訊號

## 12-1. 搜尋彙整(同 §二)

## 12-2. Tier1-A 正式進攻(每檔 ≤ 7 行)

```
【{代號}】Tier1-A {🟢/🟡/🔴} | 模式{A/B/C}
進場 {price 範圍}({進場條件})
停損 {stop} | ATR {n}% | RR {n}
催化:{要點;標日期/來源}
🔥 動能評分:RS{n}/ADX{n}/RSI{n}/BB{n}/Vol{n} = {🚀爆發/💪續強/⚡偏熱/🌊普通}
🤖 我為什麼選它:{≤25字 動能右側視角}
👁️ 監控訊號:跌破{n}(前低)/量縮<0.7/RSI跌破70 → 立即降階或停損
```

## 12-3. Tier1-B 高動能小部位(每檔 ≤ 6 行)

```
【{代號}】Tier1-B | 模式 D 高動能小部位
進場:{觸發 1/2/3 中選一}
停損:{跌回突破點 / 前低 / -1ATR} | 部位:{n}%
🔥 動能評分:RS{n}/ADX{n}/RSI{n}/BB{n}/Vol{n} = {等級}
🤖 為何可攻:RS強+ADX強+趨勢強,但過熱所以小部位
👁️ 監控訊號:跌破{n}/量縮<0.7/RSI<70 → 立即降階
```

## 12-4. Tier2-B 續強試單(每檔 ≤ 5 行)

```
【{代號}】Tier2-B | 模式 D 續強試單
觸發:{觸發 1/2/3 中選一}
停損:{前低 / 突破點 / -1ATR} | 部位:0.5-1.5%
🔥 動能評分:{簡寫} = {等級}
👁️ 監控訊號:{跌破 X / 量縮 / RSI 反轉} → 取消試單
```

## 12-5. Tier2-A 觀察(每檔 ≤ 3 行)

```
【{代號}】Tier2-A 觀察
觸發條件:{拉回 MA5 不破 / 站回 X 帶量 / 突破 Y 站穩}
🤖 為何只觀察:{≤20字}
```

## 12-6. 軟排除(每檔一行)

```
{代號}:軟排除,原因{RSI高/BB高/RR低/背離/當沖熱};若 T+1~T+3 放量突破可翻案
```

## 12-7. 硬排除(每檔一行)

```
{代號}:硬排除,原因{破 MA20/跌破 ST/RS弱無轉強/法人倒貨/利空衝擊}
```

---

# 十三、資金配置

```
現金 {套用 M 建議}%
Tier1-A 正式進攻 {n}%
Tier1-B 高動能小部位 {n}%
Tier2-B 試單 {n}%
Tier2-A 觀察 0%
軟排除觀察 0%
──
單日新增曝險上限:{n}%
今日進攻意圖:{≤30字}
```

---

# 十四、紀律分層 ⭐ v3.1 重新整理

## 🔴 違反即拒絕分析(5 條,絕對不可違反)

1. trend_state = strong_downtrend → 強制硬排除
2. supertrend_bullish = false 且 weekly ≠ uptrend → 硬排除
3. ma20_dev_pct < -3 → 不進場
4. 利空消息直接衝擊基本面 → 受影響股排除
5. 全不合格 → 輸出「建議觀望」,不湊人頭

## 🟡 觸發降階(個別處理,不全排除)

- 川普口頭威脅 ≠ 基本面崩壞;行政命令才需降階
- RR < 0.8 + 背離 → 不得列 Tier1-A,可降 Tier2-A/B
- RSI 75-85 → 不直接排除,判斷是否高動能續強
- BB > 100 → 不直接排除,判斷是否噴出延伸
- 創高股 → 不可只用舊壓力計算 RR
- Tier2-A 不是永久觀察,T+1~T+3 續強可升級
- 軟排除可翻案,硬排除不可翻案

## 📋 風格紀律(輸出前自檢)

1. **禁止參考 A 的分數**(B 獨立判斷)
2. **禁止逆勢逢低**(Fib/POC/破均線買)
3. 消息必標來源/時間
4. 每檔行數遵守(Tier1≤7 / Tier1-B≤6 / Tier2-B≤5 / Tier2-A≤3)
5. 不寫冗長總結

## ✅ 最終檢查(輸出前必過)

每檔自問:
1. 順勢推進 vs 逆勢摸底? 後者排除
2. 進場條件是站穩 vs 跌深? 後者排除
3. 停損是趨勢破壞 vs 亂設保底? 必為前者
4. RSI/BB 高 → 噴出延伸 vs 末端陷阱? 後者排除
5. 大盤激進度低 → 部位有縮小嗎?

---

# 十五、動能評分定義 ⭐ v3.1 新增

每檔 Tier1-A / Tier1-B / Tier2-B 必須輸出:

```
🔥 動能評分:RS{n}/ADX{n}/RSI{n}/BB{n}/Vol{n} = 等級
```

## 子分數對照(0-20 各)

- **RS** = rs_vs_bench_20d:>15→20、10-15→15、5-10→10、0-5→5、<0→0
- **ADX** = ADX:>40→20、35-40→15、30-35→10、25-30→5、<25→0
- **RSI** = 動能視角:65-75→20、75-80→15、80-85→10、85+→5、<65→0
- **BB** = bb_position_pct:60-90→20、90-100→15、100-120→10、>120→0、<60→5
- **Vol** = vol_ratio_5d:>1.5→20、1.2-1.5→15、1.0-1.2→10、<1.0→5

## 等級判定(總分 0-100)

- **🚀 爆發**:總分 ≥ 80(全綠燈,動能完美)
- **💪 續強**:60-79(健康強勢)
- **⚡ 偏熱**:40-59(動能仍在但警戒)
- **🌊 普通**:< 40(動能不足,可能是 Tier2-A)

---

# 十六、資料

【貼上 Prompt M 的 🤖 AI 區段】
【貼上 sector_候選名單 JSON】
(不需貼 A 的輸出 — B 獨立判斷)
"""
        # 動態組合：若有 M 結果，自動插入到 prompt 開頭
        m_result = st.session_state.get("prompt_m_ai_result", "").strip()
        if m_result:
            st.success("✅ 已自動帶入 Prompt M 結果")
            final_prompt_b = (
                "## 🌐 今日市況（來自 Prompt M）\n\n"
                f"{m_result}\n\n"
                "---\n\n"
                f"{prompt_b}"
            )
        else:
            st.warning("⚠️ 尚未在 Tab 6 貼上 Prompt M 結果，建議先跑 M 再回來複製")
            final_prompt_b = prompt_b
        st.code(final_prompt_b, language="text")

    with st.expander("🌸 提示詞 C：動能右側持股風控（需開網頁搜尋）", expanded=False):
        prompt_c = r"""# 🌸 提示詞 C：動能右側持股風控

你是台股**動能右側組合風控專家**,原則「灌溉鮮花、砍掉雜草、順勢加碼、不留戀弱勢」。

## 動能右側風控核心信條

✅ **守鮮花**:強勢延續、量價齊揚、籌碼穩定 → 續抱或加碼
✅ **早砍雜草**:動能轉弱的**早期警訊**就該行動,不等到雙破
✅ **順勢加碼**:突破前高、拉回不破均線 → 加碼放大
❌ **不為「成本」留戀**:跌破停損就是出,不問報酬率
❌ **不在下降趨勢攤平**

⚠️ **先開啟網頁搜尋**。每檔 ≤ 6 行。

---

## 第零步:填入市況(從 Prompt M 結果複製整段 AI 區段)

```
═══════ 市況快照 {YYYY-MM-DD}(給 AI 用)═══════
[整段貼上]
═══════ 以上請複製貼到 Prompt A/B/C 開頭 ═══════
```

## 第一步:必填「市況套用聲明」(缺則拒絕分析)

```
【市況套用聲明】
今日激進度:{n}/100 → {🟢全力/🟢積極/🟡中性/🟠防禦/🔴強防/🚨危機}
建議現金比:{n}%(套用 M 建議)

我在這次風控中會做的調整:
├─ 鮮花處理:{加碼/續抱/減 X%}
├─ 警戒處理:{續抱/減 15%/減 30%}
├─ 雜草處理:{減 30%/減 50%/全出}
├─ 加碼條件:{依激進度自填}
└─ 大盤約束:{1-2 句具體}
   例:「激進度 48 偏防禦 → 鮮花改設追漲停損鎖獲利、停加碼」
   例:「大盤週 RSI 80 過熱 → 高 beta 持股優先減 20%」
```

## 第二步:網頁搜尋(≤6 行)

**S1 系統性風險**:`VIX`、`Fed 利率`、`美中關係`
**S2 川普動態**:`Trump latest`、`Trump China Taiwan`
**S3 每檔持股**:`[代號] 最新消息`、`[公司] 法說會`、`[代號] 營收`

### 搜尋彙整輸出

```
📡 系統補充:VIX動向/Fed{方向}/地緣{摘要}
🇺🇸 川普:{48hr Y/N}/{性質}/{影響持股}
📰 持股消息:
  {代號}:{🟢🟡🔴}{要點}
```

## 第三步:動能右側分類(不依賴 A 分數)

獨立判斷,不參考 A 評級。

### 🌸 鮮花(動能延續中)

至少滿足 **5 項**:

```
□ trend_state ∈ (uptrend / strong_uptrend)(不含 strong_bottom_bounce)
□ supertrend_bullish = true
□ weekly_trend_state = uptrend
□ rs_vs_bench_20d > 0
□ MA5 > MA20(短期動能延續)
□ flag_price_up_vol_up=true 或 vol_ratio_5d > 1.0
□ foreign_accumulation_trend=accumulating 或 flag_inst_consensus_buy
□ 消息面 🟢 需求驅動 或 中性
```

### ⚠️ 警戒(動能轉弱早期警訊)⭐ 動能右側的關鍵

任一觸發 → 警戒(**不要等到雙破才行動**):

```
□ rs_vs_bench_20d 從正轉負(題材退潮)
□ flag_price_up_vol_down=true(漲價量縮,動能枯竭)
□ MA5 跌破 MA20(短期多頭失守)
□ flag_bearish_divergence_macd 或 _rsi(背離警告)
□ flag_supertrend_flip_bear(剛轉空,最早警訊)
□ vol_ratio_5d 連 3 日 < 0.7(明顯量縮)
□ trend_state 從 uptrend 變 top_pullback 或 consolidation
□ flag_inst_consensus_sell(法人共識賣超)
□ 消息面 🟡 中性偏空(題材熄火)
□ trend_state = strong_bottom_bounce(左側反彈,動能交易者不該長抱)
```

### 🪓 雜草(結構已破,立即處理)

任一觸發:

```
□ trend_state ∈ (downtrend / strong_downtrend)
□ supertrend_bullish = false(單一就破,不要等雙破)
□ weekly_trend_state = downtrend
□ flag_inst_consensus_sell 連 3 日
□ foreign_accumulation_trend=reducing 且 rs_vs_bench_20d<-3
□ 消息面 🔴 利空未消化
□ 報酬 < -7%(不問結構,停損紀律)
□ 跌破個股原進場時設定的停損點
```

## 第四步:依激進度決定操作(動能右側專用)

| 持股分類 | 激進度≥65 全力/積極 | 激進度50-64 中性 | 激進度35-49 防禦 | 激進度<35 強防 |
|---|---|---|---|---|
| 🌸 鮮花(基礎)| 續抱 | 續抱 | 續抱+追漲停損 | 減 20% 鎖獲利 |
| 🌸 鮮花+加碼訊號 | **加碼 10-20%** | **加碼 5-10%** | 續抱(不加)| 減 20% 鎖獲利 |
| 🌸 鮮花+獲利>20% | 部分鎖利 5-10% | 部分鎖利 10-15% | 部分鎖利 20% | 減 30% 鎖獲利 |
| ⚠️ 警戒 | 續抱不加 | 減 15% | 減 30% | 減 50% |
| 🪓 雜草 | 減 30% | 減 50% | **全出** | 全出 |

### 動能加碼訊號(鮮花持股額外檢查)

任一符合 + 鮮花 → 可加碼:

```
□ 突破前高帶量(pos_52w_pct>92 + flag_price_up_vol_up)
□ flag_supertrend_flip_bull=true(剛轉多)
□ flag_bb_squeeze_fire=true(壓縮後爆發)
□ 拉回 MA5 不破 + 量縮乾淨(vol_ratio_5d<0.8 且收紅 K)
```

## 第五步:追漲停損計算(動能右側紀律)⭐

對所有 🌸 鮮花持股,計算追漲停損:

```
追漲停損公式(取較高者):
A. 個股:close - 1.5 × ATR14
B. 個股:max(MA20, MA5 × 0.97)
C. 個股報酬 > 20%:成本 × 1.10(鎖至少 10% 獲利)

最終追漲停損 = max(A, B, C)
```

範例:
- 0050 成本 75.9,現價 89.95(+18.5%),ATR 1.5%
- A: 89.95 - 1.5×1.35 = 87.93
- B: max(MA20=86, MA5×0.97=85.8) = 86
- C: 75.9 × 1.10 = 83.49
- **追漲停損 = max(87.93, 86, 83.49) = 87.93**

## 第六步:每檔輸出(6 行)

```
【{代號} {symbol}】🌸/⚠️/🪓 ({加碼訊號:Y/N})
持倉 {shares}@{avg} 現{close} 報酬 {ret}% | 持有 {n} 日
🔍 結構訊號:trend{n}/ST{n}/週{n}/RS{n}/MA5/MA20{n}/量價{n} → {綠/黃/紅燈數}
🌐 消息:{要點+影響}(標來源/時間)
🤖 動能視角:{≤30字 解讀動能狀態 + 市況約束}
   例:「強勢延續+突破前高+外資累積=鮮花+加碼訊號;激進度 48 → 加碼 10%」
   例:「MA5 跌破 MA20+量縮+RS轉負=警戒早期;激進度 48 → 減 30% 不留戀」
💭 決策:{續抱/加碼X%/減X%/全出} 停損{追漲停損計算結果} 風險:{≤15字}
```

## 第七步:持股總表

| 代號 | 🌸⚠️🪓 | 加碼Y/N | 報酬% | 持有天數 | 結構 | 消息 | 動能視角 | 決策 |
|---|---|---|---|---|---|---|---|---|

## 反向審計(2 行)

- 續抱最大風險:{≤15字}
- 賣出可能錯過:{≤15字}

## 組合總結(5 行)

- 🌸/⚠️/🪓 比例:X / Y / Z(鮮花<30% 警示)
- 加碼候選:{n} 檔
- 川普曝險:{高/中/低}
- 建議現金比:**{直接套用 Prompt M 建議}**
- 集中度:{用 股數×現價 計算;單股>30% 警示;禁用 position_size_pct}

## 硬規則(動能右側專用)

1. supertrend_bullish=false → **單一條件就減碼**(不等雙破)
2. 跌破個股進場時設的停損點 → **無條件全出**(紀律第一)
3. 報酬 < -7% 不問結構 → **強制停損**
4. 川普行政命令直接衝擊 + 技術已破 → 即減 50%
5. 雜草 >50% → 「組合需大幅調整」
6. 激進度 < 20 → 鮮花也減 ≥30%(保留現金)
7. 川普口頭威脅 ≠ 實質政策(行政命令才算)
8. 每檔 ≤ 6 行
9. **集中度計算用實際股數×現價,禁用 position_size_pct**
10. **C 獨立判斷,不參考 A 的分數**
11. **strong_bottom_bounce 不歸鮮花**(左側反轉,動能交易者警戒)
12. 加碼僅在「鮮花 + 加碼訊號 + 激進度 ≥ 50」時才執行

## 持股資料

【貼上 Prompt M 的 🤖 AI 區段】
【貼上 sector_持股 JSON】
"""
        # 動態組合：若有 M 結果，自動插入到 prompt 開頭
        m_result = st.session_state.get("prompt_m_ai_result", "").strip()
        if m_result:
            st.success("✅ 已自動帶入 Prompt M 結果")
            final_prompt_c = (
                "## 🌐 今日市況（來自 Prompt M）\n\n"
                f"{m_result}\n\n"
                "---\n\n"
                f"{prompt_c}"
            )
        else:
            st.warning("⚠️ 尚未在 Tab 6 貼上 Prompt M 結果，建議先跑 M 再回來複製")
            final_prompt_c = prompt_c
        st.code(final_prompt_c, language="text")



# -------------------------
# Main Content
# -------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
    ["Stock Analysis", "Sector Analysis", "Custom Sectors", "Regulatory Lists", "Holdings", "📊 Market Snapshot", "🎯 進攻名單追蹤"])

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
                _v = st.session_state.get("_sect_ver", 0)
                edit_group = st.selectbox(
                    "Select sector",
                    list(st.session_state.custom_sectors.keys()),
                    key=f"mgmt_select_v{_v}",
                )
                current_list = st.session_state.custom_sectors.get(edit_group, [])

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
                            st.session_state.custom_sectors[edit_group] = current_list
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
                                st.session_state.custom_sectors[edit_group] = current_list
                                save_sectors_to_db(st.session_state.custom_sectors)
                                st.rerun()

                st.divider()
                if st.button("Delete this sector", type="primary"):
                    del st.session_state.custom_sectors[edit_group]
                    save_sectors_to_db(st.session_state.custom_sectors)
                    st.session_state["_sect_ver"] = _v + 1
                    st.rerun()

# --- Tab 4: Regulatory Lists ---
with tab4:
    st.header("Regulatory Lists (Attention / Disposition)")
    st.caption(
        "Upload CSV files downloaded from TWSE or TPEx. "
        "Stocks on these lists will be flagged during analysis."
    )

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


    def _is_new_file(uploaded_file, info_key):
        """Return True only if this file hasn't been processed yet."""
        if uploaded_file is None:
            return False
        existing = st.session_state.regulatory_upload_info.get(info_key)
        if existing is None:
            return True
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

    if attn_count or disp_count:
        with st.expander("View loaded stock codes", expanded=False):
            if attn_count:
                attn_sorted = sorted(reg["attention_stocks"])
                st.markdown(f"**Attention ({attn_count})**: {', '.join(attn_sorted)}")
            if disp_count:
                disp_sorted = sorted(reg["disposition_stocks"].keys())
                st.markdown(f"**Disposition ({disp_count})**: {', '.join(disp_sorted)}")

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
            save_regulatory_to_db()
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

# --- Tab 6: Market Snapshot (大盤狀態快照) ---
with tab6:
    st.header("📊 Market Snapshot - 大盤狀態快照")
    st.caption("抓取 ^TWII 加權指數 + ^VIX，產出 AI 判讀「現在是什麼盤」所需的客觀數據")

    col_m1, col_m2, col_m3 = st.columns([1, 1, 1])

    with col_m1:
        market_date = st.date_input(
            "分析日期",
            value=st.session_state.as_of_date,
            key="market_snapshot_date"
        )

    with col_m2:
        st.write("")
        st.write("")
        fetch_market_btn = st.button(
            "🌐 抓取大盤狀態",
            type="primary",
            use_container_width=True,
            key="fetch_market_btn"
        )

    with col_m3:
        st.write("")
        st.write("")
        if st.button("🔄 清除快取", use_container_width=True, key="clear_market_cache"):
            if "market_features_cache" in st.session_state:
                del st.session_state.market_features_cache
            st.rerun()

    # 執行抓取
    if fetch_market_btn:
        with st.spinner("正在下載 ^TWII / ^VIX 並計算大盤指標..."):
            try:
                features = compute_market_features(
                    as_of_date=market_date.strftime("%Y-%m-%d")
                )
                st.session_state.market_features_cache = features
            except Exception as e:
                st.error(f"執行錯誤：{e}")
                st.exception(e)

    # 顯示結果
    if "market_features_cache" in st.session_state:
        features = st.session_state.market_features_cache

        if features.get("data_quality", {}).get("twii_available"):
            snapshot_date = features.get("snapshot_date", "?")
            actual_data_date = features.get("actual_data_date")
            data_lag_days = features.get("data_lag_days", 0)
            
            if data_lag_days and data_lag_days > 2:
                st.error(
                    f"⚠️ **資料延遲警示** — 你選的日期 **{snapshot_date}**，"
                    f"但 Yahoo Finance 此刻能抓到的最新 TWII 資料是 **{actual_data_date}**"
                    f"（落後 {data_lag_days} 天）。"
                )
                st.warning(
                    "💡 **可能原因 + 解法**：\n"
                    "1. Yahoo Finance 對 `^TWII` 的更新偶發延遲（特別是非美股交易時段）\n"
                    "2. Streamlit Cloud 進程內 yfinance 套件持有舊快取\n"
                    "3. **解法 A**：等 5-10 分鐘再點「🌐 抓取大盤狀態」重試\n"
                    "4. **解法 B**：點側邊欄 ⋮ → Reboot app 重啟 Streamlit 進程\n"
                    "5. **解法 C**：把日期改選為 `actual_data_date` ({}) 接受目前資料".format(actual_data_date)
                )
            else:
                st.success(f"✓ 大盤資料抓取成功（{snapshot_date}）")
            
            # 顯示資料來源
            data_source = features.get("data_source", "FinMind")
            if data_source == "FinMind":
                st.caption(f"📡 資料來源：FinMind（主要）")
            elif "yfinance" in data_source and "備援" in data_source:
                st.warning(f"⚠️ 資料來源：**{data_source}**（FinMind 失敗時的後備，資料可能較舊）")
            else:
                st.caption(f"📡 資料來源：{data_source}")

            # 第一排：價格與波動指標
            st.markdown("### 📈 大盤關鍵指標")
            c1, c2, c3, c4 = st.columns(4)

            twii_close = features.get("twii_close", 0)
            twii_change = features.get("twii_change_pct", 0)
            c1.metric(
                "TWII 收盤",
                f"{twii_close:,.2f}",
                f"{twii_change:+.2f}%"
            )

            rsi_d = features.get("market_rsi14", 0)
            rsi_w = features.get("market_weekly_rsi14") or 0
            c2.metric(
                "RSI 14 / 週RSI",
                f"{rsi_d:.1f}",
                f"週 {rsi_w:.1f}"
            )

            vix_val = features.get("vix_latest")
            vix_lvl = features.get("vix_level", "?")
            if vix_val is not None:
                c3.metric(
                    "VIX",
                    f"{vix_val:.2f}",
                    vix_lvl
                )
            else:
                c3.metric("VIX", "N/A", "data unavailable")

            adx_val = features.get("market_adx", 0)
            c4.metric(
                "ADX (趨勢強度)",
                f"{adx_val:.1f}",
                "強勢" if adx_val > 25 else ("中等" if adx_val > 20 else "弱勢")
            )

            # 第二排：趨勢與波動 regime
            st.markdown("### 🎯 趨勢狀態")
            tc1, tc2, tc3, tc4 = st.columns(4)

            trend_state = features.get("market_trend_state", "?")
            trend_color = "🟢" if "uptrend" in trend_state else (
                "🔴" if "downtrend" in trend_state else "🟡"
            )
            tc1.metric("日線趨勢", f"{trend_color} {trend_state}")

            weekly_trend = features.get("market_weekly_trend_state", "?")
            tc2.metric("週線趨勢", weekly_trend or "N/A")

            vol_regime = features.get("market_volatility_regime", "?")
            vol_color = "🔴" if vol_regime == "expansion" else (
                "🟢" if vol_regime == "contraction" else "🟡"
            )
            tc3.metric("波動 Regime", f"{vol_color} {vol_regime}")

            suggested = features.get("suggested_regime", "?")
            regime_emoji = {
                "trend": "📈 趨勢盤",
                "range": "↔️ 震盪盤",
                "late_trend": "🔥 末段過熱",
                "late_trend_or_blowoff": "🚀 末段過熱/權值噴發",
                "crisis": "⚠️ 黑天鵝",
                "transition": "❓ 轉折期",
            }.get(suggested, suggested)
            tc4.metric("建議市況", regime_emoji)

            # 第三排：補充資訊
            st.markdown("### 📊 量價與位置")
            qc1, qc2, qc3, qc4 = st.columns(4)

            ma20 = features.get("twii_ma20", 0)
            qc1.metric("MA20", f"{ma20:,.0f}")

            bb_pos = features.get("market_bb_position_pct", 50)
            qc2.metric("布林位置 %", f"{bb_pos:.1f}",
                       "偏上軌" if bb_pos > 80 else ("偏下軌" if bb_pos < 20 else "區間內"))

            atr_pct = features.get("market_atr14_pct", 0)
            qc3.metric("ATR 14 %", f"{atr_pct:.2f}%")

            vol_ratio = features.get("market_vol_ratio_5d", 1.0)
            qc4.metric("量比 5D", f"{vol_ratio:.2f}",
                       "爆量" if vol_ratio > 1.3 else ("量縮" if vol_ratio < 0.7 else "正常"))

            # 警示區
            warnings = features.get("data_quality", {}).get("warnings", [])
            if warnings:
                st.markdown("### ⚠️ 資料警示")
                for w in warnings:
                    st.warning(w)

            # 下載按鈕
            st.divider()
            json_str = json.dumps(features, ensure_ascii=False, indent=2, default=str)
            st.download_button(
                label="📥 下載 market_features JSON（給 AI 用）",
                data=json_str,
                file_name=f"market_features_{snapshot_date}.json",
                mime="application/json",
                key="download_market_json",
                use_container_width=True
            )

            # 完整 JSON 檢視
            with st.expander("查看完整 JSON 內容（可直接複製貼給 AI）", expanded=False):
                st.code(json_str, language="json")

        else:
            st.error("⚠️ 大盤資料抓取失敗")
            warnings = features.get("data_quality", {}).get("warnings", [])
            for w in warnings:
                st.warning(w)
    else:
        st.info("👆 點上方「抓取大盤狀態」開始")

        # 使用說明
        with st.expander("📖 使用說明", expanded=True):
            st.markdown("""
### 這個功能在做什麼？

抓取**台股大盤**（^TWII 加權指數）+ **VIX 恐慌指數**的客觀數據，產出 AI 判讀
「**現在是什麼盤**」所需的所有指標。

### 為什麼需要？

之前 AI 判市況時要靠網頁搜尋估算大盤狀態，**容易誤判**。
有了這份 JSON，AI 就能客觀判斷：

- 趨勢盤、震盪盤、末段過熱、權值噴發、黑天鵝、轉折期
- 自動推算策略權重（S1 趨勢延續 / S2 反轉捕捉 / S4 避險防禦 等）

### 使用流程

1. 點「🌐 抓取大盤狀態」
2. 下載 `market_features_YYYY-MM-DD.json`
3. 把它跟 sector JSON **一起**貼到 Prompt A
4. AI 同時做：客觀市況判讀 + 個股海選評分

### 包含哪些欄位？

- **價格**：TWII 收盤、漲跌幅、週漲跌幅
- **趨勢**：market_trend_state、weekly_trend_state、SuperTrend、MA20/60
- **動能**：RSI、週 RSI、ADX、+DI/-DI
- **波動**：ATR、Volatility Regime（expansion/contraction/normal）
- **位置**：Bollinger 位置、量比
- **VIX**：恐慌指數及等級（low/normal/elevated/panic）
- **建議**：suggested_regime（自動推算當前市況）
""")

    # ============================================
    # Prompt M 結果輸入框（給 A/B/C 自動帶入用）
    # ============================================
    st.divider()
    st.subheader("🤖 Prompt M 結果輸入區")
    st.caption(
        "跑完 Prompt M 後，把「🤖 AI 區段」整段貼進來。"
        "之後在 Sidebar 點開 Prompt A/B/C 時會自動把 M 結果帶到最前面，一鍵複製就能用。"
    )

    m_result_input = st.text_area(
        "貼上 Prompt M 的 🤖 AI 區段（從『═══════ 市況快照』開始到『═══════ 以上請複製...』結束）：",
        value=st.session_state.get("prompt_m_ai_result", ""),
        height=400,
        key="m_result_textarea",
        placeholder="""═══════ 市況快照 2026-04-25（給 AI 用）═══════

🌐 市場狀態：...

📊 大盤客觀：
   TWII 38,932 (+3.23%) / 日RSI 76 / 週RSI 80 / VIX 19 normal
   ...

🌡️ 激進度量表：48/100 → 🟠 偏向防禦
   ...

═══════ 以上請複製貼到 Prompt A/B/C 開頭 ═══════"""
    )

    # 三個按鈕：儲存 / 清除 / 預覽
    btn_save_col, btn_clear_col, btn_preview_col = st.columns(3)

    with btn_save_col:
        if st.button("💾 儲存 M 結果", type="primary", use_container_width=True, key="save_m_result"):
            st.session_state.prompt_m_ai_result = m_result_input.strip()
            if m_result_input.strip():
                st.success("✓ 已儲存！A/B/C 會自動帶入此結果")
            else:
                st.warning("輸入框是空的，未儲存")
            st.rerun()

    with btn_clear_col:
        if st.button("🗑️ 清除 M 結果", use_container_width=True, key="clear_m_result"):
            st.session_state.prompt_m_ai_result = ""
            st.success("✓ 已清除")
            st.rerun()

    with btn_preview_col:
        if st.button("👁️ 預覽組合 Prompt", use_container_width=True, key="preview_combined"):
            st.session_state["_show_combined_preview"] = True

    # 顯示目前狀態
    saved_m = st.session_state.get("prompt_m_ai_result", "").strip()
    if saved_m:
        st.success(f"✅ 目前已儲存 M 結果（{len(saved_m)} 字元）— A/B/C 會自動帶入")
        with st.expander("查看已儲存的 M 結果", expanded=False):
            st.code(saved_m, language="text")
    else:
        st.info("⚠️ 尚未儲存 M 結果。A/B/C 會顯示原始模板（缺市況前綴）")

    # 預覽組合 Prompt（A/B/C 三個各看一眼）
    if st.session_state.get("_show_combined_preview") and saved_m:
        st.markdown("---")
        st.markdown("### 👁️ 組合 Prompt 預覽")
        preview_tab1, preview_tab2, preview_tab3 = st.tabs(["📋 Prompt A 預覽", "🔥 Prompt B 預覽", "🌸 Prompt C 預覽"])

        m_block = f"## 🌐 今日市況（來自 Prompt M）\n\n{saved_m}\n\n---\n\n"
        with preview_tab1:
            st.caption("（這是你按 Sidebar 的 Prompt A 會看到的完整內容，已帶 M 結果）")
            st.code(m_block + "[此處會接 Prompt A 的完整模板]", language="text")
        with preview_tab2:
            st.caption("（這是你按 Sidebar 的 Prompt B 會看到的完整內容，已帶 M 結果）")
            st.code(m_block + "[此處會接 Prompt B 的完整模板]", language="text")
        with preview_tab3:
            st.caption("（這是你按 Sidebar 的 Prompt C 會看到的完整內容，已帶 M 結果）")
            st.code(m_block + "[此處會接 Prompt C 的完整模板]", language="text")

        if st.button("關閉預覽", key="close_preview"):
            st.session_state["_show_combined_preview"] = False
            st.rerun()

# --- Tab 7: 進攻名單追蹤 ---
with tab7:
    st.header("🎯 進攻名單追蹤")
    st.caption("記錄不同 AI（Claude/GPT/Gemini 等）對每檔股票的 Prompt B 判斷，方便後續比對誰準")

    # ============================================
    # 日期選擇 + 名單管理
    # ============================================
    al_col1, al_col2, al_col3 = st.columns([1.2, 1, 1])

    with al_col1:
        attack_date = st.date_input(
            "進攻名單日期",
            value=st.session_state.as_of_date,
            key="attack_list_date"
        )
        attack_date_str = attack_date.strftime("%Y-%m-%d")

    with al_col2:
        st.write("")
        st.write("")
        if st.button("📥 載入該日清單", use_container_width=True, key="load_attack_list"):
            st.session_state.attack_list = load_attack_list_from_db()
            st.rerun()

    with al_col3:
        st.write("")
        st.write("")
        # 顯示該日有幾筆紀錄
        today_list = st.session_state.attack_list.get(attack_date_str, [])
        st.metric("該日股票數", len(today_list))

    st.divider()

    # ============================================
    # 新增股票區
    # ============================================
    with st.expander("➕ 新增股票到當日進攻名單", expanded=(len(today_list) == 0)):
        ns_col1, ns_col2, ns_col3 = st.columns([1, 2, 1])
        with ns_col1:
            new_stock_id = st.text_input(
                "股票代號",
                key=f"new_stock_id_{attack_date_str}",
                placeholder="例：2330"
            )
        with ns_col2:
            new_stock_name = st.text_input(
                "名稱（選填）",
                key=f"new_stock_name_{attack_date_str}",
                placeholder="例：台積電"
            )
        with ns_col3:
            st.write("")
            st.write("")
            if st.button("➕ 加入", use_container_width=True, key="btn_add_attack_stock"):
                sid = new_stock_id.strip().upper()
                if not sid:
                    st.error("請輸入股票代號")
                else:
                    # 檢查是否已存在
                    today_list_check = st.session_state.attack_list.get(attack_date_str, [])
                    existing = [s["stock_id"] for s in today_list_check]
                    if sid in existing:
                        st.warning(f"{sid} 已存在於 {attack_date_str} 名單中")
                    else:
                        if attack_date_str not in st.session_state.attack_list:
                            st.session_state.attack_list[attack_date_str] = []
                        st.session_state.attack_list[attack_date_str].append({
                            "stock_id": sid,
                            "stock_name": new_stock_name.strip(),
                            "ai_judgments": []
                        })
                        save_attack_list_to_db(st.session_state.attack_list)
                        st.success(f"已新增 {sid}")
                        st.rerun()

    # ============================================
    # 顯示該日所有股票 + AI 判斷
    # ============================================
    today_list = st.session_state.attack_list.get(attack_date_str, [])

    if not today_list:
        st.info(f"📭 {attack_date_str} 尚無進攻名單。請先在上方新增股票。")
    else:
        st.subheader(f"📋 {attack_date_str} 進攻名單（{len(today_list)} 檔）")

        for stock_idx, stock_entry in enumerate(today_list):
            sid = stock_entry["stock_id"]
            sname = stock_entry.get("stock_name", "")
            judgments = stock_entry.get("ai_judgments", [])

            # 標題列
            title_text = f"【{sid}】{sname}" if sname else f"【{sid}】"
            if judgments:
                # 顯示有幾個 AI 判斷
                ai_sources = ", ".join([j["ai_source"] for j in judgments])
                title_text += f"  —  {len(judgments)} 個判斷（{ai_sources}）"

            with st.expander(title_text, expanded=False):
                # 顯示已有的 AI 判斷
                if judgments:
                    st.markdown("##### 📊 AI 判斷比較")

                    # 比較表（簡要）
                    comparison_data = []
                    for j in judgments:
                        comparison_data.append({
                            "AI": j.get("ai_source", "?"),
                            "Tier": j.get("tier", "?"),
                            "進場區": f"{j.get('entry_low', '-')} ~ {j.get('entry_high', '-')}",
                            "停損": j.get("stop_loss", "-"),
                            "部位%": j.get("position_pct", "-"),
                            "動能評分": f"{j.get('momentum_score', '-')} {j.get('momentum_level', '')}",
                            "edge": j.get("dominant_edge", "-"),
                        })
                    st.dataframe(comparison_data, use_container_width=True)

                    st.markdown("##### 📝 詳細記錄")
                    for j_idx, j in enumerate(judgments):
                        with st.container(border=True):
                            jc1, jc2, jc3 = st.columns([2, 2, 1])
                            with jc1:
                                st.markdown(f"**🤖 {j.get('ai_source', '?')}**")
                                st.caption(f"建立於：{j.get('created_at', '-')}")
                            with jc2:
                                st.markdown(f"**Tier**: `{j.get('tier', '?')}` | **Edge**: `{j.get('dominant_edge', '?')}`")
                                st.markdown(f"**進場**: {j.get('entry_low', '-')} ~ {j.get('entry_high', '-')} | **停損**: {j.get('stop_loss', '-')}")
                                st.markdown(f"**部位**: {j.get('position_pct', '-')}% | **動能**: {j.get('momentum_score', '-')} {j.get('momentum_level', '')}")
                            with jc3:
                                if st.button("🗑️ 刪除", key=f"del_judgment_{stock_idx}_{j_idx}_{attack_date_str}"):
                                    st.session_state.attack_list[attack_date_str][stock_idx]["ai_judgments"].pop(j_idx)
                                    save_attack_list_to_db(st.session_state.attack_list)
                                    st.rerun()

                            # 完整原文
                            raw = j.get("raw_text", "").strip()
                            if raw:
                                with st.expander("📄 完整 AI 輸出", expanded=False):
                                    st.code(raw, language="text")

                else:
                    st.info("尚未有任何 AI 判斷。請在下方新增。")

                # ----------------------------------------
                # 新增 AI 判斷區
                # ----------------------------------------
                st.markdown("---")
                st.markdown("##### ➕ 新增 AI 判斷")

                form_key = f"add_judgment_{stock_idx}_{attack_date_str}"
                with st.form(key=form_key, clear_on_submit=True):
                    fc1, fc2, fc3 = st.columns(3)

                    with fc1:
                        ai_source = st.selectbox(
                            "AI 來源",
                            ["Claude", "GPT", "Gemini", "Grok", "其他"],
                            key=f"ai_src_{form_key}"
                        )
                        tier = st.selectbox(
                            "Tier 分類",
                            ["Tier1-A", "Tier1-B", "Tier2-A", "Tier2-B", "軟排除", "硬排除"],
                            key=f"tier_{form_key}"
                        )

                    with fc2:
                        entry_low = st.number_input(
                            "進場區下限",
                            min_value=0.0, step=0.5, format="%.2f",
                            key=f"entry_low_{form_key}"
                        )
                        entry_high = st.number_input(
                            "進場區上限",
                            min_value=0.0, step=0.5, format="%.2f",
                            key=f"entry_high_{form_key}"
                        )
                        stop_loss = st.number_input(
                            "停損價",
                            min_value=0.0, step=0.5, format="%.2f",
                            key=f"stop_{form_key}"
                        )

                    with fc3:
                        position_pct = st.number_input(
                            "部位 %",
                            min_value=0.0, max_value=30.0, step=0.5, format="%.2f",
                            key=f"pos_{form_key}"
                        )
                        momentum_score = st.number_input(
                            "動能評分（0-100）",
                            min_value=0, max_value=100, step=5,
                            key=f"mscore_{form_key}"
                        )
                        momentum_level = st.selectbox(
                            "動能等級",
                            ["🚀 爆發", "💪 續強", "⚡ 偏熱", "🌊 普通", "（無）"],
                            key=f"mlevel_{form_key}"
                        )

                    fc4, fc5 = st.columns(2)
                    with fc4:
                        dominant_edge = st.selectbox(
                            "主導 edge",
                            ["S1+", "S1", "S3", "S4", "S5", "（無）"],
                            key=f"edge_{form_key}"
                        )

                    raw_text = st.text_area(
                        "完整 AI 原文（必填，貼整段 Prompt B 對該股的輸出）",
                        height=200,
                        key=f"raw_{form_key}",
                        placeholder="""【2330】Tier1-A 🟢需求驅動 | 模式 A
進場 1850-1880（突破前高+量增）
停損 1820 | ATR 2.4% | RR 1.8
催化：法說會 4/30、外資+8K
🔥 動能評分：RS 18/ADX 38/RSI 72/BB 85/Vol 1.4 = 💪續強（總分 75）
🤖 我為什麼選它：強勢突破前高、AVWAP 上方、模式 A 追進
👁️ 監控訊號：跌破 1820 / 量縮 < 0.7 / RSI 跌破 70 → 立即降階
部位：4%（基礎 5% × 激進度 ×0.85 × 個股 ×1.0）"""
                    )

                    submit_btn = st.form_submit_button("💾 儲存 AI 判斷", type="primary", use_container_width=True)

                    if submit_btn:
                        if not raw_text.strip():
                            st.error("請貼上完整 AI 原文")
                        else:
                            new_judgment = {
                                "ai_source": ai_source,
                                "tier": tier,
                                "entry_low": float(entry_low) if entry_low > 0 else None,
                                "entry_high": float(entry_high) if entry_high > 0 else None,
                                "stop_loss": float(stop_loss) if stop_loss > 0 else None,
                                "position_pct": float(position_pct) if position_pct > 0 else None,
                                "momentum_score": int(momentum_score) if momentum_score > 0 else None,
                                "momentum_level": momentum_level if momentum_level != "（無）" else "",
                                "dominant_edge": dominant_edge if dominant_edge != "（無）" else "",
                                "raw_text": raw_text.strip(),
                                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            }
                            st.session_state.attack_list[attack_date_str][stock_idx]["ai_judgments"].append(new_judgment)
                            save_attack_list_to_db(st.session_state.attack_list)
                            st.success(f"已儲存 {ai_source} 的判斷")
                            st.rerun()

                # 刪除整檔股票
                st.markdown("---")
                if st.button(f"🗑️ 從清單移除 {sid}", key=f"del_stock_{stock_idx}_{attack_date_str}", type="secondary"):
                    st.session_state.attack_list[attack_date_str].pop(stock_idx)
                    if not st.session_state.attack_list[attack_date_str]:
                        del st.session_state.attack_list[attack_date_str]
                    save_attack_list_to_db(st.session_state.attack_list)
                    st.rerun()

    # ============================================
    # 跨日期歷史總覽
    # ============================================
    st.divider()
    with st.expander("📅 歷史日期總覽（所有日期的進攻名單）", expanded=False):
        all_dates = sorted(st.session_state.attack_list.keys(), reverse=True)
        if not all_dates:
            st.info("尚無歷史紀錄")
        else:
            history_summary = []
            for d in all_dates:
                stocks = st.session_state.attack_list[d]
                stock_count = len(stocks)
                judgment_count = sum(len(s.get("ai_judgments", [])) for s in stocks)
                stock_list = ", ".join([s["stock_id"] for s in stocks])
                history_summary.append({
                    "日期": d,
                    "股票數": stock_count,
                    "AI 判斷總數": judgment_count,
                    "股票清單": stock_list[:50] + ("..." if len(stock_list) > 50 else "")
                })
            st.dataframe(history_summary, use_container_width=True)

            # 下載備份
            json_str = json.dumps(st.session_state.attack_list, ensure_ascii=False, indent=2, default=str)
            st.download_button(
                label="📥 下載完整進攻名單 JSON 備份",
                data=json_str,
                file_name=f"attack_list_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                key="download_attack_list_backup",
                use_container_width=True
            )

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
