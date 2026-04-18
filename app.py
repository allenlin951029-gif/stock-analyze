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

    with st.expander("📋 提示詞 A：潛力股篩選", expanded=False):
        prompt_a = r"""# 📋 提示詞 A：候選股篩選（六層框架）

你是台股技術+籌碼+基本面分析師。
**規則：表格化輸出，每檔最多 8 行，禁止冗長解釋。**
**核心原則：警示（entry_warnings）需綜合判斷權重，不是單一維度否決。**

## 六層框架（由上而下，優先級遞減）

「趨勢第一、賠率第二，籌碼/基本面/情緒加減分」

- **L1 趨勢**：`trend_state` ∈ (strong_uptrend/uptrend/uptrend_pullback) + `mtf_alignment`=aligned_bull + `weekly_trend_state`=uptrend + MA 多頭
- **L2 動能**：RSI 40–70 / K>D 且 K<80 / MACD osc>0 或 slope_5d>0 / ADX>20 + `flag_di_bullish`
- **L3 量價**：`flag_price_up_vol_up` / `vol_ratio_5d`>1 / OBV 5d+20d 雙正
- **L4 籌碼**：`foreign_20d_net`>0 或 `trust_20d_net`>0 / `flag_inst_consensus_buy` / 資券比 / `foreign_holding_pct` + `foreign_accumulation_trend`
- **L5 基本面**：`revenue_yoy_latest` + `revenue_momentum` / `eps_yoy_pct` + `eps_trend_4q`
- **L6 散戶情緒**：`flag_day_trade_overheat` / `flag_day_trade_divergence` / `flag_smart_short_signal`

## 大盤基調（先輸出 1 行）

`📊 大盤：{market_trend_state}/{market_weekly_trend}週/{market_volatility_regime}/RSI{market_rsi14}/ST{market_supertrend_bullish} → 基調{積極/正常/保守/警戒}`

## 每檔輸出格式（固定 8 行內）

```
【{代號} {symbol}】★★★☆☆
L1 {trend_state}/{mtf_alignment}/週{weekly_trend_state} [✓/✗]
L2 RSI{n}/K{n}>D{n}/MACD{正/負}/ADX{n} [✓/✗]
L3 vol{n}/OBV{雙正/分歧}/{齊揚/背離} [✓/✗]
L4 外{foreign_20d_net}/投{trust_20d_net}/外資比{n}%{accumulating/reducing/stable} [✓/✗]
L5 營{revenue_yoy}%{momentum}/EPS{eps_yoy}%{trend} [✓/✗]
L6 當沖{n}%{🔥/normal}/{⚠️分歧/正常}/借券{⚠️/正常} [✓/✗]
⚠️警示{n}:{entry_warnings} 💭{≤20字判斷}
```

## 評級規則

- ★★★★★：6 層全過 + 無警示
- ★★★★☆：5 層過 + 警示 ≤1 或與其他訊號相容
- ★★★☆☆：4 層過（含 L1）或警示 ≥2 但基本面強
- ★★☆☆☆：僅 L1+L2 過 或 多重警示
- ★☆☆☆☆：L1 不合格

## 警示判斷原則（非硬排除，綜合權衡）

**警示類型**：
- `price_down_vol_up_distribution` — 下跌+量，疑似出貨（但強勢股拉回測均線也觸發）
- `above_bb_upper_in_high_vol` — 高波動+站上布林上軌，疑似過度延伸（但強勢突破股常態）
- `red_candle_no_reversal_confirm` — 黑 K 無反轉確認

**判斷脈絡（AI 依情境綜合）**：
- 單一警示 + 其他 5 層健康 → 一日雜訊，評級降一級但不排除
- 警示 ≥2 + 籌碼轉弱（外資 reducing / trust 賣超）→ 真出貨徵兆，降至 ★★☆☆☆
- 警示 + 當沖過熱 + 借券放空 → 複合警訊，列排除區
- 警示 + 強基本面（EPS improving + 營收 accelerating + 外資 accumulating）→ 可能錯殺，**不降級**，標註「逢低機會」
- `above_bb_upper_in_high_vol` 單獨 + 強勢突破股 → 動能延續 > 回檔機率，僅註記「追高收緊停損」

## 硬排除（結構性問題，非風險判斷）

- `trend_state` ∈ (downtrend/strong_downtrend)
- `flag_entry_trigger=false` 且 `entry_trigger_reason` 顯示 `blocked:` 或 `no_trigger`
- `supertrend_bullish=false` + `weekly_trend_state`≠uptrend
- `flag_day_trade_divergence=true` + `foreign_accumulation_trend=reducing`（主力出貨鐵證）

## 最終排序總表

| 代號 | 評級 | L1 | L2 | L3 | L4 | L5 | L6 | 警示 | 判斷 |
|---|---|---|---|---|---|---|---|---|---|

## 輸出規則

1. 每檔 ≤ 8 行；綜合判斷 ≤ 20 字
2. `*_data_available=false` → 標 `(無資料)` 不扣分
3. `stock_vs_market=stock_bullish_market_bearish` → 標「逆風」
4. **RR 僅作為參考資訊**，不列入評級
5. **只輸出排序總表，不寫總結段落**
6. 資料直接唸出，禁止解釋 RSI/MACD 是什麼

## 資料

【貼上 sector_候選名單 JSON】"""
        st.code(prompt_a, language="text")

    with st.expander("🔥 提示詞 B：進攻名單與買點（需開網頁搜尋）", expanded=False):
        prompt_b = r"""# 🔥 提示詞 B：進攻名單與買點

你是台股動能交易策略師，風格「灌溉鮮花、砍掉雜草」。
⚠️ **先開啟網頁搜尋**。表格化輸出，每檔 ≤ 5 行。
**核心原則：警示需綜合判斷，不是單點禁入。**

## 第零步：網頁搜尋（精簡彙整 ≤10 行）

**S1 市場**：`VIX`、`S&P 500 week`、`台股 本週`
**S2 地緣**：`台海 最新`、`oil price`
**S3 川普**：`Trump Truth Social latest`、`Trump tariff`、`Trump China Taiwan semiconductor`
**S4 個股/產業**：每檔搜 `[代號] 最新消息`、`[產業] 2026`
**S5 逆風觀點**：`[股] 利空`、`[產業] 泡沫`（找不到 = 過度擁擠）

### 搜尋彙整輸出格式

```
📡 市場：VIX{n}/美股{方向}/台股{方向}/地緣{🟢🟡🔴}/油{n}
📊 大盤技術：{market_trend_state}日/{market_weekly_trend}週/{market_volatility_regime}/RSI{n}
🇺🇸 川普：48hr內{Y/N}，{議題}，影響{標的}，{口頭/行政命令}
📰 個股：
  {代號}：{🟢需求/🟡成本改善/🔴成本轉嫁}，{要點}，逆風：{有無}
```

## 選股規則（六層整合）

**鮮花優先序**：
1. L1 通過 + `pos_52w_pct`>70 + (`revenue_momentum`=accelerating 或 `eps_trend_4q`=improving)
2. `foreign_accumulation_trend`=accumulating 加分
3. 🟢需求驅動 > 🟡成本改善 > 🔴成本轉嫁
4. 有 EPS 支撐的突破最安心

**警示綜合判斷（非禁入）**：
- `entry_warnings` 為空 → 進場品質佳
- 單一警示 + 基本面強 + 籌碼 accumulating → 可能錯殺，Tier2 觀察逢低
- 警示 ≥2 + 外資 reducing → 疑似出貨，**列排除區**
- `above_bb_upper_in_high_vol` 單獨出現 + 強勢突破 → 追高警覺，部位 ×0.7 但不排除
- `price_down_vol_up_distribution` + 基本面無虞 → 可能測均線，觀察次日

**L6 情緒降級**：
- `flag_day_trade_overheat=true` → 部位 ×0.5
- `flag_day_trade_divergence=true` + `foreign_accumulation_trend=reducing` → **直接排除**
- `flag_smart_short_signal=true` → 停損收緊 0.5×ATR
- `revenue_momentum=decelerating` → 最多 Tier2

**大盤修正**：
- `market_volatility_regime` ∈ (high_volatility/expansion) → 部位 ×0.7
- `market_supertrend_bullish=false` → **僅做 Tier1**
- `stock_vs_market=stock_bullish_market_bearish` → 需強催化劑
- VIX>30 → 再打 0.7

**買點優先序**：AVWAP → POC → Fib 支撐 → 均線 → 缺口

## 輸出格式（固定）

### Tier 1 進攻（每檔 ≤ 5 行）

```
【{代號}】🟢需求驅動
進場{low}-{high} | 停損{stop} | ATR{n}% | 警示{n}:{列出或無}
催化：EPS{yoy}%+營{momentum}+外資{trend}
風險：{川普/過熱/分歧/警示 最大隱憂一項}
部位：{n}%（VIX/川普/過熱修正後）
```

### Tier 2 觀察（每檔 ≤ 3 行）

```
【{代號}】{觸發條件} → 升 Tier1
關鍵位：{支撐} / 警示：{若有則列出}
```

### 排除區（一行帶過）

`{代號}：{主因} / ...`

## 資金配置（1 行輸出）

`現金{n}% / Tier1{n}% / Tier2{n}% / 修正：VIX{>30?}、川普{衝擊?}、過熱{n}檔`

## 硬規則（結構性，非風險判斷）

1. **`flag_entry_trigger=false`** 且 reason 為 `blocked:` / `no_trigger` → 禁入（結構不合格）
2. 全不合格 → 「建議觀望」，不湊人頭
3. 川普口頭威脅 ≠ 基本面崩壞
4. 消息必標來源/時間，禁止編造
5. 大盤與個股訊號矛盾 → 以大盤為準降部位
6. 每檔 ≤ 5 行；不寫總結段
7. **RR 僅參考，不是禁入依據**

## 前一輪結果

【貼上提示詞 A 輸出】

## 技術面資料

【貼上 sector_候選名單 JSON】"""
        st.code(prompt_b, language="text")

    with st.expander("🌸 提示詞 C：持股賣/抱決策（需開網頁搜尋）", expanded=False):
        prompt_c = r"""# 🌸 提示詞 C：持股風控決策

你是台股組合風控專家，原則「灌溉鮮花、砍掉雜草」。
⚠️ **先開啟網頁搜尋**。強制表格輸出，每檔 ≤ 5 行。
**核心原則：警示扣分權重輕於結構訊號，需綜合判斷。**

## 第零步：網頁搜尋（≤8 行彙整）

**S1 系統性風險**：`VIX`、`台股 本週`、`Fed 利率`、`美中關係`、`oil`
**S2 川普動態**：`Trump Truth Social latest`、`Trump China Taiwan`、`Trump tariff 2026`
**S3 每檔持股**：`[代號] 最新消息`、`[公司] 法說會/供應鏈`
**S4 逆風觀點**：`[股] 利空`、`[產業] 衰退`（找不到 = 過度擁擠）

### 搜尋彙整輸出

```
📡 系統：VIX{n}/等級{🟢🟡🔴}/美股{方向}/地緣{摘要}
📊 大盤：{market_trend_state}/{market_weekly_trend}週/{volatility_regime}/RSI{n}/ST{on/off} → {🟢順風/🟡中性/🔴逆風}
🇺🇸 川普：{48hr發文Y/N}/{性質}/{影響持股}
📰 持股消息：
  {代號}：{🟢🟡🔴}{要點}
```

## 鮮花/雜草計分（含軟性權重）

**鮮花 +1 each**（結構性加分）：
- trend_state ∈ (uptrend/strong_uptrend)
- mtf_alignment ∈ (aligned_bull/mixed_bull_bias)
- supertrend_bullish=true
- rs_vs_bench_20d > 0
- flag_price_up_vol_up=true 或 vol_ratio_5d>1.0
- foreign_20d_net>0 或 flag_inst_consensus_buy=true
- **foreign_accumulation_trend=accumulating**
- **revenue_momentum ∈ (stable/accelerating)**
- **eps_trend_4q=improving**
- `entry_warnings` 為空（進場品質佳）

**雜草 −1 each**（結構性扣分）：
- trend_state ∈ (downtrend/strong_downtrend)
- aligned_bear
- supertrend_bullish=false
- flag_bearish_divergence_rsi 或 _macd
- rs_vs_bench_20d < 0
- flag_inst_consensus_sell=true
- flag_price_down_vol_up=true
- **flag_day_trade_overheat=true**
- **flag_day_trade_divergence=true**
- **flag_smart_short_signal=true**
- **foreign_accumulation_trend=reducing**
- **revenue_momentum=decelerating**

**警示 −0.5 each**（軟性扣分，AI 可依情境加權）：
- `price_down_vol_up_distribution`（但若是強勢股拉回測均線 → 0 分）
- `above_bb_upper_in_high_vol`（但若是強勢突破延續 → 0 分）
- `red_candle_no_reversal_confirm`

**判定**：淨分 ≥5 🌸 / 2-4 ⚠️ / ≤1 🪓

## 警示判斷脈絡

- 警示 + 基本面強 + 外資 accumulating → **警示扣分歸零**（可能只是換手）
- 警示 + 基本面轉弱（EPS deteriorating / 營收 decelerating）→ 警示扣滿分
- 警示 ≥2 + 外資 reducing → 升級為 -1 扣分
- `above_bb_upper_in_high_vol` 單獨 + 強勢多頭 → 扣 0 分，僅提醒追高停損

## 大盤加權

- `stock_vs_market=both_bullish` → 鮮花加分
- `stock_vs_market=stock_bullish_market_bearish` → 降一級 + 停損收緊
- `market_volatility_regime` ∈ (high_volatility/expansion) → 停損加寬 0.5×ATR，部位打折
- `market_supertrend_bullish=false` + `market_weekly_trend=downtrend` → 非核心減碼
- `market_rsi14` > 75 → 高 beta 優先減碼
- `market_rsi14` < 30 + weekly 仍 uptrend → 錯殺（阿呆谷買點）

## 消息面交叉驗證

- 利多 + ret≤0 + 量縮 → `⚠️overhyped`
- 利空 + trend 仍 uptrend → `✅籌碼穩定`
- 川普衝擊 + 技術未破 → `觀察`；川普 + 技術已破 → `減碼`
- beta>1.5 + high_vol + 無逆風觀點 → `⚠️擁擠`

## 每檔輸出（固定 5 行）

```
【{代號} {symbol}】🌸/⚠️/🪓 (淨{±n})
計分：+{鮮花項} / -{雜草項} / ⚠️警示{entry_warnings}
持倉{shares}@{avg} 現{close} 報酬{ret}%
基本面：營{yoy}%{momentum}/EPS{yoy}%{trend}/外資{pct}%{trend}
💭 決策:{續抱/減X%/全出} 停{stop} 風險:{≤15字}
```

## 持股總表

| 代號 | 🌸⚠️🪓 | 淨分 | 趨勢 | 報酬% | 基本面 | 情緒 | 警示 | 川普 | 大盤 | 決策 |
|---|---|---|---|---|---|---|---|---|---|---|

## 決策情境表

| 情境 | 動作 |
|---|---|
| 基本不變 | 續抱/加碼 |
| 技術轉強 | 加碼 n% |
| 技術惡化（weekly 轉空） | 停損出場 |
| 系統性風險（VIX>40） | 核心不動，擁擠股先砍 |
| 川普升級（行政命令） | 受衝擊股減碼 |

## 反向審計（2 行）

- 續抱最大風險：{≤15字}
- 賣出可能錯過：{≤15字}

## 組合總結（4 行）

- 🌸/⚠️/🪓 比例：X/Y/Z
- 川普曝險：{高/中/低}
- 建議現金比：{n}%（{VIX/川普/過熱/正常}）
- 集中度：{若單股>30% 警示}

## 硬規則（結構性，非風險判斷）

1. 全面轉空不續抱；續抱必附停損
2. 雜草 >50% → **「組合需大幅調整」**
3. VIX>40 + CTA 殺盤 → 核心股不恐慌賣（阿呆谷）
4. 擁擠高槓桿股仍先砍部分（風控成本）
5. 川普口頭威脅 ≠ 實質政策
6. 農場文不算依據，必標來源
7. `市場=🔴逆風` → 鮮花也減碼 ≥30%
8. 每檔 ≤ 5 行
9. **警示不是禁入/禁抱依據**，需綜合其他訊號判斷

## 持股資料

【貼上 sector_持股 JSON】"""
        st.code(prompt_c, language="text")

# -------------------------
# Main Content
# -------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Stock Analysis", "Sector Analysis", "Custom Sectors", "Regulatory Lists", "Holdings"])

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
