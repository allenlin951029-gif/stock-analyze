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

# 引入 stock.py 中的函式與變數

from stock import analyze_stock_technical, analyze_sector_performance, SECTOR_DICT, build_ai_features, format_text_report

st.set_page_config(page_title=“Stock Analyze”, layout=“wide”)
st.title(“Stock Analyze (雲端資料庫 + AI 雙模式版)”)

# ———————––

# Firestore Configuration (雲端資料庫設定)

# ———————––

FS_COLLECTION = “stock_app_data”
FS_DOCUMENT = “config”

@st.cache_resource
def get_db():
“”“初始化 Firestore 連線”””
if “firebase” in st.secrets:
try:
key_dict = json.loads(st.secrets[“firebase”][“text_key”], strict=False)
creds = service_account.Credentials.from_service_account_info(key_dict)
db = firestore.Client(credentials=creds, project=key_dict[“project_id”])
return db
except Exception as e:
st.error(f”Firebase 連線失敗: {e}”)
return None
return None

def load_sectors_from_db():
“”“從 Firestore 讀取自選設定”””
db = get_db()
if db:
try:
doc_ref = db.collection(FS_COLLECTION).document(FS_DOCUMENT)
doc = doc_ref.get()
if doc.exists:
data = doc.to_dict()
return data.get(“custom_sectors”, {})
else:
return {}
except Exception as e:
st.warning(f”讀取資料庫失敗 (暫用空白設定): {e}”)
return {}
else:
return st.session_state.get(”_temp_local_sectors”, {})

def save_sectors_to_db(data):
“”“將自選設定寫入 Firestore”””
db = get_db()
if db:
try:
doc_ref = db.collection(FS_COLLECTION).document(FS_DOCUMENT)
doc_ref.set({“custom_sectors”: data}, merge=True)
except Exception as e:
st.error(f”寫入資料庫失敗: {e}”)
else:
st.session_state[”_temp_local_sectors”] = data
st.warning(“⚠️ 未設定 Firebase，資料僅暫存於記憶體，App 休眠後將消失。”)

# ———————––

# Cookies (只保留歷史搜尋，不存自選股)

# ———————––

cookies = EncryptedCookieManager(
prefix=“stock_analyze_”,
password=st.secrets.get(“COOKIE_PASSWORD”, “dev_password_change_me_32chars_min_____”),
)
if not cookies.ready():
st.stop()

HIST_KEY = “history”
CUR_KEY = “current_id”

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
return (v or “0050”).strip().upper()

st.session_state[”_cookie_saved_this_run”] = False
def commit_cookies_once():
if not st.session_state.get(”_cookie_saved_this_run”, False):
cookies.save()
st.session_state[”_cookie_saved_this_run”] = True

def save_history_to_cookie(history_list):
cookies[HIST_KEY] = json.dumps(history_list, ensure_ascii=False)
commit_cookies_once()

def save_current_to_cookie(sid):
cookies[CUR_KEY] = sid.strip().upper()
commit_cookies_once()

# ———————––

# Session init

# ———————––

if “history” not in st.session_state:
st.session_state.history = load_history_from_cookie()
if “current_id” not in st.session_state:
st.session_state.current_id = load_current_from_cookie()

if “custom_sectors” not in st.session_state:
st.session_state.custom_sectors = load_sectors_from_db()

if “results_archive” not in st.session_state:
st.session_state.results_archive = []
if “view_index” not in st.session_state:
st.session_state.view_index = 0
if “_last_debug” not in st.session_state:
st.session_state._last_debug = “”
if “last_tick” not in st.session_state:
st.session_state.last_tick = 0
if “as_of_date” not in st.session_state:
st.session_state.as_of_date = datetime.now().date()
if “sector_as_of_date” not in st.session_state:
st.session_state.sector_as_of_date = datetime.now().date()

# 報告模式 (預設 Human)

if “report_mode” not in st.session_state:
st.session_state.report_mode = “👤 Human（閱讀版）”

# ———————––

# Helpers

# ———————––

def push_history_cookie(stock_id: str):
sid = stock_id.strip().upper()
if not sid:
return
st.session_state.history = [x for x in st.session_state.history if x != sid]
st.session_state.history.insert(0, sid)
st.session_state.history = st.session_state.history[:5]
save_history_to_cookie(st.session_state.history)

def save_to_archive(display_title, display_date, content):
record = {
“id”: display_title,
“date”: str(display_date),
“content”: content,
“created_at”: datetime.now().strftime(”%H:%M:%S”)
}
st.session_state.results_archive.append(record)
if len(st.session_state.results_archive) > 10:
st.session_state.results_archive.pop(0)
st.session_state.view_index = len(st.session_state.results_archive) - 1

def is_ai_mode():
return st.session_state.report_mode == “🤖 AI（JSON 數據版）”

def run_analysis(stock_id: str, as_of_date, write_history: bool):
“”“個股分析 (支援 Human/AI 雙模式回傳)”””
sid = stock_id.strip().upper()
if not sid:
return

```
st.session_state.current_id = sid
save_current_to_cookie(sid)

if write_history:
    push_history_cookie(sid)

final_result = None
try:
    final_result = analyze_stock_technical(sid, as_of_date=as_of_date)
except Exception as e:
    err_msg = f"⚠️ 分析失敗：{type(e).__name__}: {e}"
    final_result = {
        "human_report": err_msg,
        "ai_report": {"error": str(e)}
    }
    st.session_state._last_debug = f"exception={type(e).__name__}"

save_to_archive(sid, as_of_date, final_result)
```

def run_sector_analysis(sector_name: str, as_of_date, custom_list=None):
“”“族群漲跌快篩 (表格模式)”””
final_report = “”
try:
final_report = analyze_sector_performance(sector_name, as_of_date=as_of_date, custom_tickers=custom_list)
except Exception as e:
final_report = f”⚠️ 族群分析失敗：{e}”
save_to_archive(f”快篩: {sector_name}”, as_of_date, final_report)

def run_full_sector_report(sector_name: str, as_of_date, custom_list=None):
“”“族群完整分析 (連發模式)”””
target_list = custom_list if custom_list else SECTOR_DICT.get(sector_name, [])

```
if not target_list:
    save_to_archive(f"完整分析: {sector_name}", as_of_date, "此族群沒有股票。")
    return

# 根據模式決定輸出格式
if is_ai_mode():
    # AI 模式：收集所有股票的 ai_report 組成一個大 JSON
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
        "human_report": f"📂 族群 [{sector_name}] AI 完整分析 ({len(target_list)} 檔)\n日期: {as_of_date}",
        "ai_report": {
            "sector": sector_name,
            "date": str(as_of_date),
            "stocks": all_reports
        }
    }
    save_to_archive(f"完整分析: {sector_name}", as_of_date, combined)
else:
    # Human 模式：拼接文字報告
    full_content = []
    full_content.append(f"📂 族群 [{sector_name}] 完整技術分析報告")
    full_content.append(f"📅 日期: {as_of_date}")
    full_content.append(f"📊 包含股票: {', '.join(target_list)}")
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
            full_content.append(f"❌ {stock} 分析失敗: {e}")
            full_content.append("-" * 60)

    combined_report = "\n".join(full_content)
    save_to_archive(f"完整分析: {sector_name}", as_of_date, combined_report)
```

# ———————––

# Sidebar

# ———————––

with st.sidebar:
st.subheader(“⚙️ 設定”)

```
# ★ 報告模式選擇 (放在最上方、最醒目)
st.markdown("---")
st.markdown("#### 📋 報告模式")
mode = st.radio(
    "選擇報告輸出模式：",
    ["👤 Human（閱讀版）", "🤖 AI（JSON 數據版）"],
    index=0 if st.session_state.report_mode == "👤 Human（閱讀版）" else 1,
    key="report_mode_radio",
    help="Human：精簡文字摘要，適合人類閱讀。\nAI：完整 JSON 數據，適合 LLM / ML 模型消費。"
)
st.session_state.report_mode = mode

# 模式說明
if is_ai_mode():
    st.info("🤖 **AI 模式**\n輸出完整 JSON 數據，包含所有指標數值、旗標、統計量，可下載供模型使用。")
else:
    st.success("👤 **Human 模式**\n輸出精簡文字摘要，包含趨勢、指標、旗標等重點資訊。")

st.markdown("---")
auto = st.toggle("即時更新（每 10 秒刷新）", value=False, key="auto_refresh")
tick = 0
if auto:
    tick = st_autorefresh(interval=10_000, key="autorefresh_10s")
    st.caption(f"autorefresh tick = {tick}")

st.divider()
st.subheader("個股搜尋歷史")
if st.session_state.history:
    picked = st.selectbox("點選回查", st.session_state.history, index=0, key="history_pick")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("回查", use_container_width=True, key="history_run"):
            with st.spinner(f"正在分析 {picked} ..."):
                run_analysis(picked, st.session_state.as_of_date, write_history=False)
                st.rerun()
    with col_b:
        if st.button("清除", use_container_width=True, key="history_clear"):
            st.session_state.history = []
            save_history_to_cookie([])
            st.rerun()
else:
    st.caption("尚無歷史紀錄")

st.divider()
if "firebase" in st.secrets:
    st.success("🟢 已連接雲端資料庫")
else:
    st.warning("🔴 未連接雲端資料庫")

st.info("💡 下方主畫面可翻頁查看分析結果。")
```

# ———————––

# Main Content

# ———————––

tab1, tab2, tab3 = st.tabs([“📊 個股技術分析”, “📈 族群分析”, “📂 自選管理”])

# — Tab 1: 個股 —

with tab1:
col1, col_mid, col2 = st.columns([2.0, 1.1, 1.0])
with col1:
stock_id = st.text_input(
“輸入股票代號（例：2330 / 0050 / 6531.TW）”,
value=st.session_state.current_id,
key=“stock_id_input”,
)
with col_mid:
as_of = st.date_input(“資料日期”, value=st.session_state.as_of_date, key=“as_of_date_input”)
st.session_state.as_of_date = as_of
with col2:
st.write(””), st.write(””)
# 按鈕文字根據模式顯示
btn_label = “🤖 開始 AI 分析” if is_ai_mode() else “👤 開始分析個股”
search = st.button(btn_label, use_container_width=True, key=“run_btn”)

```
if search:
    with st.spinner(f"正在分析 {stock_id.strip().upper()} （{'AI模式' if is_ai_mode() else 'Human模式'}）..."):
        run_analysis(stock_id, st.session_state.as_of_date, write_history=True)
        st.rerun()
```

# — Tab 2: 族群分析 —

with tab2:
source_type = st.radio(“選擇來源:”, [“內建族群”, “自選族群”], horizontal=True)
c1, c2 = st.columns([2, 1])

```
selected_sector = None
target_list = []

with c1:
    if source_type == "內建族群":
        opts = list(SECTOR_DICT.keys())
        selected_sector = st.selectbox("選擇族群", opts, key="sector_select_builtin")
        if selected_sector:
            target_list = SECTOR_DICT[selected_sector]
    else:
        custom_opts = list(st.session_state.custom_sectors.keys())
        if not custom_opts:
            st.warning("目前沒有自選族群。")
        else:
            selected_sector = st.selectbox("選擇自選族群", custom_opts, key="sector_select_custom")
            if selected_sector:
                target_list = st.session_state.custom_sectors[selected_sector]

with c2:
    sector_date = st.date_input("選擇日期", value=st.session_state.sector_as_of_date, key="sector_date")
    st.session_state.sector_as_of_date = sector_date

if selected_sector:
    st.markdown(f"**包含股票**: `{', '.join(target_list) if target_list else '(無)'}`")
    
    # 顯示當前模式提示
    if is_ai_mode():
        st.caption("🤖 目前為 AI 模式 — 完整報告將輸出 JSON 格式")
    else:
        st.caption("👤 目前為 Human 模式 — 完整報告將輸出文字摘要")
    
    b1, b2 = st.columns(2)
    with b1:
        if st.button("📊 生成「漲跌快篩表」", use_container_width=True):
            with st.spinner(f"正在分析 {selected_sector} ..."):
                clist = target_list if source_type == "自選族群" else None
                run_sector_analysis(selected_sector, sector_date, custom_list=clist)
                st.rerun()
    with b2:
        report_btn_label = "📑 生成「完整分析報告」"
        if st.button(report_btn_label, use_container_width=True):
            with st.spinner(f"正在生成 {selected_sector} 完整報告（{'AI' if is_ai_mode() else 'Human'} 模式）..."):
                clist = target_list if source_type == "自選族群" else None
                run_full_sector_report(selected_sector, sector_date, custom_list=clist)
                st.rerun()
```

# — Tab 3: 自選管理 —

with tab3:
st.header(“📂 自選族群管理”)
col_mgmt_1, col_mgmt_2 = st.columns(2)

```
# 1. 新增
with col_mgmt_1:
    with st.container(border=True):
        st.subheader("新增族群")
        new_group = st.text_input("輸入新族群名稱")
        if st.button("建立"):
            if not new_group.strip():
                st.error("名稱不能為空")
            elif new_group in st.session_state.custom_sectors:
                st.error("名稱已存在")
            else:
                st.session_state.custom_sectors[new_group] = []
                save_sectors_to_db(st.session_state.custom_sectors)
                st.success(f"已建立 {new_group}")
                st.rerun()

# 2. 編輯
with col_mgmt_2:
    with st.container(border=True):
        st.subheader("編輯族群")
        if not st.session_state.custom_sectors:
            st.info("暫無資料")
        else:
            edit_group = st.selectbox("選擇族群", list(st.session_state.custom_sectors.keys()), key="mgmt_select")
            current_list = st.session_state.custom_sectors[edit_group]
            
            c_add1, c_add2 = st.columns([3, 1])
            with c_add1:
                stock_to_add = st.text_input("輸入股票代號", key="mgmt_add_input")
            with c_add2:
                st.write(""), st.write("")
                if st.button("➕ 加入"):
                    val = stock_to_add.strip().upper()
                    if val and val not in current_list:
                        current_list.append(val)
                        save_sectors_to_db(st.session_state.custom_sectors)
                        st.success(f"已加入 {val}")
                        st.rerun()
            
            st.divider()
            if not current_list:
                st.caption("(空)")
            else:
                for s in current_list:
                    cr1, cr2 = st.columns([4, 1])
                    with cr1: st.text(f"• {s}")
                    with cr2:
                        if st.button("移除", key=f"del_{edit_group}_{s}"):
                            current_list.remove(s)
                            save_sectors_to_db(st.session_state.custom_sectors)
                            st.rerun()
            
            st.divider()
            if st.button("🗑️ 刪除此族群", type="primary"):
                del st.session_state.custom_sectors[edit_group]
                save_sectors_to_db(st.session_state.custom_sectors)
                st.rerun()
```

# ———————––

# Auto Refresh Logic

# ———————––

if auto and tick != st.session_state.last_tick:
st.session_state.last_tick = tick
with st.spinner(f”自動更新中：{st.session_state.current_id} …”):
run_analysis(st.session_state.current_id, st.session_state.as_of_date, write_history=False)
st.rerun()

st.divider()

# ———————––

# Pagination Display (根據全域模式自動切換)

# ———————––

archive_len = len(st.session_state.results_archive)

if archive_len > 0:
if st.session_state.view_index < 0:
st.session_state.view_index = 0
if st.session_state.view_index >= archive_len:
st.session_state.view_index = archive_len - 1

```
current_idx = st.session_state.view_index
record = st.session_state.results_archive[current_idx]

# 模式標籤
mode_badge = "🤖 AI" if is_ai_mode() else "👤 Human"

# 標題區塊
st.markdown(
    f"""
    <div style="text-align: center; background-color: #262730; padding: 10px; border-radius: 5px; border: 1px solid #464b5c; margin-bottom: 10px;">
        <span style="font-size: 1.2em; font-weight: bold; color: #ffffff;">{record['id']}</span>
        <span style="color: #cccccc; font-size: 0.9em; margin-left: 10px;">({record['date']})</span>
        <span style="background-color: {'#1a73e8' if is_ai_mode() else '#2e7d32'}; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; margin-left: 8px;">{mode_badge}</span>
        <br>
        <span style="font-size: 0.8em; color: #aaaaaa;">第 {current_idx + 1} / {archive_len} 筆紀錄 ({record['created_at']})</span>
    </div>
    """, 
    unsafe_allow_html=True
)

# 翻頁按鈕
c_prev, c_mode_switch, c_next = st.columns([1, 1, 1])
with c_prev:
    if st.button("⬅️ 上一頁", disabled=(current_idx == 0), use_container_width=True):
        st.session_state.view_index -= 1
        st.rerun()
with c_mode_switch:
    # 快速切換按鈕（無需回側邊欄）
    content = record['content']
    if isinstance(content, dict) and "human_report" in content and "ai_report" in content:
        switch_label = "切換至 👤 Human" if is_ai_mode() else "切換至 🤖 AI"
        if st.button(switch_label, use_container_width=True, key="inline_mode_switch"):
            if is_ai_mode():
                st.session_state.report_mode = "👤 Human（閱讀版）"
            else:
                st.session_state.report_mode = "🤖 AI（JSON 數據版）"
            st.rerun()
with c_next:
    if st.button("下一頁 ➡️", disabled=(current_idx == archive_len - 1), use_container_width=True):
        st.session_state.view_index += 1
        st.rerun()

# ===========================
# 內容顯示 (根據全域模式)
# ===========================
content = record['content']
st.write("---")

if isinstance(content, dict) and "human_report" in content and "ai_report" in content:
    # 雙模式資料 (個股分析 / 族群完整AI報告)
    if is_ai_mode():
        # ── AI 模式 ──
        st.markdown("### 🤖 AI Features JSON Data")
        
        ai_data = content["ai_report"]
        
        # 如果是族群 AI 報告（包含多檔股票）
        if isinstance(ai_data, dict) and "stocks" in ai_data:
            st.markdown(f"**族群**: {ai_data.get('sector', '?')} — **日期**: {ai_data.get('date', '?')}")
            st.markdown(f"**包含 {len(ai_data['stocks'])} 檔股票**")
            
            # 可展開查看每檔
            for stock_key, stock_data in ai_data["stocks"].items():
                with st.expander(f"📄 {stock_key}", expanded=False):
                    st.json(stock_data)
            
            # 整包下載
            json_str = json.dumps(ai_data, indent=2, default=str, ensure_ascii=False)
            st.download_button(
                label="📥 下載整份 JSON",
                data=json_str,
                file_name=f"sector_{ai_data.get('sector','unknown')}_{ai_data.get('date','')}_.json",
                mime="application/json",
                key=f"dl_sector_json_{current_idx}"
            )
        else:
            # 單檔個股 AI 資料
            st.json(ai_data)
            
            json_str = json.dumps(ai_data, indent=2, default=str, ensure_ascii=False)
            st.download_button(
                label="📥 下載 JSON 檔案",
                data=json_str,
                file_name=f"{record['id']}_{record['date']}_ai_features.json",
                mime="application/json",
                key=f"dl_json_{current_idx}"
            )
    else:
        # ── Human 模式 ──
        st.markdown("### 👤 Human Report（閱讀版）")
        st.code(content["human_report"], language="text")

else:
    # 純文字報告 (例如族群快篩表、舊版資料)
    st.code(str(content), language="text")
```

else:
st.info(“尚未分析或目前沒有紀錄。請在上方選擇「個股」或「族群」並開始分析。”)
