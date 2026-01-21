import json
import io
import os
from contextlib import redirect_stdout
from datetime import datetime

import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_cookies_manager import EncryptedCookieManager

# 引入 stock.py 中的函式與變數
# 請確保您的 stock.py 已經更新（支援 custom_tickers 參數）
from stock import analyze_stock_technical, analyze_sector_performance, SECTOR_DICT

st.set_page_config(page_title="Stock Analyze", layout="wide")
st.title("Stock Analyze (翻頁紀錄版)")

# -------------------------
# Cookies (只保留歷史搜尋，不存自選股)
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
# Local File Storage for Custom Sectors (本地檔案儲存)
# -------------------------
SECTORS_FILE = "sectors.json"

def load_sectors_file():
    """從 sectors.json 讀取自選設定"""
    if not os.path.exists(SECTORS_FILE):
        return {}
    try:
        with open(SECTORS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_sectors_file(data):
    """將自選設定寫入 sectors.json"""
    try:
        with open(SECTORS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"存檔失敗: {e}")

# -------------------------
# Session init
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = load_history_from_cookie()
if "current_id" not in st.session_state:
    st.session_state.current_id = load_current_from_cookie()

# 載入自定義族群 (從檔案)
if "custom_sectors" not in st.session_state:
    st.session_state.custom_sectors = load_sectors_file()

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

# -------------------------
# Helpers
# -------------------------
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
        "id": display_title,
        "date": str(display_date),
        "content": content,
        "created_at": datetime.now().strftime("%H:%M:%S")
    }
    st.session_state.results_archive.append(record)
    if len(st.session_state.results_archive) > 10:
        st.session_state.results_archive.pop(0)
    st.session_state.view_index = len(st.session_state.results_archive) - 1

def run_analysis(stock_id: str, as_of_date, write_history: bool):
    """個股分析"""
    sid = stock_id.strip().upper()
    if not sid:
        return
    
    st.session_state.current_id = sid
    save_current_to_cookie(sid)
    
    if write_history:
        push_history_cookie(sid)
    
    buf = io.StringIO()
    ret = None
    final_report = ""
    try:
        with redirect_stdout(buf):
            ret = analyze_stock_technical(sid, as_of_date=as_of_date)
        stdout_text = buf.getvalue()
        if isinstance(ret, str) and ret.strip():
            final_report = ret
        elif stdout_text.strip():
            final_report = stdout_text
        else:
            final_report = "（函式沒有有效輸出）"
    except Exception as e:
        final_report = f"⚠️ 分析失敗：{type(e).__name__}: {e}"
        st.session_state._last_debug = f"exception={type(e).__name__}"
    
    save_to_archive(sid, as_of_date, final_report)

def run_sector_analysis(sector_name: str, as_of_date, custom_list=None):
    """族群漲跌快篩 (表格模式)"""
    final_report = ""
    try:
        # 呼叫 stock.py，若有 custom_list 則優先使用
        final_report = analyze_sector_performance(sector_name, as_of_date=as_of_date, custom_tickers=custom_list)
    except Exception as e:
        final_report = f"⚠️ 族群分析失敗：{e}"
    save_to_archive(f"快篩: {sector_name}", as_of_date, final_report)

def run_full_sector_report(sector_name: str, as_of_date, custom_list=None):
    """族群完整分析 (連發模式) - 針對清單內每一檔做完整分析"""
    target_list = custom_list if custom_list else SECTOR_DICT.get(sector_name, [])
    
    if not target_list:
        save_to_archive(f"完整分析: {sector_name}", as_of_date, "此族群沒有股票。")
        return

    full_content = []
    full_content.append(f"📂 族群 [{sector_name}] 完整技術分析報告")
    full_content.append(f"📅 日期: {as_of_date}")
    full_content.append(f"📊 包含股票: {', '.join(target_list)}")
    full_content.append("=" * 60)
    full_content.append("")

    for stock in target_list:
        try:
            res = analyze_stock_technical(stock, as_of_date=as_of_date)
            full_content.append(res)
            full_content.append("")
            full_content.append("=" * 60) # 分隔線
            full_content.append("")
        except Exception as e:
            full_content.append(f"❌ {stock} 分析失敗: {e}")
            full_content.append("-" * 60)
    
    combined_report = "\n".join(full_content)
    save_to_archive(f"完整分析: {sector_name}", as_of_date, combined_report)

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

    st.divider()
    st.subheader("個股搜尋歷史（前 5 筆）")
    if st.session_state.history:
        picked = st.selectbox("點選回查", st.session_state.history, index=0, key="history_pick")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("回查這筆", use_container_width=True, key="history_run"):
                with st.spinner(f"正在分析 {picked} ..."):
                    run_analysis(picked, st.session_state.as_of_date, write_history=False)
                    st.rerun()
        with col_b:
            if st.button("清除歷史", use_container_width=True, key="history_clear"):
                st.session_state.history = []
                save_history_to_cookie([])
    else:
        st.caption("尚無歷史紀錄")
    st.divider()
    st.info("💡 下方主畫面可翻頁查看最近 10 次的分析結果。")

# -------------------------
# Main Content
# -------------------------
tab1, tab2, tab3 = st.tabs(["📊 個股技術分析", "📈 族群分析 (內建/自選)", "📂 自選族群管理"])

# --- Tab 1: 個股 ---
with tab1:
    col1, col_mid, col2 = st.columns([2.0, 1.1, 1.0])
    with col1:
        stock_id = st.text_input(
            "輸入股票代號（例：0050 / 2330 / 2330.TW / 6223.TWO）",
            value=st.session_state.current_id,
            key="stock_id_input",
        )
    with col_mid:
        as_of = st.date_input("資料日期", value=st.session_state.as_of_date, key="as_of_date_input")
        st.session_state.as_of_date = as_of
    with col2:
        st.write(""), st.write("")
        search = st.button("開始分析個股", use_container_width=True, key="run_btn")
    
    if search:
        with st.spinner(f"正在分析 {stock_id.strip().upper()} ..."):
            run_analysis(stock_id, st.session_state.as_of_date, write_history=True)
            st.rerun()

# --- Tab 2: 族群分析 ---
with tab2:
    st.write("選擇「內建族群」或「自選族群」，產生快篩表或完整報告。")
    
    source_type = st.radio("選擇來源:", ["內建族群", "自選族群 (我的最愛)"], horizontal=True)
    
    c1, c2 = st.columns([2, 1])
    
    selected_sector = None
    target_list = []

    with c1:
        if source_type == "內建族群":
            opts = list(SECTOR_DICT.keys())
            selected_sector = st.selectbox("選擇內建族群", opts, key="sector_select_builtin")
            if selected_sector:
                target_list = SECTOR_DICT[selected_sector]
        else:
            custom_opts = list(st.session_state.custom_sectors.keys())
            if not custom_opts:
                st.warning("目前沒有自選族群，請至「自選族群管理」分頁新增。")
            else:
                selected_sector = st.selectbox("選擇自選族群", custom_opts, key="sector_select_custom")
                if selected_sector:
                    target_list = st.session_state.custom_sectors[selected_sector]

    with c2:
        sector_date = st.date_input("選擇日期", value=st.session_state.sector_as_of_date, key="sector_date")
        st.session_state.sector_as_of_date = sector_date

    if selected_sector:
        st.markdown(f"**目前包含股票**: `{', '.join(target_list) if target_list else '(無)'}`")
        
        b1, b2 = st.columns(2)
        with b1:
            if st.button("📊 生成「漲跌快篩表」", use_container_width=True):
                with st.spinner(f"正在分析 {selected_sector} ..."):
                    # 如果是自選，傳入 custom_list
                    clist = target_list if source_type == "自選族群 (我的最愛)" else None
                    run_sector_analysis(selected_sector, sector_date, custom_list=clist)
                    st.rerun()
        with b2:
            if st.button("📑 生成「完整分析報告」", use_container_width=True, help="會針對清單內每一檔股票跑一次完整分析"):
                with st.spinner(f"正在生成 {selected_sector} 完整報告 (需時較久) ..."):
                    clist = target_list if source_type == "自選族群 (我的最愛)" else None
                    run_full_sector_report(selected_sector, sector_date, custom_list=clist)
                    st.rerun()

# --- Tab 3: 自選族群管理 (後台) ---
with tab3:
    st.header("📂 自選族群管理 (存檔於 sectors.json)")
    st.info("您可以在此新增、編輯自選的股票組合。資料會儲存在伺服器端的檔案中。")
    
    col_mgmt_1, col_mgmt_2 = st.columns(2)
    
    # 1. 新增族群
    with col_mgmt_1:
        with st.container(border=True):
            st.subheader("1. 新增族群")
            new_group = st.text_input("輸入新族群名稱 (例: 觀察名單)")
            if st.button("建立族群"):
                if not new_group.strip():
                    st.error("名稱不能為空")
                elif new_group in st.session_state.custom_sectors:
                    st.error("名稱已存在")
                else:
                    st.session_state.custom_sectors[new_group] = []
                    save_sectors_file(st.session_state.custom_sectors)
                    st.success(f"已建立 {new_group}")
                    st.rerun()
    
    # 2. 編輯族群
    with col_mgmt_2:
        with st.container(border=True):
            st.subheader("2. 編輯族群")
            if not st.session_state.custom_sectors:
                st.info("暫無資料，請先新增族群")
            else:
                edit_group = st.selectbox("選擇要編輯的族群", list(st.session_state.custom_sectors.keys()), key="mgmt_select")
                current_list = st.session_state.custom_sectors[edit_group]
                
                # Add Stock
                c_add1, c_add2 = st.columns([3, 1])
                with c_add1:
                    stock_to_add = st.text_input("輸入股票代號 (例: 2330)", key="mgmt_add_input")
                with c_add2:
                    st.write(""), st.write("")
                    if st.button("➕ 加入"):
                        val = stock_to_add.strip().upper()
                        if val:
                            if val not in current_list:
                                current_list.append(val)
                                save_sectors_file(st.session_state.custom_sectors)
                                st.success(f"已加入 {val}")
                                st.rerun()
                            else:
                                st.warning("已存在清單中")
                
                st.divider()
                st.write(f"**{edit_group}** 成分股:")
                
                # List & Remove
                if not current_list:
                    st.caption("(空)")
                else:
                    for s in current_list:
                        cr1, cr2 = st.columns([4, 1])
                        with cr1:
                            st.text(f"• {s}")
                        with cr2:
                            if st.button("移除", key=f"del_{edit_group}_{s}"):
                                current_list.remove(s)
                                save_sectors_file(st.session_state.custom_sectors)
                                st.rerun()
                
                st.divider()
                if st.button("🗑️ 刪除此族群", type="primary"):
                    del st.session_state.custom_sectors[edit_group]
                    save_sectors_file(st.session_state.custom_sectors)
                    st.warning(f"已刪除 {edit_group}")
                    st.rerun()

# -------------------------
# Auto Refresh Logic
# -------------------------
if auto and tick != st.session_state.last_tick:
    st.session_state.last_tick = tick
    # 自動刷新只針對 Tab 1 的個股
    with st.spinner(f"自動更新中：{st.session_state.current_id} ..."):
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
    
    st.markdown(
        f"""
        <div style="text-align: center; background-color: #262730; padding: 10px; border-radius: 5px; border: 1px solid #464b5c; margin-bottom: 10px;">
            <span style="font-size: 1.2em; font-weight: bold; color: #ffffff;">
                {record['id']}
            </span>
            <span style="color: #cccccc; font-size: 0.9em; margin-left: 10px;">
                ({record['date']})
            </span>
            <br>
            <span style="font-size: 0.8em; color: #aaaaaa;">
                第 {current_idx + 1} / {archive_len} 筆紀錄 (分析時間: {record['created_at']})
            </span>
        </div>
        """, 
        unsafe_allow_html=True
    )

    c_space_l, c_prev, c_next, c_space_r = st.columns([2, 1, 1, 2])
    
    with c_prev:
        if st.button("⬅️ 上一頁", disabled=(current_idx == 0), use_container_width=True):
            st.session_state.view_index -= 1
            st.rerun()
            
    with c_next:
        if st.button("下一頁 ➡️", disabled=(current_idx == archive_len - 1), use_container_width=True):
            st.session_state.view_index += 1
            st.rerun()

    st.code(record['content'], language="text")

else:
    st.info("尚未分析或目前沒有紀錄。請在上方選擇「個股」或「族群」並開始分析。")
