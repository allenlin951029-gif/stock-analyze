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
from stock import analyze_stock_technical, analyze_sector_performance, SECTOR_DICT

st.set_page_config(page_title="Stock Analyze", layout="wide")
st.title("Stock Analyze (雲端資料庫版)")

# -------------------------
# Firestore Configuration (雲端資料庫設定)
# -------------------------
FS_COLLECTION = "stock_app_data"
FS_DOCUMENT = "config"

@st.cache_resource
def get_db():
    if "firebase" in st.secrets:
        try:
            # strict=False 允許 json 中出現換行符號
            key_dict = json.loads(st.secrets["firebase"]["text_key"], strict=False)
            creds = service_account.Credentials.from_service_account_info(key_dict)
            db = firestore.Client(credentials=creds, project=key_dict["project_id"])
            return db
        except Exception as e:
            st.error(f"Firebase 連線失敗: {e}")
            return None
    return None

def load_sectors_from_db():
    """優先從 Firestore 讀取，失敗則回傳空字典"""
    db = get_db()
    if db:
        try:
            doc_ref = db.collection(FS_COLLECTION).document(FS_DOCUMENT)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict().get("custom_sectors", {})
            return {}
        except Exception as e:
            st.warning(f"讀取資料庫失敗: {e}")
            return {}
    # 若無 DB，嘗試讀取 Session 暫存 (重整後會還原)
    return st.session_state.get("_temp_local_sectors", {})

def save_sectors_to_db(data):
    """寫入 Firestore"""
    db = get_db()
    if db:
        try:
            doc_ref = db.collection(FS_COLLECTION).document(FS_DOCUMENT)
            doc_ref.set({"custom_sectors": data}, merge=True)
        except Exception as e:
            st.error(f"寫入資料庫失敗: {e}")
    else:
        st.session_state["_temp_local_sectors"] = data
        st.warning("⚠️ 未設定 Firebase，資料僅暫存於記憶體。")

# -------------------------
# Cookies (僅保留搜尋歷史)
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
    if not raw: return []
    try:
        v = json.loads(raw)
        return [str(x).strip().upper() for x in v if str(x).strip()] if isinstance(v, list) else []
    except: return []

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
# Session Init
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
if "as_of_date" not in st.session_state:
    st.session_state.as_of_date = datetime.now().date()
if "sector_as_of_date" not in st.session_state:
    st.session_state.sector_as_of_date = datetime.now().date()

# -------------------------
# Helpers
# -------------------------
def push_history_cookie(stock_id: str):
    sid = stock_id.strip().upper()
    if not sid: return
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
    sid = stock_id.strip().upper()
    if not sid: return
    st.session_state.current_id = sid
    save_current_to_cookie(sid)
    if write_history: push_history_cookie(sid)
    
    try:
        with st.spinner(f"正在分析 {sid} ..."):
            # 這裡對接新的 stock.py，它會直接回傳字串
            report = analyze_stock_technical(sid, as_of_date=as_of_date)
            save_to_archive(sid, as_of_date, report)
    except Exception as e:
        save_to_archive(sid, as_of_date, f"分析失敗: {e}")

def run_sector_analysis(sector_name: str, as_of_date, custom_list=None):
    try:
        # 呼叫 stock.py 的族群分析
        report = analyze_sector_performance(sector_name, as_of_date=as_of_date, custom_tickers=custom_list)
        save_to_archive(f"快篩: {sector_name}", as_of_date, report)
    except Exception as e:
        save_to_archive(f"快篩: {sector_name}", as_of_date, f"失敗: {e}")

def run_full_sector_report(sector_name: str, as_of_date, custom_list=None):
    target_list = custom_list if custom_list else SECTOR_DICT.get(sector_name, [])
    if not target_list:
        save_to_archive(f"完整: {sector_name}", as_of_date, "無成分股")
        return
    
    full_content = [f"📂 {sector_name} 完整報告 ({as_of_date})", "="*40]
    with st.spinner(f"正在生成 {sector_name} 完整報告 (需時較久)..."):
        for stock in target_list:
            try:
                res = analyze_stock_technical(stock, as_of_date=as_of_date)
                full_content.append(res)
                full_content.append("\n" + "="*40 + "\n")
            except Exception as e:
                full_content.append(f"❌ {stock}: {e}")
    
    save_to_archive(f"完整: {sector_name}", as_of_date, "\n".join(full_content))

# -------------------------
# UI Sidebar
# -------------------------
with st.sidebar:
    st.subheader("設定")
    auto = st.toggle("即時更新（每 10 秒刷新）", value=False, key="auto_refresh")
    tick = 0
    if auto:
        tick = st_autorefresh(interval=10_000, key="autorefresh_10s")
        st.caption(f"refresh tick: {tick}")

    st.divider()
    st.subheader("個股搜尋歷史")
    if st.session_state.history:
        picked = st.selectbox("回查歷史", st.session_state.history)
        if st.button("回查"):
            run_analysis(picked, st.session_state.as_of_date, False)
            st.rerun()
    else:
        st.caption("尚無歷史")
    
    st.divider()
    if "firebase" in st.secrets:
        st.success("🟢 Firebase 連線中")
    else:
        st.warning("🔴 無 Firebase")

# -------------------------
# UI Main Tabs
# -------------------------
tab1, tab2, tab3 = st.tabs(["📊 個股分析", "📈 族群分析", "📂 族群管理"])

# --- Tab 1: 個股 ---
with tab1:
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        sid = st.text_input("股票代號", st.session_state.current_id, key="t1_sid")
    with c2:
        odate = st.date_input("日期", st.session_state.as_of_date, key="t1_date")
        st.session_state.as_of_date = odate
    with c3:
        st.write(""); st.write("")
        if st.button("分析", key="t1_btn"):
            run_analysis(sid, odate, True)
            st.rerun()

# --- Tab 2: 族群分析 ---
with tab2:
    st.write("選擇來源與模式")
    src = st.radio("來源", ["內建", "自選"], horizontal=True, label_visibility="collapsed")
    
    c1, c2 = st.columns([2, 1])
    sel_sector, targets = None, []
    
    with c1:
        if src == "內建":
            sel_sector = st.selectbox("內建族群", list(SECTOR_DICT.keys()))
            if sel_sector: targets = SECTOR_DICT[sel_sector]
        else:
            opts = list(st.session_state.custom_sectors.keys())
            if opts:
                sel_sector = st.selectbox("自選族群", opts)
                if sel_sector: targets = st.session_state.custom_sectors[sel_sector]
            else:
                st.warning("無自選族群，請至「族群管理」新增")
    
    with c2:
        sdate = st.date_input("日期", st.session_state.sector_as_of_date, key="t2_date")
        st.session_state.sector_as_of_date = sdate

    if sel_sector:
        st.caption(f"成分股: {', '.join(targets)}")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("生成快篩表", key="t2_btn1"):
                c_list = targets if src == "自選" else None
                run_sector_analysis(sel_sector, sdate, c_list)
                st.rerun()
        with b2:
            if st.button("生成完整報告", key="t2_btn2"):
                c_list = targets if src == "自選" else None
                run_full_sector_report(sel_sector, sdate, c_list)
                st.rerun()

# --- Tab 3: 族群管理 ---
with tab3:
    st.subheader("自選族群管理 (後台)")
    c1, c2 = st.columns(2)
    with c1:
        new_name = st.text_input("新增族群名稱")
        if st.button("新增", key="t3_add_grp"):
            if new_name and new_name not in st.session_state.custom_sectors:
                st.session_state.custom_sectors[new_name] = []
                save_sectors_to_db(st.session_state.custom_sectors)
                st.success(f"已新增 {new_name}")
                st.rerun()
    with c2:
        if st.session_state.custom_sectors:
            edit_grp = st.selectbox("編輯族群", list(st.session_state.custom_sectors.keys()))
            curr_lst = st.session_state.custom_sectors[edit_grp]
            add_stk = st.text_input("加入代號").strip().upper()
            if st.button("加入", key="t3_add_stk") and add_stk:
                if add_stk not in curr_lst:
                    curr_lst.append(add_stk)
                    save_sectors_to_db(st.session_state.custom_sectors)
                    st.success(f"已加入 {add_stk}")
                    st.rerun()
            
            st.write("目前成分股:")
            for s in curr_lst:
                c_a, c_b = st.columns([4, 1])
                c_a.text(s)
                if c_b.button("刪", key=f"del_{edit_grp}_{s}"):
                    curr_lst.remove(s)
                    save_sectors_to_db(st.session_state.custom_sectors)
                    st.rerun()
            
            st.divider()
            if st.button("刪除此族群", type="primary"):
                del st.session_state.custom_sectors[edit_grp]
                save_sectors_to_db(st.session_state.custom_sectors)
                st.rerun()

# -------------------------
# Auto Refresh (Tab 1 only)
# -------------------------
if auto and tick != st.session_state.last_tick:
    st.session_state.last_tick = tick
    with st.spinner(f"自動更新中：{st.session_state.current_id} ..."):
        run_analysis(st.session_state.current_id, st.session_state.as_of_date, write_history=False)
        st.rerun()

st.divider()

# -------------------------
# Result Display
# -------------------------
if st.session_state.results_archive:
    idx = st.session_state.view_index
    total = len(st.session_state.results_archive)
    if idx < 0: idx = 0
    if idx >= total: idx = total - 1
    st.session_state.view_index = idx
    
    rec = st.session_state.results_archive[idx]
    
    st.markdown(
        f"""
        <div style="text-align: center; background-color: #262730; padding: 10px; border-radius: 5px; border: 1px solid #464b5c; margin-bottom: 10px;">
            <span style="font-size: 1.2em; font-weight: bold; color: #ffffff;">
                {rec['id']}
            </span>
            <span style="color: #cccccc; font-size: 0.9em; margin-left: 10px;">
                ({rec['date']})
            </span>
            <br>
            <span style="font-size: 0.8em; color: #aaaaaa;">
                第 {idx+1} / {total} 筆紀錄 (分析時間: {rec['created_at']})
            </span>
        </div>
        """, 
        unsafe_allow_html=True
    )

    c_space_l, c_prev, c_next, c_space_r = st.columns([2, 1, 1, 2])
    with c_prev:
        if st.button("⬅️ 上一頁", disabled=idx==0):
            st.session_state.view_index -= 1
            st.rerun()
    with c_next:
        if st.button("下一頁 ➡️", disabled=idx==total-1):
            st.session_state.view_index += 1
            st.rerun()

    st.code(rec['content'], language="text")
else:
    st.info("尚未分析或目前沒有紀錄。請在上方選擇「個股」或「族群」並開始分析。")



