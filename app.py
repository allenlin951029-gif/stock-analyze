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


def load_holdings_from_db():
    db = get_db()
    if db:
        try:
            doc_ref = db.collection(FS_COLLECTION).document(FS_DOCUMENT)
            doc = doc_ref.get()
            if doc.exists:
                return doc.to_dict().get("holdings", [])
            return []
        except Exception as e:
            st.warning(f"DB read (holdings) failed: {e}")
            return []
    return st.session_state.get("_temp_local_holdings", [])


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

    mode = st.session_state.report_mode  # 'human' or 'ai'

    if mode == "ai":  # Deep Dive
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

    # ==========================================
    # AI 操盤手提示詞複製區（可折疊）
    # ==========================================
    st.divider()
    st.subheader("🤖 AI 提示詞")
    st.caption("展開複製，連同 JSON 貼給 AI")

    # --- Prompt A: 潛力股篩選 ---
    with st.expander("📋 提示詞 A：潛力股篩選", expanded=False):
        prompt_a = r"""你是一位台股技術分析專家，擅長多時間框架趨勢分析與量價結構判讀。
我將上傳一份 JSON 格式的候選股技術面資料，請依照以下框架嚴格篩選出具潛力的標的。

═══ 分析框架（按優先級排序）═══

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
⚠️ 反向審計：價格上漲但 vol_ratio_5d < 0.8 或 obv_slope_20d < 0 → 標註「量能不足，突破可信度存疑」

【第四層：籌碼面 — 修正項，非決定項】
- foreign_20d_net > 0 或 trust_20d_net > 0
- flag_inst_consensus_buy = true 最佳
- flag_foreign_divergence = true → 標註背離
- flag_inst_consensus_sell = true → 降級

【第五層：風險評估】
- risk_reward_ratio ≥ 1.5 合格，≥ 2.0 優秀
- flag_poor_risk_reward = true 直接排除
- atr_stop_loss_pct < -15%、max_drawdown_20d > -20% 需警示
- beta_60d > 2.0 標註「高 Beta」

═══ 輸出格式 ═══
每檔股票：潛力評級（5星制）、趨勢/動能/量價/籌碼/風險一句話摘要、綜合判斷、風險提醒。
最後做排序總表。

═══ 重要規則 ═══
1. 必須交叉驗證至少 3 個維度
2. 每個看多結論附反面檢查：「如果我看錯了，最可能錯在哪？」
3. supertrend_bullish = false 必須標註
4. entry_trigger_veto 不為空時逐條列出

═══ 我的資料如下 ═══
【貼上 sector_候選名單 JSON】"""
        st.code(prompt_a, language="text")

    # --- Prompt B: 進攻名單（含網頁搜尋 + 川普推文）---
    with st.expander("🔥 提示詞 B：進攻名單與買點（需開網頁搜尋）", expanded=False):
        prompt_b = r"""你是一位台股短中線交易策略師，風格為「灌溉鮮花、砍掉雜草」。
我將提供候選股技術面 JSON + 前一輪篩選結果，請制定進攻名單與買點。

⚠️ 請先開啟「網頁搜尋」功能，分析前先完成以下搜尋任務。

═══ 第零步：網頁搜尋任務（先做完再分析）═══

【搜尋 1：國際市場與恐慌指標】
- "VIX index today"
- "S&P 500 this week"、"美股 本週"、"台股 大盤 本週"
- "Fed interest rate latest"
→ VIX > 25 標註警戒，VIX > 40 標註「恐慌 — CTA 大逃殺買點」

【搜尋 2：地緣政治與總經風險】
- "geopolitical risk 2026"、"台海 地緣政治 最新"
- "oil price today"、"global supply chain disruption"

【搜尋 3：川普社群發文與市場反應 🇺🇸】
- "Trump Truth Social latest post"
- "Trump tariff latest"
- "Trump trade policy 2026"
- "川普 關稅 最新"
- "Trump statement market reaction"
目的：
a) 川普最近是否有發文提及關稅、制裁、中國/台灣、科技業、Fed？
b) 這些發文是否已引發市場反應（美股期貨波動、特定板塊暴跌/暴漲）？
c) 對我候選名單中的標的有無直接衝擊？（例如加徵半導體關稅 → 影響台積電供應鏈）
⚠️ 川普發文的特性：時間不固定、內容經常反覆、市場反應可能過度。
→ 已反映在股價的舊消息不算新風險
→ 但剛發布且市場尚未完全反應的新推文 = 重大變數

【搜尋 4：個股產業消息（Bottom-up 供應鏈研調）】
每檔候選股搜尋：
- "[股票代號] [公司名] 最新消息 2026"
- "[公司名] 供應鏈 拉貨"、"[公司名] 營收 EPS"
- "[所屬產業] 產業趨勢 2026"

【搜尋 5：逆風觀點（Contrarian View）】
- "[公司名] 利空 風險"、"[所屬產業] 泡沫 過熱"

⚠️ 搜尋紀律：忽略社群農場文，優先採信法說會/財報/供應鏈數據/官方統計。

═══ 搜尋結果彙整（先輸出）═══

## 📡 市場環境掃描
**VIX**：XX | **狀態**：正常/警戒/恐慌
**美股**：[摘要] | **台股**：[摘要]
**地緣風險**：[摘要] | **油價**：$XX

## 🇺🇸 川普動態
**最近發文摘要**：[關鍵推文內容與時間]
**涉及議題**：[關稅/Fed/中國/科技業/其他]
**市場已反應程度**：[已充分反應/部分反應/尚未反應]
**對候選名單影響**：[直接衝擊/間接影響/無關]

## 📰 個股消息面
（每檔：產業趨勢、驅動力🟢🟡🔴、重大消息、供應鏈拉貨、利多出盡風險、逆風觀點）

═══ 選股與買入邏輯 ═══
1. 只收鮮花：trend_state="uptrend" + aligned_bull 首選，pos_52w_pct > 70
2. 需求驅動優先：🟢需求驅動 > 🟡成本改善 > 🔴成本轉嫁
3. 基本面搭配技術面：有 EPS 支撐的突破更安心
4. 捕捉市場犯錯：大跌但基本面無惡化 → 「潛在錯殺」
5. 利多出盡檢查：搜尋到利多但 stock_ret_20d ≤ 0 → 「⚠️ overhyped」
6. 川普風險折價：若川普最新發文直接衝擊某標的所屬產業且市場尚未反應 → 該標的部位自動打 5 折或暫緩進場
7. 禁止在 downtrend 攤平

【買點】依序：AVWAP → POC → Fibonacci → 均線 → 缺口
【停損】atr_stop_loss | 【目標】target_resistance → fib_ext_1272/1618
【部位】VIX > 30 → 所有部位打 7 折；川普新推文衝擊 → 再打折

═══ 輸出 ═══
Tier 1（立即進場）/ Tier 2（監控中）/ ❌ 排除
每檔含：進場區間、停損、目標、RR、驅動力、消息面、逆風觀點、川普風險、最大隱憂
最後：資金配置建議（含現金部位 15-20%，高風險時 25%+）

═══ 重要規則 ═══
1. RR < 1.5 禁入進攻名單
2. 全部不合格就直接說「建議觀望持有現金」
3. 每個買入建議附最壞情境
4. 搜尋結果全是看多 → 標註「擁擠風險」
5. 消息必須標明來源，不編造新聞

═══ 前一輪篩選結果 ═══
【貼上提示詞 A 輸出】

═══ 原始技術面資料 ═══
【貼上 sector_候選名單 JSON】"""
        st.code(prompt_b, language="text")

    # --- Prompt C: 持股賣/抱決策（含網頁搜尋 + 川普推文）---
    with st.expander("🌸 提示詞 C：持股賣/抱決策（需開網頁搜尋）", expanded=False):
        prompt_c = r"""你是一位台股投資組合風控專家，核心原則「灌溉鮮花、砍掉雜草」。
我將上傳持股技術面 JSON，請對每檔做「賣出 vs 續抱」決策分析。

⚠️ 請先開啟「網頁搜尋」功能。

═══ 第零步：網頁搜尋任務 ═══

【搜尋 1：系統性風險掃描】
- "VIX index today"、"台股加權指數 本週"、"Fed 利率 最新"
- "美中關係 最新"、"oil price today"
→ 風險等級：🟢低 / 🟡中 / 🔴高

【搜尋 2：川普社群發文 🇺🇸】
- "Trump Truth Social latest post"
- "Trump tariff latest 2026"
- "川普 關稅 最新"
- "Trump China Taiwan semiconductor"
- "Trump statement market impact"
重點關注：
a) 最近 48 小時內是否有新發文？涉及什麼議題？
b) 對我的持股產業有無直接衝擊？
c) 市場已反應多少？（已消化 vs 剛發布）
d) 過去類似發文後市場的反應模式（通常過度反應後回彈？還是趨勢性轉變？）
⚠️ 判斷原則：
→ 川普推文引發的恐慌殺盤 ≠ 基本面崩壞，不要因此建議賣出基本面無虞的持股
→ 但如果推文內容是「實質政策變化」（如正式簽署行政命令）而非口頭威脅 → 風險等級上調

【搜尋 3：每檔持股個股消息】
每檔搜尋：
- "[股票代號] [公司名] 最新消息 2026"
- "[公司名] 法說會 財報"、"[公司名] 供應鏈 訂單 拉貨"

【搜尋 4：逆風觀點】
- "[公司名] 風險 利空"、"[所屬產業] 衰退 泡沫"
→ 找不到任何看空觀點 = 過度擁擠警訊

═══ 搜尋結果彙整（先輸出）═══

## 📡 系統性風險
**VIX**：XX | **等級**：🟢/🟡/🔴
**美股**：[摘要] | **台股**：[摘要] | **地緣**：[摘要]

## 🇺🇸 川普動態
**最近發文**：[時間 + 內容摘要]
**性質判斷**：[口頭威脅 / 政策預告 / 正式行政命令]
**影響持股**：[列出直接受衝擊的持股代號及影響路徑]
**歷史模式**：[過去類似推文後市場表現]
**建議應對**：[不需反應 / 密切觀察 / 高beta先減碼]

## 📰 個股消息面
（每檔：重大消息、供應鏈狀況、利多出盡、逆風觀點、消息vs技術一致性）

═══ 技術面分析框架 ═══

【鮮花指標】trend_state=uptrend、aligned_bull、supertrend_bullish=true、rs_vs_bench_20d>0、
flag_price_up_vol_up=true、RR≥1.5、法人買超

【雜草指標】downtrend、aligned_bear、supertrend_bullish=false、頂背離、
rs弱於大盤且惡化、RR<1.0、flag_poor_risk_reward=true、法人共識賣超、放量殺盤

═══ 消息面交叉驗證 ═══

【利多不漲⚠️】搜尋到利多 + stock_ret_20d ≤ 0 + 量縮 → overhyped
【利空不跌✅】搜尋到利空 + trend仍uptrend + 量穩 → 籌碼穩定已消化
【川普衝擊判斷】推文衝擊但技術面未破 → 觀察；技術面已破 + 推文加壓 → 減碼
【擁擠風險】beta>1.5 + high_vol + 找不到逆風觀點 → 連鎖停損風險高
【戰略彈性】技術弱但供應鏈拉貨實際數據支撐 → 可短轉長；weekly也轉空+消息面無支撐 → 出場

═══ 輸出格式 ═══

每檔持股：
### [代號] — 🌸鮮花 / ⚠️邊緣 / 🪓雜草
- 趨勢：日線/週線/MTF/Supertrend
- 鮮花 vs 雜草計分
- 風險：RR、停損、目標、Beta、回撤
- 籌碼：外資/投信 20d、法人共識
- 📰 消息面：驅動力🟢🟡🔴、利多出盡、技術vs消息一致性、逆風觀點
- 🇺🇸 川普影響：[無/間接/直接] — [說明]
- 決策表：基本情境/轉強/惡化/系統性風險/川普升級 各情境的動作與觸發條件
- 反向審計：續抱最大風險？賣出可能錯過什麼？

持股體質總表（含消息面與川普影響欄位）

整體組合建議：
- 鮮花/雜草比例
- 川普政策風險曝險度（幾檔持股在衝擊範圍內）
- 系統性風險等級對應的行動
- 現金部位建議（🔴時 ≥ 25%）

═══ 重要規則 ═══
1. 技術面全面轉空不要續抱
2. 戰略彈性只限週線仍完好的持股
3. 每個續抱必附停損價位
4. 雜草超過 50% → 「組合需大幅調整」
5. ETF 側重趨勢 + 總經判斷
6. VIX>40 + CTA大逃殺 → 不恐慌賣出基本面無虞的持股（阿呆谷）
7. 但擁擠高槓桿股仍建議先砍部分部位（風控成本）
8. 川普推文恐慌 ≠ 基本面崩壞 → 區分口頭威脅與實質政策
9. 社群農場文不算分析依據
10. 消息必須標明來源

═══ 我的持股資料如下 ═══
【貼上 sector_持股 JSON】"""
        st.code(prompt_c, language="text")
# -------------------------
# Main Content
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Stock Analysis", "Sector Analysis", "Custom Sectors", "Holdings"])

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
                    # Clear stale selectbox key to prevent KeyError on rerun
                    if "mgmt_select" in st.session_state:
                        del st.session_state["mgmt_select"]
                    st.rerun()

# --- Tab 4: Holdings ---
with tab4:
    st.header("📊 Holdings Management")
    st.caption("管理你的持股清單（股票代碼、成交均價、股數），方便搭配分析器使用")

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
                h_shares = st.number_input("股數（張）", min_value=0, step=1, key="h_shares")

            if st.button("➕ 新增", key="h_add_btn", use_container_width=True):
                h_val = h_stock.strip().upper()
                if not h_val:
                    st.error("請輸入股票代碼")
                elif h_avg_price <= 0:
                    st.error("成交均價須 > 0")
                elif h_shares <= 0:
                    st.error("股數須 > 0")
                else:
                    # Check duplicate
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
                # Build holdings as custom sector for quick analysis
                holdings_tickers = [h["stock_id"] for h in st.session_state.holdings]
                st.markdown(f"**持股清單**：`{', '.join(holdings_tickers)}`")
                if st.button("🔍 用分析器跑持股報告", key="h_run_sector", use_container_width=True):
                    # Temporarily inject holdings as a custom sector and run
                    st.session_state.custom_sectors["持股"] = holdings_tickers
                    save_sectors_to_db(st.session_state.custom_sectors)
                    st.success("已建立「持股」族群，請切到 Sector Analysis → Custom → 持股 執行分析")
                    st.rerun()
            else:
                st.info("尚無持股資料")

    # Holdings table display
    st.divider()
    if st.session_state.holdings:
        st.subheader("目前持股")

        # Table header
        hdr1, hdr2, hdr3, hdr4, hdr5 = st.columns([2, 2, 2, 2, 1])
        hdr1.markdown("**股票代碼**")
        hdr2.markdown("**成交均價**")
        hdr3.markdown("**股數（張）**")
        hdr4.markdown("**成本金額**")
        hdr5.markdown("**操作**")

        total_cost = 0.0
        indices_to_delete = []

        for i, h in enumerate(st.session_state.holdings):
            cost = h["avg_price"] * h["shares"] * 1000  # 1張 = 1000股
            total_cost += cost
            r1, r2, r3, r4, r5 = st.columns([2, 2, 2, 2, 1])
            r1.text(h["stock_id"])
            r2.text(f"${h['avg_price']:,.2f}")
            r3.text(f"{h['shares']}")
            r4.text(f"${cost:,.0f}")
            with r5:
                if st.button("🗑️", key=f"h_del_{i}_{h['stock_id']}"):
                    indices_to_delete.append(i)

        # Process deletions
        if indices_to_delete:
            for idx in sorted(indices_to_delete, reverse=True):
                removed = st.session_state.holdings.pop(idx)
            save_holdings_to_db(st.session_state.holdings)
            st.rerun()

        st.divider()
        st.markdown(f"**持股總成本**：`${total_cost:,.0f}`　|　**持股檔數**：`{len(st.session_state.holdings)}`")

        # Clear all
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


