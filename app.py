import streamlit as st
import subprocess
import sys
import time

# --- 分析師除錯模組 (Auto-Fix) ---
# 這是針對 Streamlit Cloud 的強制修復方案
# 如果系統找不到 yfinance，程式會自動執行 pip install
try:
    import yfinance as yf
except ModuleNotFoundError:
    placeholder = st.empty()
    placeholder.warning("⚠️ 偵測到環境缺少 yfinance 套件，正在嘗試自動安裝... (這可能需要幾秒鐘)")
    try:
        # 使用 subprocess 強制安裝
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
        import yfinance as yf

        placeholder.success("✅ yfinance 安裝成功！正在載入分析工具...")
        time.sleep(1)
        placeholder.empty()  # 清除訊息
    except Exception as e:
        st.error(f"❌ 自動安裝失敗。請確認 requirements.txt 位於專案根目錄。\n詳細錯誤: {e}")
        st.stop()

import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 設定頁面配置 (展現專業素養) ---
st.set_page_config(
    page_title="專業股票分析儀表板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 側邊欄：參數設定 (第二層：量化分析技能) ---
st.sidebar.title("🔍 分析參數設定")

# 預設為台積電 (2330.TW)，但也支援美股 (如 AAPL)
default_ticker = "2330.TW"
ticker_input = st.sidebar.text_input("請輸入股票代號 (例如: 2330.TW, AAPL, NVDA)", value=default_ticker)
ticker_symbol = ticker_input.upper()

# 時間區間選擇
start_date = st.sidebar.date_input("開始日期", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("結束日期", datetime.now())

st.sidebar.markdown("---")
st.sidebar.info(
    "**分析師筆記：**\n\n"
    "此工具結合量化數據與基本面資訊，"
    "協助您進行杜邦分析與趨勢判斷。"
)


# --- 核心邏輯 ---
def get_stock_data(symbol, start, end):
    try:
        stock = yf.Ticker(symbol)
        # 取得歷史股價
        df_history = stock.history(start=start, end=end)
        # 取得基本資料
        info = stock.info
        return stock, df_history, info
    except Exception as e:
        st.error(f"無法取得數據，請確認代號是否正確。錯誤訊息: {e}")
        return None, None, None


stock, df_history, info = get_stock_data(ticker_symbol, start_date, end_date)

if stock is not None and not df_history.empty:

    # --- 標題區與即時數據 ---
    st.title(f"📈 {info.get('longName', ticker_symbol)} ({ticker_symbol}) 深度分析報告")

    current_price = df_history['Close'].iloc[-1]
    prev_price = df_history['Close'].iloc[-2]
    delta = current_price - prev_price
    delta_percent = (delta / prev_price) * 100

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("最新收盤價", f"{current_price:.2f}", f"{delta:.2f} ({delta_percent:.2f}%)")
    with col2:
        st.metric("本益比 (PE)", f"{info.get('trailingPE', 'N/A')}")
    with col3:
        st.metric("股價淨值比 (PB)", f"{info.get('priceToBook', 'N/A')}")
    with col4:
        # 第一層：財務會計 - 股東權益報酬率
        st.metric("股東權益報酬率 (ROE)",
                  f"{info.get('returnOnEquity', 0) * 100:.2f}%" if info.get('returnOnEquity') else "N/A")

    # --- 分析模組 ---
    tab1, tab2, tab3 = st.tabs(["📊 技術趨勢與量價", "📑 財務報表分析 (Level 1)", "🏢 公司基本面與評價"])

    with tab1:
        st.subheader("價格走勢與移動平均線 (MA)")

        # 計算移動平均線 (MA)
        df_history['MA20'] = df_history['Close'].rolling(window=20).mean()
        df_history['MA60'] = df_history['Close'].rolling(window=60).mean()

        fig = go.Figure()

        # K線圖
        fig.add_trace(go.Candlestick(
            x=df_history.index,
            open=df_history['Open'],
            high=df_history['High'],
            low=df_history['Low'],
            close=df_history['Close'],
            name='K線'
        ))

        # MA線
        fig.add_trace(go.Scatter(x=df_history.index, y=df_history['MA20'], line=dict(color='orange', width=1),
                                 name='月線 (20MA)'))
        fig.add_trace(
            go.Scatter(x=df_history.index, y=df_history['MA60'], line=dict(color='blue', width=1), name='季線 (60MA)'))

        fig.update_layout(
            height=600,
            xaxis_rangeslider_visible=False,
            title_text=f"{ticker_symbol} 股價走勢圖",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**分析師觀點：** 觀察月線與季線的黃金交叉或死亡交叉，配合成交量變化，可判斷短期與中期的多空力道。")

    with tab2:
        st.subheader("三大財務報表核心數據 (Financial Accounting)")
        st.markdown("透過損益表與資產負債表，我們可以識別財報中的警訊 (Red Flags)。")

        # 取得財務數據
        financials = stock.financials
        balance_sheet = stock.balance_sheet
        cashflow = stock.cashflow

        col_fin1, col_fin2 = st.columns(2)

        with col_fin1:
            st.markdown("### 損益表摘要 (Income Statement)")
            if not financials.empty:
                # 顯示最近幾期的營收與獲利
                try:
                    display_fin = financials.loc[['Total Revenue', 'Net Income', 'Operating Income']].transpose()
                    # 格式化數字
                    st.dataframe(display_fin.style.format("{:,.0f}"))
                except KeyError:
                    st.dataframe(financials)
            else:
                st.warning("無法取得詳細損益表數據")

        with col_fin2:
            st.markdown("### 現金流量表摘要 (Cash Flow)")
            if not cashflow.empty:
                try:
                    # 重點觀察營業現金流
                    display_cf = cashflow.loc[['Operating Cash Flow',
                                               'Free Cash Flow']].transpose() if 'Free Cash Flow' in cashflow.index else \
                    cashflow.loc[['Operating Cash Flow']].transpose()
                    st.dataframe(display_cf.style.format("{:,.0f}"))

                    st.info(
                        "💡 **財報品質辨識**：若「淨利」為正，但「營業現金流」長期為負，可能代表獲利品質不佳，需注意應收帳款過高的風險。")
                except KeyError:
                    st.dataframe(cashflow.head())
            else:
                st.warning("無法取得詳細現金流數據")

    with tab3:
        col_b1, col_b2 = st.columns([1, 2])

        with col_b1:
            st.subheader("公司概況")
            st.markdown(f"**產業:** {info.get('industry', 'N/A')}")
            st.markdown(f"**板塊:** {info.get('sector', 'N/A')}")
            st.markdown(f"**員工數:** {info.get('fullTimeEmployees', 'N/A')}")
            st.markdown(f"**說明:**")
            st.caption(info.get('longBusinessSummary', '無詳細說明'))

        with col_b2:
            st.subheader("評價模型指標 (Valuation)")

            val_data = {
                "指標": ["市值 (Market Cap)", "本益比 (Trailing PE)", "預估本益比 (Forward PE)", "PEG Ratio",
                         "企業價值/EBITDA"],
                "數值": [
                    f"{info.get('marketCap', 0):,.0f}",
                    f"{info.get('trailingPE', 'N/A')}",
                    f"{info.get('forwardPE', 'N/A')}",
                    f"{info.get('pegRatio', 'N/A')}",
                    f"{info.get('enterpriseToEbitda', 'N/A')}"
                ]
            }
            val_df = pd.DataFrame(val_data)
            st.table(val_df)

            st.markdown("---")
            st.markdown("**分析師觀點 - PEG 指標解讀：**")
            st.markdown("- **PEG < 1**: 股價可能被低估（相對於成長性）。")
            st.markdown("- **PEG ≈ 1**: 估值合理。")
            st.markdown("- **PEG > 1**: 股價可能被高估，或市場給予極高的成長溢價。")

else:
    st.warning("請在左側輸入有效的股票代號以開始分析。")

# --- 頁尾：專業聲明 ---
st.markdown("---")
st.caption("免責聲明：本工具僅供量化分析輔助，不構成投資建議。投資人應運用批判性思考 (Level 2) 並獨立判斷 (Level 3)。")