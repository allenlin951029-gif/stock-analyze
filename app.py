import streamlit as st
from stock import analyze_stock_technical

st.set_page_config(page_title="Stock Analyze", layout="wide")
st.title("Stock Analyze")

stock_id = st.text_input("輸入股票代號（例：0050 / 2330 / 6223）", value="0050")
run = st.button("開始分析")

if run:
    sid = stock_id.strip()
    if not sid:
        st.warning("請輸入股票代號")
    else:
        with st.spinner(f"正在分析 {sid}..."):
            report = analyze_stock_technical(sid)

        if not report or str(report).strip() == "":
            st.error("沒有產生輸出（可能代號無資料或抓取失敗）")
        else:
            st.code(report, language="text")
