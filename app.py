import io
from contextlib import redirect_stdout

import streamlit as st

# 直接用你現有的函式
from stock import analyze_stock_technical

st.set_page_config(page_title="股票分析器", layout="wide")

st.title("股票分析器（Streamlit 版）")

stock_id = st.text_input("輸入股票代號（例如 2330 / 2330.TW / 6223.TWO）", value="2330")

run = st.button("開始分析")

if run:
    if not stock_id.strip():
        st.warning("請輸入股票代號")
    else:
        buf = io.StringIO()
        with redirect_stdout(buf):
            analyze_stock_technical(stock_id.strip())
        report = buf.getvalue()

        if report.strip():
            st.code(report, language="text")
        else:
            st.error("沒有產生輸出（可能代號無資料或抓取失敗）")
