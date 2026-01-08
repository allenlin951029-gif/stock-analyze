# -*- coding: utf-8 -*-
import sys
import subprocess
import importlib
import random
import time
import io
from datetime import datetime, timedelta

# --- 0. 自動安裝套件功能 (Auto-Install) ---
def install_and_import(package_name, import_name=None):
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
    except ImportError:
        print(f"📦 偵測到缺少 {package_name}，正在為您自動安裝中...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} 安裝成功！")
        except Exception as e:
            print(f"❌ 安裝失敗: {e}")
            sys.exit(1)

# --- 1. 正式匯入套件 ---
import truststore
truststore.inject_into_ssl()

import yfinance as yf
import pandas as pd
import numpy as np
import requests


# --- 數據清理與計算函式 ---
def clean_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0)
        lv1 = df.columns.get_level_values(1)
        if "Close" in set(lv0):
            df.columns = lv0
        elif "Close" in set(lv1):
            df.columns = lv1
        else:
            df.columns = df.columns.droplevel(1)
    return df


def get_k_status(open_p, close_p):
    if close_p > open_p:
        return "🔴 紅棒"
    elif close_p < open_p:
        return "🟢 綠棒"
    else:
        return "➖ 十字"


# --- 新增：K棒型態（當天） ---
def describe_candle(open_p, high_p, low_p, close_p):
    """
    回傳較具體的K棒型態（簡化常用型態）
    - 長紅/長黑
    - 十字星/蜻蜓十字/墓碑十字
    - 錘頭/吊人
    - 倒錘/流星
    - 小紅/小黑 + 影線特徵
    """
    o, h, l, c = float(open_p), float(high_p), float(low_p), float(close_p)
    rng = h - l
    if rng <= 0:
        return "一字線"

    body = abs(c - o)
    upper = h - max(o, c)
    lower = min(o, c) - l

    body_r = body / rng
    upper_r = upper / rng
    lower_r = lower / rng

    bullish = c > o

    # 十字系（本體極小）
    if body_r <= 0.10:
        if upper_r >= 0.65 and lower_r <= 0.15:
            return "墓碑十字"
        if lower_r >= 0.65 and upper_r <= 0.15:
            return "蜻蜓十字"
        return "十字星"

    # 長實體
    if body_r >= 0.60:
        return "長紅K" if bullish else "長黑K"

    # 下影長：錘頭/吊人
    if lower_r >= 0.60 and body_r <= 0.30 and upper_r <= 0.20:
        return "錘頭線" if bullish else "吊人線"

    # 上影長：倒錘/流星
    if upper_r >= 0.60 and body_r <= 0.30 and lower_r <= 0.20:
        return "倒錘線" if bullish else "流星線"

    base = "小紅K" if bullish else "小黑K"
    if upper_r >= 0.35 and lower_r >= 0.35:
        return base + "（上下影明顯）"
    if upper_r >= 0.45:
        return base + "（上影偏長）"
    if lower_r >= 0.45:
        return base + "（下影偏長）"
    return base


# --- 官方來源抓三大法人（取代 Yahoo 爬蟲）---
def _safe_int(x):
    try:
        s = str(x).strip().replace(",", "")
        if s in ("", "--", "—", "NaN", "nan", "None"):
            return None
        return int(float(s))
    except Exception:
        return None


def _find_net_col(cols, keyword, exclude=None):
    """
    找同時含 keyword 與 買賣超 的欄位；可排除關鍵字（避免誤抓「外資自營商」）
    """
    exclude = exclude or []
    cands = []
    for c in cols:
        s = str(c)
        if (keyword in s) and ("買賣超" in s) and not any(e in s for e in exclude):
            cands.append(c)
    return cands[0] if cands else None


def _parse_twse_t86_csv(text):
    text = text.replace("\r", "").replace("=", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]

    start = None
    for i, ln in enumerate(lines):
        if ("證券代號" in ln) and ("證券名稱" in ln):
            start = i
            break
    if start is None:
        return None

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("說明") or lines[j].startswith("備註"):
            end = j
            break

    csv_text = "\n".join(lines[start:end])
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return None


def _parse_tpex_csv(text):
    text = text.replace("\ufeff", "").replace("\r", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]

    start = None
    for i, ln in enumerate(lines):
        if ("代號" in ln) and ("名稱" in ln):
            start = i
            break
    if start is None:
        return None

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if lines[j].startswith("說明") or lines[j].startswith("備註"):
            end = j
            break

    csv_text = "\n".join(lines[start:end])
    try:
        return pd.read_csv(io.StringIO(csv_text))
    except Exception:
        return None


def get_institutional_data(stock_id, trade_date=None, market_hint=None):
    """
    用 TWSE / TPEx 官方 CSV 取三大法人買賣超（較 Yahoo 股市頁面穩定很多）
    trade_date：建議傳入「最後交易日」(date)，避免假日/休市無資料
    market_hint：傳入 'xxxx.TW' 或 'xxxx.TWO'，用來決定先試哪個市場
    """
    stock_no = stock_id.strip().upper().replace(".TW", "").replace(".TWO", "")
    if trade_date is None:
        trade_date = datetime.now().date()

    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept-Language": "zh-TW,zh;q=0.9,en-US;q=0.8,en;q=0.7",
    }

    prefer_twse = True
    if isinstance(market_hint, str) and market_hint.upper().endswith(".TWO"):
        prefer_twse = False

    last_error = None

    def try_twse(d_):
        url = f"https://www.twse.com.tw/fund/T86?response=csv&date={d_.strftime('%Y%m%d')}&selectType=ALLBUT0999"
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200 or len(r.text) < 200:
            return None, f"TWSE HTTP {r.status_code}"
        if "沒有符合條件的資料" in r.text or "很抱歉" in r.text:
            return None, "TWSE 無資料(可能休市)"
        df = _parse_twse_t86_
