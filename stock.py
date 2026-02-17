import io
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import yfinance as yf

try:
    import truststore

    truststore.inject_into_ssl()
except Exception:
    pass


# =========================
# Utils
# =========================
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


def _safe_int(x):
    try:
        s = str(x).strip().replace(",", "")
        if s in ("", "--", "—", "NaN", "nan", "None"):
            return None
        return int(float(s))
    except Exception:
        return None


def _resolve_yf_ticker(stock_id: str) -> str:
    s = stock_id.strip().upper()
    if any(s.endswith(sfx) for sfx in [".TW", ".TWO", ".KS", ".T", ".HK"]):
        return s
    if len(s) > 0 and s[0].isdigit():
        return f"{s}.TW"
    return s


def _strip_suffix(stock_id: str) -> str:
    return (stock_id.strip().upper()
            .replace(".TW", "").replace(".TWO", "")
            .replace(".KS", "").replace(".T", "").replace(".HK", ""))


def _norm_col(s: str) -> str:
    s = str(s).replace("\ufeff", "").replace("\u3000", "")
    s = s.replace("（", "(").replace("）", ")")
    return re.sub(r"\s+", "", s)


def _ensure_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        if getattr(df.index, "tz", None) is not None:
            df = df.copy()
            df.index = df.index.tz_localize(None)
    except Exception:
        pass
    return df


def _nearest_trading_ts(df: pd.DataFrame, target_date):
    if df is None or df.empty:
        return None
    if target_date is None:
        target_date = datetime.now().date()
    target = pd.Timestamp(target_date)
    idx = df.index
    try:
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
    except Exception:
        pass
    pos = idx.searchsorted(target + pd.Timedelta(days=1)) - 1
    return None if pos < 0 else idx[pos]


def _parse_roc_date(date_str: str) -> str:
    """將民國年字串 '114/05/13' 轉為西元年 '2025-05-13'"""
    try:
        parts = str(date_str).split('/')
        if len(parts) == 3:
            year = int(parts[0]) + 1911
            return f"{year}-{parts[1]}-{parts[2]}"
        return str(date_str)
    except Exception:
        return str(date_str)


# ── AI 分析用工具函數 ──
def _fmt_series(series, n=5, fmt=".2f"):
    """回傳最近 n 筆數值清單，讓 AI 看趨勢方向"""
    vals = series.dropna().iloc[-n:]
    return "[" + ", ".join(f"{v:{fmt}}" for v in vals) + "]"


def _percentile_rank(series, window=252):
    """最新值在過去 window 天的歷史百分位（0~100）"""
    s = series.dropna()
    if len(s) < 10:
        return None
    current = s.iloc[-1]
    hist = s.iloc[-window:]
    return round((hist < current).sum() / len(hist) * 100, 1)


def _trend_label(vals):
    """給一串數值，判斷趨勢方向文字"""
    if len(vals) < 2:
        return "資料不足"
    diffs = [vals[i] - vals[i - 1] for i in range(1, len(vals))]
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    if pos >= len(diffs) * 0.8: return "持續上升"
    if neg >= len(diffs) * 0.8: return "持續下降"
    return "偏向上升" if pos > neg else ("偏向下降" if neg > pos else "橫盤震盪")


# =========================
# Candle
# =========================
def get_k_status(open_p, close_p, high_p=None, low_p=None):
    try:
        o, c = float(open_p), float(close_p)
        if high_p is not None and low_p is not None:
            rng = float(high_p) - float(low_p)
            if rng > 0 and abs(c - o) / rng <= 0.10:
                return "➖ 十字"
        return "🔴 紅棒" if c > o else ("🟢 綠棒" if c < o else "➖ 十字")
    except Exception:
        return "➖"


def describe_candle(open_p, high_p, low_p, close_p):
    o, h, l, c = float(open_p), float(high_p), float(low_p), float(close_p)
    rng = h - l
    if rng <= 0:
        return "一字線"
    body = abs(c - o)
    upper, lower = h - max(o, c), min(o, c) - l
    body_r, upper_r, lower_r = body / rng, upper / rng, lower / rng
    bullish = c > o
    if body_r <= 0.10:
        if upper_r >= 0.65 and lower_r <= 0.15: return "墓碑十字（看空）"
        if lower_r >= 0.65 and upper_r <= 0.15: return "蜻蜓十字（看多）"
        return "十字星（多空均衡）"
    if body_r >= 0.60:
        return "長紅K（強勢上漲）" if bullish else "長黑K（強勢下跌）"
    if lower_r >= 0.60 and body_r <= 0.30 and upper_r <= 0.20:
        return "錘頭線（底部反轉訊號）" if bullish else "吊人線（高位風險）"
    if upper_r >= 0.60 and body_r <= 0.30 and lower_r <= 0.20:
        return "倒錘線（底部反轉訊號）" if bullish else "流星線（頂部反轉訊號）"
    base = "小紅K" if bullish else "小黑K"
    if upper_r >= 0.35 and lower_r >= 0.35: return base + "（上下影明顯，多空拉鋸）"
    if upper_r >= 0.45: return base + "（上影偏長，上方賣壓）"
    if lower_r >= 0.45: return base + "（下影偏長，下方支撐）"
    return base


# =========================
# Institutional (三大法人)
# =========================
def _parse_twse_t86_csv(text: str):
    text = text.replace("\r", "").replace("=", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    start = next((i for i, ln in enumerate(lines) if "證券代號" in ln and "證券名稱" in ln), None)
    if start is None: return None
    end = next((j for j in range(start + 1, len(lines))
                if lines[j].startswith("說明") or lines[j].startswith("備註")), len(lines))
    try:
        return pd.read_csv(io.StringIO("\n".join(lines[start:end])))
    except Exception:
        return None


def _parse_tpex_csv(text: str):
    text = text.replace("\ufeff", "").replace("\r", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    start = next((i for i, ln in enumerate(lines) if "代號" in ln and "名稱" in ln), None)
    if start is None: return None
    end = next((j for j in range(start + 1, len(lines))
                if lines[j].startswith("說明") or lines[j].startswith("備註")), len(lines))
    try:
        return pd.read_csv(io.StringIO("\n".join(lines[start:end])))
    except Exception:
        return None


def get_institutional_data(stock_id: str, trade_date, market_hint=None, max_back: int = 10):
    stock_no = _strip_suffix(stock_id)
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "zh-TW,zh;q=0.9"}
    prefer_twse = not (isinstance(market_hint, str) and market_hint.upper().endswith(".TWO"))
    last_error = None

    def try_twse(d_):
        url = f"https://www.twse.com.tw/fund/T86?response=csv&date={d_.strftime('%Y%m%d')}&selectType=ALLBUT0999"
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code != 200 or len(r.text) < 200:
            return None, f"TWSE HTTP {r.status_code}"
        if "沒有符合條件的資料" in r.text or "很抱歉" in r.text:
            return None, "TWSE 無資料(休市)"
        df = _parse_twse_t86_csv(r.text)
        return (df, None) if df is not None else (None, "TWSE 解析失敗")

    def try_tpex(d_):
        roc_year = d_.year - 1911
        roc_date = f"{roc_year:03d}/{d_.month:02d}/{d_.day:02d}"
        url = (
            "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge_result.php"
            f"?l=zh-tw&o=csv&se=EW&t=D&d={roc_date}&s=0,asc"
        )
        h2 = {**headers, "Referer": "https://www.tpex.org.tw/web/stock/3insti/daily_trade/3itrade_hedge.php"}
        r = requests.get(url, headers=h2, timeout=15)
        if r.status_code != 200 or len(r.text) < 200:
            return None, f"TPEx HTTP {r.status_code}"
        if "沒有符合條件的資料" in r.text or "很抱歉" in r.text:
            return None, "TPEx 無資料(休市)"
        df = _parse_tpex_csv(r.text)
        return (df, None) if df is not None else (None, "TPEx 解析失敗")

    markets = [("TWSE", try_twse), ("TPEx", try_tpex)]
    if not prefer_twse:
        markets = [("TPEx", try_tpex), ("TWSE", try_twse)]

    def find_col(colmap, pred):
        return next((orig for nk, orig in colmap.items() if pred(nk)), None)

    def is_foreign(nk):
        if "外資自營商" in nk and "不含外資自營商" not in nk: return False
        return ("外陸資" in nk) or ("外資及陸資" in nk) or ("外資" in nk)

    for back in range(max_back + 1):
        d = trade_date - timedelta(days=back)
        for mkt_name, fn in markets:
            df, err = fn(d)
            if df is None:
                last_error = err
                continue
            cols = list(df.columns)
            colmap = {_norm_col(c): c for c in cols}
            code_col = next((c for c in cols if _norm_col(c) in ("證券代號", "代號")), None)
            if code_col is None:
                last_error = f"{mkt_name} 找不到代號欄"
                continue
            row = df[df[code_col].astype(str).str.strip() == stock_no]
            if row.empty:
                last_error = f"{mkt_name} 找不到 {stock_no}"
                continue

            foreign_col = find_col(colmap, lambda nk: is_foreign(nk) and "買賣超" in nk)
            trust_col = find_col(colmap, lambda nk: "投信" in nk and "買賣超" in nk)
            dealer_total = find_col(colmap, lambda nk: "自營商" in nk and "買賣超" in nk
                                                       and "外資" not in nk and "自行買賣" not in nk and "避險" not in nk)
            dealer_self = find_col(colmap, lambda nk: "自營商" in nk and "自行買賣" in nk
                                                      and "買賣超" in nk and "外資" not in nk)
            dealer_hedge = find_col(colmap, lambda nk: "自營商" in nk and "避險" in nk
                                                       and "買賣超" in nk and "外資" not in nk)

            foreign = _safe_int(row.iloc[0][foreign_col]) if foreign_col else None
            trust = _safe_int(row.iloc[0][trust_col]) if trust_col else None
            dealer = None
            if dealer_total:
                dealer = _safe_int(row.iloc[0][dealer_total])
            elif dealer_self or dealer_hedge:
                a = _safe_int(row.iloc[0][dealer_self]) if dealer_self else 0
                b = _safe_int(row.iloc[0][dealer_hedge]) if dealer_hedge else 0
                dealer = (a or 0) + (b or 0)

            if foreign is None:
                buy_c = find_col(colmap, lambda nk: is_foreign(nk)
                                                    and ("買進" in nk or "買入" in nk) and "買賣超" not in nk)
                sell_c = find_col(colmap, lambda nk: is_foreign(nk)
                                                     and "賣出" in nk and "買賣超" not in nk)
                bv = _safe_int(row.iloc[0][buy_c]) if buy_c else None
                sv = _safe_int(row.iloc[0][sell_c]) if sell_c else None
                if bv is not None and sv is not None:
                    foreign = bv - sv

            yf_suffix = ".TW" if mkt_name == "TWSE" else ".TWO"
            return {"id": f"{stock_no}{yf_suffix}", "date": d.strftime("%Y-%m-%d"),
                    "foreign": foreign, "trust": trust, "dealer": dealer, "error": None}

    return {"error": last_error or "未知錯誤", "id": f"{stock_no}.TW"}


def get_institutional_multi_days(stock_id: str, end_date, market_hint=None, days=5):
    """抓近 N 個交易日三大法人，回傳 list（最舊在前）"""
    results, d, attempts = [], end_date, 0
    while len(results) < days and attempts < days * 4:
        attempts += 1
        rec = get_institutional_data(stock_id, d, market_hint=market_hint, max_back=0)
        if rec.get("error") is None:
            results.insert(0, rec)
            d = datetime.strptime(rec["date"], "%Y-%m-%d").date() - timedelta(days=1)
        else:
            d = d - timedelta(days=1)
    return results


def get_foreign_holding_ratio(stock_no: str) -> dict:
    """從 TWSE 抓外資持股比率（僅支援上市股票）
    修正：RWD API 回傳 tables 格式，需用欄位名稱定位持股比率欄，
          不能用 last[-1]（可能是日期或其他欄位）。
    """
    headers = {"User-Agent": "Mozilla/5.0"}
    urls = [
        f"https://www.twse.com.tw/rwd/zh/fund/MI_QFIIS?response=json&stockNo={stock_no}&queryType=1",
        f"https://www.twse.com.tw/fund/MI_QFIIS?response=json&stockNo={stock_no}&queryType=1",
    ]
    last_err = None
    for url in urls:
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}"
                continue
            j = r.json()

            # ── 方式 A：用 fields + data 找到欄位名稱 ──
            fields, data = None, None

            # RWD tables 格式
            if isinstance(j.get("tables"), list):
                for tbl in j["tables"]:
                    f = tbl.get("fields", [])
                    d = tbl.get("data", [])
                    if f and d:
                        # 選有「持股比」或「比率」或「比例」的表
                        joined = "".join(str(x) for x in f)
                        if any(kw in joined for kw in ("持股比", "比率", "比例")):
                            fields, data = f, d
                            break
                        # 備用：選有「外資」的表
                        if any("外資" in str(x) for x in f) and fields is None:
                            fields, data = f, d

            # 傳統 root data 格式
            if fields is None and j.get("fields") and j.get("data"):
                fields, data = j["fields"], j["data"]

            # 無欄位定義 → fallback 到 root data（舊邏輯）
            if fields is None and isinstance(j.get("data"), list) and j["data"]:
                data = j["data"]

            if not data:
                last_err = "無資料"
                continue

            last = data[-1]
            if not last:
                last_err = "空資料行"
                continue

            # ── 取日期（第一欄通常是民國日期）──
            date_str = _parse_roc_date(str(last[0]))

            # ── 用欄位名稱找持股比率 ──
            ratio = None
            if fields:
                norm_fields = [_norm_col(f) for f in fields]
                ratio_idx = None
                for i, nf in enumerate(norm_fields):
                    if any(kw in nf for kw in ("持股比", "比率", "比例", "Percentage")):
                        ratio_idx = i
                        break
                if ratio_idx is not None and ratio_idx < len(last):
                    raw = str(last[ratio_idx]).replace("%", "").replace(",", "").strip()
                    if raw and re.match(r"^-?\d+(\.\d+)?$", raw):
                        ratio = float(raw)
            else:
                # 沒有 fields 定義 → 逐欄嘗試找像比率的數值（0~100 之間的小數）
                for val in reversed(last):
                    raw = str(val).replace("%", "").replace(",", "").strip()
                    if raw and re.match(r"^-?\d+(\.\d+)?$", raw):
                        v = float(raw)
                        if 0 < v < 100:  # 持股比率通常在 0~100
                            ratio = v
                            break

            if ratio is not None:
                return {"ratio": ratio, "date": date_str, "error": None}
            else:
                last_err = f"找不到持股比率欄（fields={fields}, last_row={last}）"
                continue

        except Exception as e:
            last_err = str(e)
            continue

    return {"ratio": None, "date": None, "error": last_err or "未知錯誤"}


def _consecutive_direction(daily_list, key):
    """計算連續幾日同方向，回傳 (天數, 方向文字)"""
    vals = [rec.get(key) for rec in reversed(daily_list) if rec.get(key) is not None]
    if not vals:
        return 0, "n/a"
    direction = "買超" if vals[0] > 0 else ("賣超" if vals[0] < 0 else "持平")
    count = next((i for i, v in enumerate(vals)
                  if not ((direction == "買超" and v > 0) or
                          (direction == "賣超" and v < 0) or
                          (direction == "持平" and v == 0))), len(vals))
    return count, direction


# =========================
# Margin Trading (融資融券)
# =========================
def _parse_twse_json_table(obj):
    if not isinstance(obj, dict): return None

    # 1. 嘗試直接從 root 抓 (exchangeReport 格式)
    if "fields" in obj and "data" in obj:
        try:
            return pd.DataFrame(obj["data"], columns=obj["fields"])
        except Exception:
            pass

    # 2. 嘗試從 tables 抓 (RWD 格式)
    if "tables" in obj and isinstance(obj["tables"], list):
        # 優先尋找欄位中明確包含「代號」或「代碼」的表格
        for tbl in obj["tables"]:
            fields = tbl.get("fields", [])
            data = tbl.get("data", [])
            if not fields or not data:
                continue
            if any("代號" in str(f) or "代碼" in str(f) for f in fields):
                try:
                    return pd.DataFrame(data, columns=fields)
                except Exception:
                    continue

        # 備用方案：抓欄位數最多的表（個股明細通常欄位最多）
        best_table = None
        max_cols = 0
        for tbl in obj["tables"]:
            if "fields" in tbl and "data" in tbl:
                if len(tbl["fields"]) > max_cols:
                    max_cols = len(tbl["fields"])
                    best_table = tbl
        if best_table and max_cols > 5:
            try:
                return pd.DataFrame(best_table["data"], columns=best_table["fields"])
            except Exception:
                pass

    return None


def _twse_margin_json(date_yyyymmdd: str, headers: dict):
    urls = [
        f"https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN?response=json&date={date_yyyymmdd}&selectType=ALL",
        f"https://www.twse.com.tw/exchangeReport/MI_MARGN?response=json&date={date_yyyymmdd}&selectType=ALL",
    ]

    h2 = headers.copy()
    h2["Referer"] = "https://www.twse.com.tw/zh/page/trading/exchange/MI_MARGN.html"

    last_err = None
    for url in urls:
        try:
            r = requests.get(url, headers=h2, timeout=15)
            if r.status_code != 200:
                last_err = f"HTTP {r.status_code}";
                continue
            j = r.json()
            df = pd.DataFrame(j) if isinstance(j, list) else _parse_twse_json_table(j)
            if df is not None and not df.empty:
                return df, None
            last_err = "空資料/格式不符"
        except Exception as e:
            last_err = str(e)
    return None, last_err or "取資料失敗"


def _parse_tpex_margin_csv(text: str):
    if not text: return None
    text = text.replace("\ufeff", "").replace("\r", "")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    start = next((i for i, ln in enumerate(lines)
                  if "代號" in ln and "名稱" in ln and ("資" in ln or "融資" in ln)), None)
    if start is None: return None
    end = next((j for j in range(start + 1, len(lines))
                if lines[j].startswith("*****") or lines[j].startswith("說明")
                or lines[j].startswith("備註")), len(lines))
    try:
        return pd.read_csv(io.StringIO("\n".join(lines[start:end])))
    except Exception:
        return None


def get_margin_short_data(stock_id: str, trade_date, market_hint: str = None, max_back: int = 10):
    stock_no = _strip_suffix(stock_id)
    headers = {"User-Agent": "Mozilla/5.0", "Accept-Language": "zh-TW,zh;q=0.9"}
    prefer_twse = not (isinstance(market_hint, str) and market_hint.upper().endswith(".TWO"))
    last_error = None

    def find_col(colmap, contains):
        return next((orig for nk, orig in colmap.items() if all(kw in nk for kw in contains)), None)

    for back in range(max_back + 1):
        d = trade_date - timedelta(days=back)

        if prefer_twse:
            df, err = _twse_margin_json(d.strftime("%Y%m%d"), headers=headers)
            if df is None or df.empty:
                last_error = err;
                continue

            colmap = {_norm_col(c): c for c in df.columns}
            # 修正：先精確匹配，再模糊匹配（含「代號」或「代碼」的欄位）
            code_col = next((v for k, v in colmap.items()
                             if k in ("股票代號", "證券代號", "證券代碼", "Code", "代號")), None)
            if not code_col:
                # 模糊匹配：找欄位名含「代號」或「代碼」的
                code_col = next((v for k, v in colmap.items()
                                 if "代號" in k or "代碼" in k), None)
            if not code_col:
                # 最後手段：檢查第一欄內容是否為股票代號格式
                first_col = df.columns[0] if len(df.columns) > 0 else None
                if first_col:
                    sample = df[first_col].astype(str).str.strip().head(10)
                    if sample.str.match(r"^\d{4,6}[A-Z]?$").any():
                        code_col = first_col
            if not code_col:
                last_error = "TWSE 找不到代號欄";
                continue

            sub = df[df[code_col].astype(str).str.strip().str.strip('"') == stock_no]
            if sub.empty:
                last_error = f"TWSE 找不到 {stock_no}";
                continue

            row = sub.iloc[0]
            m_bal_c = find_col(colmap, ["融資", "餘額"]) or find_col(colmap, ["資", "餘額"])
            m_chg_c = find_col(colmap, ["融資", "增減"]) or find_col(colmap, ["資", "增減"])
            m_lim_c = find_col(colmap, ["融資", "限額"]) or find_col(colmap, ["資", "限額"])
            s_bal_c = find_col(colmap, ["融券", "餘額"]) or find_col(colmap, ["券", "餘額"])
            s_chg_c = find_col(colmap, ["融券", "增減"]) or find_col(colmap, ["券", "增減"])
            m_bal = _safe_int(row[m_bal_c]) if m_bal_c else None
            m_chg = _safe_int(row[m_chg_c]) if m_chg_c else None
            m_lim = _safe_int(row[m_lim_c]) if m_lim_c else None
            s_bal = _safe_int(row[s_bal_c]) if s_bal_c else None
            s_chg = _safe_int(row[s_chg_c]) if s_chg_c else None
            usage = round(m_bal / m_lim * 100, 1) if (m_bal and m_lim and m_lim > 0) else None
            return {"id": f"{stock_no}.TW", "date": d.strftime("%Y-%m-%d"),
                    "margin_balance": m_bal, "margin_change": m_chg,
                    "margin_limit": m_lim, "margin_usage_rate": usage,
                    "short_balance": s_bal, "short_change": s_chg, "error": None}

        # TPEx
        roc_year = d.year - 1911
        roc_date = f"{roc_year:03d}/{d.month:02d}/{d.day:02d}"
        url_csv = (
            "https://www.tpex.org.tw/web/stock/margin_trading/margin_balance/margin_bal_result.php"
            f"?l=zh-tw&d={roc_date}&o=csv&s=0,asc"
        )
        try:
            r = requests.get(url_csv, headers=headers, timeout=15)
            df = _parse_tpex_margin_csv(r.text) if r.status_code == 200 and len(r.text) > 200 else None
            if df is None or df.empty:
                last_error = f"TPEx 無資料";
                continue
            colmap = {_norm_col(c): c for c in df.columns}
            code_col = colmap.get("代號") or colmap.get("股票代號")
            if not code_col:
                last_error = "TPEx 找不到代號欄";
                continue
            sub = df[df[code_col].astype(str).str.strip() == stock_no]
            if sub.empty:
                last_error = f"TPEx 找不到 {stock_no}";
                continue
            row = sub.iloc[0]
            m_bal_c = colmap.get("資餘額(張)") or colmap.get("資餘額") or find_col(colmap, ["資", "餘額"])
            m_chg_c = colmap.get("資餘額增減(張)") or find_col(colmap, ["資", "增減"])
            s_bal_c = colmap.get("券餘額(張)") or colmap.get("券餘額") or find_col(colmap, ["券", "餘額"])
            s_chg_c = colmap.get("券餘額增減(張)") or find_col(colmap, ["券", "增減"])
            return {"id": f"{stock_no}.TWO", "date": d.strftime("%Y-%m-%d"),
                    "margin_balance": _safe_int(row[m_bal_c]) if m_bal_c else None,
                    "margin_change": _safe_int(row[m_chg_c]) if m_chg_c else None,
                    "margin_limit": None, "margin_usage_rate": None,
                    "short_balance": _safe_int(row[s_bal_c]) if s_bal_c else None,
                    "short_change": _safe_int(row[s_chg_c]) if s_chg_c else None,
                    "error": None}
        except Exception as e:
            last_error = f"TPEx 失敗: {e}";
            continue

    return {"error": last_error or "未知錯誤"}


# =========================
# Indicators
# =========================
def calculate_rsi(series, period):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_adx(df, period=14):
    df = df.copy()
    df["H-L"] = df["High"] - df["Low"]
    df["H-PC"] = abs(df["High"] - df["Close"].shift(1))
    df["L-PC"] = abs(df["Low"] - df["Close"].shift(1))
    df["TR"] = df[["H-L", "H-PC", "L-PC"]].max(axis=1)
    df["UpMove"] = df["High"] - df["High"].shift(1)
    df["DownMove"] = df["Low"].shift(1) - df["Low"]
    df["+DM"] = np.where((df["UpMove"] > df["DownMove"]) & (df["UpMove"] > 0), df["UpMove"], 0.0)
    df["-DM"] = np.where((df["DownMove"] > df["UpMove"]) & (df["DownMove"] > 0), df["DownMove"], 0.0)
    alpha = 1 / period
    df["TR_s"] = df["TR"].ewm(alpha=alpha, adjust=False).mean()
    df["+DM_s"] = df["+DM"].ewm(alpha=alpha, adjust=False).mean()
    df["-DM_s"] = df["-DM"].ewm(alpha=alpha, adjust=False).mean()
    df["+DI"] = 100 * df["+DM_s"] / df["TR_s"].replace(0, np.nan)
    df["-DI"] = 100 * df["-DM_s"] / df["TR_s"].replace(0, np.nan)
    df["DX"] = 100 * abs(df["+DI"] - df["-DI"]) / (df["+DI"] + df["-DI"]).replace(0, np.nan)
    df["ADX"] = df["DX"].ewm(alpha=alpha, adjust=False).mean()
    return df["ADX"], df["+DI"], df["-DI"]


def calculate_bbw(df, period=20, std_dev=2):
    ma = df["Close"].rolling(window=period).mean()
    std = df["Close"].rolling(window=period).std()
    upper = ma + std * std_dev
    lower = ma - std * std_dev
    return (upper - lower) / ma * 100, upper, lower


def calculate_mfi(df, period=14):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    rmf = tp * df["Volume"]
    pos = pd.Series(np.where(tp > tp.shift(1), rmf, 0.0), index=df.index).rolling(period).sum()
    neg = pd.Series(np.where(tp < tp.shift(1), rmf, 0.0), index=df.index).rolling(period).sum()
    return 100 - (100 / (1 + pos / neg.abs().replace(0, np.nan)))


def calculate_avwap(df, anchor_date):
    mask = df.index >= anchor_date
    if not mask.any(): return None
    sub = df.loc[mask].copy()
    tp = (sub["High"] + sub["Low"] + sub["Close"]) / 3
    return (tp * sub["Volume"]).cumsum() / sub["Volume"].cumsum().replace(0, np.nan)


def calculate_obv(df):
    """On Balance Volume — 累計量能方向"""
    return (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()


def calculate_supertrend(df, period=10, multiplier=3.0):
    """SuperTrend，回傳 (line, direction)；direction +1=多頭，-1=空頭"""
    df = df.copy()
    df["TR"] = pd.concat([
        df["High"] - df["Low"],
        abs(df["High"] - df["Close"].shift(1)),
        abs(df["Low"] - df["Close"].shift(1))
    ], axis=1).max(axis=1)
    df["ATR"] = df["TR"].ewm(span=period, adjust=False).mean()
    hl2 = (df["High"] + df["Low"]) / 2
    upper_band = (hl2 + multiplier * df["ATR"]).copy()
    lower_band = (hl2 - multiplier * df["ATR"]).copy()
    supertrend = pd.Series(np.nan, index=df.index)
    direction = pd.Series(0, index=df.index)

    for i in range(1, len(df)):
        ub_p = upper_band.iloc[i - 1]
        lb_p = lower_band.iloc[i - 1]
        c_p = df["Close"].iloc[i - 1]
        upper_band.iloc[i] = min(upper_band.iloc[i], ub_p) if c_p <= ub_p else upper_band.iloc[i]
        lower_band.iloc[i] = max(lower_band.iloc[i], lb_p) if c_p >= lb_p else lower_band.iloc[i]
        c_now = df["Close"].iloc[i]
        st_p = supertrend.iloc[i - 1]
        if pd.isna(st_p) or st_p >= ub_p:
            if c_now > upper_band.iloc[i]:
                supertrend.iloc[i] = lower_band.iloc[i];
                direction.iloc[i] = 1
            else:
                supertrend.iloc[i] = upper_band.iloc[i];
                direction.iloc[i] = -1
        else:
            if c_now < lower_band.iloc[i]:
                supertrend.iloc[i] = upper_band.iloc[i];
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i];
                direction.iloc[i] = 1

    supertrend.iloc[0] = upper_band.iloc[0]
    direction.iloc[0] = -1
    return supertrend, direction


def calculate_fibonacci(df, lookback=120):
    """計算近 lookback 日的斐波那契回撤位"""
    sub = df.tail(lookback)
    high = float(sub["High"].max())
    low = float(sub["Low"].min())
    diff = high - low
    return {
        "high": high, "high_date": sub["High"].idxmax().strftime("%Y-%m-%d"),
        "low": low, "low_date": sub["Low"].idxmin().strftime("%Y-%m-%d"),
        "0.236": round(high - 0.236 * diff, 2),
        "0.382": round(high - 0.382 * diff, 2),
        "0.500": round(high - 0.500 * diff, 2),
        "0.618": round(high - 0.618 * diff, 2),
        "0.786": round(high - 0.786 * diff, 2),
    }


def calc_relative_strength(stock_df, benchmark_ticker="0050.TW", period=20):
    """個股相對大盤近 period 日強弱對比"""
    try:
        bench = clean_yf_columns(_ensure_naive_index(
            yf.download(benchmark_ticker, period="3mo", progress=False, auto_adjust=False)))
        if bench.empty or len(stock_df) < period:
            return None
        s_ret = (float(stock_df["Close"].iloc[-1]) / float(stock_df["Close"].iloc[-period]) - 1) * 100
        b_ret = (float(bench["Close"].iloc[-1]) / float(bench["Close"].iloc[-period]) - 1) * 100
        rs = round(s_ret - b_ret, 2)
        label = ("明顯強於大盤（主動性買盤）" if rs > 3 else
                 ("略強於大盤" if rs > 0 else
                  ("略弱於大盤" if rs > -3 else "明顯弱於大盤（資金流出）")))
        return {"stock_ret": round(s_ret, 2), "bench_ret": round(b_ret, 2), "rs": rs, "label": label}
    except Exception:
        return None


def detect_gaps(df, lookback=30):
    """偵測近 lookback 日的跳空缺口"""
    sub = df.tail(lookback + 1)
    gaps = []
    for i in range(1, len(sub)):
        prev, curr = sub.iloc[i - 1], sub.iloc[i]
        date_str = sub.index[i].strftime("%Y-%m-%d")
        if float(curr["Low"]) > float(prev["High"]):
            future = sub.iloc[i + 1:]
            filled = any(float(r["Low"]) <= float(prev["High"]) for _, r in future.iterrows())
            gaps.append({"date": date_str, "type": "向上跳空缺口（強勢）",
                         "upper": float(curr["Low"]), "lower": float(prev["High"]),
                         "filled": "已回補" if filled else "未回補"})
        elif float(curr["High"]) < float(prev["Low"]):
            future = sub.iloc[i + 1:]
            filled = any(float(r["High"]) >= float(prev["Low"]) for _, r in future.iterrows())
            gaps.append({"date": date_str, "type": "向下跳空缺口（弱勢）",
                         "upper": float(prev["Low"]), "lower": float(curr["High"]),
                         "filled": "已回補" if filled else "未回補"})
    return gaps


# =========================
# Pattern Detection
# =========================
def _find_pivots(df, left=3, right=3):
    if df is None or df.empty or len(df) < left + right + 5:
        return [], []
    highs = df["High"].to_numpy(dtype=float)
    lows = df["Low"].to_numpy(dtype=float)
    idx = df.index
    hp, lp = [], []
    for i in range(left, len(df) - right):
        hw = highs[i - left: i + right + 1]
        lw = lows[i - left: i + right + 1]
        if np.isfinite(highs[i]) and highs[i] == np.nanmax(hw):
            if not hp or hp[-1][0] < i - right:
                hp.append((i, idx[i], highs[i]))
        if np.isfinite(lows[i]) and lows[i] == np.nanmin(lw):
            if not lp or lp[-1][0] < i - right:
                lp.append((i, idx[i], lows[i]))
    return hp, lp


def _pct_diff(a, b):
    if a == 0 or not np.isfinite(a) or not np.isfinite(b): return np.inf
    return abs(a - b) / abs(a)


def detect_double_top(df, lookback=120, pivot_lr=3, peak_tol=0.018,
                      min_gap=8, max_gap=60, confirm_margin=0.003):
    if df is None or df.empty: return None
    sub = df.tail(lookback).copy()
    hp, _ = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(hp) < 2: return None
    p2 = hp[-1]
    p1 = next((c for c in reversed(hp[:-1]) if min_gap <= p2[0] - c[0] <= max_gap), None)
    if p1 is None or _pct_diff(float(p1[2]), float(p2[2])) > peak_tol: return None
    lo, hi = min(p1[0], p2[0]), max(p1[0], p2[0])
    neckline = float(sub.iloc[lo:hi + 1]["Low"].min())
    confirmed = float(sub["Close"].iloc[-1]) < neckline * (1 - confirm_margin)
    height = max(float(p1[2]), float(p2[2])) - neckline
    return {"pattern": "M頭（Double Top）",
            "status": "已確認（跌破頸線）" if confirmed else "形成中", "confirmed": confirmed,
            "peak1_date": str(p1[1].date()), "peak2_date": str(p2[1].date()),
            "peak1": float(p1[2]), "peak2": float(p2[2]),
            "neckline": neckline, "target": round(neckline - height, 2)}


def detect_double_bottom(df, lookback=160, pivot_lr=3, trough_tol=0.020,
                         min_gap=8, max_gap=80, confirm_margin=0.003):
    if df is None or df.empty: return None
    sub = df.tail(lookback).copy()
    _, lp = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(lp) < 2: return None
    t2 = lp[-1]
    t1 = next((c for c in reversed(lp[:-1]) if min_gap <= t2[0] - c[0] <= max_gap), None)
    if t1 is None or _pct_diff(float(t1[2]), float(t2[2])) > trough_tol: return None
    lo, hi = min(t1[0], t2[0]), max(t1[0], t2[0])
    neckline = float(sub.iloc[lo:hi + 1]["High"].max())
    confirmed = float(sub["Close"].iloc[-1]) > neckline * (1 + confirm_margin)
    height = neckline - min(float(t1[2]), float(t2[2]))
    return {"pattern": "W底（Double Bottom）",
            "status": "已確認（突破頸線）" if confirmed else "形成中", "confirmed": confirmed,
            "trough1_date": str(t1[1].date()), "trough2_date": str(t2[1].date()),
            "trough1": float(t1[2]), "trough2": float(t2[2]),
            "neckline": neckline, "target": round(neckline + height, 2)}


def detect_head_and_shoulders(df, lookback=180, pivot_lr=3,
                              shoulder_tol=0.025, confirm_margin=0.003):
    if df is None or df.empty: return None
    sub = df.tail(lookback).copy()
    hp, lp = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(hp) < 3 or len(lp) < 2: return None
    ls, head, rs = hp[-3], hp[-2], hp[-1]
    if not (float(head[2]) > float(ls[2]) and float(head[2]) > float(rs[2])): return None
    if _pct_diff(float(ls[2]), float(rs[2])) > shoulder_tol: return None
    mid_lows = [p for p in lp if ls[0] <= p[0] <= rs[0]]
    if len(mid_lows) < 2: return None
    neckline = float(np.mean([p[2] for p in mid_lows]))
    confirmed = float(sub["Close"].iloc[-1]) < neckline * (1 - confirm_margin)
    height = float(head[2]) - neckline
    return {"pattern": "頭肩頂（H&S Top）",
            "status": "已確認（跌破頸線）" if confirmed else "形成中", "confirmed": confirmed,
            "head": float(head[2]), "head_date": str(head[1].date()),
            "left_shoulder": float(ls[2]), "right_shoulder": float(rs[2]),
            "neckline": round(neckline, 2), "target": round(neckline - height, 2)}


def detect_inverse_head_and_shoulders(df, lookback=180, pivot_lr=3,
                                      shoulder_tol=0.025, confirm_margin=0.003):
    if df is None or df.empty: return None
    sub = df.tail(lookback).copy()
    hp, lp = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(lp) < 3 or len(hp) < 2: return None
    ls, head, rs = lp[-3], lp[-2], lp[-1]
    if not (float(head[2]) < float(ls[2]) and float(head[2]) < float(rs[2])): return None
    if _pct_diff(float(ls[2]), float(rs[2])) > shoulder_tol: return None
    mid_highs = [p for p in hp if ls[0] <= p[0] <= rs[0]]
    if len(mid_highs) < 2: return None
    neckline = float(np.mean([p[2] for p in mid_highs]))
    confirmed = float(sub["Close"].iloc[-1]) > neckline * (1 + confirm_margin)
    height = neckline - float(head[2])
    return {"pattern": "頭肩底（Inverse H&S）",
            "status": "已確認（突破頸線）" if confirmed else "形成中", "confirmed": confirmed,
            "head": float(head[2]), "head_date": str(head[1].date()),
            "left_shoulder": float(ls[2]), "right_shoulder": float(rs[2]),
            "neckline": round(neckline, 2), "target": round(neckline + height, 2)}


def detect_wedge(df, lookback=120, pivot_lr=3, breakout_margin=0.003):
    if df is None or df.empty: return None
    sub = df.tail(lookback).copy()
    hp, lp = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(hp) < 3 or len(lp) < 3: return None
    highs, lows = hp[-3:], lp[-3:]
    xh, yh = np.array([p[0] for p in highs], float), np.array([p[2] for p in highs], float)
    xl, yl = np.array([p[0] for p in lows], float), np.array([p[2] for p in lows], float)
    ah, bh = np.polyfit(xh, yh, 1)
    al, bl = np.polyfit(xl, yl, 1)
    n = len(sub) - 1
    upper_now = ah * n + bh
    lower_now = al * n + bl
    close_now = float(sub["Close"].iloc[-1])

    if ah > 0 and al > 0 and al > ah:  # 上升楔形（看空）
        status = ("已確認（向下跌破）" if close_now < lower_now * (1 - breakout_margin)
                  else ("假突破警示" if close_now > upper_now * (1 + breakout_margin) else "形成中"))
        return {"pattern": "上升楔形（Rising Wedge，看空信號）", "status": status,
                "upper_now": round(upper_now, 2), "lower_now": round(lower_now, 2)}

    if ah < 0 and al < 0 and ah < al:  # 下降楔形（看多）
        status = ("已確認（向上突破）" if close_now > upper_now * (1 + breakout_margin)
                  else ("假跌破警示" if close_now < lower_now * (1 - breakout_margin) else "形成中"))
        return {"pattern": "下降楔形（Falling Wedge，看多信號）", "status": status,
                "upper_now": round(upper_now, 2), "lower_now": round(lower_now, 2)}
    return None


def detect_sym_triangle(df, lookback=180, pivot_lr=3, need_points=3,
                        tighten_ratio=0.65, breakout_margin=0.003):
    if df is None or df.empty: return None
    sub = df.tail(lookback).copy()
    hp, lp = _find_pivots(sub, left=pivot_lr, right=pivot_lr)
    if len(hp) < need_points or len(lp) < need_points: return None
    highs, lows = hp[-need_points:], lp[-need_points:]
    hp_p, lp_p = [x[2] for x in highs], [x[2] for x in lows]
    if not (hp_p[0] > hp_p[1] > hp_p[2]): return None
    if not (lp_p[0] < lp_p[1] < lp_p[2]): return None
    xh, yh = np.array([p[0] for p in highs], float), np.array([p[2] for p in highs], float)
    xl, yl = np.array([p[0] for p in lows], float), np.array([p[2] for p in lows], float)
    ah, bh = np.polyfit(xh, yh, 1)
    al, bl = np.polyfit(xl, yl, 1)
    if not (ah < 0 and al > 0): return None
    early = sub.iloc[:max(30, len(sub) // 3)]
    late = sub.iloc[-max(30, len(sub) // 3):]
    er, lr = float(early["High"].max() - early["Low"].min()), float(late["High"].max() - late["Low"].min())
    if er <= 0 or lr / er > tighten_ratio: return None
    n = len(sub) - 1
    upper_now, lower_now = ah * n + bh, al * n + bl
    close_now = float(sub["Close"].iloc[-1])
    if close_now > upper_now * (1 + breakout_margin):
        status, direction = "已確認（向上突破）", "向上"
    elif close_now < lower_now * (1 - breakout_margin):
        status, direction = "已確認（向下跌破）", "向下"
    else:
        status, direction = "形成中", "未突破"
    return {"pattern": "收斂三角（Sym Triangle）", "status": status, "direction": direction,
            "upper_now": float(upper_now), "lower_now": float(lower_now),
            "range_shrink": float(lr / er)}


def detect_patterns(df):
    results = []
    for fn in [detect_double_top, detect_double_bottom,
               detect_head_and_shoulders, detect_inverse_head_and_shoulders,
               detect_wedge, detect_sym_triangle]:
        try:
            p = fn(df)
            if p: results.append(p)
        except Exception:
            pass
    return results


# =========================
# Sector Analysis
# =========================
SECTOR_DICT = {
    "記憶體族群": ["MU", "WDC", "000660.KS"],
    "被動元件族群": ["VSH", "6981.T", "6976.T"],
    "AI 與伺服器族群": ["NVDA", "SMCI", "TSM", "VRT"],
    "手機與 IC 設計": ["QCOM", "ARM", "1810.HK"],
    "太空組": ["ARKX", "UFO"]
}


def analyze_sector_performance(sector_key: str, as_of_date=None, custom_tickers: list = None):
    tickers = custom_tickers if custom_tickers else SECTOR_DICT.get(sector_key, [])
    if not tickers:
        return f"❌ 無效族群：{sector_key}"
    if as_of_date is None:
        as_of_date = datetime.now().date()

    out = [f"📊 {sector_key} 族群行情快篩", f"📅 資料日期: {as_of_date}", "=" * 45]
    total_pct, valid_count, results = 0.0, 0, []

    for t in tickers:
        try:
            resolved = _resolve_yf_ticker(t)
            df = clean_yf_columns(_ensure_naive_index(
                yf.download(resolved, period="1y", interval="1d", progress=False, auto_adjust=False)))
            if df.empty and resolved.endswith(".TW"):
                alt = resolved.replace(".TW", ".TWO")
                df = clean_yf_columns(_ensure_naive_index(
                    yf.download(alt, period="1y", interval="1d", progress=False, auto_adjust=False)))
            ts = _nearest_trading_ts(df, as_of_date)
            if ts is None:
                results.append((t, None, None, "無資料"));
                continue
            loc = df.index.get_loc(ts)
            if loc < 1:
                results.append((t, float(df.iloc[loc]["Close"]), None, "無前日"));
                continue
            price = float(df.iloc[loc]["Close"])
            prev_price = float(df.iloc[loc - 1]["Close"])
            pct = (price - prev_price) / prev_price * 100
            results.append((t, price, pct, None))
            total_pct += pct
            valid_count += 1
        except Exception as e:
            results.append((t, None, None, str(e)))

    avg_pct = total_pct / valid_count if valid_count > 0 else 0.0
    icon = "🔴" if avg_pct > 0 else ("🟢" if avg_pct < 0 else "➖")
    out.append(f"族群均漲跌: {icon} {avg_pct:+.2f}%")
    out.append("-" * 45)
    out.append(f"{'代號':<10}{'收盤':>10}{'漲跌幅':>10}")
    out.append("-" * 45)
    for t, price, pct, err in results:
        if err:
            out.append(f"{t:<10}{'N/A':>10}  {err}")
        else:
            out.append(f"{t:<10}{price:>10.2f}{pct:>+10.2f}%")
    out.append("-" * 45)
    return "\n".join(out)


def analyze_stock_technical(stock_id: str, as_of_date=None) -> str:
    stock_id = stock_id.strip().upper()
    if not stock_id:
        return "請輸入股票代號"

    yf_ticker = _resolve_yf_ticker(stock_id)
    stock_no = _strip_suffix(stock_id)
    out = [f"🔄 正在分析 {stock_id} ..."]

    # ── 下載日線 ──
    df_daily = yf.download(yf_ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
    df_daily = clean_yf_columns(_ensure_naive_index(df_daily))

    if df_daily.empty and yf_ticker.endswith(".TW"):
        alt = yf_ticker.replace(".TW", ".TWO")
        out.append(f"⚠️ .TW 無資料，改試 {alt}")
        yf_ticker = alt
        df_daily = yf.download(yf_ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
        df_daily = clean_yf_columns(_ensure_naive_index(df_daily))

    if df_daily.empty:
        return "\n".join(out + [f"❌ 找不到 {stock_id}"])

    latest_overall_ts = df_daily.index[-1]
    chosen_ts = _nearest_trading_ts(df_daily, as_of_date)
    if chosen_ts is None:
        return "\n".join(out + ["❌ 選定日期前無資料"])

    df_upto = df_daily.loc[:chosen_ts].copy()
    if df_upto.empty:
        return "\n".join(out + ["❌ 無可用資料"])

    # ── 下載週線/月線 ──
    df_weekly = clean_yf_columns(_ensure_naive_index(
        yf.download(yf_ticker, period="2y", interval="1wk", progress=False, auto_adjust=False)))
    df_monthly = clean_yf_columns(_ensure_naive_index(
        yf.download(yf_ticker, period="5y", interval="1mo", progress=False, auto_adjust=False)))
    df_weekly_upto = df_weekly.loc[:chosen_ts] if (df_weekly is not None and not df_weekly.empty) else None
    df_monthly_upto = df_monthly.loc[:chosen_ts] if (df_monthly is not None and not df_monthly.empty) else None

    latest = df_upto.iloc[-1]
    prev = df_upto.iloc[-2] if len(df_upto) >= 2 else latest
    trade_date = chosen_ts.date()

    # ── 抓外部資料 ──
    chips_multi = get_institutional_multi_days(stock_id, trade_date, market_hint=yf_ticker, days=5)
    margin_data = get_margin_short_data(
        stock_id, trade_date=latest_overall_ts.date(), market_hint=yf_ticker, max_back=7)
    foreign_holding = (get_foreign_holding_ratio(stock_no)
                       if yf_ticker.endswith(".TW")
                       else {"ratio": None, "error": "僅支援上市股"})

    # ── 計算指標 ──
    close = df_upto["Close"]
    ma5 = close.rolling(5).mean()
    ma20 = close.rolling(20).mean()
    ma60 = close.rolling(60).mean()

    rsi6 = calculate_rsi(close, 6)
    rsi14 = calculate_rsi(close, 14)

    low_min = df_upto["Low"].rolling(9).min()
    high_max = df_upto["High"].rolling(9).max()
    rsv = (close - low_min) / (high_max - low_min).replace(0, np.nan) * 100
    k_val = rsv.ewm(com=2).mean()
    d_val = k_val.ewm(com=2).mean()

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False).mean()
    osc = dif - dea

    adx, pdi, mdi = calculate_adx(df_upto)
    bbw, bb_upper, bb_lower = calculate_bbw(df_upto)
    mfi_series = calculate_mfi(df_upto)
    obv = calculate_obv(df_upto)

    try:
        st_line, st_dir = calculate_supertrend(df_upto)
        st_val = float(st_line.iloc[-1])
        st_direction = int(st_dir.iloc[-1])
    except Exception:
        st_val = st_direction = None

    fib = calculate_fibonacci(df_upto)
    rs_data = calc_relative_strength(df_upto, benchmark_ticker="0050.TW", period=20)
    gaps = detect_gaps(df_upto, lookback=30)
    avwap = calculate_avwap(df_upto, f"{chosen_ts.year}-01-01")

    # ── 常用現值 ──
    c_now = float(close.iloc[-1])
    ma5_now = float(ma5.iloc[-1])
    ma20_now = float(ma20.iloc[-1])
    ma60_now = float(ma60.iloc[-1])

    # ── 量價關係 ──
    vol_today = float(latest["Volume"]) if not pd.isna(latest.get("Volume", float("nan"))) else 0
    vol_prev = float(prev["Volume"]) if not pd.isna(prev.get("Volume", float("nan"))) else 0
    use_prev = vol_today == 0 and len(df_upto) >= 2
    vol_used = vol_prev if use_prev else vol_today
    vol_note = "（改用前日量）" if use_prev else ""
    avg_vol_5 = float(df_upto["Volume"].tail(5).mean())
    vol_ratio = vol_used / avg_vol_5 if avg_vol_5 > 0 else 1.0
    price_up = c_now > float(prev["Close"])
    vol_up = vol_used > avg_vol_5

    if price_up and vol_up:
        vol_price_label = "價漲量增（健康上漲，多方積極）"
    elif price_up and not vol_up:
        vol_price_label = "價漲量縮（動能不足，需觀察）"
    elif not price_up and vol_up:
        vol_price_label = "價跌量增（主力出貨警訊，偏空）"
    else:
        vol_price_label = "價跌量縮（正常回檔，賣壓有限）"

    # ── 趨勢階段判斷 ──
    if c_now > ma20_now > ma60_now and ma5_now > ma20_now:
        trend_stage = "上升趨勢（均線多頭排列，多方主導）"
    elif c_now < ma20_now < ma60_now and ma5_now < ma20_now:
        trend_stage = "下降趨勢（均線空頭排列，空方主導）"
    elif c_now > ma20_now and ma20_now < ma60_now:
        trend_stage = "底部反彈（價格回升但均線未翻多，待確認）"
    elif c_now < ma20_now and ma20_now > ma60_now:
        trend_stage = "高位回落（價格跌破短均，注意風險）"
    else:
        trend_stage = "盤整階段（均線糾結，方向不明）"

    # ── 交叉訊號 ──
    def _cross(fast, slow, buy_msg, sell_msg):
        f, s = fast.dropna(), slow.dropna()
        if len(f) < 2 or len(s) < 2:
            return ""
        if f.iloc[-2] < s.iloc[-2] and f.iloc[-1] > s.iloc[-1]:
            return f"  ⚡ {buy_msg}"
        if f.iloc[-2] > s.iloc[-2] and f.iloc[-1] < s.iloc[-1]:
            return f"  ⚡ {sell_msg}"
        return ""

    kd_cross = _cross(k_val, d_val, "KD 黃金交叉（買訊）", "KD 死亡交叉（賣訊）")
    macd_cross = _cross(dif, dea, "MACD 黃金交叉（買訊）", "MACD 死亡交叉（賣訊）")

    # ============================================================
    #  開始輸出報告
    # ============================================================
    SEP = "=" * 56
    LINE = "-" * 56

    out.append("")
    out.append(SEP)
    out.append(f"  {yf_ticker}  技術分析報告（AI 分析用）")
    out.append(f"  資料日期: {trade_date}  /  查詢日: {as_of_date}")
    out.append(SEP)

    # ───────────────────────────────
    # 1. 近 5 日行情
    # ───────────────────────────────
    out.append("【1. 近 5 日行情】")
    out.append(f"  {'日期':<10} {'收盤':>8} {'漲跌':>8} {'量(張)':>9}  K棒型態")
    out.append(f"  {LINE}")
    for idx, row in df_upto.tail(5).iterrows():
        loc = df_upto.index.get_loc(idx)
        chg = float(row["Close"]) - float(df_upto.iloc[loc - 1]["Close"]) if loc > 0 else 0.0
        v = int(float(row["Volume"]) / 1000) if not pd.isna(row.get("Volume", float("nan"))) else 0
        out.append(
            f"  {idx.strftime('%m-%d'):<10} {float(row['Close']):>8.2f} {chg:>+8.2f} {v:>9,}"
            f"  {get_k_status(row['Open'], row['Close'], row['High'], row['Low'])}"
        )
    out.append(f"  當日K棒型態: {describe_candle(latest['Open'], latest['High'], latest['Low'], latest['Close'])}")
    out.append(LINE)

    # ───────────────────────────────
    # 2. 趨勢概況
    # ───────────────────────────────
    out.append("【2. 趨勢概況】")
    out.append(f"  ▸ 趨勢階段: {trend_stage}")
    out.append(
        f"  ▸ 現價 {c_now:.2f}"
        f"  MA5={ma5_now:.2f}（乖離 {(c_now - ma5_now) / ma5_now * 100:+.2f}%）"
        f"  MA20={ma20_now:.2f}（{(c_now - ma20_now) / ma20_now * 100:+.2f}%）"
        f"  MA60={ma60_now:.2f}（{(c_now - ma60_now) / ma60_now * 100:+.2f}%）"
    )
    out.append(f"  ▸ MA5  近5日: {_fmt_series(ma5)}  趨勢方向: {_trend_label(ma5.dropna().iloc[-5:].tolist())}")
    out.append(f"  ▸ MA20 近5日: {_fmt_series(ma20)}  趨勢方向: {_trend_label(ma20.dropna().iloc[-5:].tolist())}")

    high_52 = float(df_upto.tail(252)["High"].max())
    low_52 = float(df_upto.tail(252)["Low"].min())
    pos_52 = (c_now - low_52) / (high_52 - low_52) * 100 if high_52 != low_52 else 50
    out.append(f"  ▸ 52 週高低: {low_52:.2f} ～ {high_52:.2f}，現價位於 {pos_52:.1f}%（0%=近低，100%=近高）")

    if df_weekly_upto is not None and not df_weekly_upto.empty:
        wma20 = float(df_weekly_upto["Close"].rolling(20).mean().iloc[-1])
        wc = float(df_weekly_upto["Close"].iloc[-1])
        out.append(
            f"  ▸ [週K] 收={wc:.2f}，20週均={wma20:.2f}，{'週線站上均線（多）' if wc > wma20 else '週線跌破均線（空）'}")
    if df_monthly_upto is not None and not df_monthly_upto.empty:
        mc = float(df_monthly_upto["Close"].iloc[-1])
        out.append(f"  ▸ [月K] 收={mc:.2f}")

    if rs_data:
        out.append(
            f"  ▸ 相對強度 vs 大盤(0050) 近20日:"
            f" 個股 {rs_data['stock_ret']:+.2f}% / 大盤 {rs_data['bench_ret']:+.2f}%"
            f" → RS={rs_data['rs']:+.2f}%，{rs_data['label']}"
        )
    out.append(LINE)

    # ───────────────────────────────
    # 3. 量價分析
    # ───────────────────────────────
    out.append("【3. 量價分析】")
    vol_lots = int(vol_used / 1000)
    avg5_lots = int(avg_vol_5 / 1000)
    out.append(f"  ▸ 今日量: {vol_lots:,} 張{vol_note}，5日均量: {avg5_lots:,} 張，量/均量比: {vol_ratio:.2f}x")
    out.append(f"  ▸ 量價關係研判: {vol_price_label}")
    vol5 = df_upto["Volume"].tail(5) / 1000
    out.append(f"  ▸ 近5日量能(張): {_fmt_series(vol5, fmt='.0f')}  趨勢: {_trend_label(vol5.tolist())}")

    obv_pct = _percentile_rank(obv, 252)
    out.append(f"  ▸ OBV 近5日: {_fmt_series(obv, fmt='.0f')}  趨勢: {_trend_label(obv.dropna().iloc[-5:].tolist())}")
    if obv_pct is not None:
        obv_interp = "偏高（資金持續流入）" if obv_pct > 60 else ("偏低（資金持續流出）" if obv_pct < 40 else "中性")
        out.append(f"    OBV 近一年第 {obv_pct:.0f} 百分位 → {obv_interp}")
    out.append(LINE)

    # ───────────────────────────────
    # 4. 動能指標（含近5日序列）
    # ───────────────────────────────
    out.append("【4. 動能指標（含近5日序列）】")

    # RSI
    rsi6_now = float(rsi6.iloc[-1])
    rsi6_pct = _percentile_rank(rsi6, 252)
    rsi6_interp = (
        "超買（>80，留意回檔）" if rsi6_now > 80 else
        "超賣（<20，可能反彈）" if rsi6_now < 20 else
        "偏強（60~80）" if rsi6_now > 60 else
        "偏弱（20~40）" if rsi6_now < 40 else
        "中性（40~60）"
    )
    out.append(f"  ▸ RSI(6) = {rsi6_now:.2f}，{rsi6_interp}")
    out.append(f"    近5日序列: {_fmt_series(rsi6)}  趨勢: {_trend_label(rsi6.dropna().iloc[-5:].tolist())}")
    if rsi6_pct is not None:
        out.append(
            f"    近一年百分位: 第 {rsi6_pct:.0f} 位（{'偏高' if rsi6_pct > 70 else ('偏低' if rsi6_pct < 30 else '正常')}）")

    rsi14_now = float(rsi14.iloc[-1])
    out.append(f"  ▸ RSI(14) = {rsi14_now:.2f}，近5日: {_fmt_series(rsi14)}")

    # KD
    k_now, d_now = float(k_val.iloc[-1]), float(d_val.iloc[-1])
    out.append(f"  ▸ KD(9,3,3): K={k_now:.2f}, D={d_now:.2f}{kd_cross}")
    out.append(f"    K 近5日: {_fmt_series(k_val)}  趨勢: {_trend_label(k_val.dropna().iloc[-5:].tolist())}")
    kd_interp = (
        "KD 超買區（>80）" if k_now > 80 and d_now > 80 else
        "KD 超賣區（<20）" if k_now < 20 and d_now < 20 else
        "多方格局（K>D 且非超買）" if k_now > d_now else
        "空方格局（K<D）"
    )
    out.append(f"    KD 研判: {kd_interp}")

    # MACD
    dif_now = float(dif.iloc[-1])
    dea_now = float(dea.iloc[-1])
    osc_now = float(osc.iloc[-1])
    osc_trend = _trend_label(osc.dropna().iloc[-5:].tolist())
    osc_pct = _percentile_rank(osc, 252)
    out.append(f"  ▸ MACD: DIF={dif_now:.3f}, DEA={dea_now:.3f}, OSC={osc_now:.3f}{macd_cross}")
    out.append(f"    OSC 近5日: {_fmt_series(osc, fmt='.3f')}  趨勢: {osc_trend}")
    if osc_now > 0:
        macd_interp = f"OSC 正值且{osc_trend} → {'多頭加速' if '上升' in osc_trend else '多頭但動能趨緩'}"
    else:
        macd_interp = f"OSC 負值且{osc_trend} → {'空頭加速' if '下降' in osc_trend else '空頭但賣壓趨緩'}"
    out.append(f"    MACD 研判: {macd_interp}")
    if osc_pct is not None:
        out.append(f"    OSC 近一年第 {osc_pct:.0f} 百分位")
    out.append(LINE)

    # ───────────────────────────────
    # 5. 趨勢強度與波動
    # ───────────────────────────────
    out.append("【5. 趨勢強度與波動】")

    # ADX
    adx_now = float(adx.iloc[-1])
    pdi_now = float(pdi.iloc[-1])
    mdi_now = float(mdi.iloc[-1])
    adx_interp = (
        "強趨勢（ADX>25，方向明確）" if adx_now > 25 else
        "趨勢醞釀中（ADX 15~25）" if adx_now > 15 else
        "無明確趨勢（ADX<15，盤整）"
    )
    adx_dir = "多方主導（+DI > -DI）" if pdi_now > mdi_now else "空方主導（-DI > +DI）"
    out.append(f"  ▸ ADX(14) = {adx_now:.2f}，{adx_interp}，{adx_dir}")
    out.append(f"    +DI={pdi_now:.2f}, -DI={mdi_now:.2f}，ADX 近5日: {_fmt_series(adx)}")

    # BB
    bbw_now = float(bbw.iloc[-1])
    bb_upper_now = float(bb_upper.iloc[-1])
    bb_lower_now = float(bb_lower.iloc[-1])
    bbw_pct = _percentile_rank(bbw, 252)
    bb_pos = (c_now - bb_lower_now) / (bb_upper_now - bb_lower_now) * 100 if (bb_upper_now - bb_lower_now) > 0 else 50
    bbw_interp = (
        "⚠️ 帶寬極度收斂（近一年最低區間），通常為大波段前兆" if bbw_pct is not None and bbw_pct < 15 else
        "帶寬偏低，波動收縮中" if bbw_now < 10 else
        "帶寬偏高（趨勢強烈）" if bbw_now > 30 else
        "帶寬正常"
    )
    out.append(f"  ▸ BB 帶寬 = {bbw_now:.2f}%，{bbw_interp}")
    out.append(f"    上軌={bb_upper_now:.2f}，下軌={bb_lower_now:.2f}，現價在帶內 {bb_pos:.1f}% 位置")
    if bbw_pct is not None:
        out.append(f"    帶寬近一年第 {bbw_pct:.0f} 百分位")

    # SuperTrend
    if st_val is not None:
        st_label = "多頭（價格在 SuperTrend 上方）" if st_direction == 1 else "空頭（價格在 SuperTrend 下方）"
        st_dist = (c_now - st_val) / st_val * 100
        out.append(f"  ▸ SuperTrend(10, 3) = {st_val:.2f}，{st_label}，距SuperTrend {st_dist:+.2f}%")

    # MFI
    mfi_now = float(mfi_series.iloc[-1])
    mfi_interp = (
        "超買（>80，大量資金流入，留意回檔）" if mfi_now > 80 else
        "超賣（<20，大量資金流出，留意反彈）" if mfi_now < 20 else
        "正常"
    )
    out.append(f"  ▸ MFI(14) = {mfi_now:.2f}，{mfi_interp}")

    # AVWAP
    if avwap is not None and not avwap.empty and np.isfinite(float(avwap.iloc[-1])):
        avwap_now = float(avwap.iloc[-1])
        avwap_dev = (c_now - avwap_now) / avwap_now * 100
        avwap_interp = "偏貴（乖離>5%）" if avwap_dev > 5 else ("偏便宜（乖離<-5%）" if avwap_dev < -5 else "合理區間")
        out.append(f"  ▸ AVWAP(YTD) = {avwap_now:.2f}，乖離 {avwap_dev:+.2f}%（{avwap_interp}）")
    out.append(LINE)

    # ───────────────────────────────
    # 6. 斐波那契支撐壓力
    # ───────────────────────────────
    out.append("【6. 斐波那契支撐壓力位（近120日高低計算）】")
    out.append(f"  ▸ 波段高點: {fib['high']:.2f}（{fib['high_date']}）  波段低點: {fib['low']:.2f}（{fib['low_date']}）")
    fib_levels = [
        ("高點", fib["high"], ""),
        ("0.236", fib["0.236"], ""),
        ("0.382", fib["0.382"], "← 重要支撐"),
        ("0.500", fib["0.500"], "← 中位支撐"),
        ("0.618", fib["0.618"], "← 黃金支撐（最關鍵）"),
        ("0.786", fib["0.786"], "← 深度回撤，趨勢轉弱警示"),
        ("低點", fib["low"], ""),
    ]
    for name, val, note in fib_levels:
        marker = " ◀ 現價區間" if name not in ("高點", "低點") and abs(c_now - val) / val < 0.015 else ""
        out.append(f"    {name:<8} = {val:.2f}  {note}{marker}")

    # 標示現價落在哪個區間
    for i in range(len(fib_levels) - 1):
        hi, lo = fib_levels[i][1], fib_levels[i + 1][1]
        if hi >= c_now >= lo:
            out.append(f"  → 現價落在 {fib_levels[i][0]} ～ {fib_levels[i + 1][0]} 區間")
            break
    out.append(LINE)

    # ───────────────────────────────
    # 7. 跳空缺口
    # ───────────────────────────────
    if gaps:
        out.append("【7. 近30日跳空缺口】")
        for g in gaps:
            out.append(
                f"  ▸ {g['date']}  {g['type']}"
                f"  缺口區間: {g['lower']:.2f} ～ {g['upper']:.2f}，{g['filled']}"
            )
        out.append(LINE)

    # ───────────────────────────────
    # 8. K線型態辨識
    # ───────────────────────────────
    out.append("【8. K線型態辨識（M頭/W底/頭肩/楔形/三角）】")
    try:
        patterns = detect_patterns(df_upto)
        if not patterns:
            out.append("  ▸ 未偵測到明確型態（資料不足或條件未符合）")
        else:
            for p in patterns:
                pname = p["pattern"]
                pstatus = p["status"]
                if "neckline" in p:
                    out.append(
                        f"  ▸ {pname}：{pstatus}"
                        f"  頸線={p['neckline']:.2f}  目標={p['target']:.2f}"
                        + (f"  頭={p['head']:.2f}（{p.get('head_date', '')}）" if "head" in p else "")
                    )
                else:
                    out.append(
                        f"  ▸ {pname}：{pstatus}"
                        f"  上緣={p['upper_now']:.2f}，下緣={p['lower_now']:.2f}"
                        + (f"，收斂比={p['range_shrink']:.2f}" if "range_shrink" in p else "")
                    )
    except Exception:
        out.append("  ▸ 型態辨識失敗（資料不足）")
    out.append(LINE)

    # ───────────────────────────────
    # 9. 籌碼面：三大法人
    # ───────────────────────────────
    out.append("【9. 籌碼面：三大法人（近5日）】")

    def _to_lots(v):
        v2 = _safe_int(v)
        return int(round(v2 / 1000)) if v2 is not None else None

    if chips_multi:
        dates_str = [r["date"][-5:] for r in chips_multi]
        f_vals = [_to_lots(r.get("foreign")) for r in chips_multi]
        t_vals = [_to_lots(r.get("trust")) for r in chips_multi]
        d_vals = [_to_lots(r.get("dealer")) for r in chips_multi]

        def _fmtl(vals):
            return "[" + ", ".join((f"{v:+,}" if v is not None else "n/a") for v in vals) + "]"

        out.append(f"  ▸ 日期        : {dates_str}")
        out.append(f"  ▸ 外資 (張)   : {_fmtl(f_vals)}")
        out.append(f"  ▸ 投信 (張)   : {_fmtl(t_vals)}")
        out.append(f"  ▸ 自營商 (張) : {_fmtl(d_vals)}")

        f_days, f_dir = _consecutive_direction(chips_multi, "foreign")
        t_days, t_dir = _consecutive_direction(chips_multi, "trust")
        d_days, d_dir = _consecutive_direction(chips_multi, "dealer")
        out.append(
            f"  ▸ 連續方向: "
            f"外資 {f_days} 日{f_dir} / 投信 {t_days} 日{t_dir} / 自營 {d_days} 日{d_dir}"
        )

        f_cum = sum(v for v in f_vals if v is not None)
        t_cum = sum(v for v in t_vals if v is not None)
        d_cum = sum(v for v in d_vals if v is not None)
        out.append(f"  ▸ 5日累計: 外資 {f_cum:+,} 張 / 投信 {t_cum:+,} 張 / 自營 {d_cum:+,} 張")

        # 籌碼背離判斷 (新增)
        c_5d_ago = float(df_upto["Close"].iloc[-5]) if len(df_upto) >= 5 else c_now
        price_is_up_5d = c_now > c_5d_ago

        divergence_msg = ""
        if price_is_up_5d and f_cum < 0:
            divergence_msg = "  ⚠️ 警示：籌碼背離（股價上漲，但外資趁高調節賣出）"

        if f_cum > 0 and t_cum > 0:
            consensus = "外資＋投信同步買超（強多頭信號，法人積極布局）"
        elif f_cum < 0 and t_cum < 0:
            consensus = "外資＋投信同步賣超（強空頭信號，法人持續撤退）"
        elif f_cum > 0 and t_cum < 0:
            consensus = "外資買超、投信賣超（方向分歧，以外資為主要參考）"
        elif f_cum < 0 and t_cum > 0:
            consensus = "外資賣超、投信買超（方向分歧，投信或逆勢承接）"
        else:
            consensus = "外資/投信無明顯方向"
        out.append(f"  ▸ 籌碼方向研判: {consensus}")

        if divergence_msg:
            out.append(divergence_msg)

    else:
        out.append("  ▸ 無法取得三大法人資料（TWSE/TPEx 可能休市或網路問題）")

    # 外資持股比率
    if foreign_holding.get("ratio") is not None:
        ratio = foreign_holding["ratio"]
        r_interp = (
            "偏高（>50%，外資為主要持有者，動向影響極大）" if ratio > 50 else
            "中等（20~50%）" if ratio > 20 else
            "偏低（<20%）"
        )
        out.append(f"  ▸ 外資持股比率: {ratio:.2f}%（{r_interp}，資料: {foreign_holding.get('date')}）")
    else:
        out.append(f"  ▸ 外資持股比率: 無法取得（{foreign_holding.get('error')}）")
    out.append(LINE)

    # ───────────────────────────────
    # 10. 融資融券（散戶槓桿）
    # ───────────────────────────────
    out.append("【10. 融資融券（散戶槓桿指標）】")
    if isinstance(margin_data, dict) and margin_data.get("error") is None:
        m_bal = _safe_int(margin_data.get("margin_balance"))
        m_chg = _safe_int(margin_data.get("margin_change"))
        m_rate = margin_data.get("margin_usage_rate")
        s_bal = _safe_int(margin_data.get("short_balance"))
        s_chg = _safe_int(margin_data.get("short_change"))

        m_str = f"{m_bal:,} 張" if m_bal is not None else "n/a"
        m_chg_str = (f"，較前日 {m_chg:+,} 張（{'加碼' if m_chg > 0 else '減碼'}）"
                     if m_chg is not None else "")
        out.append(f"  ▸ 融資餘額: {m_str}{m_chg_str}")

        if m_rate is not None:
            rate_interp = (
                "⚠️ 使用率偏高（>70%），散戶槓桿集中，下跌時斷頭風險大" if m_rate > 70 else
                "使用率中等（40~70%）" if m_rate > 40 else
                "使用率正常（<40%）"
            )
            out.append(f"  ▸ 融資使用率: {m_rate:.1f}%（{rate_interp}）")

        s_str = f"{s_bal:,} 張" if s_bal is not None else "n/a"
        s_chg_str = f"，較前日 {s_chg:+,} 張" if s_chg is not None else ""

        if m_bal and s_bal and m_bal > 0:
            sr = s_bal / m_bal * 100
            sr_interp = (
                "偏高（>15%，空方力道強，需留意軋空可能）" if sr > 15 else
                "中等（5~15%）" if sr > 5 else
                "偏低（<5%，空方力道弱）"
            )
            out.append(f"  ▸ 融券餘額: {s_str}{s_chg_str}，券資比: {sr:.1f}%（{sr_interp}）")
        else:
            out.append(f"  ▸ 融券餘額: {s_str}{s_chg_str}")

        out.append(f"    （資料日期: {margin_data.get('date')}，市場: {margin_data.get('id')}）")
    else:
        err_msg = margin_data.get("error") if isinstance(margin_data, dict) else "未知"
        out.append(f"  ▸ 無法取得融資融券（{err_msg}）")

    out.append(SEP)
    out.append("✅ 以上為完整技術面數據，請根據上述各項指標進行綜合研判。")
    out.append("   重點關注：趨勢階段 + 量價關係 + 籌碼方向 + 指標交叉訊號 + 型態確認")
    return "\n".join(out)
