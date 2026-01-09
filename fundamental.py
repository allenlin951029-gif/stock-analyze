# -*- coding: utf-8 -*-
import io
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


def _resolve_yf_ticker(stock_id: str) -> str:
    s = (stock_id or "").strip().upper()
    if s.endswith(".TW") or s.endswith(".TWO"):
        return s
    if s.isdigit():
        return f"{s}.TW"
    return s


def _safe_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (int, float)):
            return float(x)
        s = str(x).strip().replace(",", "")
        # 常見缺值符號（含全形/長破折號）
        if s in ("", "--", "—", "－", "N/A", "n/a", "None", "nan", "NaN"):
            return None
        return float(s)
    except Exception:
        return None


def _nearest_trading_date(df: pd.DataFrame, dt: datetime):
    if df is None or df.empty:
        return None
    idx = pd.to_datetime(df.index)
    target = pd.to_datetime(dt)
    diffs = (idx - target).to_numpy(dtype="timedelta64[ns]")
    pos = int(np.abs(diffs).argmin())
    return idx[pos].to_pydatetime()


def _http_get(url: str, timeout: int = 25):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; stock-analyze/1.0)",
        "Accept": "*/*",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    r.raise_for_status()
    return r


def _decode_bytes(b: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp950"):
        try:
            return b.decode(enc)
        except Exception:
            pass
    return b.decode(errors="ignore")


def _extract_by_keywords(rec: dict, keywords):
    for k, v in rec.items():
        ks = str(k)
        if any(kw in ks for kw in keywords):
            return v
    return None


def _bwibbu_latest_openapi(stock_no: str):
    """
    TWSE OpenAPI：最新交易日 全市場（BWIBBU_ALL）
    再用 stock_no 篩一筆
    """
    url = "https://openapi.twse.com.tw/v1/exchangeReport/BWIBBU_ALL"
    r = _http_get(url, timeout=25)
    data = r.json()
    if not isinstance(data, list):
        return None

    # 嘗試各種可能的代號欄位名稱
    for row in data:
        if not isinstance(row, dict):
            continue
        code = (
            row.get("證券代號")
            or row.get("股票代號")
            or row.get("公司代號")
            or row.get("Code")
            or row.get("code")
        )
        if str(code).strip() != stock_no:
            continue

        pe_raw = _extract_by_keywords(row, ["本益比"])
        pb_raw = _extract_by_keywords(row, ["股價淨值比", "淨值比"])
        dy_raw = _extract_by_keywords(row, ["殖利率"])

        return {
            "pe": _safe_float(pe_raw),
            "pb": _safe_float(pb_raw),
            "dy": _safe_float(dy_raw),
        }
    return None


def _bwibbu_one_day_twse_rwd(stock_no: str, yyyymmdd: str):
    """
    TWSE 舊 API（依日期+代碼）
    """
    url = (
        "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"
        f"?date={yyyymmdd}&stockNo={stock_no}&response=json"
    )
    r = _http_get(url, timeout=25)
    js = r.json()
    fields = js.get("fields") or []
    rows = js.get("data") or []
    if not rows:
        return None

    row = rows[0]
    rec = {fields[i]: row[i] for i in range(len(fields))} if isinstance(row, list) else row

    pe = _safe_float(_extract_by_keywords(rec, ["本益比"]))
    pb = _safe_float(_extract_by_keywords(rec, ["股價淨值比", "淨值比"]))
    dy = _safe_float(_extract_by_keywords(rec, ["殖利率"]))

    return {"pe": pe, "pb": pb, "dy": dy}


def _bwibbu_with_backoff(stock_no: str, start_dt: datetime, max_back_days: int = 10):
    """
    先 OpenAPI（最新）→ 再回退舊 API（最多往前 N 天）
    """
    # 1) 最新（不吃 date）
    got = _bwibbu_latest_openapi(stock_no)
    if got and any(got.get(k) is not None for k in ("pe", "pb", "dy")):
        return got

    # 2) 需要 date 的回退（避免當天尚未更新）
    for i in range(max_back_days + 1):
        dt = start_dt - timedelta(days=i)
        ymd = dt.strftime("%Y%m%d")
        try:
            d = _bwibbu_one_day_twse_rwd(stock_no, ymd)
            if d and any(d.get(k) is not None for k in ("pe", "pb", "dy")):
                return d
        except Exception:
            continue
    return None


def _mops_monthly_revenue_latest(stock_no: str):
    """
    MOPS opendata 月營收：
    上市：t187ap05_L.csv
    上櫃：t187ap05_O.csv
    """
    urls = [
        "http://mopsfin.twse.com.tw/opendata/t187ap05_L.csv",
        "http://mopsfin.twse.com.tw/opendata/t187ap05_O.csv",
    ]

    for url in urls:
        try:
            r = _http_get(url, timeout=30)
            text = _decode_bytes(r.content)
            df = pd.read_csv(io.StringIO(text), dtype=str)
            if df.empty:
                continue

            # 兼容欄位名稱變動/編碼
            code_col = next((c for c in df.columns if "公司代號" in str(c) or "代號" in str(c)), None)
            ym_col = next((c for c in df.columns if "資料年月" in str(c) or "年月" in str(c)), None)
            yoy_col = next((c for c in df.columns if "去年同月增減" in str(c) or "年增" in str(c)), None)

            if not code_col:
                continue

            hit = df[df[code_col].astype(str).str.strip() == stock_no]
            if hit.empty:
                continue

            rec = hit.iloc[0].to_dict()
            ym = rec.get(ym_col) if ym_col else None
            yoy = _safe_float(rec.get(yoy_col)) if yoy_col else None
            return {"ym": ym, "yoy": yoy}
        except Exception:
            continue

    return None


def analyze_fundamental(stock_id: str) -> str:
    stock_id = (stock_id or "").strip().upper()
    if not stock_id:
        return "請輸入股票代號"

    yf_ticker = _resolve_yf_ticker(stock_id)
    code_only = stock_id.replace(".TW", "").replace(".TWO", "")

    out = []
    out.append("==================================================")
    out.append(f"🏢 {stock_id} 基本面分析")
    out.append("==================================================")

    df = yf.download(yf_ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return "\n".join(out + ["⚠️ 抓不到價格資料（Yahoo Finance 無數據或連線失敗）"])

    latest_dt = pd.to_datetime(df.index[-1]).to_pydatetime()
    out.append(f"📅 價格基準日: {latest_dt.strftime('%Y-%m-%d')}")
    out.append("")

    close_now = _safe_float(df["Close"].iloc[-1])
    one_year_dt = _nearest_trading_date(df, latest_dt - timedelta(days=365))
    close_1y = _safe_float(df.loc[pd.to_datetime(one_year_dt), "Close"]) if one_year_dt else None

    pe = pb = dy = None
    eps_yoy = None

    # ---- 估值（台股個股才有）
    if code_only.isdigit():
        bw_now = _bwibbu_with_backoff(code_only, latest_dt, max_back_days=10)
        bw_1y = _bwibbu_with_backoff(code_only, one_year_dt, max_back_days=10) if one_year_dt else None

        if bw_now:
            pe, pb, dy = bw_now.get("pe"), bw_now.get("pb"), bw_now.get("dy")

        eps_now = (close_now / pe) if (close_now is not None and pe not in (None, 0)) else None
        eps_1y = (close_1y / bw_1y["pe"]) if (close_1y is not None and bw_1y and bw_1y.get("pe") not in (None, 0)) else None
        if eps_now is not None and eps_1y not in (None, 0):
            eps_yoy = (eps_now / eps_1y - 1) * 100

    # ---- 月營收 YoY
    rev_yoy = None
    rev_ym = None
    if code_only.isdigit():
        rev = _mops_monthly_revenue_latest(code_only)
        if rev:
            rev_yoy = rev.get("yoy")
            rev_ym = rev.get("ym")

    # ---- yfinance info：毛利率/ROE
    gm = roe = None
    try:
        info = yf.Ticker(yf_ticker).info or {}
        gm = _safe_float(info.get("grossMargins"))
        roe = _safe_float(info.get("returnOnEquity"))
    except Exception:
        pass

    # ---- PE 河流（近 36 月：用月頻回推、每次用 backoff）
    pe_samples = []
    if code_only.isdigit():
        try:
            for m in range(36):
                target = latest_dt - timedelta(days=30 * m)
                tdt = _nearest_trading_date(df, target)
                if not tdt:
                    continue
                bw = _bwibbu_with_backoff(code_only, tdt, max_back_days=10)
                if bw and bw.get("pe") not in (None, 0):
                    pe_samples.append(float(bw["pe"]))
        except Exception:
            pe_samples = []

    def pct(x):
        return "n/a" if x is None else f"{x:.2f}%"

    def num(x):
        return "n/a" if x is None else f"{x:.2f}"

    out.append("1) 成長性指標 (Growth) - 股價上漲的燃料")
    if rev_yoy is not None:
        out.append(f"• 月營收年增率 (Revenue YoY%): {rev_yoy:+.2f}%  (資料年月: {rev_ym or 'n/a'})")
    else:
        out.append("• 月營收年增率 (Revenue YoY%): n/a（可能資料源暫時取用失敗或代號不在檔內）")

    if eps_yoy is not None:
        out.append(f"• EPS 成長率（以 PE+股價推估）: {eps_yoy:+.2f}%")
    else:
        out.append("• EPS 成長率: n/a（需要近一年 PE/價格都可取得）")

    out.append("")
    out.append("2) 獲利能力指標 (Profitability) - 公司的護城河")
    out.append(f"• 毛利率 (Gross Margin): {pct(None if gm is None else gm * 100)}")
    out.append(f"• ROE: {pct(None if roe is None else roe * 100)}")

    out.append("")
    out.append("3) 估值指標 (Valuation) - 買得便不便宜")
    out.append(f"• 本益比 (PE): {num(pe)}")
    out.append(f"• 股價淨值比 (PB): {num(pb)}")

    if pe_samples and pe is not None:
        s = pd.Series(pe_samples)
        q25, q50, q75 = s.quantile([0.25, 0.5, 0.75]).tolist()
        pct_rank = (s.lt(pe).sum() / len(s)) * 100
        out.append(
            "• 本益比河流圖（文字版，近36月月頻抽樣）: "
            f"min={s.min():.2f}, Q1={q25:.2f}, median={q50:.2f}, Q3={q75:.2f}, max={s.max():.2f}"
        )
        out.append(f"• 目前 PE 位於樣本約 {pct_rank:.1f} 百分位（越高代表越貴）")
    else:
        out.append("• 本益比河流圖: n/a（歷史 PE 抓不到足夠樣本）")

    out.append("")
    out.append("4) 存股防禦指標 (Safety)")
    out.append(f"• 現金殖利率 (Dividend Yield): {pct(dy)}")
    out.append("==================================================")
    return "\n".join(out)
