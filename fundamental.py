# -*- coding: utf-8 -*-
import io
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf

# Optional: helps some SSL envs on Streamlit Cloud
try:
    import truststore
    truststore.inject_into_ssl()
except Exception:
    pass


def _resolve_yf_ticker(stock_id: str) -> str:
    s = (stock_id or "").strip().upper()
    if s.endswith(".TW") or s.endswith(".TWO"):
        return s
    # 台股預設 TW
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
        if s in ("", "--", "N/A", "n/a", "None", "nan", "NaN"):
            return None
        return float(s)
    except Exception:
        return None


def _http_get_json(url: str, timeout: int = 20):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; stock-analyze/1.0)",
        "Accept": "application/json,text/plain,*/*",
        "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    }
    r = requests.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.json()


def _bwibbu_one_day(stock_no: str, yyyymmdd: str):
    """
    TWSE: 個股本益比、殖利率及股價淨值比（BWIBBU_d）
    回傳 pe/pb/dy
    """
    url = (
        "https://www.twse.com.tw/rwd/zh/afterTrading/BWIBBU_d"
        f"?date={yyyymmdd}&stockNo={stock_no}&response=json"
    )
    js = _http_get_json(url, timeout=20)
    fields = js.get("fields") or []
    data = js.get("data") or []
    if not data:
        return None

    row = data[0]
    rec = {fields[i]: row[i] for i in range(len(fields))} if isinstance(row, list) else row
    pe = _safe_float(rec.get("本益比"))
    pb = _safe_float(rec.get("股價淨值比"))
    dy = _safe_float(rec.get("殖利率(%)"))
    return {"pe": pe, "pb": pb, "dy": dy}


def _mops_monthly_revenue_latest(stock_no: str):
    """
    MOPS opendata: t187ap05_L.csv（上市公司月營收）
    回傳最近一筆：yoy, ym
    """
    url = "https://mopsfin.twse.com.tw/opendata/t187ap05_L.csv"
    r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=25)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), dtype=str)
    if df.empty:
        return None

    code_col = next((c for c in df.columns if "公司代號" in c), None)
    if not code_col:
        return None

    hit = df[df[code_col].astype(str).str.strip() == stock_no]
    if hit.empty:
        return None

    rec = hit.iloc[0].to_dict()
    ym_col = next((c for c in df.columns if "資料年月" in c), None)
    yoy_col = next((c for c in df.columns if "去年同月增減" in c or "年增率" in c), None)

    ym = rec.get(ym_col) if ym_col else None
    yoy = _safe_float(rec.get(yoy_col)) if yoy_col else None

    return {"ym": ym, "yoy": yoy}


def _nearest_trading_date(df: pd.DataFrame, dt: datetime):
    if df is None or df.empty:
        return None
    idx = pd.to_datetime(df.index)
    target = pd.to_datetime(dt)
    pos = (idx - target).abs().argmin()
    return idx[pos].to_pydatetime()


def analyze_fundamental(stock_id: str) -> str:
    """
    基本面文字輸出：
    1) Growth: 月營收YoY、EPS YoY（用 PE+價推估）
    2) Profitability: Gross Margin、ROE（yfinance info，缺就 n/a）
    3) Valuation: PE/PB、近 36 月 PE 分布（近似河流圖文字版）
    4) Safety: Dividend Yield
    """
    stock_id = (stock_id or "").strip().upper()
    if not stock_id:
        return "請輸入股票代號"

    yf_ticker = _resolve_yf_ticker(stock_id)
    code_only = stock_id.replace(".TW", "").replace(".TWO", "")

    out = []
    out.append("==================================================")
    out.append(f"🏢 {stock_id} 基本面分析")
    out.append("==================================================")

    # 價格資料
    df = yf.download(yf_ticker, period="2y", interval="1d", progress=False, auto_adjust=False)
    if df is None or df.empty:
        return "\n".join(out + ["⚠️ 抓不到價格資料（Yahoo Finance 無數據或連線失敗）"])

    latest_dt = pd.to_datetime(df.index[-1]).to_pydatetime()
    out.append(f"📅 價格基準日: {latest_dt.strftime('%Y-%m-%d')}")
    out.append("")

    close_now = _safe_float(df["Close"].iloc[-1])
    one_year_dt = _nearest_trading_date(df, latest_dt - timedelta(days=365))
    close_1y = _safe_float(df.loc[pd.to_datetime(one_year_dt), "Close"]) if one_year_dt else None

    # 估值/殖利率：BWIBBU（台股才有）
    pe = pb = dy = None
    eps_yoy = None
    if code_only.isdigit():
        bw_now = _bwibbu_one_day(code_only, latest_dt.strftime("%Y%m%d"))
        bw_1y = _bwibbu_one_day(code_only, one_year_dt.strftime("%Y%m%d")) if one_year_dt else None

        if bw_now:
            pe, pb, dy = bw_now.get("pe"), bw_now.get("pb"), bw_now.get("dy")

        # EPS YoY（推估）：eps = price / pe
        eps_now = (close_now / pe) if (close_now is not None and pe not in (None, 0)) else None
        eps_1y = (close_1y / bw_1y["pe"]) if (close_1y is not None and bw_1y and bw_1y.get("pe") not in (None, 0)) else None
        if eps_now is not None and eps_1y not in (None, 0):
            eps_yoy = (eps_now / eps_1y - 1) * 100

    # 月營收 YoY（台股公司才有；ETF 通常沒有）
    rev_yoy = None
    rev_ym = None
    if code_only.isdigit():
        rev = _mops_monthly_revenue_latest(code_only)
        if rev:
            rev_yoy = rev.get("yoy")
            rev_ym = rev.get("ym")

    # yfinance info：毛利率/ROE
    gm = roe = None
    try:
        info = yf.Ticker(yf_ticker).info or {}
        gm = _safe_float(info.get("grossMargins"))
        roe = _safe_float(info.get("returnOnEquity"))
    except Exception:
        pass

    # 歷史 PE（近 36 月月頻抽樣）
    pe_samples = []
    if code_only.isdigit():
        try:
            for m in range(36):
                target = latest_dt - timedelta(days=30 * m)
                tdt = _nearest_trading_date(df, target)
                if not tdt:
                    continue
                bw = _bwibbu_one_day(code_only, tdt.strftime("%Y%m%d"))
                if bw and bw.get("pe") not in (None, 0):
                    pe_samples.append(float(bw["pe"]))
        except Exception:
            pe_samples = []

    def pct(x):
        return "n/a" if x is None else f"{x:.2f}%"

    def num(x):
        return "n/a" if x is None else f"{x:.2f}"

    # ---- 1) Growth
    out.append("1) 成長性指標 (Growth) - 股價上漲的燃料")
    if rev_yoy is not None:
        out.append(f"• 月營收年增率 (Revenue YoY%): {rev_yoy:+.2f}%  (資料年月: {rev_ym or 'n/a'})")
    else:
        out.append("• 月營收年增率 (Revenue YoY%): n/a（ETF/或資料源未涵蓋）")

    if eps_yoy is not None:
        out.append(f"• EPS 成長率（以 PE+股價推估）: {eps_yoy:+.2f}%")
    else:
        out.append("• EPS 成長率: n/a（需要近一年 PE/價格都可取得）")

    out.append("")

    # ---- 2) Profitability
    out.append("2) 獲利能力指標 (Profitability) - 公司的護城河")
    out.append(f"• 毛利率 (Gross Margin): {pct(None if gm is None else gm * 100)}")
    out.append(f"• ROE: {pct(None if roe is None else roe * 100)}")
    out.append("")

    # ---- 3) Valuation
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
        out.append("• 本益比河流圖: n/a（非台股/或 BWIBBU 抓不到足夠歷史）")

    out.append("")

    # ---- 4) Safety
    out.append("4) 存股防禦指標 (Safety)")
    out.append(f"• 現金殖利率 (Dividend Yield): {pct(dy)}")

    out.append("==================================================")
    return "\n".join(out)
