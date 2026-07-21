"""
Minimal yfinance shim backed by Yahoo chart API via requests.

Reason: in this remote environment yfinance's curl_cffi transport fails TLS
through the agent proxy, but `requests` (which honors REQUESTS_CA_BUNDLE) works.
This shim provides just enough surface (yf.Ticker(sym).history(...)) for
backtesting/data_loader.py to run unchanged.

Installed via:  sys.modules['yfinance'] = yf_shim
"""
import os
import time
import calendar
from datetime import datetime, timedelta, date

import pandas as pd
import requests

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
})

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "yahoo_cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


def _to_unix(d):
    if isinstance(d, datetime):
        dt = d
    elif isinstance(d, date):
        dt = datetime(d.year, d.month, d.day)
    else:
        dt = pd.Timestamp(d).to_pydatetime()
    return int(calendar.timegm(dt.timetuple()))


def _fetch_chart(symbol, period1, period2):
    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {
        "period1": period1,
        "period2": period2,
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
    }
    last = None
    for attempt in range(6):
        try:
            r = _SESSION.get(url, params=params, timeout=25)
            if r.status_code == 200:
                return r.json()
            last = f"HTTP {r.status_code}"
        except Exception as e:  # noqa: BLE001
            last = str(e)
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Yahoo fetch failed for {symbol}: {last}")


def _json_to_df(js, auto_adjust):
    res = js.get("chart", {}).get("result")
    if not res:
        return pd.DataFrame()
    r = res[0]
    ts = r.get("timestamp") or []
    q = r["indicators"]["quote"][0]
    opens = q.get("open", [])
    highs = q.get("high", [])
    lows = q.get("low", [])
    closes = q.get("close", [])
    vols = q.get("volume", [])
    adj = None
    if "adjclose" in r["indicators"]:
        adj = r["indicators"]["adjclose"][0].get("adjclose")
    rows = []
    idx = []
    for i, t in enumerate(ts):
        c = closes[i] if i < len(closes) else None
        if c is None:
            continue
        o = opens[i] if i < len(opens) and opens[i] is not None else c
        h = highs[i] if i < len(highs) and highs[i] is not None else c
        low = lows[i] if i < len(lows) and lows[i] is not None else c
        v = vols[i] if i < len(vols) and vols[i] is not None else 0
        if auto_adjust and adj is not None and i < len(adj) and adj[i] and c:
            factor = adj[i] / c
            o, h, low, c = o * factor, h * factor, low * factor, adj[i]
        idx.append(pd.Timestamp(t, unit="s"))
        rows.append((o, h, low, c, v))
    df = pd.DataFrame(rows, index=pd.DatetimeIndex(idx),
                      columns=["Open", "High", "Low", "Close", "Volume"])
    return df


class Ticker:
    def __init__(self, symbol):
        self.symbol = symbol

    def history(self, start=None, end=None, period=None, auto_adjust=True, **kwargs):
        if start is None:
            start = date.today() - timedelta(days=365)
        if end is None:
            end = date.today() + timedelta(days=1)
        p1, p2 = _to_unix(start), _to_unix(end)
        safe = self.symbol.replace("^", "IDX_").replace("/", "_")
        cache = os.path.join(_CACHE_DIR, f"{safe}_{p1}_{p2}_{int(auto_adjust)}.parquet")
        if os.path.exists(cache):
            return pd.read_parquet(cache)
        js = _fetch_chart(self.symbol, p1, p2)
        df = _json_to_df(js, auto_adjust)
        if len(df):
            df.to_parquet(cache)
        time.sleep(0.4)  # be polite to Yahoo
        return df


def download(symbols, start=None, end=None, auto_adjust=True, progress=False, **kwargs):
    if isinstance(symbols, str):
        symbols = [symbols]
    frames = {}
    for s in symbols:
        frames[s] = Ticker(s).history(start=start, end=end, auto_adjust=auto_adjust)
    if len(symbols) == 1:
        return frames[symbols[0]]
    return pd.concat(frames, axis=1)
