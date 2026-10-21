"""
Free price/volume data for the local backtester (yfinance), cached on disk.

Ticker formats
  TQX US    AAPL.NB   -> Yahoo AAPL      (".NB" stripped; share classes BRK.B -> BRK-B)
  TQX HK    0700.HK   -> Yahoo 0700.HK   (4-digit code kept; 00700.HK is normalised to 0700.HK)
  Plain tickers (AAPL, SPY, 0700.HK) pass through unchanged.

Prices are split/dividend ADJUSTED (auto_adjust=True). TQX may serve unadjusted bars,
so results are not expected to match platform runs to the cent.
"""
import json
from pathlib import Path

import pandas as pd
import yfinance as yf

CACHE = Path(__file__).resolve().parent / "cache"
FIELDS = ["Open", "High", "Low", "Close", "Volume"]

# Historical S&P 500 lists use the ticker valid ON THAT DATE; Yahoo files the whole
# history under the CURRENT ticker. Each alias below was checked (2026-09-23) to return
# full 2022-2026 history under the new symbol.
YF_ALIAS = {
    "ABC": "COR",     # AmerisourceBergen -> Cencora
    "ANTM": "ELV",    # Anthem -> Elevance Health
    "BK": "BNY",      # Bank of New York Mellon
    "FBHS": "FBIN",   # Fortune Brands
    "FI": "FISV",     # Fiserv
    "FLT": "CPAY",    # Fleetcor -> Corpay
    "MMC": "MRSH",    # Marsh McLennan
    "NLOK": "GEN",    # NortonLifeLock -> Gen Digital
    "PEAK": "DOC",    # Healthpeak
    "PKI": "RVTY",    # PerkinElmer -> Revvity
    "RE": "EG",       # Everest Re -> Everest Group
    "WRK": "SW",      # WestRock -> Smurfit WestRock (Yahoo carries WRK history on SW)
}


def to_yf(symbol):
    s = str(symbol).strip().upper()
    if s.endswith(".HK"):
        code = s[:-3]
        return f"{int(code):04d}.HK" if code.isdigit() else s
    if s.endswith(".NB"):
        s = s[:-3]
    s = YF_ALIAS.get(s, s)
    return s.replace(".", "-")


def _ymd(d):
    return pd.Timestamp(str(d)).strftime("%Y-%m-%d")


def _meta_path():
    return CACHE / "_ranges.json"


def _read_meta():
    p = _meta_path()
    return json.loads(p.read_text()) if p.exists() else {}


def load_prices(tickers, start, end):
    """{ticker: DataFrame[Open, High, Low, Close, Volume]} indexed by int YYYYMMDD.

    Reuses the on-disk cache when it already covers [start, end]; otherwise downloads
    the missing tickers in one batch. Tickers Yahoo has no data for are left out and
    reported, never silently filled.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    start, end = _ymd(start), _ymd(end)
    meta = _read_meta()
    tickers = sorted({to_yf(t) for t in tickers})

    stale = [t for t in tickers
             if t not in meta or meta[t][0] > start or meta[t][1] < end
             or not (CACHE / f"{t}.csv").exists()]
    if stale:
        # yfinance's end is exclusive -> ask for one extra day
        raw = yf.download(stale, start=start, end=_ymd(pd.Timestamp(end) + pd.Timedelta(days=1)),
                          auto_adjust=True, progress=False, group_by="ticker", threads=True)
        for t in stale:
            if isinstance(raw.columns, pd.MultiIndex):
                if t not in raw.columns.get_level_values(0):
                    continue
                f = raw[t]
            else:
                f = raw
            f = f[[c for c in FIELDS if c in f.columns]].dropna(how="all")
            if f.empty:
                continue
            f.to_csv(CACHE / f"{t}.csv")
            meta[t] = [start, end]
        _meta_path().write_text(json.dumps(meta, indent=1))

    out, missing = {}, []
    for t in tickers:
        p = CACHE / f"{t}.csv"
        if t not in meta or not p.exists():
            missing.append(t)
            continue
        f = pd.read_csv(p, index_col=0, parse_dates=True)
        f = f.loc[(f.index >= start) & (f.index <= end)]
        if f.empty:
            missing.append(t)
            continue
        f.index = f.index.strftime("%Y%m%d").astype(int)
        out[t] = f
    if missing:
        print(f"[data] no Yahoo data for {len(missing)} ticker(s): {', '.join(missing[:15])}"
              f"{' ...' if len(missing) > 15 else ''}")
    return out
