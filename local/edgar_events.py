#!/usr/bin/env python3
"""
Earnings-release dates from SEC EDGAR (free): 8-K filings carrying Item 2.02
("Results of Operations and Financial Condition"), for every ticker that was in the
universe, written in the format local/backtest.py --events expects.

  set SEC_USER_AGENT="Your Name your@email.com"      (SEC requires a contact; no default)
  python local/edgar_events.py --universe local/ref/sp500_components_by_date.csv
         --start 2022-10-01 --end 2026-09-30 --out local/ref/events_sec_8k_202.csv

KNOWN DATE = the filing's EDGAR acceptance timestamp, converted to the first US session
on which the market could act on it:
  accepted before 16:00 ET on a trading day -> that day
  accepted at/after 16:00 ET, or on a non-trading day -> the next trading day
event_date is set to the same session (the release IS the event), so the strategy's -1
leg never fires on these -- a filing isn't known the day before it's filed.

Delisted companies are usually absent from SEC's current ticker->CIK map, so they get no
events; the script reports how many universe tickers it could not map.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as dl  # noqa: E402
from backtest import Membership  # noqa: E402

ET = ZoneInfo("America/New_York")
MIN_GAP = 0.12            # seconds between requests: SEC allows <= 10/s


class Sec:
    def __init__(self, ua):
        self.s = requests.Session()
        self.s.headers.update({"User-Agent": ua, "Accept-Encoding": "gzip, deflate"})
        self.last = 0.0

    def get(self, url):
        wait = MIN_GAP - (time.time() - self.last)
        if wait > 0:
            time.sleep(wait)
        self.last = time.time()
        r = self.s.get(url, timeout=30)
        if r.status_code == 403:
            raise SystemExit("SEC returned 403 -- the User-Agent must include a contact "
                             "(name + email); see https://www.sec.gov/os/accessing-edgar-data")
        r.raise_for_status()
        return r.json()


def acceptance_et(raw):
    """EDGAR's acceptanceDateTime ("2024-02-01T21:30:30.000Z") is genuine UTC.
    Verified 2026-09-23 against known release times: AAPL's 16:30 ET after-close
    releases show 20:30Z (EDT) / 21:30Z (EST); JPM's ~06:45 ET pre-market ones 10:46Z."""
    return pd.Timestamp(raw).tz_convert(ET)


def session_for(ts, sessions):
    """First trading session on which a filing accepted at `ts` (ET) is actionable."""
    d = int(ts.strftime("%Y%m%d"))
    after_close = ts.hour >= 16
    later = [s for s in sessions if s > d or (s == d and not after_close)]
    if later:
        return min(later)
    # beyond the cached calendar (future): next weekday as a fallback
    nxt = ts.normalize() + pd.Timedelta(days=1 if after_close else 0)
    while nxt.weekday() >= 5:
        nxt += pd.Timedelta(days=1)
    return int(nxt.strftime("%Y%m%d"))


def filings(sec, cik, start, end):
    """All (form, items, acceptanceDateTime, accession) for a CIK between start/end."""
    j = sec.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json")
    blocks = [j["filings"]["recent"]]
    for f in j["filings"].get("files", []):
        if f.get("filingTo", "9999") >= start:
            blocks.append(sec.get(f"https://data.sec.gov/submissions/{f['name']}"))
    for b in blocks:
        for form, items, acc, fd, an in zip(b["form"], b["items"], b["acceptanceDateTime"],
                                            b["filingDate"], b["accessionNumber"]):
            if start <= fd <= end:
                yield form, items or "", acc, fd, an


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default=str(HERE / "ref" / "sp500_components_by_date.csv"))
    ap.add_argument("--start", default="2022-10-01")
    ap.add_argument("--end", default="2026-09-30")
    ap.add_argument("--out", default=str(HERE / "ref" / "events_sec_8k_202.csv"))
    ap.add_argument("--user-agent", default=os.environ.get("SEC_USER_AGENT"))
    a = ap.parse_args()
    if not a.user_agent:
        raise SystemExit("set SEC_USER_AGENT or pass --user-agent 'Name email' (SEC requires it)")

    sec = Sec(a.user_agent)
    tick = sec.get("https://www.sec.gov/files/company_tickers.json")
    cik_of = {}
    for v in tick.values():
        cik_of.setdefault(v["ticker"].upper(), int(v["cik_str"]))

    m = Membership(a.universe)
    lo = int(pd.Timestamp(a.start).strftime("%Y%m%d"))
    hi = int(pd.Timestamp(a.end).strftime("%Y%m%d"))
    yf_tickers = m.all_between(lo, hi)                       # already alias-mapped
    sessions = dl.load_prices(["SPY"], a.start, a.end)["SPY"].index.tolist()

    rows, unmapped = [], []
    for k, t in enumerate(yf_tickers, 1):
        cik = cik_of.get(t) or cik_of.get(t.replace("-", ".")) or cik_of.get(t.replace("-", ""))
        if cik is None:
            unmapped.append(t)
            continue
        for form, items, acc, fd, an in filings(sec, cik, a.start, a.end):
            if form != "8-K" or "2.02" not in items.split(","):
                continue
            ts = acceptance_et(acc)
            s = session_for(ts, sessions)
            rows.append({"symbol": t, "event_date": s, "known_date": s,
                         "source": "sec_8k_item_2.02", "period": "",
                         "acceptance_et": ts.strftime("%Y-%m-%d %H:%M"), "filing_date": fd,
                         "cik": cik, "accession": an})
        if k % 50 == 0:
            print(f"[edgar] {k}/{len(yf_tickers)} tickers, {len(rows)} filings", flush=True)

    df = pd.DataFrame(rows).drop_duplicates(["symbol", "event_date"]).sort_values(["symbol", "event_date"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    per_year = df.assign(y=df["event_date"] // 10000).groupby(["symbol", "y"]).size()
    print(json.dumps({"tickers": len(yf_tickers), "mapped": len(yf_tickers) - len(unmapped),
                      "unmapped": unmapped, "filings": len(df),
                      "symbols_with_filings": int(df["symbol"].nunique()),
                      "median_filings_per_symbol_year": float(per_year.median()),
                      "out": a.out}, indent=2))


if __name__ == "__main__":
    main()
