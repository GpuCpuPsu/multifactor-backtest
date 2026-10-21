"""Why is the catalyst gate letting through ~1 name instead of ~8?"""
from panda_backtest.api.api import *
from panda_backtest.api.stock_us_api import *
import tqx_data
import pandas as pd
import datetime


def _shift(d, days):
    dt = datetime.date(int(d[:4]), int(d[4:6]), int(d[6:8]))
    return (dt - datetime.timedelta(days=days)).strftime("%Y%m%d")


def initialize(context):
    as_of = "20250801"

    # rebuild the same top-100 universe the pipeline uses
    df = tqx_data.get_us_daily(symbol=[], start_date=_shift(as_of, 8),
                               end_date=_shift(as_of, 1), market="nb")
    df = df[df["date"] == df["date"].max()].copy()
    for c in ("amount", "volume", "close", "high", "low"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["symbol", "amount", "volume", "close"])
    df = df[(df["volume"] > 0) & (df["close"] > 0) & (df["amount"] > 0) & (df["high"] > df["low"])]
    uni = [s for s in df.sort_values("amount", ascending=False).head(100)["symbol"].tolist()
           if s.endswith(".NB")]
    print(f"[U] universe={len(uni)} sample={uni[:5]}")

    # 1) does the event endpoint cope with 100 symbols at once?
    ev = tqx_data.get_event_fina_disclosure(
        symbol=uni, market="us", start_date=_shift(as_of, 400), end_date=_shift(as_of, -120))
    print(f"[V] event rows for 100 syms = {0 if ev is None else len(ev)}")
    if ev is not None and len(ev) > 0:
        rel = ev[ev["event_type"] == "EarningsReleases"]
        print(f"[V] EarningsReleases={len(rel)} distinct_symbols={rel['symbol'].nunique()}")

    # 2) compare against querying in small batches
    total, syms_seen = 0, set()
    for i in range(0, len(uni), 20):
        chunk = uni[i:i + 20]
        try:
            e2 = tqx_data.get_event_fina_disclosure(
                symbol=chunk, market="us",
                start_date=_shift(as_of, 400), end_date=_shift(as_of, -120))
        except Exception as exc:
            print(f"[W] chunk {i} failed: {exc}")
            continue
        if e2 is None or len(e2) == 0:
            continue
        r2 = e2[e2["event_type"] == "EarningsReleases"]
        total += len(r2)
        syms_seen |= set(r2["symbol"].tolist())
    print(f"[W] BATCHED EarningsReleases={total} distinct_symbols={len(syms_seen)}")

    # 3) trading calendar sanity
    cal = tqx_data.get_trading_calendar(start_date=_shift(as_of, 400),
                                        end_date=_shift(as_of, -120),
                                        market="us", is_trading_day=1)
    print(f"[X] calendar rows={0 if cal is None else len(cal)} "
          f"cols={list(cal.columns) if cal is not None else None}")
    if cal is not None and len(cal) > 0:
        col = "date" if "date" in cal.columns else cal.columns[0]
        days = sorted(set(int(x) for x in pd.to_numeric(cal[col], errors="coerce").dropna()))
        print(f"[X] parsed_days={len(days)} first={days[0]} last={days[-1]} "
              f"contains_{as_of}={int(as_of) in days}")

        # 4) with the PIT gate applied, how many names are in-window on this date?
        if ev is not None and len(ev) > 0:
            rel = ev[ev["event_type"] == "EarningsReleases"].copy()
            rel["_ev"] = pd.to_numeric(rel["start_date"], errors="coerce")
            rel["_info"] = pd.to_numeric(rel["info_date"], errors="coerce")
            rel = rel.dropna(subset=["_ev", "_info"])
            visible = rel[rel["_info"] <= int(as_of)]
            print(f"[Y] events visible as of {as_of}: {len(visible)} "
                  f"(of {len(rel)} total)")
            t_idx = days.index(int(as_of))
            hits = []
            for sym, g in visible.groupby("symbol"):
                for e in g["_ev"].tolist():
                    if int(e) not in days:
                        continue
                    d = t_idx - days.index(int(e))
                    if -1 <= d <= 3:
                        hits.append((sym, int(e), d))
            print(f"[Y] IN-WINDOW on {as_of}: {len(hits)} -> {hits[:8]}")

            # how wide would the window need to be to reach ~8 names?
            for w in (3, 5, 10, 21, 42):
                n = 0
                for sym, g in visible.groupby("symbol"):
                    for e in g["_ev"].tolist():
                        if int(e) not in days:
                            continue
                        d = t_idx - days.index(int(e))
                        if -1 <= d <= w:
                            n += 1
                            break
                print(f"[Z] window [-1,+{w}] would gate in {n} names")


def handle_data(context, data):
    pass
