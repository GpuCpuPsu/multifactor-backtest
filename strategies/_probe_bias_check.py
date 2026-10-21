"""Quantify the survivorship gap and validate a rules-based, PIT-safe universe."""
from panda_backtest.api.api import *
from panda_backtest.api.stock_us_api import *
import tqx_data
import pandas as pd


def initialize(context):
    # Full column list of get_us_daily -- looking for a size/market-cap field
    try:
        d = tqx_data.get_us_daily(symbol=[], start_date="20250102", end_date="20250102",
                                  market="nb")
        print(f"[D] get_us_daily cols({len(d.columns)}) = {list(d.columns)}")
    except Exception as exc:
        print(f"[D] FAILED: {exc}")
        return

    # Today's ACTIVE tickers, per stock_detail
    try:
        det = tqx_data.get_stock_detail(market="us", fields=["symbol", "status", "asset_state"])
        active_now = set(det[det["status"] == 1]["symbol"].dropna().tolist())
        print(f"[E] stock_detail active(status=1)={len(active_now)} total={len(det)}")
        if "asset_state" in det.columns:
            print(f"[E] asset_state counts={det['asset_state'].value_counts().head(6).to_dict()}")
    except Exception as exc:
        print(f"[E] FAILED: {exc}")
        return

    # How many names trading in Jan-2025 are NOT in today's active list?
    traded_2025 = set(d["symbol"].dropna().tolist())
    gone = traded_2025 - active_now
    print(f"[F] symbols trading 20250102={len(traded_2025)}  "
          f"not active today={len(gone)}  ({len(gone)/len(traded_2025)*100:.1f}%)")

    # Now the number that matters: restrict to the LIQUID top-100 as of Jan-2025
    dd = d.dropna(subset=["amount"]).copy()
    dd["amount"] = pd.to_numeric(dd["amount"], errors="coerce")
    dd = dd.dropna(subset=["amount"]).sort_values("amount", ascending=False)
    top100 = dd.head(100)["symbol"].tolist()
    top100_gone = [s for s in top100 if s not in active_now]
    print(f"[G] top-100 by dollar volume on 20250102: {len(top100_gone)} of 100 "
          f"are NOT active today -> {top100_gone[:15]}")

    # Overlap between that liquidity universe and today's NDX list
    try:
        ndx = tqx_data.get_index_component(index_symbol="NDX", market="us")
        ndx_syms = set(ndx["stock_symbol"].dropna().tolist())
        ov = len(set(top100) & ndx_syms)
        print(f"[H] overlap top-100-by-$vol vs today's NDX list = {ov}/100")
    except Exception as exc:
        print(f"[H] FAILED: {exc}")

    print(f"[I] sample top-15 by $vol 20250102: {top100[:15]}")


def handle_data(context, data):
    pass
