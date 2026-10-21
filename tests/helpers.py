"""Shared test helpers. Every test runs offline on small synthetic price series."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

import backtest as bt  # noqa: E402  (also stubs the TQX-only imports of strategy.py)

STRAT = bt.STRAT


def trading_days(n, start="2024-01-02"):
    return [int(d.strftime("%Y%m%d")) for d in pd.bdate_range(start, periods=n)]


def trending_prices(days, drifts, volume=1_000_000.0, start=100.0):
    """{ticker: OHLCV frame} where ticker i compounds at drifts[i] per day."""
    out = {}
    for t, mu in drifts.items():
        close = start * (1 + mu) ** np.arange(len(days))
        out[t] = pd.DataFrame({"Open": close, "High": close * 1.01, "Low": close * 0.99,
                               "Close": close, "Volume": volume}, index=days)
    return out


def membership_csv(tmp_path, rows):
    """rows: [(yyyymmdd, [tickers])] -> path of a date,tickers membership CSV."""
    p = tmp_path / "members.csv"
    pd.DataFrame({"date": [pd.Timestamp(str(d)).strftime("%Y-%m-%d") for d, _ in rows],
                  "tickers": [",".join(t) for _, t in rows]}).to_csv(p, index=False)
    return p


def strategy_context(hist, universe=None, days=None, events=None):
    from types import SimpleNamespace
    days = days or []
    return SimpleNamespace(universe=universe or list(hist), hist=hist, events=events or {},
                           calendar=days, cal_idx={d: i for i, d in enumerate(days)},
                           cat_total_hits=0, cat_minus1_hits=0)
