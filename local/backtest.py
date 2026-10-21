#!/usr/bin/env python3
"""
Local backtester for pipeline/strategy.py -- free data, no tqx-cli, no platform credits.

  python local/backtest.py --start 20230101 --end 20241231 --grid "5:10:off,5:10:on"
         --universe local/universe_us.txt (--events events.csv | --no-catalyst)
         [--label IS] [--market us|hk] [--out local/results/sweep.jsonl]

Grid entries are top_n:rebalance_bars:stop(on|off). Every combo appends one labeled row
to --out, with the same metric keys run_pipeline.py writes (net_yield, sharpe, max_dd,
profit_factor, win_rate, gatc_*) plus engine="local".

WHAT IS SHARED WITH THE PLATFORM VERSION
  The signal layer is imported from pipeline/strategy.py, not copied: compute_scores(),
  _catalyst_flag(), _atr() and every constant (TREND_*, VOL_*, UNIVERSE_SIZE, ATR_MULT,
  MAX_DEPLOYED, ...). Change the strategy file and this engine follows.

WHAT THIS ENGINE REPLACES (and how it may differ from TQX)
  * Prices: yfinance, split/dividend adjusted (see local/data.py).
  * Fills: market orders at the bar's close, +/- slippage (default 0.001, as on TQX),
    optional proportional commission. Scoring uses the PRIOR close (strategy.py fix),
    so filling at today's close is not a same-bar peek.
  * Universe: top UNIVERSE_SIZE by dollar volume (close*volume -- Yahoo has no 'amount')
    on the last completed session, refreshed every UNIVERSE_REFRESH_BARS, ranked within
    --universe. Pass local/ref/sp500_components_by_date.csv for POINT-IN-TIME S&P 500
    membership (fja05680/sp500, rebuilt from Wikipedia's change log) -- each refresh
    ranks only the names that were members that day. Residual bias: Yahoo has no prices
    for ~30 names that left the index (acquired / delisted / failed, e.g. SIVB, FRC), so
    those can't be picked; rows report member_price_coverage (~96%). A .txt list is a
    static universe of today's names and is heavily survivorship-biased.
  * Delisting: a held name whose Yahoo history ends is closed at its last price.
  * --placebo N reruns each combo N times ranking the SAME catalyst-eligible names
    randomly: p_net_vs_placebo = share of random runs that matched the composite score.
  * Calendar: trading days of SPY (us) / 2800.HK (hk).

CATALYST DATES (--events CSV)
  Columns: symbol, event_date            required
           known_date                    optional; defaults to event_date
           source, period                optional; see Q4 rule below
  symbol accepts AAPL, AAPL.NB or 0700.HK. Dates: YYYYMMDD or YYYY-MM-DD.
  Point-in-time rule (unchanged from strategy.py): an event is visible on day T only if
  known_date <= T. Filing announcements are known when published, so for them
  known_date == event_date and the -1 (day-before) leg of the window never fires --
  that is correct, not a bug.
  Q4 rule: rows with source containing "financial_statement" whose period is Q4 / annual
  (Q4, ANNUAL, 10-K, or a bare FY2024) are DROPPED -- that date lags the release by ~20
  days. Filing-announcement rows are always kept.
"""
import argparse
import importlib.util
import json
import math
import random
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import data as dl  # noqa: E402

CAL_PROXY = {"us": "SPY", "hk": "2800.HK"}
WARMUP_DAYS = 220          # calendar days before --start: covers HISTORY_BARS + ATR
TAIL_DAYS = 10             # calendar days after --end so the catalyst window can see T+1


def _load_module(name, path, stubs=()):
    for s in stubs:
        sys.modules.setdefault(s, types.ModuleType(s))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Platform-only imports are stubbed; only the pure signal functions are used here.
STRAT = _load_module("tqx_strategy", ROOT / "pipeline" / "strategy.py",
                     stubs=("panda_backtest", "panda_backtest.api", "panda_backtest.api.api",
                            "panda_backtest.api.stock_us_api", "tqx_data"))
PIPE = _load_module("run_pipeline", ROOT / "pipeline" / "run_pipeline.py")


# ------------------------------------------------------------------ inputs
class Membership:
    """Candidate list per date. A .csv with columns date,tickers ("A,AAPL,...") is
    point-in-time index membership (e.g. local/ref/sp500_components_by_date.csv); a .txt
    is one static list for every date (survivorship-biased if it's today's names)."""

    def __init__(self, path):
        self.path = Path(path)
        if self.path.suffix.lower() == ".csv":
            df = pd.read_csv(self.path)
            df["d"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d").astype(int)
            df = df.sort_values("d")
            self.rows = [(d, sorted({dl.to_yf(t) for t in s.split(",") if t.strip()}))
                         for d, s in zip(df["d"], df["tickers"])]
            self.point_in_time = True
        else:
            out = []
            for line in self.path.read_text().splitlines():
                line = line.split("#", 1)[0].strip()
                if line:
                    out.append(dl.to_yf(line))
            self.rows = [(0, sorted(set(out)))]
            self.point_in_time = False

    def on(self, day):
        """Members per the latest list dated on or before `day`."""
        cur = self.rows[0][1]
        for d, members in self.rows:
            if d > day:
                break
            cur = members
        return cur

    def all_between(self, lo, hi):
        """Every ticker that was a member at any point in [lo, hi] -- what must be priced."""
        out = set(self.on(lo))
        for d, members in self.rows:
            if lo < d <= hi:
                out.update(members)
        return sorted(out)


_Q4 = r"Q4|ANNUAL|10-K|^FY\s*\d{4}$"


def load_events(path):
    df = pd.read_csv(path, dtype=str)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = {"symbol", "event_date"} - set(df.columns)
    if missing:
        raise SystemExit(f"--events {path}: missing column(s) {sorted(missing)}")
    if "known_date" not in df.columns:
        df["known_date"] = df["event_date"]
    df["known_date"] = df["known_date"].fillna(df["event_date"])

    n_in, n_q4 = len(df), 0
    if "source" in df.columns:
        src = df["source"].fillna("").str.lower()
        per = df["period"].fillna("").str.upper().str.strip() if "period" in df.columns \
            else pd.Series("", index=df.index)
        drop = src.str.contains("financial_statement") & per.str.contains(_Q4, regex=True)
        n_q4 = int(drop.sum())
        df = df[~drop]

    ev = pd.to_datetime(df["event_date"], format="mixed", errors="coerce")
    kn = pd.to_datetime(df["known_date"], format="mixed", errors="coerce")
    out = pd.DataFrame({"symbol": df["symbol"].map(dl.to_yf),
                        "ev": ev.dt.strftime("%Y%m%d"),
                        "known": kn.dt.strftime("%Y%m%d")}).dropna()
    out["ev"] = out["ev"].astype(int)
    out["known"] = out["known"].astype(int)
    print(f"[events] {path}: {n_in} rows, {n_q4} dropped by Q4/annual financial_statement "
          f"rule, {n_in - n_q4 - len(out)} unparseable, {len(out)} used "
          f"({out['symbol'].nunique()} symbols)")
    return out


def events_as_of(events, today, universe):
    """The strategy's point-in-time gate: only events with known_date <= today."""
    e = events[(events["known"] <= today) & (events["symbol"].isin(universe))]
    return {s: sorted(set(g["ev"].tolist())) for s, g in e.groupby("symbol")}


# ------------------------------------------------------------------ engine
class Panel:
    """Close/High/Low/Volume as calendar-aligned frames (rows = trading days)."""

    def __init__(self, prices, days):
        idx = pd.Index(days)
        pick = lambda f: pd.DataFrame({t: p[f] for t, p in prices.items()}).reindex(idx)
        self.C, self.H, self.L, self.V = (pick(f) for f in ("Close", "High", "Low", "Volume"))
        # same bar filter strategy.py applies when appending bars
        self.valid = (self.C > 0) & (self.H >= self.L) & (self.V >= 0) & \
            self.C.notna() & self.H.notna() & self.L.notna() & self.V.notna()

    def hist(self, sym, i):
        """Last HISTORY_BARS valid bars up to and including row i (oldest first)."""
        if sym not in self.C.columns:
            return None
        m = self.valid[sym].iloc[:i + 1]
        rows = m[m].index[-STRAT.HISTORY_BARS:]
        if len(rows) == 0:
            return None
        return {"close": self.C.loc[rows, sym].tolist(), "high": self.H.loc[rows, sym].tolist(),
                "low": self.L.loc[rows, sym].tolist(), "vol": self.V.loc[rows, sym].tolist()}

    def rank_universe(self, i, candidates):
        """Top UNIVERSE_SIZE by dollar volume on row i (the last COMPLETED session),
        among `candidates` (the index members on that date)."""
        cols = [t for t in candidates if t in self.C.columns]
        c, h, l, v = (x.iloc[i][cols] for x in (self.C, self.H, self.L, self.V))
        amt = c * v
        ok = (v > 0) & (c > 0) & (amt > 0) & (h > l)
        return amt[ok].sort_values(ascending=False).head(STRAT.UNIVERSE_SIZE).index.tolist()


def run(panel, days, start, end, top_n, rebal, stop_on, events, capital, slippage, commission,
        membership, coverage=None, placebo_seed=None):
    """placebo_seed: rank the SAME eligible set randomly instead of by composite score --
    a null distribution for "does the score pick better than chance?"."""
    S = STRAT
    rng = random.Random(placebo_seed) if placebo_seed is not None else None
    last_valid = {t: (int(np.flatnonzero(panel.valid[t].to_numpy())[-1])
                      if panel.valid[t].any() else -1) for t in panel.C.columns}
    ew_equity, ew_val = [], 1.0
    ctx = SimpleNamespace(universe=[], hist={}, events={}, calendar=days,
                          cal_idx={d: i for i, d in enumerate(days)},
                          cat_total_hits=0, cat_minus1_hits=0)
    lo = next(i for i, d in enumerate(days) if d >= start)
    hi = max(i for i, d in enumerate(days) if d <= end)

    cash, pos, stops, last_px = float(capital), {}, {}, {}
    trades, equity, eligible = [], [], []
    bar_count, last_refresh, n_stopped = 0, -999, 0

    def fill(sym, qty, px, side, today):
        nonlocal cash
        f = px * (1 + slippage) if side == 0 else px * (1 - slippage)
        gross = f * qty
        cash += (-gross * (1 + commission)) if side == 0 else (gross * (1 - commission))
        trades.append({"trade_date": str(today), "trade_id": len(trades) + 1,
                       "contract_code": sym, "price": f, "volume": qty, "direction": side})

    def sell_all(sym, px, today):
        fill(sym, pos.pop(sym), px, 1, today)
        stops.pop(sym, None)

    for i in range(lo, hi + 1):
        today = days[i]
        bar = bar_count + 1
        closes = panel.C.iloc[i]
        for sym, c in closes.items():
            if c == c and c > 0:
                last_px[sym] = c

        # ---- before_trading: universe refresh on the last completed session ----
        if bar - last_refresh >= S.UNIVERSE_REFRESH_BARS and i > 0:
            members = membership.on(days[i - 1])
            if coverage is not None:
                priced = sum(1 for t in members if t in panel.C.columns
                             and panel.valid[t].iloc[i - 1])
                coverage.append(priced / len(members) if members else 0.0)
            uni = panel.rank_universe(i - 1, members)
            if len(uni) >= S.MIN_UNIVERSE:
                ctx.universe, last_refresh = uni, bar
        rebal_day = bar % rebal == 0
        if rebal_day and ctx.universe and events is not None:
            ctx.events = events_as_of(events, today, ctx.universe)

        # ---- handle_data ----
        bar_count += 1

        # equal-weight buy-and-hold of the current universe: the "no skill" benchmark
        if ctx.universe and i > 0:
            r = (panel.C.iloc[i][ctx.universe] / panel.C.iloc[i - 1][ctx.universe] - 1).dropna()
            if len(r):
                ew_val *= 1 + float(r.mean())
        ew_equity.append(ew_val)

        # a held name whose price history has ended (delisted / acquired) is closed at
        # its last price rather than carried forever
        for sym in list(pos):
            if i > last_valid.get(sym, -1):
                sell_all(sym, last_px[sym], today)

        if stop_on:
            for sym in list(pos):
                c, st = closes.get(sym), stops.get(sym)
                if st is not None and c == c and c is not None and 0 < c <= st:
                    sell_all(sym, c, today)
                    n_stopped += 1

        if rebal_day and ctx.universe:
            ctx.hist = {s: h for s in ctx.universe if (h := panel.hist(s, i))}
            scores, _ = S.compute_scores(ctx, str(today))
            eligible.append(len(scores))
            if rng is not None:
                scores = {s: rng.random() for s in sorted(scores)}
            targets = sorted(scores, key=scores.get, reverse=True)[:top_n]

            for sym in list(pos):
                c = closes.get(sym)
                if sym not in targets and c == c and c is not None and c > 0:
                    sell_all(sym, c, today)

            total = cash + sum(q * last_px.get(s, 0.0) for s, q in pos.items())
            budget = total * S.MAX_DEPLOYED / top_n
            for sym in targets:
                c = closes.get(sym)
                if sym in pos or c is None or c != c or c <= 0:
                    continue
                qty = int(min(budget, cash) // (c * (1 + slippage) * (1 + commission)))
                if qty <= 0:
                    continue
                fill(sym, qty, c, 0, today)
                pos[sym] = qty
                if stop_on:
                    atr = S._atr(ctx.hist[sym])
                    if atr is not None:
                        stops[sym] = c - S.ATR_MULT * atr

        equity.append(cash + sum(q * last_px.get(s, 0.0) for s, q in pos.items()))

    return {"equity": equity, "trades": trades, "eligible": eligible, "stops": n_stopped,
            "cat_hits": ctx.cat_total_hits, "cat_pre": ctx.cat_minus1_hits,
            "first_day": days[lo], "last_day": days[hi], "ew_return": ew_val - 1.0}


# ------------------------------------------------------------------ scoring
def metrics(res, capital):
    eq = pd.Series(res["equity"], dtype=float)
    net = eq.iloc[-1] / capital - 1.0
    years = len(eq) / 252.0
    ann = (1 + net) ** (1 / years) - 1 if years > 0 and net > -1 else -1.0
    rets = pd.concat([pd.Series([eq.iloc[0] / capital - 1]), eq.pct_change().dropna()])
    sd = rets.std()
    sharpe = rets.mean() / sd * math.sqrt(252) if sd > 0 else 0.0
    max_dd = float((eq / eq.cummax().clip(lower=capital) - 1).min())

    pnl = PIPE.round_trips(res["trades"])
    wins, losses = [p for p in pnl if p > 0], [p for p in pnl if p < 0]
    win_rate = len(wins) / len(pnl) if pnl else 0.0
    gl = abs(sum(losses))
    pf = (sum(wins) / gl) if gl > 1e-9 else (0.0 if not wins else 10.0)
    calmar = ann / abs(max_dd) if abs(max_dd) > 1e-9 else 0.0
    lit, inner, norm = PIPE.gatc(net, pf, win_rate, sharpe, max_dd, calmar)
    return {"net_yield": round(net, 4), "annualised": round(ann, 4), "sharpe": round(sharpe, 4),
            "max_dd": round(max_dd, 4), "profit_factor": round(pf, 4),
            "win_rate": round(win_rate, 4), "risk_adj_calmar": round(calmar, 4),
            "round_trips": len(pnl), "gatc_literal": round(lit, 5),
            "gatc_inner_weights": round(inner, 5), "gatc_normalised": round(norm, 5)}


# ------------------------------------------------------------------ cli
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--grid", required=True, help='e.g. "5:10:off,3:5:on" (top_n:rebal:stop)')
    ap.add_argument("--universe", required=True,
                    help="membership CSV (date,tickers) for point-in-time, or .txt static list")
    ap.add_argument("--placebo", type=int, default=0,
                    help="also run N random-rank placebos per combo (same eligible set)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--events", help="catalyst-date CSV (see module docstring)")
    g.add_argument("--no-catalyst", action="store_true",
                   help="disable the earnings gate (every scored name is eligible)")
    ap.add_argument("--market", choices=sorted(CAL_PROXY), default="us")
    ap.add_argument("--label", default="LOCAL")
    ap.add_argument("--capital", type=float, default=1_000_000)
    ap.add_argument("--slippage", type=float, default=0.001)
    ap.add_argument("--commission", type=float, default=0.0)
    ap.add_argument("--out", default=str(HERE / "results" / "sweep.jsonl"))
    a = ap.parse_args()

    start = int(pd.Timestamp(a.start).strftime("%Y%m%d"))
    end = int(pd.Timestamp(a.end).strftime("%Y%m%d"))
    grid = PIPE.parse_grid(a.grid)
    fetch_lo = pd.Timestamp(str(start)) - pd.Timedelta(days=WARMUP_DAYS)
    fetch_hi = pd.Timestamp(str(end)) + pd.Timedelta(days=TAIL_DAYS)
    membership = Membership(a.universe)
    universe = membership.all_between(int(fetch_lo.strftime("%Y%m%d")), end)
    proxy = CAL_PROXY[a.market]

    events = None
    if a.no_catalyst:
        STRAT._catalyst_flag = lambda ctx, sym, today: (1, False)
    else:
        events = load_events(a.events)

    prices = dl.load_prices(universe + [proxy], fetch_lo, fetch_hi)
    if proxy not in prices:
        raise SystemExit(f"no calendar proxy data ({proxy})")
    days = prices[proxy].index.tolist()
    bench = prices[proxy]["Close"]
    stock_prices = {t: p for t, p in prices.items() if t in universe}
    panel = Panel(stock_prices, days)
    print(f"[local] {len(stock_prices)}/{len(universe)} tickers with data, "
          f"{sum(start <= d <= end for d in days)} sessions {start}-{end}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for k, (n, r, s) in enumerate(grid, 1):
        args = (panel, days, start, end, n, r, s == "on", events,
                a.capital, a.slippage, a.commission, membership)
        coverage = []
        res = run(*args, coverage=coverage)
        m = metrics(res, a.capital)
        b = bench.loc[res["first_day"]:res["last_day"]]
        elig = res["eligible"]
        placebo = {}
        if a.placebo:
            pm = [metrics(run(*args, placebo_seed=seed), a.capital) for seed in range(a.placebo)]
            pn = np.array([x["net_yield"] for x in pm])
            ps = np.array([x["sharpe"] for x in pm])
            placebo = {"placebo_n": a.placebo,
                       "placebo_net_mean": round(float(pn.mean()), 4),
                       "placebo_net_p05_p95": [round(float(np.percentile(pn, 5)), 4),
                                               round(float(np.percentile(pn, 95)), 4)],
                       "placebo_sharpe_mean": round(float(ps.mean()), 4),
                       # share of random-pick runs that did at least as well as the score
                       "p_net_vs_placebo": round(float((pn >= m["net_yield"]).mean()), 3),
                       "p_sharpe_vs_placebo": round(float((ps >= m["sharpe"]).mean()), 3)}
        row = {"label": f"{a.label}_n{n}_r{r}_stop{s}", "engine": "local", "market": a.market,
               "stop": s, "top_n": n, "rebalance": r, "window": f"{start}-{end}",
               **m, **placebo,
               "bench_return": round(float(b.iloc[-1] / b.iloc[0] - 1), 4),
               "bench": proxy,
               "universe_ew_return": round(res["ew_return"], 4),
               "universe_point_in_time": membership.point_in_time,
               "member_price_coverage": round(float(np.mean(coverage)), 4) if coverage else None,
               "avg_eligible": round(sum(elig) / len(elig), 2) if elig else 0.0,
               "empty_rebalances": sum(1 for x in elig if x == 0),
               "stops_fired": res["stops"],
               "catalyst": "off" if events is None else Path(a.events).name,
               "catalyst_hits": res["cat_hits"], "pre_earnings_hits": res["cat_pre"],
               "universe_file": Path(a.universe).name, "universe_size": len(stock_prices),
               "slippage": a.slippage, "commission": a.commission}
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[{k}/{len(grid)}] {json.dumps(row)}", flush=True)


if __name__ == "__main__":
    main()
