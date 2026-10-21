#!/usr/bin/env python3
"""
SPY at its 30-day high / low: does it bounce or break?  (yfinance, no platform calls)

  python local/spy_levels.py [--ticker SPY] [--start 2005-01-01] [--end YYYY-MM-DD]
         [--lookback 30] [--tol 0.005] [--horizon 5] [--vol-mult 1.5] [--vol-window 20]
         [--rev-pct 0.01] [--out local/results/spy_levels_events.csv]

EVENT -- bar t comes near a level built ONLY from earlier bars:
  resistance  High_t >= R*(1 - tol),  R = max(High[t-lookback .. t-1])
  support     Low_t  <= S*(1 + tol),  S = min(Low[t-lookback .. t-1])
  The level is frozen at the touch. Next touch on the same side is only counted after
  the previous one resolved, so one test of a level is one event, not five.

OUTCOME -- the first of these on bars t .. t+horizon:
  break         a CLOSE beyond the level on volume >= vol_mult x the mean volume of the
                vol_window bars before that bar, AND the next `hold` closes (default 2)
                also stay beyond the level
  failed_break  that close-on-volume happened, but a close within the next `hold` bars
                came back inside the level (with --hold 0 this can't occur)
  weak_break    a close beyond the level WITHOUT that volume
  bounce      a close back inside the level by >= rev_pct, with no close beyond before it
  unresolved  none of the above within the horizon

READ THIS BEFORE TRADING ON IT
  Outcome labels use bars after the touch, so "return | bounce" is NOT tradable -- you
  can't know it bounced at the touch. The tradable numbers are the base rates (how often
  each outcome follows a touch) and the forward returns over ALL touches.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as dl  # noqa: E402

FWD = (1, 5, 10, 20)


def find_events(df, lookback, tol, horizon, vol_mult, vol_window, rev_pct, hold=2):
    hi_lvl = df["High"].rolling(lookback).max().shift(1)
    lo_lvl = df["Low"].rolling(lookback).min().shift(1)
    avg_vol = df["Volume"].rolling(vol_window).mean().shift(1)   # prior bars only
    close, high, low, vol = df["Close"], df["High"], df["Low"], df["Volume"]
    n = len(df)
    events = []

    for side in ("resistance", "support"):
        level_s = hi_lvl if side == "resistance" else lo_lvl
        sign = 1 if side == "resistance" else -1          # +1: "beyond" means above
        t, busy_until = 0, -1
        while t < n:
            lvl = level_s.iloc[t]
            if t <= busy_until or lvl != lvl:
                t += 1
                continue
            touched = (high.iloc[t] >= lvl * (1 - tol)) if sign > 0 else (low.iloc[t] <= lvl * (1 + tol))
            if not touched:
                t += 1
                continue

            outcome, k_hit, judgeable = "unresolved", None, True
            last = min(t + horizon, n - 1)
            for k in range(t, last + 1):
                c = close.iloc[k]
                if (c - lvl) * sign > 0:                                  # closed beyond
                    av = avg_vol.iloc[k]
                    strong = av == av and vol.iloc[k] >= vol_mult * av
                    if not strong:
                        outcome = "weak_break"
                    elif k + hold >= n:
                        judgeable = False                 # hold window runs past the data
                    elif all((close.iloc[k + j] - lvl) * sign > 0 for j in range(1, hold + 1)):
                        outcome = "break"                 # volume AND held `hold` closes
                    else:
                        outcome = "failed_break"          # volume, but closed back inside
                    k_hit = k
                    break
                if (lvl - c) * sign >= rev_pct * lvl:                     # reversed inside
                    outcome, k_hit = "bounce", k
                    break
            if not judgeable or (outcome == "unresolved" and t + horizon >= n):
                t += 1  # not enough future bars to judge -- drop, don't label
                continue
            row = {"date": int(df.index[t]), "side": side, "level": round(float(lvl), 4),
                   "touch_close": round(float(close.iloc[t]), 4), "outcome": outcome,
                   "bars_to_outcome": (k_hit - t) if k_hit is not None else None,
                   "vol_ratio_at_outcome": round(float(vol.iloc[k_hit] / avg_vol.iloc[k_hit]), 3)
                   if k_hit is not None and avg_vol.iloc[k_hit] == avg_vol.iloc[k_hit] else None}
            for h in FWD:
                row[f"fwd_{h}d"] = round(float(close.iloc[t + h] / close.iloc[t] - 1), 5) \
                    if t + h < n else None
            events.append(row)
            busy_until = k_hit if k_hit is not None else t + horizon
            t += 1
    return pd.DataFrame(events).sort_values("date").reset_index(drop=True)


def summarise(ev):
    out = {}
    for side, g in ev.groupby("side"):
        rates = g["outcome"].value_counts(normalize=True).round(3).to_dict()
        fwd_all = {f"fwd_{h}d": round(float(g[f"fwd_{h}d"].mean()), 5) for h in FWD}
        by_outcome = {o: {"n": int(len(x)),
                          "median_bars": None if x["bars_to_outcome"].isna().all()
                          else float(x["bars_to_outcome"].median()),
                          **{f"fwd_{h}d": round(float(x[f"fwd_{h}d"].mean()), 5) for h in FWD}}
                      for o, x in g.groupby("outcome")}
        out[side] = {"touches": int(len(g)), "outcome_rates": rates,
                     "fwd_mean_all_touches": fwd_all, "by_outcome_LOOKAHEAD": by_outcome}
    return out


def main():
    ap = argparse.ArgumentParser(description="SPY 30-day high/low bounce-vs-break study")
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--start", default="2005-01-01")
    ap.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--tol", type=float, default=0.005, help="'near' = within this fraction")
    ap.add_argument("--horizon", type=int, default=5, help="N bars to resolve")
    ap.add_argument("--vol-mult", type=float, default=1.5)
    ap.add_argument("--vol-window", type=int, default=20)
    ap.add_argument("--rev-pct", type=float, default=0.01, help="reversal needed for a bounce")
    ap.add_argument("--hold", type=int, default=2,
                    help="closes after the break that must stay beyond the level (0 = off)")
    ap.add_argument("--out", default=str(HERE / "results" / "spy_levels_events.csv"))
    a = ap.parse_args()

    t = dl.to_yf(a.ticker)
    df = dl.load_prices([t], a.start, a.end).get(t)
    if df is None:
        raise SystemExit(f"no data for {t}")
    ev = find_events(df, a.lookback, a.tol, a.horizon, a.vol_mult, a.vol_window, a.rev_pct,
                     a.hold)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ev.to_csv(out, index=False)
    params = {k: v for k, v in vars(a).items() if k != "out"}
    summary = {"params": params, "bars": len(df),
               "first": int(df.index[0]), "last": int(df.index[-1]), **summarise(ev)}
    out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\n{len(ev)} events -> {out}")


if __name__ == "__main__":
    main()
