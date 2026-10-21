#!/usr/bin/env python3
"""
Is "SPY does better after touching its 30-day LOW than after touching its 30-day HIGH"
real, or an artefact of overlapping samples?

  python local/spy_levels_stats.py [--events local/results/spy_levels_events.csv]
         [--horizon 20] [--reps 5000]

Touches overlap heavily: a 20-day forward window after one touch shares most of its days
with the next touch's window, and touches cluster in the same few selloffs/rallies. So
the touches are NOT independent samples. Two tests that respect that:

1. Circular block bootstrap on CALENDAR TIME. The whole daily timeline (every day's
   forward return plus its touch flags) is resampled in contiguous blocks of b trading
   days, so overlap and clustering inside a block are preserved. Repeated for several b;
   if the answer depends on b, that's itself informative. p-values are two-sided from
   the centred bootstrap distribution.
2. Non-overlapping thinning: keep a touch only if it's >= horizon trading days after the
   previously kept touch ON THE SAME SIDE, so no two kept windows share a day. Welch t
   (normal approximation) on what's left.
Plus stability: the same difference by era and with the 2008-09 / 2020 crashes removed.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as dl  # noqa: E402


def block_bootstrap(fwd, lo, hi, block, reps, rng, chunk=500):
    """Circular block bootstrap of (mean fwd | low touch) - (mean fwd | high touch)."""
    n = len(fwd)
    nblocks = -(-n // block)
    out = []
    for c0 in range(0, reps, chunk):
        r = min(chunk, reps - c0)
        starts = rng.integers(0, n, size=(r, nblocks))
        idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(r, -1)[:, :n]
        f, l, h = fwd[idx], lo[idx], hi[idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append((f * l).sum(1) / l.sum(1) - (f * h).sum(1) / h.sum(1))
    d = np.concatenate(out)
    return d[np.isfinite(d)]


def thin(dates_idx, gap):
    kept, last = [], -10 ** 9
    for i in sorted(dates_idx):
        if i - last >= gap:
            kept.append(i)
            last = i
    return kept


def welch(a, b):
    a, b = np.asarray(a), np.asarray(b)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    z = (a.mean() - b.mean()) / se
    p = math.erfc(abs(z) / math.sqrt(2))          # two-sided, normal approximation
    return float(a.mean() - b.mean()), float(z), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default=str(HERE / "results" / "spy_levels_events.csv"))
    ap.add_argument("--ticker", default="SPY")
    ap.add_argument("--horizon", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5000)
    ap.add_argument("--blocks", default="20,60,120,250")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    ev = pd.read_csv(a.events)
    first, last = str(ev["date"].min()), pd.Timestamp.today().strftime("%Y%m%d")
    px = dl.load_prices([a.ticker], "2004-06-01", last)[dl.to_yf(a.ticker)]["Close"]
    fwd_all = (px.shift(-a.horizon) / px - 1)
    fwd_all = fwd_all[fwd_all.index >= int(first)].dropna()
    days = fwd_all.index.to_numpy()
    pos = {d: i for i, d in enumerate(days)}
    fwd = fwd_all.to_numpy()

    lo = np.zeros(len(days)); hi = np.zeros(len(days))
    for d, side in zip(ev["date"], ev["side"]):
        if d in pos:
            (lo if side == "support" else hi)[pos[d]] = 1
    obs = fwd[lo == 1].mean() - fwd[hi == 1].mean()
    res = {"horizon": a.horizon, "days": int(len(days)),
           "n_low_touches": int(lo.sum()), "n_high_touches": int(hi.sum()),
           "mean_fwd_low": round(float(fwd[lo == 1].mean()), 5),
           "mean_fwd_high": round(float(fwd[hi == 1].mean()), 5),
           "mean_fwd_all_days": round(float(fwd.mean()), 5),
           "median_fwd_low": round(float(np.median(fwd[lo == 1])), 5),
           "median_fwd_high": round(float(np.median(fwd[hi == 1])), 5),
           "diff_low_minus_high": round(float(obs), 5)}

    rng = np.random.default_rng(a.seed)
    boot = {}
    for b in (int(x) for x in a.blocks.split(",")):
        d = block_bootstrap(fwd, lo, hi, b, a.reps, rng)
        p = float((np.abs(d - obs) >= abs(obs)).mean())
        boot[f"block_{b}d"] = {"ci95": [round(float(np.percentile(d, 2.5)), 5),
                                        round(float(np.percentile(d, 97.5)), 5)],
                               "p_two_sided": round(p, 4), "reps": int(len(d))}
    res["block_bootstrap"] = boot

    tl = thin(np.flatnonzero(lo), a.horizon)
    th = thin(np.flatnonzero(hi), a.horizon)
    diff, z, p = welch(fwd[tl], fwd[th])
    res["non_overlapping"] = {"n_low": len(tl), "n_high": len(th),
                              "mean_low": round(float(fwd[tl].mean()), 5),
                              "mean_high": round(float(fwd[th].mean()), 5),
                              "diff": round(diff, 5), "z": round(z, 3),
                              "p_two_sided_normal": round(p, 4)}

    def sub(mask):
        l, h = (lo == 1) & mask, (hi == 1) & mask
        if l.sum() < 5 or h.sum() < 5:
            return None
        return {"n_low": int(l.sum()), "n_high": int(h.sum()),
                "diff": round(float(fwd[l].mean() - fwd[h].mean()), 5),
                "median_diff": round(float(np.median(fwd[l]) - np.median(fwd[h])), 5)}

    yr = days // 10000
    crash = ((days >= 20080901) & (days <= 20090630)) | ((days >= 20200220) & (days <= 20200630))
    res["stability"] = {"2005-2012": sub((yr >= 2005) & (yr <= 2012)),
                        "2013-2019": sub((yr >= 2013) & (yr <= 2019)),
                        "2020-2026": sub(yr >= 2020),
                        "ex_2008-09_and_2020_crashes": sub(~crash)}

    out = Path(a.events).with_name("spy_levels_significance.json")
    out.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
