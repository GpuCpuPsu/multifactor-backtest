#!/usr/bin/env python3
"""
README charts -> docs/img/*.png.

  python local/make_charts.py            # reuse cached curves in local/results/curves.json
  python local/make_charts.py --refresh  # rerun the backtests (about 15 minutes)

All four charts come from the same engine and data as the results tables: point-in-time
S&P 500 membership, SEC 8-K earnings dates, 5 stocks, 10-day rebalance, no stop,
0.001 slippage, 50 random-pick placebos per configuration.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import backtest as bt  # noqa: E402
import data as dl  # noqa: E402
import spy_levels_stats as st  # noqa: E402

OUT = ROOT / "docs" / "img"
CACHE = HERE / "results" / "curves.json"
WINDOWS = {"2023-24 (in-sample)": (20230101, 20241231),
           "2025-26 (out-of-sample)": (20250101, 20260922)}
N_PLACEBO = 50

# reference palette (validated: adjacent CVD dE >= 9.1; aqua/yellow carry direct labels)
SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "axes.edgecolor": BASE, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9,
})


# ------------------------------------------------------------------ data
def compute_curves():
    lo = int((pd.Timestamp("2023-01-01") - pd.Timedelta(days=bt.WARMUP_DAYS)).strftime("%Y%m%d"))
    hi = max(e for _, e in WINDOWS.values())
    pit = bt.Membership(HERE / "ref" / "sp500_components_by_date.csv")
    static = bt.Membership(HERE / "universe_us.txt")
    tickers = sorted(set(pit.all_between(lo, hi)) | set(static.all_between(lo, hi)))
    prices = dl.load_prices(tickers + ["SPY"], pd.Timestamp(str(lo)),
                            pd.Timestamp(str(hi)) + pd.Timedelta(days=bt.TAIL_DAYS))
    days = prices["SPY"].index.tolist()
    spy = prices["SPY"]["Close"]
    panel = bt.Panel({t: p for t, p in prices.items() if t != "SPY"}, days)
    events = bt.load_events(HERE / "ref" / "events_sec_8k_202.csv")
    gate = bt.STRAT._catalyst_flag

    def curve(start, end, membership, use_gate, seed=None):
        bt.STRAT._catalyst_flag = gate if use_gate else (lambda c, s, t: (1, False))
        r = bt.run(panel, days, start, end, 5, 10, False, events if use_gate else None,
                   1e6, 0.001, 0.0, membership, placebo_seed=seed)
        bt.STRAT._catalyst_flag = gate
        return r

    out = {"walkforward": {}, "survivorship": {}}
    for label, (s, e) in WINDOWS.items():
        for use_gate in (True, False):
            real = curve(s, e, pit, use_gate)
            plac = [curve(s, e, pit, use_gate, seed)["equity"] for seed in range(N_PLACEBO)]
            d = real["days"]
            b = spy.loc[d[0]:d[-1]]
            out["walkforward"][f"{label}|{'filter' if use_gate else 'nofilter'}"] = {
                "days": d, "strategy": real["equity"], "placebo": plac,
                "spy": (b / b.iloc[0] * 1e6).tolist(), "ew": real["ew_equity"]}
            print(f"[charts] {label} filter={use_gate} done", flush=True)
    s, e = WINDOWS["2023-24 (in-sample)"]
    for name, m in (("static_today", static), ("point_in_time", pit)):
        out["survivorship"][name] = curve(s, e, m, False)["equity"]
    out["survivorship"]["days"] = curve(s, e, pit, False)["days"]
    b = spy.loc[out["survivorship"]["days"][0]:out["survivorship"]["days"][-1]]
    out["survivorship"]["spy"] = (b / b.iloc[0] * 1e6).tolist()
    CACHE.write_text(json.dumps(out))
    return out


def _dates(days):
    return pd.to_datetime([str(d) for d in days])


def _date_axis(ax, x):
    import matplotlib.dates as mdates
    ticks = [t for t in pd.date_range(x[0], x[-1], freq="MS") if t.month in (1, 7)]
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))


def _pct(eq):
    return (np.asarray(eq, dtype=float) / 1e6 - 1) * 100


def _end_label(ax, x, y, text, color, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points",
                color=INK2, fontsize=9, va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none"))
    ax.plot([x], [y], "o", ms=5, color=color, mec=SURFACE, mew=1.5, zorder=5)


# ------------------------------------------------------------------ charts
def chart_walkforward(c):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey="row")
    rows = [("filter", "With earnings filter"), ("nofilter", "Without earnings filter")]
    for r, (key, row_title) in enumerate(rows):
        for col, label in enumerate(WINDOWS):
            ax = axes[r][col]
            d = c["walkforward"][f"{label}|{key}"]
            x = _dates(d["days"])
            P = np.array([_pct(p) for p in d["placebo"]])
            lo, mid, hi = np.percentile(P, [5, 50, 95], axis=0)
            ax.fill_between(x, lo, hi, color=GRID, lw=0, label="50 random-pick runs (5–95%)")
            ax.plot(x, mid, color=MUTED, lw=1, label="random-pick median")
            ax.plot(x, _pct(d["spy"]), color=INK2, lw=1.5, ls=(0, (4, 2)), label="SPY")
            s = _pct(d["strategy"])
            ax.plot(x, s, color=BLUE, lw=2, label="strategy")
            beat = (P[:, -1] >= s[-1]).mean()
            _end_label(ax, x[-1], s[-1], f"{s[-1]:+.0f}%", BLUE)
            ax.set_title(f"{row_title} · {label}", fontsize=10)
            ax.text(0.02, 0.95, f"random picks did as well: {beat:.0%} of runs",
                    transform=ax.transAxes, color=INK2, fontsize=9, va="top")
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
            ax.axhline(0, color=BASE, lw=0.8)
            _date_axis(ax, x)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Walk-forward: cumulative return, 5 stocks, 10-day rebalance",
                 x=0.01, ha="left", fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0.04, 0.97, 0.97))
    fig.savefig(OUT / "walkforward.png", dpi=150)


def chart_survivorship(c):
    s = c["survivorship"]
    x = _dates(s["days"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for key, color, name in (("static_today", ORANGE, "today's large caps (hindsight)"),
                             ("point_in_time", BLUE, "members on each date"),
                             ("spy", INK2, "SPY")):
        y = _pct(s[key])
        ax.plot(x, y, color=color, lw=2 if key != "spy" else 1.5,
                ls="-" if key != "spy" else (0, (4, 2)), label=name)
        _end_label(ax, x[-1], y[-1], f"{name}  {y[-1]:+.0f}%", color)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:+.0f}%"))
    ax.axhline(0, color=BASE, lw=0.8)
    ax.set_title("Same strategy, same settings, 2023-24: which stocks it may pick from")
    ax.legend(loc="upper left")
    ax.set_xlim(x[0], x[-1] + pd.Timedelta(days=200))
    _date_axis(ax, x)
    fig.tight_layout()
    fig.savefig(OUT / "survivorship.png", dpi=150)


def chart_spy_significance():
    ev = pd.read_csv(HERE / "results" / "spy_levels_events.csv")
    px = dl.load_prices(["SPY"], "2004-06-01", "2026-09-23")["SPY"]["Close"]
    fwd_s = (px.shift(-20) / px - 1)
    fwd_s = fwd_s[fwd_s.index >= ev["date"].min()].dropna()
    pos = {d: i for i, d in enumerate(fwd_s.index)}
    fwd = fwd_s.to_numpy()
    lo = np.zeros(len(fwd)); hi = np.zeros(len(fwd))
    for d, side in zip(ev["date"], ev["side"]):
        if d in pos:
            (lo if side == "support" else hi)[pos[d]] = 1
    obs = (fwd[lo == 1].mean() - fwd[hi == 1].mean()) * 100
    boot = st.block_bootstrap(fwd, lo, hi, 60, 5000, np.random.default_rng(7)) * 100
    p = (np.abs(boot - obs) >= abs(obs)).mean()
    tl, th = st.thin(np.flatnonzero(lo), 20), st.thin(np.flatnonzero(hi), 20)
    thin_diff = (fwd[tl].mean() - fwd[th].mean()) * 100

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(boot, bins=60, color=BLUE, alpha=0.85, edgecolor=SURFACE, linewidth=0.6)
    ax.axvline(0, color=INK, lw=1.2)
    ax.axvline(obs, color=ORANGE, lw=2)
    ax.axvline(thin_diff, color=INK2, lw=1.5, ls=(0, (4, 2)))
    ax.set_ylim(0, ax.get_ylim()[1] * 1.25)          # headroom: labels sit above the bars
    top = ax.get_ylim()[1]
    box = dict(boxstyle="round,pad=0.25", fc=SURFACE, ec="none", alpha=0.92)
    ax.text(-0.05, top * 0.97, "no difference", color=INK, fontsize=9, va="top", ha="right",
            bbox=box)
    ax.text(obs + 0.05, top * 0.97, f"measured gap {obs:+.2f} pp  (p = {p:.2f})", color=INK2,
            fontsize=9, va="top", bbox=box)
    ax.text(thin_diff + 0.05, top * 0.88, f"non-overlapping touches only: {thin_diff:+.2f} pp",
            color=INK2, fontsize=9, va="top", bbox=box)
    ax.set_xlabel("20-day return after 30-day-low touches minus after 30-day-high touches (pp)")
    ax.set_ylabel("bootstrap samples")
    ax.set_title("SPY: is the low-vs-high gap real? Block bootstrap, 60-day blocks")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "spy_significance.png", dpi=150)


def chart_big_movers():
    s = json.loads((HERE / "results" / "big_movers_summary.json").read_text())["summary"]
    periods = ["2023-2024", "2025-2026"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})

    w, x = 0.36, np.arange(2)
    for k, (key, color, name) in enumerate((("tech_per_1000", BLUE, "tech"),
                                            ("nontech_per_1000", ORANGE, "everything else"))):
        vals = [s[p][key] for p in periods]
        bars = a1.bar(x + (k - 0.5) * (w + 0.03), vals, w, color=color, label=name)
        for b, v in zip(bars, vals):
            a1.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center",
                    color=INK2, fontsize=9)
    a1.set_ylim(0, max(s[p]["tech_per_1000"] for p in periods) * 1.15)
    a1.set_xticks(x, periods)
    a1.set_ylabel("±10% days per 1,000 stock-days")
    a1.set_title("Big one-day moves became 3x more common in tech")
    a1.legend(loc="upper left")
    a1.grid(axis="x", visible=False)

    causes = [("earnings", BLUE), ("sector_day", ORANGE), ("market_day", AQUA),
              ("stock_specific", YELLOW)]
    names = {"earnings": "earnings", "sector_day": "whole sector moved",
             "market_day": "whole market moved", "stock_specific": "stock-specific"}
    for row, p in enumerate(periods):
        left = 0.0
        for cause, color in causes:
            v = s[p]["tech_cause_share"].get(cause, 0) * 100
            a2.barh(row, v, left=left, color=color, height=0.55, edgecolor=SURFACE, linewidth=2,
                    label=names[cause] if row == 0 else None)
            if v >= 6:
                a2.text(left + v / 2, row, f"{v:.0f}%", ha="center", va="center",
                        color=INK, fontsize=9, fontweight="bold")
            left += v
    a2.set_yticks([0, 1], periods)
    a2.invert_yaxis()
    a2.set_xlim(0, 100)
    a2.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    a2.set_title("What caused tech's big moves")
    a2.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.32))
    a2.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUT / "big_movers.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    c = compute_curves() if a.refresh or not CACHE.exists() else json.loads(CACHE.read_text())
    chart_walkforward(c)
    chart_survivorship(c)
    chart_spy_significance()
    chart_big_movers()
    print(f"[charts] wrote {sorted(p.name for p in OUT.glob('*.png'))}")


if __name__ == "__main__":
    main()
