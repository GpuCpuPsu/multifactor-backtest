#!/usr/bin/env python3
"""
Where do +/-10% one-day moves come from, and can they be seen coming?  (free data only)

  python local/big_movers.py [--threshold 0.10] [--recent 20] [--news]

Universe: point-in-time S&P 500 members (a name counts on day t only if it was a member
on t-1), prices from yfinance, earnings timing from the EDGAR 8-K Item 2.02 file.
"Tech" = GICS Information Technology or Communication Services, plus AMZN and TSLA.

Every feature is measured at the PREVIOUS close (knowable the evening before):
  earn     an earnings release lands on day t (after-close t-1 or pre-market t).
           Uses the EDGAR filing date as a stand-in for the SCHEDULED date -- companies
           announce report dates weeks ahead, but this file only has the filing itself.
  vol20    20-day realized volatility of daily returns
  ret5 / ret20        prior 5- / 20-day return
  vratio   prior day's volume / its 20-day average
  off_hi   distance below the 52-week closing high
  vix      VIX close
  tech     sector flag

Cause of each move (after the fact, for attribution only -- not a predictor):
  earnings      earn flag set
  market_day    >= 10 members moved >= 10% that day, or SPY moved >= 3%
  sector_day    >= 3 other same-sector names moved >= 5% the same direction
  stock_specific  none of the above -- the "no reason" moves; --news fetches headlines
Model: logistic regression, fit on 2023-2024, scored on 2025-2026 (recent regime).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data as dl  # noqa: E402
from backtest import Membership  # noqa: E402

FEATS = ["earn", "vol20", "ret5", "ret20", "vratio", "off_hi", "vix", "tech"]
TECH_EXTRA = {"AMZN", "TSLA"}


def load(threshold, end):
    m = Membership(HERE / "ref" / "sp500_components_by_date.csv")
    tickers = m.all_between(20220101, end)
    px = dl.load_prices(tickers + ["SPY", "^VIX"], "2021-12-01", str(end))
    spy, vix = px.pop("SPY")["Close"], px.pop("^VIX")["Close"]
    days = spy.index
    C = pd.DataFrame({t: p["Close"] for t, p in px.items()}).reindex(days)
    V = pd.DataFrame({t: p["Volume"] for t, p in px.items()}).reindex(days)
    R = C.pct_change(fill_method=None)

    # membership as of the previous session
    M = pd.DataFrame(False, index=days, columns=C.columns)
    for i in range(1, len(days)):
        mem = [t for t in m.on(int(days[i - 1])) if t in M.columns]
        M.iloc[i, M.columns.get_indexer(mem)] = True

    ev = pd.read_csv(HERE / "ref" / "events_sec_8k_202.csv")
    E = pd.DataFrame(False, index=days, columns=C.columns)
    for s, d in zip(ev["symbol"], ev["event_date"]):
        if s in E.columns and d in E.index:
            E.at[d, s] = True

    sect = pd.read_csv("https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv")
    sector = {dl.to_yf(s): g for s, g in zip(sect["Symbol"], sect["GICS Sector"])}
    tech = {t: (sector.get(t) in ("Information Technology", "Communication Services")
                or t in TECH_EXTRA) for t in C.columns}

    prev = C.shift(1)
    F = {
        "earn": E.astype(float),
        "vol20": R.rolling(20).std().shift(1),
        "ret5": prev / C.shift(6) - 1,
        "ret20": prev / C.shift(21) - 1,
        "vratio": V.shift(1) / V.shift(2).rolling(20).mean(),
        "off_hi": prev / prev.rolling(252, min_periods=200).max() - 1,
        "vix": pd.DataFrame(np.repeat(vix.shift(1).reindex(days).to_numpy()[:, None],
                                      C.shape[1], 1), index=days, columns=C.columns),
        "tech": pd.DataFrame(np.repeat(np.array([[float(tech[t]) for t in C.columns]]),
                                       len(days), 0), index=days, columns=C.columns),
    }
    spy_r = spy.pct_change()
    fwd = {h: (C.shift(-h) / C - 1).sub(spy.shift(-h) / spy - 1, axis=0) for h in (1, 5, 20)}

    long = pd.DataFrame({"ret": R.stack(future_stack=True)})
    long["member"] = M.stack(future_stack=True)
    for k, f in F.items():
        long[k] = f.stack(future_stack=True)
    for h, f in fwd.items():
        long[f"xfwd{h}"] = f.stack(future_stack=True)
    long = long[long["member"] & long["ret"].notna()].drop(columns="member")
    long.index.names = ["date", "symbol"]
    long = long.reset_index()
    long = long[long["date"] >= 20230101]
    long["big"] = long["ret"].abs() >= threshold
    long["sector"] = long["symbol"].map(sector).fillna("Former member")

    # ---- attribution (after the fact) ----
    big = long[long["big"]]
    n_big_day = big.groupby("date").size()
    long["market_day"] = long["date"].map(n_big_day).fillna(0).ge(10) | \
        long["date"].map(spy_r.abs()).ge(0.03)
    moved = long[long["ret"].abs() >= 0.05]
    peer = moved.groupby(["date", "sector", np.sign(moved["ret"])]).size()
    key = list(zip(long["date"], long["sector"], np.sign(long["ret"])))
    long["peers_same_dir"] = [peer.get(k, 0) for k in key]
    long["peers_same_dir"] -= (long["ret"].abs() >= 0.05).astype(int)   # exclude itself
    long["cause"] = np.select(
        [long["earn"] == 1, long["market_day"], long["peers_same_dir"] >= 3],
        ["earnings", "market_day", "sector_day"], "stock_specific")
    return long, spy_r


def summarize(long):
    out = {}
    for period, g in (("2023-2024", long[long.date < 20250101]),
                      ("2025-2026", long[long.date >= 20250101])):
        b = g[g.big]
        tb = b[b.tech == 1]
        out[period] = {
            "stock_days": int(len(g)), "big_moves": int(len(b)),
            "per_1000_stock_days": round(1000 * len(b) / len(g), 2),
            "up_share": round(float((b.ret > 0).mean()), 3),
            "tech_per_1000": round(1000 * len(tb) / max(1, (g.tech == 1).sum()), 2),
            "nontech_per_1000": round(1000 * (len(b) - len(tb)) / max(1, (g.tech == 0).sum()), 2),
            "cause_share": b["cause"].value_counts(normalize=True).round(3).to_dict(),
            "tech_cause_share": tb["cause"].value_counts(normalize=True).round(3).to_dict(),
            "p_big_given_earnings": round(float(g[g.earn == 1].big.mean()), 4),
            "p_big_given_no_earnings": round(float(g[g.earn == 0].big.mean()), 5),
        }
    return out


def lift_table(long, col, q=5):
    g = long[long.date >= 20250101].dropna(subset=[col]).copy()
    g["bucket"] = pd.qcut(g[col], q, labels=False, duplicates="drop")
    t = g.groupby("bucket").agg(lo=(col, "min"), hi=(col, "max"), n=("big", "size"),
                                p_big=("big", "mean"))
    t["lift"] = t["p_big"] / g["big"].mean()
    return t.round(4)


def model(long):
    d = long.dropna(subset=FEATS).copy()
    d["lvol"] = np.log(d["vol20"].clip(lower=1e-4))
    d["lvr"] = np.log(d["vratio"].clip(lower=1e-3))
    X = ["earn", "lvol", "ret5", "ret20", "lvr", "off_hi", "vix", "tech"]
    tr, te = d[d.date < 20250101], d[d.date >= 20250101]
    mu, sd = tr[X].mean(), tr[X].std()
    z = lambda f: (f[X] - mu) / sd
    res = {}

    clf = LogisticRegression(max_iter=1000).fit(z(tr), tr["big"])
    p = clf.predict_proba(z(te))[:, 1]
    top = te.assign(p=p).nlargest(max(1, len(te) // 100), "p")      # top 1% of stock-days
    res["size_of_move"] = {
        "test_auc": round(float(roc_auc_score(te["big"], p)), 3),
        "base_rate": round(float(te["big"].mean()), 4),
        "top1pct_hit_rate": round(float(top["big"].mean()), 3),
        "top1pct_lift": round(float(top["big"].mean() / te["big"].mean()), 1),
        "top1pct_earnings_share": round(float(top["earn"].mean()), 3),
        "coef_standardized": dict(zip(X, np.round(clf.coef_[0], 3))),
        "no_earnings_test_auc": round(float(roc_auc_score(
            te.loc[te.earn == 0, "big"], p[te["earn"].to_numpy() == 0])), 3),
    }

    btr, bte = tr[tr.big], te[te.big]
    dclf = LogisticRegression(max_iter=1000).fit(z(btr), btr["ret"] > 0)
    dp = dclf.predict_proba(z(bte))[:, 1]
    res["direction_given_big_move"] = {
        "test_auc": round(float(roc_auc_score(bte["ret"] > 0, dp)), 3),
        "test_n": int(len(bte)), "up_share_test": round(float((bte["ret"] > 0).mean()), 3),
        "coef_standardized": dict(zip(X, np.round(dclf.coef_[0], 3)))}
    return res


def follow_through(long):
    b = long[long.big & (long.date >= 20250101)].copy()
    b["dir"] = np.where(b.ret > 0, "up", "down")
    b["kind"] = np.where(b.cause == "earnings", "earnings", "non_earnings")
    g = b.groupby(["dir", "kind"])
    t = g.agg(n=("ret", "size"), **{f"mean_x{h}d": (f"xfwd{h}", "mean") for h in (1, 5, 20)},
              med_x5d=("xfwd5", "median"),
              hit_x5d=("xfwd5", lambda s: (s > 0).mean()))
    return t.round(4)


def recent(long, sessions, with_news):
    days = sorted(long["date"].unique())[-sessions:]
    r = long[long.big & long.date.isin(days)].sort_values(["date", "ret"], ascending=[False, False])
    cols = ["date", "symbol", "ret", "cause", "sector", "earn", "vol20", "ret20", "off_hi"]
    r = r[cols].copy()
    if with_news:
        import yfinance as yf
        heads = []
        for d, s in zip(r["date"], r["symbol"]):
            day = pd.Timestamp(str(d)).tz_localize("America/New_York")
            got = []
            for it in (yf.Ticker(s).news or []):
                ct = it.get("content", it)
                ts = pd.Timestamp(ct.get("pubDate") or 0, tz="UTC") if ct.get("pubDate") else None
                if ts is not None and day - pd.Timedelta(days=1) <= ts.tz_convert("America/New_York") <= day + pd.Timedelta(days=1, hours=12):
                    got.append(ct.get("title", "")[:110])
            heads.append(" | ".join(got[:2]) if got else "(no headline in Yahoo's recent feed)")
        r["headlines"] = heads
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.10)
    ap.add_argument("--end", default="20260923")
    ap.add_argument("--recent", type=int, default=20, help="sessions to list at the end")
    ap.add_argument("--news", action="store_true", help="attach Yahoo headlines to recent moves")
    a = ap.parse_args()

    long, _ = load(a.threshold, int(a.end))
    out = {"threshold": a.threshold, "summary": summarize(long)}
    out["model"] = model(long)
    print(json.dumps(out, indent=2))
    for col in ("vol20", "vratio", "ret20", "off_hi"):
        print(f"\n2025-2026 P(big move) by {col} quintile (feature at prior close):")
        print(lift_table(long, col).to_string())
    print("\n2025-2026 what happened AFTER a big move (return minus SPY):")
    ft = follow_through(long)
    print(ft.to_string())
    rec = recent(long, a.recent, a.news)
    print(f"\nlast {a.recent} sessions, moves >= {a.threshold:.0%}:")
    with pd.option_context("display.max_colwidth", 240, "display.width", 400):
        print(rec.to_string(index=False))

    res = HERE / "results"
    res.mkdir(exist_ok=True)
    (res / "big_movers_summary.json").write_text(json.dumps(out, indent=2))
    long[long.big].to_csv(res / "big_movers_events.csv", index=False)
    rec.to_csv(res / "big_movers_recent.csv", index=False)
    ft.to_csv(res / "big_movers_follow_through.csv")


if __name__ == "__main__":
    main()
