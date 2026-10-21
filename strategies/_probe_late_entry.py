"""Diagnostic: did the >20% trailing-revenue-growth rule exclude the big 2023-24 winners
during the run, and only admit them afterwards?"""
from panda_backtest.api.api import *
from panda_backtest.api.stock_us_api import *
import tqx_data
import pandas as pd

WATCH = ["NVDA.NB", "META.NB", "AVGO.NB", "AMD.NB", "MU.NB", "SMCI.NB", "ANET.NB", "PLTR.NB"]
DATES = ["20230103", "20230403", "20230703", "20231002", "20240102", "20240401",
         "20240701", "20241001", "20250102", "20250701", "20260102"]

REV = ["is_revenue_business_total", "is_revenue_goods_services", "is_sales_goods_services_net",
       "is_reported_revenue", "is_reported_op_revenue"]
GP = ["is_gross_profit", "is_reported_gross_profit"]
OCF = ["cfs_net_cf_operating", "cfs_reported_cf_operating", "cfs_cf_operating_before_wc"]
FIELDS = ["symbol", "fy_period", "date"] + list(dict.fromkeys(REV + GP + OCF))


def _fp(row, cands):
    for n in cands:
        if n in row and row[n] is not None:
            try:
                v = float(row[n])
            except (TypeError, ValueError):
                continue
            if v == v and abs(v) != float("inf"):
                return v
    return None


def _prior(fy):
    try:
        return f"FY{int(fy[2:6]) - 1}{fy[6:]}"
    except Exception:
        return None


def initialize(context):
    print("date       symbol     fy_period  rev_growth  gross_mgn  ocf>0  PASSES?")
    for asof in DATES:
        y = int(asof[:4]); m = int(asof[4:6]); q = (m - 1) // 3 + 1
        sq = f"{y-2}q{q}"
        eq = f"{y}q{q+1}" if q < 4 else f"{y+1}q1"
        try:
            df = tqx_data.get_financial_statement(
                sq, eq, symbol=WATCH, date=asof, is_latest=False,
                market="nb", interim_type="cumulative", fields=FIELDS)
        except Exception as exc:
            print(f"{asof}  QUERY FAILED: {exc}")
            continue
        if df is None or len(df) == 0:
            print(f"{asof}  no rows")
            continue
        df = df.dropna(subset=["symbol", "fy_period"])
        by_sp_df = df.drop_duplicates(subset=["symbol", "fy_period"], keep="last")
        latest = by_sp_df.sort_values("fy_period").drop_duplicates(subset=["symbol"], keep="last")
        by_sp = {(r["symbol"], r["fy_period"]): r for _, r in by_sp_df.iterrows()}

        for _, row in latest.iterrows():
            sym = row["symbol"]
            fy = row["fy_period"]
            prow = by_sp.get((sym, _prior(fy)))
            rn = _fp(row, REV); gp = _fp(row, GP); oc = _fp(row, OCF)
            rp = _fp(prow, REV) if prow is not None else None
            if rn is None or rp is None or rp <= 0 or gp is None or rn <= 0 or oc is None:
                print(f"{asof} {sym:10s} {fy:9s} {'--':>10s} {'--':>10s} {'--':>6s}  NO DATA")
                continue
            g = rn / rp - 1.0
            mgn = gp / rn
            ok = g > 0.20 and mgn > 0.20 and oc > 0
            print(f"{asof} {sym:10s} {fy:9s} {g*100:9.1f}% {mgn*100:9.1f}% "
                  f"{'yes' if oc > 0 else 'no':>6s}  {'PASS' if ok else 'fail'}")


def handle_data(context, data):
    pass
