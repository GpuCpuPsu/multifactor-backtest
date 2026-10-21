"""The local backtest engine: fills, sizing, delistings, placebo and benchmark mechanics."""
import numpy as np
import pytest

from helpers import STRAT, bt, membership_csv, trading_days, trending_prices

DAYS = trading_days(160)
MEMBERS = [f"M{i:02d}" for i in range(16)]
DRIFTS = {t: 0.0005 * i for i, t in enumerate(MEMBERS)}      # M15 trends hardest


def _run(tmp_path, prices=None, top_n=3, rebal=5, slippage=0.0, placebo_seed=None):
    prices = prices or trending_prices(DAYS, DRIFTS)
    m = bt.Membership(membership_csv(tmp_path, [(DAYS[0], MEMBERS)]))
    panel = bt.Panel(prices, DAYS)
    return bt.run(panel, DAYS, DAYS[100], DAYS[-1], top_n, rebal, False, None, 1e6,
                  slippage, 0.0, m, placebo_seed=placebo_seed)


def test_buys_the_top_n_by_composite_score(tmp_path, no_catalyst):
    res = _run(tmp_path, top_n=3)
    first_buys = {t["contract_code"] for t in res["trades"] if t["direction"] == 0}
    assert first_buys == {"M15", "M14", "M13"}


def test_fills_pay_slippage_both_ways(tmp_path, no_catalyst):
    prices = trending_prices(DAYS, DRIFTS)
    res = _run(tmp_path, prices=prices, slippage=0.001)
    for t in res["trades"]:
        close = prices[t["contract_code"]].loc[int(t["trade_date"]), "Close"]
        side = 1 + 0.001 if t["direction"] == 0 else 1 - 0.001
        assert t["price"] == pytest.approx(close * side)


def test_deploys_at_most_90_percent_split_across_top_n(tmp_path, no_catalyst):
    res = _run(tmp_path, top_n=3)
    first_day = res["trades"][0]["trade_date"]
    spent = sum(t["price"] * t["volume"] for t in res["trades"]
                if t["trade_date"] == first_day and t["direction"] == 0)
    assert spent <= STRAT.MAX_DEPLOYED * 1e6
    assert spent > STRAT.MAX_DEPLOYED * 1e6 * 0.99           # integer shares, nearly all used


def test_held_stock_whose_data_ends_is_closed_at_its_last_price(tmp_path, no_catalyst):
    prices = trending_prices(DAYS, DRIFTS)
    last = 130
    prices["M15"].iloc[last + 1:] = np.nan                   # delisted after day 130
    res = _run(tmp_path, prices=prices)
    sells = [t for t in res["trades"] if t["contract_code"] == "M15" and t["direction"] == 1]
    assert sells and int(sells[0]["trade_date"]) == DAYS[last + 1]
    assert sells[0]["price"] == pytest.approx(prices["M15"]["Close"].iloc[last])


def test_placebo_ranks_the_same_eligible_set_randomly(tmp_path, no_catalyst):
    real = _run(tmp_path)
    fake = _run(tmp_path, placebo_seed=1)
    assert fake["eligible"] == real["eligible"]               # same candidates each rebalance
    pick = lambda r: {t["contract_code"] for t in r["trades"] if t["direction"] == 0}
    assert pick(fake) != pick(real)                           # different, random choices


def test_equal_weight_benchmark_compounds_the_mean_daily_member_return(tmp_path, no_catalyst):
    """Rebalanced to equal weight daily: (1 + mean daily return)^n, not the mean of each
    stock's compounded return."""
    res = _run(tmp_path)
    n = len(res["ew_equity"])
    expected = (1 + np.mean([DRIFTS[t] for t in MEMBERS])) ** n - 1
    assert res["ew_return"] == pytest.approx(expected, rel=1e-9)
