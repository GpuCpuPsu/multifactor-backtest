"""Survivorship bias: on each date the strategy may only pick stocks that were index
members on that date -- not today's winners."""
import pandas as pd

import data as dl
from helpers import bt, membership_csv, trading_days, trending_prices


def test_membership_is_point_in_time(tmp_path):
    p = membership_csv(tmp_path, [(20240101, ["AAA", "BBB"]), (20240301, ["AAA", "CCC"])])
    m = bt.Membership(p)
    assert m.point_in_time
    assert m.on(20240215) == ["AAA", "BBB"]          # before CCC joined
    assert m.on(20240301) == ["AAA", "CCC"]          # BBB removed, CCC added
    assert m.all_between(20240101, 20240401) == ["AAA", "BBB", "CCC"]


def test_static_list_is_flagged_as_not_point_in_time(tmp_path):
    p = tmp_path / "today.txt"
    p.write_text("# comment\nAAA\nBBB\n")
    m = bt.Membership(p)
    assert not m.point_in_time and m.on(19990101) == ["AAA", "BBB"]


def test_renamed_tickers_map_to_current_yahoo_symbol():
    assert dl.to_yf("ANTM") == "ELV"          # Anthem -> Elevance: history lives under ELV
    assert dl.to_yf("BRK.B") == "BRK-B"
    assert dl.to_yf("AAPL.NB") == "AAPL"
    assert dl.to_yf("00700.HK") == "0700.HK"


def _run(tmp_path, drifts, rows, days, start, end, top_n=3, rebal=5):
    prices = trending_prices(days, drifts)
    m = bt.Membership(membership_csv(tmp_path, rows))
    panel = bt.Panel(prices, days)
    return bt.run(panel, days, start, end, top_n, rebal, False, None, 1e6, 0.0, 0.0, m)


def test_non_members_are_never_traded_even_if_they_are_the_best_performers(tmp_path, no_catalyst):
    days = trading_days(160)
    members = [f"M{i}" for i in range(16)]
    drifts = {t: 0.0005 * i for i, t in enumerate(members)}
    drifts["WINNER"] = 0.02                           # by far the strongest trend, never a member
    res = _run(tmp_path, drifts, [(days[0], members)], days, days[100], days[-1])
    traded = {t["contract_code"] for t in res["trades"]}
    assert traded and "WINNER" not in traded


def test_a_stock_can_only_be_bought_after_it_joins_the_index(tmp_path, no_catalyst):
    days = trading_days(200)
    members = [f"M{i}" for i in range(16)]
    drifts = {t: 0.0005 * i for i, t in enumerate(members)}
    drifts["JOINER"] = 0.02
    join = days[150]
    res = _run(tmp_path, drifts, [(days[0], members), (join, members + ["JOINER"])],
               days, days[100], days[-1])
    buys = [int(t["trade_date"]) for t in res["trades"]
            if t["contract_code"] == "JOINER" and t["direction"] == 0]
    assert buys and min(buys) > join
