"""Tech big-mover study: every predictor is known by the previous close, and each +/-10%
move gets one cause (earnings > market day > sector day > stock-specific)."""
import numpy as np
import pandas as pd
import pytest

import big_movers as bm


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    days = list(range(1, 301))
    cols = ["NVDA", "AMD", "XOM"]
    C = pd.DataFrame(100 * np.cumprod(1 + rng.normal(0, 0.02, (300, 3)), axis=0),
                     index=days, columns=cols)
    V = pd.DataFrame(rng.uniform(1e6, 2e6, (300, 3)), index=days, columns=cols)
    vix = pd.Series(rng.uniform(12, 30, 300), index=days)
    E = pd.DataFrame(False, index=days, columns=cols)
    return C, V, vix, E, {"NVDA": True, "AMD": True, "XOM": False}


def test_features_for_day_t_ignore_day_t_prices(frames):
    C, V, vix, E, tech = frames
    base = bm.build_features(C, V, vix, E, tech)
    C2, V2, vix2 = C.copy(), V.copy(), vix.copy()
    C2.loc[280] *= 2.0; V2.loc[280] *= 20; vix2.loc[280] = 80     # day 280 shock
    shocked = bm.build_features(C2, V2, vix2, E, tech)
    for k in base:
        pd.testing.assert_series_equal(base[k].loc[280], shocked[k].loc[280])
        if k not in ("earn", "tech"):
            assert not base[k].loc[281].equals(shocked[k].loc[281]), k   # seen the next day


def test_tech_flag_follows_the_sector_map(frames):
    f = bm.build_features(*frames)
    assert list(f["tech"].iloc[0]) == [1.0, 1.0, 0.0]


def _rows(rows):
    df = pd.DataFrame(rows, columns=["date", "symbol", "ret", "earn", "sector"])
    df["big"] = df["ret"].abs() >= 0.10
    return df


def _causes(rows, spy=None):
    df = _rows(rows)
    spy_r = pd.Series(spy or {}, dtype=float)
    out = bm.attribute_causes(df, spy_r)
    return dict(zip(out["symbol"] + "@" + out["date"].astype(str), out["cause"]))


def test_earnings_wins_even_on_a_market_wide_day():
    rows = [(1, f"S{i}", 0.12, 0, "Tech") for i in range(10)] + [(1, "NVDA", 0.15, 1, "Tech")]
    assert _causes(rows)["NVDA@1"] == "earnings"


def test_ten_big_movers_or_a_3pct_spy_day_is_a_market_day():
    rows = [(1, f"S{i}", 0.12, 0, f"Sec{i}") for i in range(10)] + [(2, "AMD", 0.11, 0, "Tech")]
    c = _causes(rows, spy={2: -0.035})
    assert c["S0@1"] == "market_day" and c["AMD@2"] == "market_day"


def test_three_same_direction_sector_peers_make_a_sector_day():
    peers = [(1, f"P{i}", 0.06, 0, "Tech") for i in range(3)]
    assert _causes([(1, "AMD", 0.12, 0, "Tech")] + peers)["AMD@1"] == "sector_day"


def test_peers_moving_the_other_way_or_too_few_do_not_count():
    opposite = [(1, f"P{i}", -0.06, 0, "Tech") for i in range(3)]
    two = [(2, f"Q{i}", 0.06, 0, "Tech") for i in range(2)]
    c = _causes([(1, "AMD", 0.12, 0, "Tech"), (2, "NVDA", 0.12, 0, "Tech")] + opposite + two)
    assert c["AMD@1"] == "stock_specific" and c["NVDA@2"] == "stock_specific"
