"""SPY 30-day high/low study: levels use only earlier bars, and each touch is classified
as bounce / break / failed_break / weak_break by the rules in local/spy_levels.py."""
import pandas as pd

import spy_levels as sl

BASE = 20240101
SPIKE, T = 30, 45     # spike sets resistance 105 / support 95 for bars 31-60; scenario at 45
RULES = dict(lookback=30, tol=0.005, horizon=5, vol_mult=1.5, vol_window=20, rev_pct=0.01)


def _series(scenario, n=80):
    """Flat 100 (high 101 / low 99, volume 1000) plus a spike bar, plus `scenario`:
    {bar: (high, low, close, volume)}."""
    rows = []
    for i in range(n):
        h, l, c, v = 101.0, 99.0, 100.0, 1000.0
        if i == SPIKE:
            h, l = 105.0, 95.0
        if i in scenario:
            h, l, c, v = scenario[i]
        rows.append({"High": h, "Low": l, "Close": c, "Volume": v})
    return pd.DataFrame(rows, index=range(BASE, BASE + n))


def _resistance_at_T(df, hold=2):
    ev = sl.find_events(df, hold=hold, **RULES)
    return ev[(ev.side == "resistance") & (ev.date == BASE + T)].iloc[0]


def test_level_is_built_from_earlier_bars_only():
    # the touch bar's own 106 high must not lift its level above the prior max of 105
    ev = _resistance_at_T(_series({T: (106, 104, 104.5, 1000), T + 1: (104, 102, 103, 1000)}))
    assert ev["level"] == 105.0


def test_bounce():
    ev = _resistance_at_T(_series({T: (105, 103, 104.5, 1000), T + 1: (104, 102, 103.5, 1000)}))
    assert ev["outcome"] == "bounce" and ev["bars_to_outcome"] == 1


def test_break_needs_volume_and_two_closes_holding_above():
    s = _series({T: (106.5, 104, 106, 3000), T + 1: (107, 105.5, 106.5, 1000),
                 T + 2: (107.5, 106, 107, 1000)})
    assert _resistance_at_T(s)["outcome"] == "break"


def test_volume_break_that_closes_back_inside_is_a_failed_break():
    s = _series({T: (106.5, 104, 106, 3000), T + 1: (106, 103, 104, 1000)})
    assert _resistance_at_T(s)["outcome"] == "failed_break"
    assert _resistance_at_T(s, hold=0)["outcome"] == "break"     # old rule: no hold check


def test_close_beyond_without_volume_is_a_weak_break():
    s = _series({T: (106.5, 104, 106, 1200), T + 1: (107, 105, 106.5, 1000)})
    assert _resistance_at_T(s)["outcome"] == "weak_break"


def test_hold_rule_relabels_breaks_without_changing_which_touches_count():
    s = _series({T: (106.5, 104, 106, 3000), T + 1: (106, 103, 104, 1000)})
    a = sl.find_events(s, hold=0, **RULES)
    b = sl.find_events(s, hold=2, **RULES)
    assert a[["date", "side"]].equals(b[["date", "side"]])


def test_touches_too_close_to_the_end_of_data_are_dropped_not_labelled_unresolved():
    # after bar 60 the flat 101 highs keep touching an unbroken 101 level: unresolved touches
    ev = sl.find_events(_series({}, n=80), hold=2, **RULES)
    unresolved = ev[ev.outcome == "unresolved"]
    assert len(unresolved) > 0
    assert (unresolved.date < BASE + 80 - RULES["horizon"]).all()
