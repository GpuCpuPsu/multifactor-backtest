"""Look-ahead in earnings timing: an SEC filing is only actionable from the first session
that opens after it was accepted, and the strategy may only see events already known."""
import pandas as pd
import pytest

import edgar_events as ee
from helpers import STRAT, bt, strategy_context

SESSIONS = [20240201, 20240202, 20240205, 20241010, 20241011, 20241014, 20241031, 20241101]


@pytest.mark.parametrize("raw, expected_et", [
    ("2024-02-01T21:30:30.000Z", "2024-02-01 16:30"),   # winter: UTC-5 (AAPL after close)
    ("2024-10-31T20:30:25.000Z", "2024-10-31 16:30"),   # summer: UTC-4
    ("2024-10-11T10:46:16.000Z", "2024-10-11 06:46"),   # JPM pre-market
])
def test_edgar_timestamps_are_utc_and_converted_to_eastern(raw, expected_et):
    assert ee.acceptance_et(raw).strftime("%Y-%m-%d %H:%M") == expected_et


@pytest.mark.parametrize("raw, session", [
    ("2024-02-01T21:30:30.000Z", 20240202),   # after the close -> next session
    ("2024-10-11T10:46:16.000Z", 20241011),   # pre-market -> same session
    ("2024-10-11T19:59:00.000Z", 20241011),   # 15:59 ET -> still tradable that day
    ("2024-10-11T20:00:00.000Z", 20241014),   # 16:00 ET Friday -> Monday
    ("2024-02-03T15:00:00.000Z", 20240205),   # Saturday -> Monday
])
def test_filing_maps_to_first_actionable_session(raw, session):
    assert ee.session_for(ee.acceptance_et(raw), SESSIONS) == session


def test_session_mapping_does_not_depend_on_calendar_order():
    ts = ee.acceptance_et("2024-10-11T10:46:16.000Z")
    assert ee.session_for(ts, sorted(SESSIONS, reverse=True)) == 20241011


def _events(tmp_path, rows):
    p = tmp_path / "events.csv"
    pd.DataFrame(rows, columns=["symbol", "event_date", "known_date", "source", "period"]
                 ).to_csv(p, index=False)
    return bt.load_events(p)


def test_events_are_invisible_before_their_known_date(tmp_path):
    ev = _events(tmp_path, [("AAPL.NB", "20240502", "20240410", "filing_announcement", "")])
    assert bt.events_as_of(ev, 20240409, ["AAPL"]) == {}
    assert bt.events_as_of(ev, 20240410, ["AAPL"]) == {"AAPL": [20240502]}


def test_q4_and_annual_financial_statement_dates_are_dropped(tmp_path):
    ev = _events(tmp_path, [
        ("MSFT", "20240125", "", "financial_statement", "FY2023Q4"),   # dropped: Q4
        ("NVDA", "20240221", "", "financial_statement", "FY2024"),     # dropped: annual
        ("MSFT", "20240425", "", "financial_statement", "FY2024Q3"),   # kept
        ("AAPL", "20240201", "", "filing_announcement", "FY2023Q4"),   # kept: not a statement
    ])
    assert sorted(zip(ev["symbol"], ev["ev"])) == [("AAPL", 20240201), ("MSFT", 20240425)]


def test_catalyst_window_is_one_day_before_to_three_days_after():
    days = [20240429, 20240430, 20240501, 20240502, 20240503, 20240506, 20240507, 20240508]
    ctx = strategy_context({}, days=days, events={"AAPL": [20240502]})
    flag = {d: STRAT._catalyst_flag(ctx, "AAPL", str(d)) for d in days}
    assert flag[20240430] == (0, False)
    assert flag[20240501] == (1, True)        # the day before: flagged as pre-earnings
    assert flag[20240502] == (1, False)
    assert flag[20240507] == (1, False)       # +3 trading days (weekend skipped)
    assert flag[20240508] == (0, False)
