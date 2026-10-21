import pytest

from helpers import STRAT


@pytest.fixture
def no_catalyst(monkeypatch):
    """Disable the earnings gate (every scored name eligible) for one test."""
    monkeypatch.setattr(STRAT, "_catalyst_flag", lambda ctx, sym, today: (1, False))
