"""Look-ahead bias: the signal on day t must not use day t's bar, because orders fill on it."""
import numpy as np
import pandas as pd
import pytest

from helpers import ROOT, STRAT, strategy_context


def _hist(seed, n=100):
    rng = np.random.default_rng(seed)
    close = list(100 * np.cumprod(1 + rng.normal(0.001, 0.02, n)))
    vol = list(rng.uniform(5e5, 2e6, n))
    return {"close": close, "high": [c * 1.01 for c in close],
            "low": [c * 0.99 for c in close], "vol": vol}


@pytest.fixture
def universe_hist():
    return {f"S{i}.NB": _hist(i) for i in range(8)}


def test_todays_close_and_volume_do_not_change_scores(universe_hist, no_catalyst):
    before, _ = STRAT.compute_scores(strategy_context(universe_hist), "20240102")
    shocked = {s: {k: list(v) for k, v in h.items()} for s, h in universe_hist.items()}
    shocked["S3.NB"]["close"][-1] *= 3.0      # today's bar: price triples...
    shocked["S3.NB"]["vol"][-1] *= 50.0       # ...on 50x volume
    after, _ = STRAT.compute_scores(strategy_context(shocked), "20240102")
    assert after == pytest.approx(before)


def test_yesterdays_close_does_change_scores(universe_hist, no_catalyst):
    """Guard against a vacuous pass: the signal must react to the last COMPLETED bar."""
    before, _ = STRAT.compute_scores(strategy_context(universe_hist), "20240102")
    shocked = {s: {k: list(v) for k, v in h.items()} for s, h in universe_hist.items()}
    shocked["S3.NB"]["close"][-2] *= 3.0
    after, _ = STRAT.compute_scores(strategy_context(shocked), "20240102")
    assert after["S3.NB"] > before["S3.NB"]


def test_names_without_enough_history_are_not_scored(universe_hist, no_catalyst):
    short = dict(universe_hist)
    short["NEW.NB"] = {k: v[-30:] for k, v in _hist(99).items()}   # listed 30 days ago
    scores, _ = STRAT.compute_scores(strategy_context(short), "20240102")
    assert "NEW.NB" not in scores


def test_qube_factor_matches_strategy_scores(universe_hist, no_catalyst):
    """The uploaded Qube factor (qube/trend_volume_factor.py) must equal compute_scores."""
    class Factor:
        pass
    ns = {"Factor": Factor}
    exec((ROOT / "qube" / "trend_volume_factor.py").read_text(), ns)

    days = list(range(1, 101))
    frames = [pd.DataFrame({"symbol": s, "date": days, "close": h["close"], "volume": h["vol"]})
              for s, h in universe_hist.items()]
    long = pd.concat(frames).set_index(["symbol", "date"]).sort_index()
    factor = ns["TrendVolumeComposite"]().calculate(long).xs(100, level="date")

    scores, _ = STRAT.compute_scores(strategy_context(universe_hist), "20240102")
    for s, v in scores.items():
        assert factor[s] == pytest.approx(v, abs=1e-12)
