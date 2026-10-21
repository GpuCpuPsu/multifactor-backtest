# Multi-Factor Equity Backtest

A momentum + volume stock-ranking strategy with an earnings-catalyst filter. It started on the [TQX](https://www.tqx.ai) quant platform and was rebuilt to run locally on free data, then stress-tested for backtest biases and for whether it beats random stock selection.

**Main finding:** after removing hindsight bias and look-ahead bias, the earnings-catalyst filter performed **no better than random stock selection**. The un-filtered momentum/volume tilt beat the market out-of-sample (2025–26), but mostly by riding one AI/semiconductor rally, so it is unproven.

## What's in here

| Path | What it is |
|---|---|
| `pipeline/strategy.py` | The strategy, in TQX's backtest format: trend + volume composite score, earnings-window filter, optional ATR stop |
| `pipeline/run_pipeline.py` | Orchestrator for TQX runs: rewrites config, creates/runs the strategy, scores results, parameter sweeps |
| `local/backtest.py` | Local backtester that reuses the strategy's signal code: point-in-time S&P 500 universe, walk-forward sweeps, random-pick placebo tests |
| `local/data.py` | Yahoo Finance price loader with disk cache and ticker-rename handling |
| `local/edgar_events.py` | Earnings dates from SEC EDGAR 8-K Item 2.02 filings, timestamped to the first tradable session |
| `local/spy_levels.py`, `local/spy_levels_stats.py` | Side study: SPY at its 30-day high/low, bounce vs. break, with a block-bootstrap significance test |
| `local/big_movers.py` | Side study: where ±10% one-day moves come from, and whether they can be predicted |
| `qube/` | The composite score as a TQX Qube factor, plus a step-by-step platform workflow |
| `strategies/` | Earlier TQX experiments (runs 1–10, benchmarks, probes) |
| `local/results/`, `pipeline/results/` | Saved results from the runs below |

## Setup

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Windows; use .venv/bin/python on macOS/Linux
```

Price data downloads on the first run and is cached in `local/cache/`.

## Running it

```bash
# Walk-forward with point-in-time S&P 500 membership and SEC earnings dates, 50 random-pick placebos per setting
python local/backtest.py --start 20230101 --end 20241231 --grid "3:5:off,5:5:off,3:10:off,5:10:off" \
    --universe local/ref/sp500_components_by_date.csv --events local/ref/events_sec_8k_202.csv \
    --placebo 50 --label IS --out local/results/sweep.jsonl

# Same strategy without the earnings filter
python local/backtest.py ... --no-catalyst

# Refresh earnings dates from SEC EDGAR (SEC requires a contact in the User-Agent)
SEC_USER_AGENT="Your Name you@example.com" python local/edgar_events.py
```

`--grid` entries are `top_n:rebalance_days:stop(on|off)`.

## Bias fixes

- **Hindsight stock selection (survivorship bias):** the first local test picked from a list of stocks that are large today. Switching to the stocks that were actually in the S&P 500 on each date cut the same backtest's return from **+141% to +40%**.
- **Look-ahead in the price signal:** the trend score used the same closing price the strategy then traded at. It now uses the previous close.
- **Look-ahead in earnings timing:** SEC acceptance timestamps are UTC. Read as Eastern time, after-close releases looked known during that day's session. They are now mapped to the next tradable session.

## Results (point-in-time S&P 500, 0.001 slippage)

| Strategy | 2023–24 (in-sample) | 2025–26 (out-of-sample) | vs. random picks |
|---|---|---|---|
| With earnings filter (5 stocks, 10-day rebalance) | +20% | +36% | no better than random (p = 0.14 / 0.42) |
| Without earnings filter | +40% | +147% | out-of-sample beat all 50 placebos on return, but not clearly on Sharpe |
| SPY buy-and-hold | +58% | +35% | — |
| Universe equal-weight | +43% | +56% | — |

**p** is the share of 50 random-pick runs (same eligible stocks, random ranking) that did at least as well as the strategy.

Side studies: the SPY 30-day-low vs. 30-day-high return gap was **not significant** after accounting for overlapping samples (block bootstrap p ≈ 0.14–0.18). ±10% one-day moves were predictable in size out-of-sample (AUC 0.90, mostly from earnings timing and volatility), but not in direction (AUC 0.53).

## Limitations

- About 4% of past S&P 500 members have no Yahoo price data (mostly acquisitions, plus the 2023 bank failures), so a little survivorship bias remains.
- Costs are modelled as slippage only. Prices are split/dividend adjusted.
- Results cover 2023–2026 only. This is research, not investment advice.

## Data credits

- S&P 500 historical membership: [fja05680/sp500](https://github.com/fja05680/sp500), MIT License (see `local/ref/LICENSE-sp500-data.txt`)
- Earnings filings: [SEC EDGAR](https://www.sec.gov/edgar) (public data)
- Prices: Yahoo Finance via [yfinance](https://github.com/ranaroussi/yfinance) (not redistributed; downloaded at run time)
