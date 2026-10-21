# Qube workflow draft — Trend + Volume composite (US)

Nothing here has been run on TQX. Steps marked **(credits)** spend platform balance.

## Where the building blocks live

Qube: <https://www.tqx.ai/agent_quant> (docs: <https://docs.tqx.ai/research/info>)

| Block | Where in Qube | CLI (`tqx.cmd` on Windows) |
|---|---|---|
| Factor (the signal) | **Factor Library → + US factor** | `tqx research factor create/save/run/result` |
| Factor analysis (grouped forward returns, IC) | factor detail page → **Run factor analysis** | `tqx research factor run <id>` |
| Factor → strategy (universe, top N, rebalance, weights, exits) | ask Qube in chat to turn the factor into a strategy; confirm the proposal | — (chat / UI only) |
| Strategy + versions | **Strategy Library → + US strategy** | `tqx research strategy create/save/versions` |
| Backtest | strategy detail → run | `tqx research strategy run <id>` |
| Parameter optimization | strategy detail → optimization | — (UI only) |
| Run history | **Run history** | `tqx research backtest list/result` |

A strategy is one market only; US symbols are `.NB`, HK `.HK`.

## The draft

### 1. Factor — `trend_volume_factor.py` (Python mode)

`0.5·z(trend) + 0.5·z(volume ratio)`: trend is the mean of the 20- and 60-day return, and volume ratio is mean volume over 5 days ÷ 20 days. Both use **completed bars only**. The factor was checked locally: it matches `strategy.py`'s `compute_scores` to 2e-16, with the same top 5 on 2024-10-15.

`trend_volume_factor.formula` is the same thing in Formula mode. It has **not** been validated, because the formula engine isn't available locally. Use it only if you prefer Formula mode, and compare it against the Python version on one date.

```bash
tqx.cmd --json research factor create --market=us --file=qube/trend_volume_factor.py --name="Trend+Volume composite" --description="0.5 z(20/60d return, prior close) + 0.5 z(vol5/vol20, prior sessions)"
```

### 2. Factor analysis — **(credits)**, same walk-forward split as local

| Field | In-sample | Out-of-sample |
|---|---|---|
| Period | 2023-01-01 → 2024-12-31 | 2025-01-01 → 2026-09-22 |
| Rebalance cycle | 10 | 10 |
| Groups | 5 | 5 |
| Direction | Positive | Positive |

```bash
tqx.cmd --json research factor run <FACTOR_ID> --startDate=20230101 --endDate=20241231 --adjustmentCycle=10 --groupNumber=5 --factorDirection=Positive
```

What to look for: the top group should beat the bottom group **monotonically** across all 5 groups, and IC should stay positive in both windows. If only the top group wins, and only out-of-sample, that matches the local result: a trend tilt that paid off in the 2025–26 rally, not a stable edge.

### 3. Factor → strategy proposal (confirm these before Qube creates it)

| Parameter | Value | Why |
|---|---|---|
| Universe | US market universe | **Differs from local**: local = point-in-time S&P 500 top 100 by $ volume |
| Top N | 5 | frozen from the original in-sample choice |
| Rebalance period | 10 trading days | frozen |
| Factor direction | higher is better | |
| Weighting | equal | as in `strategy.py` (90% deployed ÷ N) |
| Position cap | 20% | 1/N |
| Take-profit / stop-loss | none | ATR stop was worse in-sample and out-of-sample |
| Minimum holding period | none | |

### 4. Backtest — **(credits)**
Run the saved version on the out-of-sample window only (slippage 0.001, 1d). Compare it with the local `wf_nocat_oos.jsonl` row. Expect the numbers to differ because of the universe and data differences, but the *sign vs. benchmark* should agree.

### 5. Optimization — in-sample window only
Grid `top N ∈ {3, 5}`, `rebalance ∈ {5, 10}`. Never optimize on 2025–26, or the out-of-sample test stops meaning anything.

## Not in this draft: the earnings catalyst gate
Qube strategies can't call `get_event_fina_disclosure`. The platform's validator rejects it. The draft is therefore the **ungated** composite, which is directly comparable to the local no-catalyst runs. To add the gate on the platform later, choose one:
- `get_financial_statement` announcement dates, excluding Q4/annual (your ~20-day lag rule);
- `get_filing_announcement` (allowed, but its fields are undocumented, so it needs one probe run);
- keep the gate local only, using the EDGAR 8-K Item 2.02 dates in `local/ref/events_sec_8k_202.csv`.
