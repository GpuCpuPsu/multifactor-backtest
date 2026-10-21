"""
RUN 4 -- ABLATION: same fundamental screen, but 60-day momentum instead of 5-day.

Run 1 showed 5-day momentum alone returned +29.43% (Sharpe 0.42) against a +59.03%
(Sharpe 1.58) buy-and-hold benchmark -- i.e. the short lookback actively destroys value,
likely by buying short-term overextension right before mean reversion. This run keeps
everything else identical and only lengthens the lookback to 60 trading days, the more
conventional momentum horizon.

Price history is pre-loaded on the first bar via stock_api_quotation so the strategy is
not handicapped by sitting in cash for a 60-bar warm-up while the benchmark is invested
from day one.

TQX cross-sectional strategy: NASDAQ-100 hard fundamental filter + 60-day momentum top-5.

Universe:        NASDAQ-100 constituents (US, `.NB` suffix)
Hard filters (point-in-time, evaluated at every rebalance date using only data disclosed
on/before that date):
  - YoY revenue growth > 20%
  - Gross margin (gross profit / revenue) > 20%
  - Operating cash flow > 0
Ranking:          5-day price momentum (close[t] / close[t-5] - 1), descending
Selection:        top 5 eligible names (fewer held if fewer than 5 pass the hard filters;
                   an empty eligible set flattens the book to cash)
Rebalance:        every 5 trading days
Weighting:        equal weight across selected names, capped at 90% of account value

API usage below was verified against the live TQX environment on 2026-07-25 by running a
diagnostic probe strategy through `tqx-cli` and reading back `inspect.signature(...)` plus
real return values for AAPL.NB / NVDA.NB (a first draft of this file guessed wrong on several
of these and silently produced a zero-trade backtest -- see notes for what changed):

  1. `get_index_component(index_symbol="NDX", market="us")` -- NOT `index_code`/`market="nb"`.
     Returns point-in-time membership: columns `index_symbol`, `stock_symbol`, `added_date`,
     `deleted_date` (110 current NDX members confirmed). `deleted_date` is None while a name
     is still a constituent, so eligibility as of any date is
     `added_date <= date and (deleted_date is None or deleted_date > date)`.
  2. `get_financial_statement(start_quarter, end_quarter, symbol=..., date=..., is_latest=...,
     market="nb", interim_type="cumulative")` -- `start_quarter`/`end_quarter` are REQUIRED
     positional-or-keyword args (format `"YYYYqN"`), omitting them raises a TypeError. There is
     no `announcement_date` column in the returned frame -- the `date=<rebalance_date>` +
     `is_latest=False` combination already does point-in-time filtering server-side, so no
     extra client-side announcement-date filter is needed or possible.
  3. Field names actually populated for real tickers (many "reported_*" fields the docs implied
     were the primary revenue field are null for AAPL/NVDA):
       revenue            -> is_revenue_business_total (fallbacks below)
       gross profit       -> is_gross_profit
       operating cash flow -> cfs_net_cf_operating
  4. `fy_period` is `"FY<year>Q<n>"` (e.g. `"FY2025Q1"`) as originally assumed. Values under
     `interim_type="cumulative"` (the default) are cumulative FISCAL-YEAR-TO-DATE, not discrete
     quarter figures (confirmed: Apple's FY2025 Q1/Q2/Q3/Q4 gross profit accumulates across the
     year). Comparing the same `fy_period` slot one year apart (what `_prior_year_period` does)
     is therefore the correct way to get a clean YoY comparison with this data convention --
     naively differencing consecutive quarters would NOT give quarterly figures.

This file is written to satisfy `tqx-cli`'s AST safety pre-check (no eval/exec/open/compile/
input/globals/locals/vars/dir/__import__) and the cross-sectional rules in
`.claude/skills/tqx-research/references/cross_sectional_strategies.md`:
  - the sell phase always runs before checking whether the target list is empty
  - `targets = ranked[:top_n]` is never guarded by a "not enough candidates" early return
"""

from panda_backtest.api.api import *
from panda_backtest.api.stock_us_api import *
import tqx_data
import pandas as pd
import datetime


NDX_INDEX_SYMBOL = "NDX"

REVENUE_FIELD_CANDIDATES = [
    "is_revenue_business_total", "is_revenue_goods_services", "is_sales_goods_services_net",
    "is_reported_revenue", "is_reported_op_revenue",
]
GROSS_PROFIT_FIELD_CANDIDATES = [
    "is_gross_profit", "is_reported_gross_profit",
]
OPERATING_CF_FIELD_CANDIDATES = [
    "cfs_net_cf_operating", "cfs_reported_cf_operating", "cfs_cf_operating_before_wc",
]
FINANCIAL_FIELDS = ["symbol", "fy_period", "date"] + list(dict.fromkeys(
    REVENUE_FIELD_CANDIDATES + GROSS_PROFIT_FIELD_CANDIDATES + OPERATING_CF_FIELD_CANDIDATES
))


def initialize(context):
    context.account = "15032863"
    context.market = "nb"

    context.top_n = 5
    context.rebalance_every = 5
    context.momentum_window = 60
    context.max_position_ratio = 0.9
    context.min_revenue_growth = 0.20
    context.min_gross_margin = 0.20

    context.bar_count = 0
    context.today_trades = []
    context.warmed_up = False

    context.membership = _load_nasdaq100_membership()
    context.universe = sorted(set(context.membership["stock_symbol"].tolist())) \
        if context.membership is not None and len(context.membership) > 0 else []
    context.close_history = {symbol: [] for symbol in context.universe}


def _load_nasdaq100_membership():
    try:
        components = tqx_data.get_index_component(index_symbol=NDX_INDEX_SYMBOL, market="us")
    except Exception as exc:
        print(f"[nasdaq100_cs] get_index_component failed: {exc}")
        return None

    if components is None or len(components) == 0:
        return None

    components = components.dropna(subset=["stock_symbol"])
    components = components[components["stock_symbol"].str.endswith(".NB")]
    # added_date/deleted_date come back as a string column when populated, but pandas infers
    # an all-null deleted_date column as numeric -- normalize both to numeric YYYYMMDD ints so
    # comparisons in _active_symbols_for_date never hit a dtype mismatch (confirmed failure
    # mode: "ufunc 'greater' did not contain a loop ... Int32DType, StrDType").
    components["added_date"] = pd.to_numeric(components["added_date"], errors="coerce")
    components["deleted_date"] = pd.to_numeric(components["deleted_date"], errors="coerce")
    return components


def _active_symbols_for_date(context, as_of_date):
    if context.membership is None or len(context.membership) == 0:
        return []
    membership = context.membership
    as_of_int = int(as_of_date)
    active = membership[membership["added_date"] <= as_of_int]
    still_in = active["deleted_date"].isna() | (active["deleted_date"] > as_of_int)
    active = active[still_in]
    return sorted(set(active["stock_symbol"].tolist()))


def _shift_days(yyyymmdd, days):
    d = datetime.date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
    return (d - datetime.timedelta(days=days)).strftime("%Y%m%d")


def _warm_up_history(context, first_date):
    """Pre-load trailing closes so the 60-bar lookback is usable from the first rebalance."""
    if not context.universe:
        return
    start = _shift_days(first_date, 150)
    try:
        df = stock_api_quotation(
            symbol_list=context.universe,
            start_date=start,
            end_date=first_date,
            fields=["symbol", "date", "close"],
            period="1d",
        )
    except Exception as exc:
        print(f"[nasdaq100_cs] warmup stock_api_quotation failed: {exc}")
        return
    if df is None or len(df) == 0:
        print("[nasdaq100_cs] warmup returned no rows")
        return
    df = df.sort_values(["symbol", "date"])
    loaded = 0
    for symbol, grp in df.groupby("symbol"):
        closes = []
        for c in grp["close"].tolist():
            try:
                v = float(c)
            except (TypeError, ValueError):
                continue
            if v == v and v > 0:
                closes.append(v)
        if closes:
            context.close_history[symbol] = closes[-(context.momentum_window + 1):]
            loaded += 1
    print(f"[nasdaq100_cs] warmup loaded {loaded} symbols from {start} to {first_date}")


def _first_present(row, candidates):
    for name in candidates:
        if name in row and row[name] is not None:
            value = row[name]
            try:
                if value == value and abs(value) != float("inf"):
                    return float(value)
            except TypeError:
                continue
    return None


def _prior_year_period(fy_period):
    try:
        year = int(fy_period[2:6])
        suffix = fy_period[6:]
        return f"FY{year - 1}{suffix}"
    except (ValueError, IndexError, TypeError):
        return None


def _quarter_bounds(rebalance_date):
    year = int(str(rebalance_date)[:4])
    month = int(str(rebalance_date)[4:6])
    quarter = (month - 1) // 3 + 1
    start_quarter = f"{year - 2}q{quarter}"
    end_year, end_quarter_num = (year, quarter + 1) if quarter < 4 else (year + 1, 1)
    end_quarter = f"{end_year}q{end_quarter_num}"
    return start_quarter, end_quarter


def _fetch_eligible_symbols(context, rebalance_date, active_symbols):
    if not active_symbols:
        return []

    start_quarter, end_quarter = _quarter_bounds(rebalance_date)
    try:
        statements = tqx_data.get_financial_statement(
            start_quarter,
            end_quarter,
            symbol=active_symbols,
            date=rebalance_date,
            is_latest=False,
            market=context.market,
            interim_type="cumulative",
            fields=FINANCIAL_FIELDS,
        )
    except Exception as exc:
        print(f"[nasdaq100_cs] get_financial_statement failed at {rebalance_date}: {exc}")
        return []

    if statements is None or len(statements) == 0:
        return []

    statements = statements.dropna(subset=["symbol", "fy_period"])

    latest_by_symbol_period = statements.drop_duplicates(
        subset=["symbol", "fy_period"], keep="last"
    )
    latest_per_symbol = latest_by_symbol_period.sort_values(
        "fy_period"
    ).drop_duplicates(subset=["symbol"], keep="last")

    by_symbol_period = {
        (row["symbol"], row["fy_period"]): row
        for _, row in latest_by_symbol_period.iterrows()
    }

    eligible = []
    for _, row in latest_per_symbol.iterrows():
        symbol = row["symbol"]
        fy_period = row["fy_period"]
        prior_period = _prior_year_period(fy_period)
        prior_row = by_symbol_period.get((symbol, prior_period))
        if prior_row is None:
            continue

        revenue_now = _first_present(row, REVENUE_FIELD_CANDIDATES)
        revenue_prior = _first_present(prior_row, REVENUE_FIELD_CANDIDATES)
        gross_profit_now = _first_present(row, GROSS_PROFIT_FIELD_CANDIDATES)
        operating_cf_now = _first_present(row, OPERATING_CF_FIELD_CANDIDATES)

        if revenue_now is None or revenue_prior is None or revenue_prior <= 0:
            continue
        if gross_profit_now is None or revenue_now <= 0:
            continue
        if operating_cf_now is None:
            continue

        revenue_growth = revenue_now / revenue_prior - 1.0
        gross_margin = gross_profit_now / revenue_now

        if (
            revenue_growth > context.min_revenue_growth
            and gross_margin > context.min_gross_margin
            and operating_cf_now > 0
        ):
            eligible.append(symbol)

    return eligible


def before_trading(context):
    context.today_trades = []


def _current_trade_date(context, data):
    for symbol in context.universe:
        bar = data.get(symbol)
        if bar is not None and getattr(bar, "trade_date", None):
            return bar.trade_date
    return getattr(context, "now", None)


def handle_data(context, data):
    context.bar_count += 1

    if not context.warmed_up:
        first_date = _current_trade_date(context, data)
        if first_date is not None:
            _warm_up_history(context, first_date)
            context.warmed_up = True

    for symbol in context.universe:
        bar = data.get(symbol)
        if bar is None or getattr(bar, "close", None) is None:
            continue
        close_price = float(bar.close)
        if close_price <= 0:
            continue
        history = context.close_history.setdefault(symbol, [])
        history.append(close_price)
        max_len = context.momentum_window + 1
        if len(history) > max_len:
            history[:] = history[-max_len:]

    if context.bar_count % context.rebalance_every != 0:
        return

    rebalance_date = _current_trade_date(context, data)
    active_symbols = _active_symbols_for_date(context, rebalance_date)
    eligible = _fetch_eligible_symbols(context, rebalance_date, active_symbols)

    scores = {}
    for symbol in eligible:
        history = context.close_history.get(symbol, [])
        if len(history) < context.momentum_window + 1:
            continue
        past = history[-(context.momentum_window + 1)]
        current = history[-1]
        if past is None or past <= 0:
            continue
        scores[symbol] = current / past - 1.0

    ranked = sorted(scores, key=scores.get, reverse=True)
    targets = ranked[: context.top_n]  # allowed to be shorter than top_n

    account = context.stock_account_dict.get(context.account)
    if account is None:
        return

    for symbol, position in list(account.positions.items()):
        if symbol not in targets and position.sellable > 0:
            order_shares(context.account, symbol, -position.sellable, style=MarketOrderStyle)
            context.today_trades.append(
                {"symbol": symbol, "side": "SELL", "qty": position.sellable}
            )

    if not targets:
        print(
            f"[nasdaq100_cs] date={rebalance_date} eligible_count=0 target_symbols=[] "
            "-- flattened to cash"
        )
        return

    budget_per_name = account.cash * context.max_position_ratio / len(targets)
    for symbol in targets:
        bar = data.get(symbol)
        if bar is None or getattr(bar, "close", None) is None:
            continue
        price = float(bar.close)
        if price <= 0:
            continue
        position = account.positions.get(symbol)
        current_qty = 0 if position is None else position.quantity
        if current_qty > 0:
            continue
        buy_qty = int(budget_per_name // price)
        if buy_qty <= 0:
            continue
        order_shares(context.account, symbol, buy_qty, style=MarketOrderStyle)
        context.today_trades.append(
            {"symbol": symbol, "side": "BUY", "qty": buy_qty, "price": price}
        )

    if context.bar_count % 105 == 0:
        print(
            f"[nasdaq100_cs] date={rebalance_date} eligible_count={len(eligible)} "
            f"target_symbols={targets}"
        )


def after_trading(context):
    account = context.stock_account_dict.get(context.account)
    if account is None:
        return
    if context.bar_count % 105 == 0:
        print(
            f"[nasdaq100_cs after_trading] date={context.now}, "
            f"total_value={account.total_value:.2f}, cash={account.cash:.2f}"
        )
