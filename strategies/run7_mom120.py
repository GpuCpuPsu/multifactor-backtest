"""
RUN 6 -- SURVIVORSHIP-FREE UNIVERSE, T-1 RANKING, COST SENSITIVITY.

Fixes the forward peek in Run 5: that version ranked the universe by dollar volume on the
SAME day it traded, but a day's full volume is not known until after the close. This
version ranks on the most recent COMPLETED session strictly before the current date, so
every input is available at decision time.

Otherwise identical to Run 5. Slippage is varied across workflows to test whether the
edge survives realistic execution costs.

--- original Run 5 header follows ---
RUN 5 -- SURVIVORSHIP-FREE UNIVERSE TEST.

Run 4 (fundamentals + 60-day momentum) produced +162% / Sharpe 2.72 on a universe built
from `get_index_component("NDX")`, which the bias audit showed carries NO deletion history
(110 rows, 0 deleted_date values). Names that were NDX constituents during the test window
but have since been removed are simply absent -- and index removals skew toward laggards.

This run keeps the strategy logic byte-for-byte identical and swaps ONLY the universe for
one that cannot suffer survivorship bias, because it is rebuilt from what actually traded:

    universe(t) = top 100 US names by dollar volume ON DATE t, from get_us_daily

`get_us_daily(symbol=[], start_date=t, end_date=t)` returns every symbol that printed a
trade that day -- 5,534 of them on 2025-01-02, including names since delisted. Ranking that
by same-day dollar volume uses only information available at the close of day t, so there
is no look-ahead, and no dependence on a vendor's current membership snapshot.

Universe + price history refresh monthly (every 21 bars) to keep the number of large
queries bounded; the rebalance cadence stays at 5 bars as in every other run.
"""

from panda_backtest.api.api import *
from panda_backtest.api.stock_us_api import *
import tqx_data
import pandas as pd
import datetime


UNIVERSE_SIZE = 100
UNIVERSE_REFRESH_BARS = 21

REVENUE_FIELD_CANDIDATES = [
    "is_revenue_business_total", "is_revenue_goods_services", "is_sales_goods_services_net",
    "is_reported_revenue", "is_reported_op_revenue",
]
GROSS_PROFIT_FIELD_CANDIDATES = ["is_gross_profit", "is_reported_gross_profit"]
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
    context.momentum_window = 120
    context.max_position_ratio = 0.9
    context.min_revenue_growth = 0.20
    context.min_gross_margin = 0.20

    context.bar_count = 0
    context.universe = []
    context.close_history = {}
    context.last_universe_refresh = -999


def _shift_days(yyyymmdd, days):
    d = datetime.date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
    return (d - datetime.timedelta(days=days)).strftime("%Y%m%d")


def _today(context, data):
    raw = getattr(context, "now", None)
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else None


def _refresh_universe(context, as_of):
    """Top-N by dollar volume on the last COMPLETED session before `as_of` (no forward peek)."""
    lookback_start = _shift_days(as_of, 8)
    lookback_end = _shift_days(as_of, 1)
    try:
        df = tqx_data.get_us_daily(symbol=[], start_date=lookback_start,
                                   end_date=lookback_end, market=context.market)
    except Exception as exc:
        print(f"[run6_liq] get_us_daily failed at {as_of}: {exc}")
        return False
    if df is None or len(df) == 0:
        print(f"[run6_liq] get_us_daily empty for {lookback_start}..{lookback_end}")
        return False

    # keep only the most recent completed session in the window
    try:
        last_session = df["date"].max()
        df = df[df["date"] == last_session]
    except Exception as exc:
        print(f"[run6_liq] could not isolate last session: {exc}")
        return False

    df = df.dropna(subset=["symbol", "amount"]).copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])
    df = df[df["amount"] > 0]
    if len(df) == 0:
        return False

    df = df.sort_values("amount", ascending=False)
    universe = df.head(UNIVERSE_SIZE)["symbol"].tolist()
    universe = [s for s in universe if isinstance(s, str) and s.endswith(".NB")]
    if not universe:
        return False

    context.universe = universe
    _reload_history(context, as_of)
    print(f"[run6_liq] date={as_of} ranked_on={last_session} traded={len(df)} "
          f"universe={len(universe)}")
    return True


def _reload_history(context, as_of):
    """Rebuild trailing closes for the current universe so 60-bar momentum is usable."""
    warm_days = max(150, int(context.momentum_window * 2.4))
    start = _shift_days(as_of, warm_days)
    try:
        q = stock_api_quotation(
            symbol_list=context.universe,
            start_date=start,
            end_date=as_of,
            fields=["symbol", "date", "close"],
            period="1d",
        )
    except Exception as exc:
        print(f"[run6_liq] history reload failed at {as_of}: {exc}")
        return
    if q is None or len(q) == 0:
        return
    q = q.sort_values(["symbol", "date"])
    hist = {}
    for symbol, grp in q.groupby("symbol"):
        closes = []
        for c in grp["close"].tolist():
            try:
                v = float(c)
            except (TypeError, ValueError):
                continue
            if v == v and v > 0:
                closes.append(v)
        if closes:
            hist[symbol] = closes[-(context.momentum_window + 1):]
    context.close_history = hist


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
        return f"FY{year - 1}{fy_period[6:]}"
    except (ValueError, IndexError, TypeError):
        return None


def _quarter_bounds(rebalance_date):
    year = int(str(rebalance_date)[:4])
    month = int(str(rebalance_date)[4:6])
    quarter = (month - 1) // 3 + 1
    end_year, end_q = (year, quarter + 1) if quarter < 4 else (year + 1, 1)
    return f"{year - 2}q{quarter}", f"{end_year}q{end_q}"


def _fetch_eligible_symbols(context, rebalance_date, active_symbols):
    if not active_symbols:
        return []
    start_quarter, end_quarter = _quarter_bounds(rebalance_date)
    try:
        statements = tqx_data.get_financial_statement(
            start_quarter, end_quarter,
            symbol=active_symbols, date=rebalance_date, is_latest=False,
            market=context.market, interim_type="cumulative", fields=FINANCIAL_FIELDS,
        )
    except Exception as exc:
        print(f"[run6_liq] get_financial_statement failed at {rebalance_date}: {exc}")
        return []
    if statements is None or len(statements) == 0:
        return []

    statements = statements.dropna(subset=["symbol", "fy_period"])
    latest_by_sp = statements.drop_duplicates(subset=["symbol", "fy_period"], keep="last")
    latest_per_symbol = latest_by_sp.sort_values("fy_period").drop_duplicates(
        subset=["symbol"], keep="last")
    by_sp = {(r["symbol"], r["fy_period"]): r for _, r in latest_by_sp.iterrows()}

    eligible = []
    for _, row in latest_per_symbol.iterrows():
        symbol = row["symbol"]
        prior_row = by_sp.get((symbol, _prior_year_period(row["fy_period"])))
        if prior_row is None:
            continue
        rev_now = _first_present(row, REVENUE_FIELD_CANDIDATES)
        rev_prior = _first_present(prior_row, REVENUE_FIELD_CANDIDATES)
        gp_now = _first_present(row, GROSS_PROFIT_FIELD_CANDIDATES)
        ocf_now = _first_present(row, OPERATING_CF_FIELD_CANDIDATES)
        if rev_now is None or rev_prior is None or rev_prior <= 0:
            continue
        if gp_now is None or rev_now <= 0 or ocf_now is None:
            continue
        if (rev_now / rev_prior - 1.0 > context.min_revenue_growth
                and gp_now / rev_now > context.min_gross_margin
                and ocf_now > 0):
            eligible.append(symbol)
    return eligible


def handle_data(context, data):
    context.bar_count += 1
    as_of = _today(context, data)
    if as_of is None:
        return

    if context.bar_count - context.last_universe_refresh >= UNIVERSE_REFRESH_BARS:
        if _refresh_universe(context, as_of):
            context.last_universe_refresh = context.bar_count

    if not context.universe:
        return

    for symbol in context.universe:
        bar = data.get(symbol)
        if bar is None or getattr(bar, "close", None) is None:
            continue
        price = float(bar.close)
        if price <= 0:
            continue
        hist = context.close_history.setdefault(symbol, [])
        hist.append(price)
        if len(hist) > context.momentum_window + 1:
            hist[:] = hist[-(context.momentum_window + 1):]

    if context.bar_count % context.rebalance_every != 0:
        return

    eligible = _fetch_eligible_symbols(context, as_of, context.universe)

    scores = {}
    for symbol in eligible:
        hist = context.close_history.get(symbol, [])
        if len(hist) < context.momentum_window + 1:
            continue
        past, current = hist[-(context.momentum_window + 1)], hist[-1]
        if past is None or past <= 0:
            continue
        scores[symbol] = current / past - 1.0

    ranked = sorted(scores, key=scores.get, reverse=True)
    targets = ranked[: context.top_n]

    account = context.stock_account_dict.get(context.account)
    if account is None:
        return

    for symbol, position in list(account.positions.items()):
        if symbol not in targets and position.sellable > 0:
            order_shares(context.account, symbol, -position.sellable, style=MarketOrderStyle)

    if not targets:
        return

    budget = account.cash * context.max_position_ratio / len(targets)
    for symbol in targets:
        bar = data.get(symbol)
        if bar is None or getattr(bar, "close", None) is None:
            continue
        price = float(bar.close)
        if price <= 0:
            continue
        position = account.positions.get(symbol)
        if position is not None and position.quantity > 0:
            continue
        qty = int(budget // price)
        if qty <= 0:
            continue
        order_shares(context.account, symbol, qty, style=MarketOrderStyle)

    if context.bar_count % 105 == 0:
        print(f"[run6_liq] date={as_of} universe={len(context.universe)} "
              f"eligible={len(eligible)} targets={targets}")


def after_trading(context):
    account = context.stock_account_dict.get(context.account)
    if account is None:
        return
    if context.bar_count % 105 == 0:
        print(f"[run6_liq] date={context.now} bar={context.bar_count} "
              f"total_value={account.total_value:,.2f}")
