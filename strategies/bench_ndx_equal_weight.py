"""
RUN 0 -- BENCHMARK CONTROL: equal-weight NASDAQ-100 buy-and-hold.

This is the control every other run gets measured against. It answers the question our
first backtest never asked: what did the universe itself do over the same window?

Design choices, all made to isolate SELECTION as the only difference vs. the live strategy:

  * Same universe (NDX), same weighting (equal), same starting capital, same costs.
  * Same 90% max deployment as the strategy, so the comparison isn't distorted by one
    side carrying more cash drag than the other.
  * Point-in-time membership: constituents are resolved as of the FIRST bar using
    `added_date <= date and (deleted_date is null or deleted_date > date)`. Using today's
    NDX membership for a backtest starting in 2025 would import survivorship bias --
    names added to the index in 2026 precisely because they performed well. That is the
    single easiest way to fake a good benchmark, so it is explicitly avoided here.
  * Buy once on the first bar, then hold. No rebalancing, no trading.

API usage verified against the live environment 2026-07-25 (see
strategies/nasdaq100_fundamental_momentum_cs.py header for the full notes).
"""

from panda_backtest.api.api import *
from panda_backtest.api.stock_us_api import *
import tqx_data
import pandas as pd


NDX_INDEX_SYMBOL = "NDX"


def initialize(context):
    context.account = "15032863"
    context.max_position_ratio = 0.9

    context.bought = False
    context.bar_count = 0
    context.n_target = 0
    context.n_ordered = 0

    context.membership = _load_membership()


def _load_membership():
    try:
        components = tqx_data.get_index_component(index_symbol=NDX_INDEX_SYMBOL, market="us")
    except Exception as exc:
        print(f"[bench] get_index_component failed: {exc}")
        return None

    if components is None or len(components) == 0:
        print("[bench] get_index_component returned nothing")
        return None

    components = components.dropna(subset=["stock_symbol"])
    components = components[components["stock_symbol"].str.endswith(".NB")]
    components["added_date"] = pd.to_numeric(components["added_date"], errors="coerce")
    components["deleted_date"] = pd.to_numeric(components["deleted_date"], errors="coerce")
    return components


def _active_symbols(context, as_of_date):
    membership = context.membership
    if membership is None or len(membership) == 0:
        return []
    as_of = int(as_of_date)
    active = membership[membership["added_date"] <= as_of]
    still_in = active["deleted_date"].isna() | (active["deleted_date"] > as_of)
    active = active[still_in]
    return sorted(set(active["stock_symbol"].tolist()))


def _today(context):
    raw = getattr(context, "now", None)
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else None


def handle_data(context, data):
    context.bar_count += 1

    if context.bought:
        return

    account = context.stock_account_dict.get(context.account)
    if account is None:
        return

    as_of = _today(context)
    if as_of is None:
        return

    universe = _active_symbols(context, as_of)
    if not universe:
        print(f"[bench] date={as_of} no active constituents resolved -- not investing")
        return

    # Only fund names that actually have a usable bar on the entry date.
    tradable = []
    for symbol in universe:
        bar = data.get(symbol)
        if bar is None:
            continue
        close_val = getattr(bar, "close", None)
        if close_val is None:
            continue
        price = float(close_val)
        if price <= 0 or price != price:
            continue
        tradable.append((symbol, price))

    if not tradable:
        print(f"[bench] date={as_of} constituents resolved but no usable bars -- waiting")
        return

    context.n_target = len(tradable)
    budget_per_name = account.cash * context.max_position_ratio / len(tradable)

    for symbol, price in tradable:
        qty = int(budget_per_name // price)
        if qty <= 0:
            continue
        order_shares(context.account, symbol, qty, style=MarketOrderStyle)
        context.n_ordered += 1

    context.bought = True
    print(
        f"[bench] date={as_of} ENTRY  constituents_active={len(universe)} "
        f"tradable={len(tradable)} orders_submitted={context.n_ordered} "
        f"budget_per_name={budget_per_name:,.0f} cash_before={account.cash:,.0f}"
    )


def after_trading(context):
    account = context.stock_account_dict.get(context.account)
    if account is None:
        return
    # Log the entry day and then roughly monthly, to keep a sanity trail without flooding.
    if context.bar_count <= 1 or context.bar_count % 21 == 0:
        held = len([p for p in account.positions.values() if p.quantity > 0])
        print(
            f"[bench] date={context.now} bar={context.bar_count} "
            f"total_value={account.total_value:,.2f} cash={account.cash:,.2f} "
            f"positions={held}"
        )
