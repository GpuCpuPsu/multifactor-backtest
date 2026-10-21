"""
RUN 1 -- ABLATION: 5-day momentum only, NO fundamental filter.

Identical to strategies/nasdaq100_fundamental_momentum_cs.py in every respect
(universe, top_n=5, 5-day rebalance, 5-day momentum window, equal weight, 90% cap,
sell-before-buy ordering) EXCEPT that the three hard fundamental conditions
-- revenue growth > 20%, gross margin > 20%, operating cash flow > 0 -- are removed.

Isolates one question: is the fundamental screen doing any work, or was the momentum
leg carrying the entire result?
"""

from panda_backtest.api.api import *
from panda_backtest.api.stock_us_api import *
import tqx_data
import pandas as pd


NDX_INDEX_SYMBOL = "NDX"


def initialize(context):
    context.account = "15032863"

    context.top_n = 5
    context.rebalance_every = 5
    context.momentum_window = 5
    context.max_position_ratio = 0.9

    context.bar_count = 0
    context.today_trades = []

    context.membership = _load_nasdaq100_membership()
    context.universe = sorted(set(context.membership["stock_symbol"].tolist())) \
        if context.membership is not None and len(context.membership) > 0 else []
    context.close_history = {symbol: [] for symbol in context.universe}


def _load_nasdaq100_membership():
    try:
        components = tqx_data.get_index_component(index_symbol=NDX_INDEX_SYMBOL, market="us")
    except Exception as exc:
        print(f"[run1_mom] get_index_component failed: {exc}")
        return None

    if components is None or len(components) == 0:
        return None

    components = components.dropna(subset=["stock_symbol"])
    components = components[components["stock_symbol"].str.endswith(".NB")]
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


def before_trading(context):
    context.today_trades = []


def _current_trade_date(context, data):
    for symbol in context.universe:
        bar = data.get(symbol)
        if bar is not None and getattr(bar, "trade_date", None):
            return bar.trade_date
    raw = getattr(context, "now", None)
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else None


def handle_data(context, data):
    context.bar_count += 1

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
    if rebalance_date is None:
        return

    # ---- the only difference from the live strategy: no fundamental gate ----
    eligible = _active_symbols_for_date(context, rebalance_date)

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
    targets = ranked[: context.top_n]

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
        print(f"[run1_mom] date={rebalance_date} eligible_count=0 target_symbols=[]")
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
            f"[run1_mom] date={rebalance_date} eligible_count={len(eligible)} "
            f"target_symbols={targets}"
        )


def after_trading(context):
    account = context.stock_account_dict.get(context.account)
    if account is None:
        return
    if context.bar_count % 105 == 0:
        print(
            f"[run1_mom] date={context.now} bar={context.bar_count} "
            f"total_value={account.total_value:,.2f} cash={account.cash:,.2f}"
        )
