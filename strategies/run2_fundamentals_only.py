"""
RUN 2 -- ABLATION: fundamental screen only, NO momentum ranking.

Holds EVERY name that passes the three hard conditions, equal-weight, rebalanced every
5 trading days. No ranking, no top_n cut.

Isolates the decisive question left by Run 1 vs Run 3: momentum-only returned +29.43%
(Sharpe 0.42) while fundamentals+momentum returned +56.49% (Sharpe 1.02). If holding the
whole eligible set beats the momentum-selected 5, the momentum leg is pure drag and
should be deleted rather than retuned.

Hard conditions (point-in-time, identical to the live strategy):
  revenue growth YoY > 20%, gross margin > 20%, operating cash flow > 0
"""

from panda_backtest.api.api import *
from panda_backtest.api.stock_us_api import *
import tqx_data
import pandas as pd


NDX_INDEX_SYMBOL = "NDX"

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

    context.rebalance_every = 5
    context.max_position_ratio = 0.9
    context.min_revenue_growth = 0.20
    context.min_gross_margin = 0.20

    context.bar_count = 0

    context.membership = _load_nasdaq100_membership()
    context.universe = sorted(set(context.membership["stock_symbol"].tolist())) \
        if context.membership is not None and len(context.membership) > 0 else []


def _load_nasdaq100_membership():
    try:
        components = tqx_data.get_index_component(index_symbol=NDX_INDEX_SYMBOL, market="us")
    except Exception as exc:
        print(f"[run2_fund] get_index_component failed: {exc}")
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
    start_quarter = f"{year - 2}q{quarter}"
    end_year, end_q = (year, quarter + 1) if quarter < 4 else (year + 1, 1)
    return start_quarter, f"{end_year}q{end_q}"


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
        print(f"[run2_fund] get_financial_statement failed at {rebalance_date}: {exc}")
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
    if context.bar_count % context.rebalance_every != 0:
        return

    rebalance_date = _current_trade_date(context, data)
    if rebalance_date is None:
        return

    active = _active_symbols_for_date(context, rebalance_date)
    targets = _fetch_eligible_symbols(context, rebalance_date, active)  # hold ALL of them

    account = context.stock_account_dict.get(context.account)
    if account is None:
        return

    for symbol, position in list(account.positions.items()):
        if symbol not in targets and position.sellable > 0:
            order_shares(context.account, symbol, -position.sellable, style=MarketOrderStyle)

    if not targets:
        print(f"[run2_fund] date={rebalance_date} eligible_count=0 -- flat")
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
        if position is not None and position.quantity > 0:
            continue
        qty = int(budget_per_name // price)
        if qty <= 0:
            continue
        order_shares(context.account, symbol, qty, style=MarketOrderStyle)

    if context.bar_count % 105 == 0:
        print(f"[run2_fund] date={rebalance_date} eligible_count={len(targets)} "
              f"holding_all_eligible")


def after_trading(context):
    account = context.stock_account_dict.get(context.account)
    if account is None:
        return
    if context.bar_count % 105 == 0:
        held = len([p for p in account.positions.values() if p.quantity > 0])
        print(f"[run2_fund] date={context.now} bar={context.bar_count} "
              f"total_value={account.total_value:,.2f} positions={held}")
