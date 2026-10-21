"""
RUN 10 -- TREND FOLLOWING WITH STOPS. Structurally different from everything before.

WHY THIS IS DIFFERENT
  Runs 1-9 all selected on BACKWARD-LOOKING FUNDAMENTALS and were always ~90% invested.
  The diagnostic proved that screen admits winners only after their move (NVDA excluded
  through all of 2023 on -13% trailing revenue, admitted Jan-2024 at +38%). This run uses
  NO fundamentals at all, and -- critically -- it can EXIT BETWEEN REBALANCES and sit in
  cash. That is the mechanic every one of the six analysts used and we never tested.

PRE-REGISTERED PARAMETERS -- fixed before any result was seen, and not to be tuned.
Sourced from what the analysts stated publicly, not from optimising on this data:

  Trend filter    price > MA50 > MA200        classic; AJay's "MAs in numerical order",
                                              Kevin's "buy strong stocks at the 200 ema"
  Ranking         120-day momentum            established from Run 7; short windows lose
  Holdings        top 15, equal weight        Run 8 showed 15 halves the drawdown vs 5
  Max deployed    90%                         same as every prior run
  Rebalance       every 21 bars (monthly)     matches a slow signal to a slow cadence
  HARD STOP       -8% from entry              Bill/AJay/FPU used -20/-25% on OPTION premium;
                                              that is roughly -3% to -8% on the underlying,
                                              so -8% is the conservative end of their range
  TRAILING STOP   -12% from high-water mark   "trim your win" / "lock in profits", mechanised
  Re-entry        blocked until next rebalance   avoids whipsawing back into a stopped name

If fewer than 15 names pass the trend filter, we hold fewer and leave the rest in CASH.
That is the point: in a downtrend this book de-risks itself, which no earlier run could do.
"""

from panda_backtest.api.api import *
from panda_backtest.api.stock_us_api import *
import tqx_data
import pandas as pd
import datetime


UNIVERSE_SIZE = 100
UNIVERSE_REFRESH_BARS = 21
MA_FAST, MA_SLOW = 50, 200
MOMENTUM_WINDOW = 120
TOP_N = 15
MAX_DEPLOYED = 0.90
HARD_STOP = -0.08
TRAIL_STOP = -0.12
HISTORY_LEN = MA_SLOW + 5


def initialize(context):
    context.account = "15032863"
    context.market = "nb"
    context.rebalance_every = UNIVERSE_REFRESH_BARS

    context.bar_count = 0
    context.last_universe_refresh = -999
    context.universe = []
    context.close_history = {}

    context.entry_price = {}
    context.high_water = {}
    context.blocked = set()

    context.n_hard = 0
    context.n_trail = 0


def _shift_days(yyyymmdd, days):
    d = datetime.date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
    return (d - datetime.timedelta(days=days)).strftime("%Y%m%d")


def _today(context):
    raw = getattr(context, "now", None)
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else None


def _refresh_universe(context, as_of):
    # end_date must land on a real trading session: get_us_daily raises on holidays
    # ("'20260525' is not in list"), so walk back until one is accepted.
    df = None
    for back in range(1, 8):
        try:
            df = tqx_data.get_us_daily(symbol=[], start_date=_shift_days(as_of, back + 7),
                                       end_date=_shift_days(as_of, back), market=context.market)
        except BaseException:
            df = None
            continue
        if df is not None and len(df) > 0:
            break
    if df is None or len(df) == 0:
        print(f"[trend] no tradable session found before {as_of}")
        return False
    try:
        last_session = df["date"].max()
        df = df[df["date"] == last_session]
    except BaseException:
        return False
    df = df.dropna(subset=["symbol", "amount"]).copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df = df.dropna(subset=["amount"])
    df = df[df["amount"] > 0].sort_values("amount", ascending=False)
    uni = [s for s in df.head(UNIVERSE_SIZE)["symbol"].tolist()
           if isinstance(s, str) and s.endswith(".NB")]
    if not uni:
        return False
    context.universe = uni
    # use the VALIDATED session date, never the engine's own date: the backtest calendar
    # and the data vendor's calendar disagree (engine runs 20260525, vendor rejects it)
    try:
        _reload_history(context, str(last_session))
    except BaseException as exc:
        print(f"[trend] history reload skipped at {as_of}: {exc}")
    return True


def _reload_history(context, as_of):
    try:
        q = stock_api_quotation(
            symbol_list=context.universe,
            start_date=_shift_days(as_of, int(MA_SLOW * 2.4)),
            end_date=as_of,
            fields=["symbol", "date", "close"],
            period="1d",
        )
    except BaseException as exc:
        print(f"[trend] history reload failed {as_of}: {exc}")
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
            hist[symbol] = closes[-HISTORY_LEN:]
    context.close_history = hist


def _ma(series, window):
    if series is None or len(series) < window:
        return None
    w = series[-window:]
    return sum(w) / float(window)


def handle_data(context, data):
    context.bar_count += 1
    as_of = _today(context)
    if as_of is None:
        return

    if context.bar_count - context.last_universe_refresh >= UNIVERSE_REFRESH_BARS:
        try:
            if _refresh_universe(context, as_of):
                context.last_universe_refresh = context.bar_count
        except BaseException as exc:
            # never let a data-layer date quirk kill the whole backtest; keep last universe
            print(f"[trend] universe refresh skipped at {as_of}: {exc}")

    account = context.stock_account_dict.get(context.account)
    if account is None:
        return

    # ---- update history for anything we track or hold ----
    watch = set(context.universe) | set(account.positions.keys())
    for symbol in watch:
        bar = data.get(symbol)
        if bar is None or getattr(bar, "close", None) is None:
            continue
        price = float(bar.close)
        if price <= 0:
            continue
        h = context.close_history.setdefault(symbol, [])
        h.append(price)
        if len(h) > HISTORY_LEN:
            h[:] = h[-HISTORY_LEN:]

    # ---- STOP CHECKS: run EVERY bar, not just on rebalance days ----
    for symbol, position in list(account.positions.items()):
        if position.quantity <= 0 or position.sellable <= 0:
            continue
        bar = data.get(symbol)
        if bar is None or getattr(bar, "close", None) is None:
            continue
        price = float(bar.close)
        if price <= 0:
            continue

        entry = context.entry_price.get(symbol)
        if entry is None or entry <= 0:
            context.entry_price[symbol] = price
            context.high_water[symbol] = price
            continue

        hw = context.high_water.get(symbol, entry)
        if price > hw:
            hw = price
            context.high_water[symbol] = hw

        hit_hard = (price / entry - 1.0) <= HARD_STOP
        hit_trail = hw > 0 and (price / hw - 1.0) <= TRAIL_STOP

        if hit_hard or hit_trail:
            order_shares(context.account, symbol, -position.sellable, style=MarketOrderStyle)
            context.blocked.add(symbol)
            context.entry_price.pop(symbol, None)
            context.high_water.pop(symbol, None)
            if hit_hard:
                context.n_hard += 1
            else:
                context.n_trail += 1

    if context.bar_count % context.rebalance_every != 0:
        return

    # ---- REBALANCE ----
    context.blocked = set()

    scores = {}
    for symbol in context.universe:
        h = context.close_history.get(symbol, [])
        if len(h) < MA_SLOW + 1 or len(h) < MOMENTUM_WINDOW + 1:
            continue
        price = h[-1]
        ma_f, ma_s = _ma(h, MA_FAST), _ma(h, MA_SLOW)
        if ma_f is None or ma_s is None or ma_s <= 0:
            continue
        if not (price > ma_f > ma_s):          # trend filter
            continue
        past = h[-(MOMENTUM_WINDOW + 1)]
        if past is None or past <= 0:
            continue
        scores[symbol] = price / past - 1.0

    ranked = sorted(scores, key=scores.get, reverse=True)
    targets = ranked[:TOP_N]

    for symbol, position in list(account.positions.items()):
        if symbol not in targets and position.sellable > 0:
            order_shares(context.account, symbol, -position.sellable, style=MarketOrderStyle)
            context.entry_price.pop(symbol, None)
            context.high_water.pop(symbol, None)

    if not targets:
        if context.bar_count % 105 == 0:
            print(f"[trend] date={as_of} qualified=0 -- fully in cash")
        return

    # size on TOP_N slots, not len(targets): fewer qualifiers => genuinely more cash
    budget = account.total_value * MAX_DEPLOYED / TOP_N
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
        qty = int(min(budget, account.cash) // price)
        if qty <= 0:
            continue
        order_shares(context.account, symbol, qty, style=MarketOrderStyle)
        context.entry_price[symbol] = price
        context.high_water[symbol] = price

    if context.bar_count % 105 == 0:
        print(f"[trend] date={as_of} qualified={len(ranked)} held={len(targets)} "
              f"stops_hard={context.n_hard} stops_trail={context.n_trail} "
              f"cash={account.cash:,.0f}")


def after_trading(context):
    account = context.stock_account_dict.get(context.account)
    if account is None:
        return
    if context.bar_count % 105 == 0:
        held = len([p for p in account.positions.values() if p.quantity > 0])
        print(f"[trend] date={context.now} value={account.total_value:,.0f} "
              f"positions={held} hard={context.n_hard} trail={context.n_trail}")
