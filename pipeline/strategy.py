"""
TQX FACTOR PIPELINE -- trend + volume, catalyst-gated, optional ATR stop.

Runs as a single uploaded strategy file (tqx-cli constraint). Structured so the
signal layer is independent of the execution layer: `compute_scores()` returns a
{symbol: composite} dict from data alone, and nothing below it knows about orders.
A live adapter replaces `execute()` only.

=============================== CONFIG (rewritten per run by run_pipeline.py) ====
"""
CFG_STOP_ENABLED = False     # ATR stop A/B toggle
CFG_TOP_N = 5                # positions held
CFG_REBALANCE_BARS = 10         # trading days between rebalances
CFG_LABEL = "SINCE_LAST"
"""
=================================================================================

FACTORS
  trend_z    mean of 20-day and 60-day simple price return, cross-sectionally z-scored.
             Measured to the last COMPLETED close, not the fill bar's close (same rule
             as volume_z below).
  volume_z   mean(vol, 5) / mean(vol, 20), z-scored. Uses ONLY sessions strictly before
             the decision date -- the spec forbids a same-day volume peek, and a day's
             volume is not final until its close.
  catalyst   1 if today is within [-1, +3] TRADING days of an earnings release.

  composite  0.5*trend_z + 0.5*volume_z, GATED by catalyst (flag 0 => not eligible).

POINT-IN-TIME CORRECTNESS (the part most likely to be faked)
  Earnings come from get_event_fina_disclosure, which carries `start_date` (the event)
  and `info_date` (when the event became knowable) as separate columns. We keep only
  rows with info_date <= decision_date. That is what makes the -1 leg legal: we flag the
  session before earnings only when the schedule was genuinely public beforehand.
  Measured on a 12-name sample over 2023-2026: 36% of releases are knowable in advance,
  34% same-day, 30% only recorded afterwards. So the -1 leg fires on a minority of events
  by construction. That is the honest number; inflating it would require using start_date
  without the info_date filter, which is exactly the leak.

UNIVERSE
  Rebuilt monthly from get_us_daily with an empty symbol list -- every symbol that
  actually printed a trade, so names later delisted are still present on historical
  dates (no survivorship). Ranked by dollar volume on the last COMPLETED session.
  Rejected before ranking: zero/NaN volume, non-positive price, and flat bars
  (high == low), which produce a degenerate ATR and a meaningless return.
  If fewer than MIN_UNIVERSE names survive, the universe is not updated.
"""

from panda_backtest.api.api import *
from panda_backtest.api.stock_us_api import *
import tqx_data
import pandas as pd
import datetime

UNIVERSE_SIZE = 100
MIN_UNIVERSE = 15
UNIVERSE_REFRESH_BARS = 21
TREND_FAST, TREND_SLOW = 20, 60
VOL_FAST, VOL_SLOW = 5, 20
ATR_PERIOD = 14
ATR_MULT = 1.5
CAT_BEFORE, CAT_AFTER = 1, 3
MAX_DEPLOYED = 0.90
HISTORY_BARS = TREND_SLOW + 40


def initialize(context):
    context.account = "15032863"
    context.market_data = "nb"      # price endpoints
    context.market_ref = "us"       # reference/event endpoints

    context.bar_count = 0
    context.last_refresh = -999
    context.stops = {}              # symbol -> stop price

    context.n_stopped = 0
    context.n_rebalances = 0
    context.eligible_counts = []
    context.cat_minus1_hits = 0
    context.cat_total_hits = 0

    init_market_data(context)


def init_market_data(context):
    # Required entry point on the platform. The universe is point-in-time (rebuilt monthly
    # from that month's dollar volume), so history can't be loaded once up front here --
    # it is (re)loaded by _refresh_universe. This only sets up the empty caches.
    context.universe = []
    context.hist = {}               # symbol -> {close:[], high:[], low:[], vol:[]}
    context.events = {}             # symbol -> [event_date ints], PIT-filtered
    context.calendar = []           # sorted trading-day ints
    context.cal_idx = {}            # trading-day int -> position in calendar


# ------------------------------------------------------------------ utilities
def _shift(yyyymmdd, days):
    d = datetime.date(int(yyyymmdd[:4]), int(yyyymmdd[4:6]), int(yyyymmdd[6:8]))
    return (d - datetime.timedelta(days=days)).strftime("%Y%m%d")


def _today(context):
    raw = getattr(context, "now", None)
    digits = "".join(ch for ch in str(raw) if ch.isdigit())
    return digits[:8] if len(digits) >= 8 else None


def _num(x):
    """float(x), or None for missing / non-numeric / NaN -- without try/except."""
    v = pd.to_numeric(x, errors="coerce")
    return None if pd.isna(v) else float(v)


def _zscore(values):
    """Cross-sectional z-score. Returns {} if dispersion is degenerate."""
    if not values:
        return {}
    xs = list(values.values())
    n = len(xs)
    if n < 2:
        return {k: 0.0 for k in values}
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    if var <= 0:
        return {k: 0.0 for k in values}
    sd = var ** 0.5
    return {k: (v - mean) / sd for k, v in values.items()}


# ------------------------------------------------------------------ data layer
# The platform only allows data calls (tqx_data.*, stock_api_quotation) directly inside
# initialize / before_trading. So every fetch sits in before_trading() below, and the
# helpers here are pure: DataFrame in, parsed structure out.
def _parse_calendar(cal):
    """Trading-calendar frame -> sorted list of trading-day ints."""
    if cal is None or len(cal) == 0:
        return []
    # get_trading_calendar returns nature_date/exchange/is_trade/next_trade_date/pretrade_date
    col = next((c for c in ("nature_date", "date", "trade_date") if c in cal.columns),
               cal.columns[0])
    days = pd.to_numeric(cal[col], errors="coerce").dropna()
    return sorted(set(int(d) for d in days.tolist()))


def _rank_universe(df):
    """Whole-market daily frame -> (top-N .NB symbols by dollar volume, session date)."""
    if df is None or len(df) == 0:
        return [], None
    last_session = df["date"].max()
    df = df[df["date"] == last_session].copy()

    for col in ("amount", "volume", "close", "high", "low"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # quality gate BEFORE ranking, per spec
    df = df.dropna(subset=["symbol", "amount", "volume", "close"])
    df = df[(df["volume"] > 0) & (df["close"] > 0) & (df["amount"] > 0)]
    if "high" in df.columns and "low" in df.columns:
        df = df[df["high"] > df["low"]]          # drop flat/untraded bars

    df = df.sort_values("amount", ascending=False)
    uni = [s for s in df.head(UNIVERSE_SIZE)["symbol"].tolist()
           if isinstance(s, str) and s.endswith(".NB")]
    return uni, str(last_session)


def _parse_history(q):
    """Quotation frame -> {symbol: {close, high, low, vol}} float lists, oldest first."""
    if q is None or len(q) == 0:
        return {}
    for col in ("close", "high", "low", "volume"):
        q[col] = pd.to_numeric(q[col], errors="coerce")
    q = q.dropna(subset=["close", "high", "low", "volume"])
    q = q[q["close"] > 0].sort_values(["symbol", "date"])
    hist = {}
    for sym, g in q.groupby("symbol"):
        rows = {"close": [float(x) for x in g["close"]],
                "high": [float(x) for x in g["high"]],
                "low": [float(x) for x in g["low"]],
                "vol": [float(x) for x in g["volume"]]}
        if rows["close"]:
            for k in rows:
                rows[k] = rows[k][-HISTORY_BARS:]
            hist[sym] = rows
    return hist


def _parse_events(df, as_of):
    """Earnings releases, filtered to what was KNOWABLE as of `as_of`."""
    if df is None or len(df) == 0:
        return {}
    df = df[df["event_type"] == "EarningsReleases"].copy()
    ev = pd.to_numeric(df["start_date"], errors="coerce")
    info = pd.to_numeric(df["info_date"], errors="coerce")
    df = df.assign(_ev=ev, _info=info).dropna(subset=["_ev", "_info"])
    # ---- THE POINT-IN-TIME GATE ----
    df = df[df["_info"] <= int(as_of)]
    out = {}
    for sym, g in df.groupby("symbol"):
        out[sym] = sorted(set(int(x) for x in g["_ev"].tolist()))
    return out


def before_trading(context):
    """All data fetches. Runs before handle_data on the same date; `bar` is the number
    handle_data is about to assign, so cadences match the old in-handle_data logic."""
    today = _today(context)
    if today is None:
        return
    bar = context.bar_count + 1

    # ---- monthly universe refresh ----
    if bar - context.last_refresh >= UNIVERSE_REFRESH_BARS:
        cal = tqx_data.get_trading_calendar(
            start_date=_shift(today, 400), end_date=_shift(today, -120),
            market=context.market_ref, is_trading_day=1)
        days = _parse_calendar(cal)
        if days:
            context.calendar = days
            context.cal_idx = {d: i for i, d in enumerate(days)}
        # get_us_daily raises on a non-trading end_date and exceptions may not be caught,
        # so ask for the last completed session per the calendar instead of probing dates.
        prior = [d for d in context.calendar if d < int(today)]
        if not prior:
            print(f"[pipe] {today} no completed session in calendar, keeping previous universe")
        else:
            df = tqx_data.get_us_daily(symbol=[], start_date=_shift(str(prior[-1]), 7),
                                       end_date=str(prior[-1]), market=context.market_data)
            uni, last_session = _rank_universe(df)
            if len(uni) < MIN_UNIVERSE:
                print(f"[pipe] {today} universe only {len(uni)} < {MIN_UNIVERSE}, keeping previous")
            else:
                q = stock_api_quotation(
                    symbol_list=uni,
                    start_date=_shift(last_session, int(HISTORY_BARS * 1.9)),
                    end_date=last_session,
                    fields=["symbol", "date", "high", "low", "close", "volume"],
                    period="1d",
                )
                context.universe = uni
                context.hist = _parse_history(q)
                context.last_refresh = bar

    # ---- earnings view, refreshed at every DECISION date ----
    # Doing this only at the monthly universe refresh hid every release whose info_date
    # landed in between -- and since 64% of releases only become knowable on/after the
    # event, that silently suppressed most of the catalyst signal (measured: 0.9
    # eligible/rebalance vs 22 on a peak day).
    if bar % CFG_REBALANCE_BARS == 0 and context.universe:
        ev = tqx_data.get_event_fina_disclosure(
            symbol=context.universe, market=context.market_ref,
            start_date=_shift(today, 400), end_date=_shift(today, -120))
        context.events = _parse_events(ev, today)


# ------------------------------------------------------------------ factors
def _catalyst_flag(context, symbol, today):
    """1 if today is within [-CAT_BEFORE, +CAT_AFTER] TRADING days of a known event."""
    evs = context.events.get(symbol)
    if not evs or not context.calendar:
        return 0, False
    t_idx = context.cal_idx.get(int(today))
    if t_idx is None:
        return 0, False
    for e in evs:
        e_idx = context.cal_idx.get(e)
        if e_idx is None:               # event on a non-trading day: skipped, as before
            continue
        delta = t_idx - e_idx           # negative => before the event
        if -CAT_BEFORE <= delta <= CAT_AFTER:
            return 1, (delta < 0)       # second value: this is a pre-earnings hit
    return 0, False


def _atr(h):
    highs, lows, closes = h["high"], h["low"], h["close"]
    n = min(len(highs), len(lows), len(closes))
    if n < ATR_PERIOD + 1:
        return None
    trs = []
    for i in range(n - ATR_PERIOD, n):
        prev_close = closes[i - 1]
        tr = max(highs[i] - lows[i], abs(highs[i] - prev_close), abs(lows[i] - prev_close))
        trs.append(tr)
    if not trs:
        return None
    atr = sum(trs) / len(trs)
    return atr if atr > 0 else None


def compute_scores(context, today):
    """Pure signal layer: data in, {symbol: composite} out. No orders, no account."""
    raw_trend, raw_vol, gated = {}, {}, []

    for sym in context.universe:
        h = context.hist.get(sym)
        if not h:
            continue
        closes, vols = h["close"], h["vol"]
        if len(closes) < TREND_SLOW + 2 or len(vols) < VOL_SLOW + 1:
            continue

        # trend: same rule as volume -- closed sessions only. closes[-1] is the bar we
        # fill on, so scoring with it would trade on a price not yet known.
        c_now = closes[-2]
        c_fast, c_slow = closes[-(TREND_FAST + 2)], closes[-(TREND_SLOW + 2)]
        if c_fast <= 0 or c_slow <= 0 or c_now <= 0:
            continue
        raw_trend[sym] = 0.5 * (c_now / c_fast - 1.0) + 0.5 * (c_now / c_slow - 1.0)

        # volume: strictly closed sessions -> drop the final (current) bar
        closed = vols[:-1]
        if len(closed) < VOL_SLOW:
            continue
        f = sum(closed[-VOL_FAST:]) / float(VOL_FAST)
        s = sum(closed[-VOL_SLOW:]) / float(VOL_SLOW)
        if s <= 0:
            continue
        raw_vol[sym] = f / s

    common = [s for s in raw_trend if s in raw_vol]
    if len(common) < 2:
        return {}, 0

    tz = _zscore({s: raw_trend[s] for s in common})
    vz = _zscore({s: raw_vol[s] for s in common})

    scores = {}
    for s in common:
        flag, pre = _catalyst_flag(context, s, today)
        if flag != 1:
            continue                      # <-- the gate
        context.cat_total_hits += 1
        if pre:
            context.cat_minus1_hits += 1
        scores[s] = 0.5 * tz.get(s, 0.0) + 0.5 * vz.get(s, 0.0)
        gated.append(s)

    return scores, len(common)


# ------------------------------------------------------------------ execution
def execute(context, data, targets):
    """Execution layer. A live adapter swaps this out; signal layer is untouched."""
    account = context.stock_account_dict.get(context.account)
    if account is None:
        return

    for sym, pos in list(account.positions.items()):
        if sym not in targets and pos.sellable > 0:
            order_shares(context.account, sym, -pos.sellable, style=MarketOrderStyle)
            context.stops.pop(sym, None)

    if not targets:
        return

    budget = account.total_value * MAX_DEPLOYED / CFG_TOP_N
    for sym in targets:
        bar = data[sym]
        if bar is None or getattr(bar, "close", None) is None:
            continue
        px = float(bar.close)
        if px <= 0:
            continue
        pos = account.positions.get(sym)
        if pos is not None and pos.quantity > 0:
            continue
        qty = int(min(budget, account.cash) // px)
        if qty <= 0:
            continue
        order_shares(context.account, sym, qty, style=MarketOrderStyle)
        if CFG_STOP_ENABLED:
            atr = _atr(context.hist.get(sym, {"high": [], "low": [], "close": []}))
            if atr is not None:
                context.stops[sym] = px - ATR_MULT * atr


def _check_stops(context, data):
    if not CFG_STOP_ENABLED:
        return
    account = context.stock_account_dict.get(context.account)
    if account is None:
        return
    for sym, pos in list(account.positions.items()):
        if pos.quantity <= 0 or pos.sellable <= 0:
            continue
        stop = context.stops.get(sym)
        if stop is None:
            continue
        bar = data[sym]
        if bar is None or getattr(bar, "close", None) is None:
            continue
        px = float(bar.close)
        if px > 0 and px <= stop:
            order_shares(context.account, sym, -pos.sellable, style=MarketOrderStyle)
            context.stops.pop(sym, None)
            context.n_stopped += 1


# ------------------------------------------------------------------ lifecycle
def handle_data(context, data):
    context.bar_count += 1
    today = _today(context)
    if today is None:
        return

    if not context.universe:
        return

    # keep history current between refreshes
    for sym in context.universe:
        bar = data[sym]
        if bar is None or getattr(bar, "close", None) is None:
            continue
        c, h, l, v = (_num(getattr(bar, f, None)) for f in ("close", "high", "low", "volume"))
        if c is None or h is None or l is None or v is None:
            continue
        if c <= 0 or h < l or v < 0:
            continue
        rec = context.hist.setdefault(sym, {"close": [], "high": [], "low": [], "vol": []})
        rec["close"].append(c); rec["high"].append(h)
        rec["low"].append(l);   rec["vol"].append(v)
        for k in rec:
            if len(rec[k]) > HISTORY_BARS:
                rec[k] = rec[k][-HISTORY_BARS:]

    _check_stops(context, data)

    if context.bar_count % CFG_REBALANCE_BARS != 0:
        return

    # universe/history/events for this date were loaded in before_trading()
    scores, n_scored = compute_scores(context, today)
    context.n_rebalances += 1
    context.eligible_counts.append(len(scores))

    ranked = sorted(scores, key=scores.get, reverse=True)
    targets = ranked[:CFG_TOP_N]
    execute(context, data, targets)

    if context.n_rebalances % 12 == 0:
        avg_elig = sum(context.eligible_counts) / max(1, len(context.eligible_counts))
        print(f"[pipe:{CFG_LABEL}] {today} scored={n_scored} eligible={len(scores)} "
              f"avg_eligible={avg_elig:.1f} held={len(targets)} stops={context.n_stopped}")


def after_trading(context):
    account = context.stock_account_dict.get(context.account)
    if account is None:
        return
    if context.bar_count % 252 == 0:
        avg = sum(context.eligible_counts) / max(1, len(context.eligible_counts))
        zero = sum(1 for x in context.eligible_counts if x == 0)
        print(f"[pipe:{CFG_LABEL}] SUMMARY bar={context.bar_count} "
              f"value={account.total_value:,.0f} rebalances={context.n_rebalances} "
              f"avg_eligible={avg:.2f} empty_rebalances={zero} "
              f"stops_fired={context.n_stopped} "
              f"catalyst_hits={context.cat_total_hits} pre_earnings={context.cat_minus1_hits}")
