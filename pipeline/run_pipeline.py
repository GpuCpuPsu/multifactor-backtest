#!/usr/bin/env python3
"""
Orchestrator + GATC scorer for the TQX factor pipeline.

  python3 pipeline/run_pipeline.py <label> <stop:on|off> <top_n> <rebal> <start> <end>
  python3 pipeline/run_pipeline.py sweep <prefix> <n:r:stop,...> <start> <end> <out.jsonl>
      e.g. sweep IS "3:5:off,5:10:off,5:10:on" 20230101 20241231 pipeline/results/sweep.jsonl
      Checks tqx-cli auth first; appends one labeled row per combo to <out.jsonl>.

Rewrites the CFG block in pipeline/strategy.py, creates and runs the workflow, pulls the
summary and every page of fills, reconstructs round-trip P&L FIFO (the platform reports
fills, not trade P&L), and scores the run with GATC's formula.

GATC formula, implemented EXACTLY as specified:
    0.55*(0.4*NetYield + 0.1*ProfitFactor + 0.05*WinRate)
  + 0.45*(0.2*Sharpe   + 0.15*MaxDD      + 0.1*RiskAdjReturn)

Two caveats reported alongside every score, not silently corrected:
  * The stated weights sum to 0.505, not 1. The inner weights alone
    (0.4+0.1+0.05+0.2+0.15+0.1) sum to exactly 1.0, which suggests the 0.55/0.45
    group multipliers may be double-applied. `score_inner` shows that reading.
  * The terms carry incompatible units (NetYield ~0.5, ProfitFactor ~1.5, Sharpe ~2.0,
    MaxDD ~-0.3), so the weighted sum is dimensionally mixed. `score_norm` maps each
    term to [0,1] before weighting. All three are printed; none is assumed correct.

RiskAdjReturn is not defined in the spec. Implemented as Calmar (annualised return /
|MaxDD|) because return/volatility would duplicate Sharpe. Flagged in output.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STRATEGY = ROOT / "pipeline" / "strategy.py"


def sh(cmd, timeout=900):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True,
                       cwd=str(ROOT), timeout=timeout)
    return r.stdout + r.stderr


def first_json(text):
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                return json.JSONDecoder().raw_decode(text[i:])[0]
            except ValueError:
                continue
    return None


def write_cfg(stop_enabled, top_n, rebal, label):
    src = STRATEGY.read_text()
    out = []
    for line in src.splitlines():
        if line.startswith("CFG_STOP_ENABLED"):
            out.append(f"CFG_STOP_ENABLED = {bool(stop_enabled)}"
                       f"      # ATR stop A/B toggle")
        elif line.startswith("CFG_TOP_N"):
            out.append(f"CFG_TOP_N = {top_n}                # positions held")
        elif line.startswith("CFG_REBALANCE_BARS"):
            out.append(f"CFG_REBALANCE_BARS = {rebal}         # trading days between rebalances")
        elif line.startswith("CFG_LABEL"):
            out.append(f'CFG_LABEL = "{label}"')
        else:
            out.append(line)
    STRATEGY.write_text("\n".join(out) + "\n")


def fetch_trades(backtest_id):
    """All fill pages -> list of dicts."""
    items, page = [], 1
    while page <= 40:
        raw = sh(f"tqx-cli backtest_result {backtest_id} --section trade "
                 f"--page {page} --page-size 100", timeout=300)
        obj = first_json(raw)
        if not obj or not obj.get("success"):
            break
        node = obj.get("results", {}).get("trade", {})
        batch = node.get("items", [])
        if not batch:
            break
        items.extend(batch)
        pg = node.get("pagination", {})
        if page >= int(pg.get("total_pages", 1)):
            break
        page += 1
    return items


def round_trips(trades):
    """FIFO-match fills into closed round trips. direction 0 = buy, 1 = sell."""
    books, closed = {}, []
    for t in sorted(trades, key=lambda x: (str(x.get("trade_date")),
                                           int(x.get("trade_id") or 0))):
        sym = t.get("contract_code")
        try:
            px = float(t.get("price"))
            qty = float(t.get("volume"))
        except (TypeError, ValueError):
            continue
        if sym is None or px <= 0 or qty <= 0:
            continue
        side = int(t.get("direction", 0))
        lots = books.setdefault(sym, [])
        if side == 0:
            lots.append([px, qty])
        else:
            remaining = qty
            while remaining > 1e-9 and lots:
                bpx, bqty = lots[0]
                take = min(bqty, remaining)
                closed.append((px - bpx) * take)
                bqty -= take
                remaining -= take
                if bqty <= 1e-9:
                    lots.pop(0)
                else:
                    lots[0][1] = bqty
    return closed


def gatc(net_yield, profit_factor, win_rate, sharpe, max_dd, risk_adj):
    prof = 0.4 * net_yield + 0.1 * profit_factor + 0.05 * win_rate
    risk = 0.2 * sharpe + 0.15 * max_dd + 0.1 * risk_adj
    literal = 0.55 * prof + 0.45 * risk
    inner = prof + risk                      # inner weights already sum to 1.0

    def unit(x, lo, hi):
        return max(0.0, min(1.0, (x - lo) / (hi - lo)))
    norm = (0.55 * (0.4 * unit(net_yield, -0.5, 1.5)
                    + 0.1 * unit(profit_factor, 0.0, 3.0)
                    + 0.05 * unit(win_rate, 0.0, 1.0))
            + 0.45 * (0.2 * unit(sharpe, -1.0, 3.0)
                      + 0.15 * unit(max_dd, -0.6, 0.0)
                      + 0.1 * unit(risk_adj, -1.0, 3.0)))
    return literal, inner, norm


def run_one(label, stop, top_n, rebal, start, end):
    """Create + run one config on the platform and score it. Returns a result dict."""
    stop_on = stop.lower() == "on"
    write_cfg(stop_on, int(top_n), int(rebal), label)

    name = f"PIPE {label} stop={stop} n={top_n} r={rebal}"
    raw = sh(f'tqx-cli strategy_create --market us --file ./pipeline/strategy.py '
             f'--name "{name}" --start-date {start} --end-date {end} '
             f'--frequency 1d --slippage 0.001 --strict-market-api')
    obj = first_json(raw)
    if not obj or not obj.get("success"):
        return {"label": label, "error": "create failed", "raw": raw[:300]}
    sid = obj["strategy_id"]

    raw = sh(f"tqx-cli strategy_run {sid} --timeout 600", timeout=900)
    obj = first_json(raw)
    if not obj or not obj.get("success"):
        return {"label": label, "strategy_id": sid, "error": "run failed", "raw": raw[:400]}

    res = obj.get("results", {})
    summ = res.get("backtest", {}).get("summary", {})
    bid = res.get("backtest_id")

    def num(k, d=0.0):
        try:
            return float(summ.get(k))
        except (TypeError, ValueError):
            return d

    net_yield = num("back_profit")
    sharpe = num("sharpe")
    max_dd = num("max_drawdown")
    ann = num("back_profit_year")

    pnl = round_trips(fetch_trades(bid))
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p < 0]
    win_rate = len(wins) / len(pnl) if pnl else 0.0
    gross_w = sum(wins)
    gross_l = abs(sum(losses))
    profit_factor = (gross_w / gross_l) if gross_l > 1e-9 else (0.0 if not wins else 10.0)
    risk_adj = (ann / abs(max_dd)) if abs(max_dd) > 1e-9 else 0.0   # Calmar

    lit, inner, norm = gatc(net_yield, profit_factor, win_rate, sharpe, max_dd, risk_adj)

    return {
        "label": label, "stop": stop, "top_n": int(top_n), "rebalance": int(rebal),
        "window": f"{start}-{end}", "strategy_id": sid, "backtest_id": bid,
        "net_yield": round(net_yield, 4), "annualised": round(ann, 4),
        "sharpe": round(sharpe, 4), "max_dd": round(max_dd, 4),
        "profit_factor": round(profit_factor, 4), "win_rate": round(win_rate, 4),
        "risk_adj_calmar": round(risk_adj, 4),
        "round_trips": len(pnl),
        "gatc_literal": round(lit, 5),
        "gatc_inner_weights": round(inner, 5),
        "gatc_normalised": round(norm, 5),
    }


def auth_ok():
    """Cheap authenticated call; the sweep refuses to start if this fails."""
    obj = first_json(sh("tqx-cli balance", timeout=60))
    return bool(obj and obj.get("success"))


def parse_grid(spec):
    """'5:10:off,3:5:on' -> [(5, 10, 'off'), (3, 5, 'on')]  (top_n:rebalance:stop)."""
    grid = []
    for item in spec.split(","):
        n, r, s = item.strip().split(":")
        if s.lower() not in ("on", "off"):
            raise ValueError(f"stop must be on|off, got {s!r} in {item!r}")
        grid.append((int(n), int(r), s.lower()))
    return grid


def sweep(prefix, grid, start, end, out_path):
    """Run every (top_n, rebalance, stop) combo; append one labeled JSONL row per run.

    Each combo is a fresh strategy_create, so it spends platform create quota --
    size the grid accordingly. Rows are appended as they finish, so a crash midway
    keeps everything completed so far.
    """
    if not auth_ok():
        print(json.dumps({"error": "tqx-cli auth check failed (tqx-cli balance); sweep not started"}))
        sys.exit(1)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    for i, (n, r, s) in enumerate(grid, 1):
        label = f"{prefix}_n{n}_r{r}_stop{s}"
        row = run_one(label, s, n, r, start, end)
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(f"[{i}/{len(grid)}] {json.dumps(row)}", flush=True)


def main():
    if sys.argv[1:2] == ["sweep"]:
        prefix, grid, start, end, out_path = sys.argv[2:7]
        sweep(prefix, parse_grid(grid), start, end, out_path)
    else:
        label, stop, top_n, rebal, start, end = sys.argv[1:7]
        print(json.dumps(run_one(label, stop, top_n, rebal, start, end)))


if __name__ == "__main__":
    main()
