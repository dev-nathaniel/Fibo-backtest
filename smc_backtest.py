#!/usr/bin/env python3
"""
smc_backtest.py - Backtest of the Smart Money Concepts (SMC) Continuation Strategy on Gold (XAUUSD).
Produces the exact report, metrics, stress tests, Monte Carlo, and equity charts as fibo_backtest.py.

THE RULE:
  1. Trend Identification:
     - Bullish trend: series of Higher Highs (HH) and Higher Lows (HL).
     - Bearish trend: series of Lower Highs (LH) and Lower Lows (LL).
  2. Setup Lifecycle (Bullish continuation):
     - Step 1 (Sweep): Price takes out a previous Higher Low (HL).
       Demand Zone is mapped to the last bearish candle of the sweep low.
     - Step 2 (BOS 1): Price rallies from sweep low and breaks above previous Higher High (HH1) -> HH2.
     - Step 3 (Pullback): From HH2, price retraces >= 50% of the [Sweep Low -> HH2] range.
     - Step 4 (BOS 2): From that >= 50% pullback, price rallies and breaks above HH2 (Armed!).
     - Step 5 (Entry): When/if price pulls back to the original Demand Zone -> BUY.
       * Stop Loss: Sweep Low - buffer.
       * Take Profit: 1:3 Risk-to-Reward ratio.
  3. Setup Lifecycle (Bearish continuation):
     - Mirror image: LH swept -> Supply Zone (last bullish candle) -> BOS 1 (LL2) ->
       Retrace >= 50% -> BOS 2 -> Entry at Supply Zone -> SL above Sweep High + buffer, TP 1:3 RR.

EVALUATION & RIGOR (Identical to fibo_backtest.py):
  * Zero look-ahead: pivots confirmed at bar t + right_bars.
  * Same-bar stop/target ambiguity: stop assumed hit first.
  * Gaps through SL or TP fill at the open.
  * Real-world costs: spread + slippage per side + overnight financing (swap) per calendar day held.
  * Position sizing = risk_pct of equity at SL, capped at max_leverage.
  * In-sample vs out-of-sample (70/30 split), year-by-year breakdown.
  * Cost stress (0x, 1x, 2x, 3x), Monte Carlo permutation test, and random-timing null test.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fibo_backtest import load_csv, demo_gold, Costs
from market_structure import find_swings, classify_structure, SwingPoint

INK, MUTED, GRID, BLUE, GRAY = "#2b2b2b", "#6b6b6b", "#e6e6e6", "#2a6fbb", "#9aa0a6"
LINES: List[str] = []


def out(s=""):
    print(s)
    LINES.append(s)


def pct(x):
    return f"{x * 100:7.1f}%"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.title.set_color(INK)


# =====================================================================================
# TRADE SCANNER & EXECUTION ENGINE
# =====================================================================================

def scan_exit(o, h, l, t, side, tp, sl):
    """Walk forward from entry bar t. Conservative: stop touch on entry bar counts. Same-bar: stop first."""
    n = len(o)
    if side == 1:
        if l[t] <= sl:
            return t, sl, "sl"
        if h[t] >= tp:
            return t, tp, "tp"
    else:
        if h[t] >= sl:
            return t, sl, "sl"
        if l[t] <= tp:
            return t, tp, "tp"
    j, step = t + 1, 512
    while j < n:
        k = min(j + step, n)
        if side == 1:
            hit_sl, hit_tp = l[j:k] <= sl, h[j:k] >= tp
        else:
            hit_sl, hit_tp = h[j:k] >= sl, l[j:k] <= tp
        any_ = hit_sl | hit_tp
        if any_.any():
            i = j + int(np.argmax(any_))
            if side == 1:
                if o[i] <= sl:
                    return i, o[i], "sl"
                if o[i] >= tp:
                    return i, o[i], "tp"
            else:
                if o[i] >= sl:
                    return i, o[i], "sl"
                if o[i] <= tp:
                    return i, o[i], "tp"
            return (i, sl, "sl") if hit_sl[i - j] else (i, tp, "tp")
        j, step = k, step * 2
    return None, None, None


def find_smc_trades(
    df: pd.DataFrame,
    left_bars: int = 3,
    right_bars: int = 3,
    sl_buffer_usd: float = 0.50,
    rr_ratio: float = 3.0,
    max_setup_bars: int = 400
) -> pd.DataFrame:
    """
    Identifies SMC continuation trades with zero look-ahead bias.
    Returns DataFrame matching fibo_backtest.py columns:
    ['entry_i', 'exit_i', 'side', 'entry', 'exit', 'sl', 'tp', 'why']
    """
    swings = find_swings(df, left_bars=left_bars, right_bars=right_bars, enforce_alternation=True)
    swings = classify_structure(swings)

    conf_swings = {}
    for s in swings:
        conf_swings.setdefault(s.confirmed_index, []).append(s)

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)

    last_h: Optional[SwingPoint] = None
    last_l: Optional[SwingPoint] = None
    trend = "NEUTRAL"

    active_setups: List[dict] = []
    rows = []
    busy_until = -1

    for i in range(n):
        # 1. Update structure from confirmed swings
        if i in conf_swings:
            for s in conf_swings[i]:
                if s.swing_type == "HIGH":
                    last_h = s
                    if s.label == "HH":
                        trend = "BULLISH"
                    elif s.label == "LH" and last_l and last_l.label == "LL":
                        trend = "BEARISH"
                else:
                    last_l = s
                    if s.label == "LL":
                        trend = "BEARISH"
                    elif s.label == "HL" and last_h and last_h.label == "HH":
                        trend = "BULLISH"

        # 2. Bullish Sweep Initiation
        if trend == "BULLISH" and last_l and last_l.label == "HL" and last_h:
            if l[i] < last_l.price:
                # Avoid re-adding same swept low
                if not any(s["swept_idx"] == last_l.index and s["side"] == 1 for s in active_setups):
                    # Last bearish candle up to bar i
                    zone_top = h[i]
                    zone_bot = l[i]
                    for bi in range(i, max(0, i - 6), -1):
                        if c[bi] < o[bi]:
                            zone_top, zone_bot = h[bi], l[bi]
                            break
                    active_setups.append({
                        "side": 1,
                        "swept_idx": last_l.index,
                        "sweep_ext": l[i],
                        "sweep_bar": i,
                        "target1": last_h.price,
                        "stage": "SWEPT",
                        "zone_top": zone_top,
                        "zone_bot": zone_bot
                    })

        # 3. Bearish Sweep Initiation
        if trend == "BEARISH" and last_h and last_h.label == "LH" and last_l:
            if h[i] > last_h.price:
                if not any(s["swept_idx"] == last_h.index and s["side"] == -1 for s in active_setups):
                    zone_top = h[i]
                    zone_bot = l[i]
                    for bi in range(i, max(0, i - 6), -1):
                        if c[bi] >= o[bi]:
                            zone_top, zone_bot = h[bi], l[bi]
                            break
                    active_setups.append({
                        "side": -1,
                        "swept_idx": last_h.index,
                        "sweep_ext": h[i],
                        "sweep_bar": i,
                        "target1": last_l.price,
                        "stage": "SWEPT",
                        "zone_top": zone_top,
                        "zone_bot": zone_bot
                    })

        # 4. Advance Setup State Machines
        rem = []
        for s in active_setups:
            st = s["stage"]
            if s["side"] == 1:
                # Bullish setup
                if st == "SWEPT":
                    if l[i] < s["sweep_ext"]:
                        s["sweep_ext"] = l[i]
                        for bi in range(i, max(0, i - 6), -1):
                            if c[bi] < o[bi]:
                                s["zone_top"], s["zone_bot"] = h[bi], l[bi]
                                break
                    if c[i] > s["target1"]:
                        s["stage"] = "BOS_1"
                        s["hh2"] = h[i]
                    elif i - s["sweep_bar"] > 150:
                        rem.append(s)

                elif st == "BOS_1":
                    if h[i] > s["hh2"]:
                        s["hh2"] = h[i]
                    fib = s["sweep_ext"] + 0.50 * (s["hh2"] - s["sweep_ext"])
                    if l[i] <= fib:
                        s["stage"] = "FIB_50"
                        s["pb_ext"] = l[i]
                    elif l[i] < s["sweep_ext"]:
                        rem.append(s)

                elif st == "FIB_50":
                    if l[i] < s["pb_ext"]:
                        s["pb_ext"] = l[i]
                    if l[i] < s["sweep_ext"]:
                        rem.append(s)
                    elif c[i] > s["hh2"]:
                        s["stage"] = "ARMED"

                elif st == "ARMED":
                    z_top = s["zone_top"]
                    sl = s["sweep_ext"] - sl_buffer_usd
                    risk_pts = z_top - sl

                    if risk_pts > 0 and l[i] <= z_top:
                        # Entry triggered!
                        if i > busy_until:
                            fill = min(o[i], z_top) if o[i] < z_top else z_top
                            tp = fill + rr_ratio * risk_pts
                            xi, xp, why = scan_exit(o, h, l, i, 1, tp, sl)
                            if xi is None:
                                xi, xp, why = n - 1, c[n - 1], "open"
                            rows.append((i, xi, 1, fill, xp, sl, tp, why))
                            busy_until = xi
                        rem.append(s)
                    elif l[i] < s["sweep_ext"]:
                        rem.append(s)
                    elif i - s["sweep_bar"] > max_setup_bars:
                        rem.append(s)

            else:
                # Bearish setup
                if st == "SWEPT":
                    if h[i] > s["sweep_ext"]:
                        s["sweep_ext"] = h[i]
                        for bi in range(i, max(0, i - 6), -1):
                            if c[bi] >= o[bi]:
                                s["zone_top"], s["zone_bot"] = h[bi], l[bi]
                                break
                    if c[i] < s["target1"]:
                        s["stage"] = "BOS_1"
                        s["ll2"] = l[i]
                    elif i - s["sweep_bar"] > 150:
                        rem.append(s)

                elif st == "BOS_1":
                    if l[i] < s["ll2"]:
                        s["ll2"] = l[i]
                    fib = s["sweep_ext"] - 0.50 * (s["sweep_ext"] - s["ll2"])
                    if h[i] >= fib:
                        s["stage"] = "FIB_50"
                        s["pb_ext"] = h[i]
                    elif h[i] > s["sweep_ext"]:
                        rem.append(s)

                elif st == "FIB_50":
                    if h[i] > s["pb_ext"]:
                        s["pb_ext"] = h[i]
                    if h[i] > s["sweep_ext"]:
                        rem.append(s)
                    elif c[i] < s["ll2"]:
                        s["stage"] = "ARMED"

                elif st == "ARMED":
                    z_bot = s["zone_bot"]
                    sl = s["sweep_ext"] + sl_buffer_usd
                    risk_pts = sl - z_bot

                    if risk_pts > 0 and h[i] >= z_bot:
                        if i > busy_until:
                            fill = max(o[i], z_bot) if o[i] > z_bot else z_bot
                            tp = fill - rr_ratio * risk_pts
                            xi, xp, why = scan_exit(o, h, l, i, -1, tp, sl)
                            if xi is None:
                                xi, xp, why = n - 1, c[n - 1], "open"
                            rows.append((i, xi, -1, fill, xp, sl, tp, why))
                            busy_until = xi
                        rem.append(s)
                    elif h[i] > s["sweep_ext"]:
                        rem.append(s)
                    elif i - s["sweep_bar"] > max_setup_bars:
                        rem.append(s)

        for r in rem:
            if r in active_setups:
                active_setups.remove(r)

    rows.sort(key=lambda r: r[0])
    return pd.DataFrame(rows, columns=["entry_i", "exit_i", "side", "entry", "exit", "sl", "tp", "why"])


# =====================================================================================
# METRICS, EVALUATION & ROBUSTNESS (fibo_backtest.py exact match)
# =====================================================================================

def evaluate(tr: pd.DataFrame, idx, costs: Costs, risk_pct: float, max_lev: float) -> pd.DataFrame:
    if tr.empty:
        return tr.assign(notional=[], net=[], R=[], hold_days=[], entry_time=[], exit_time=[])
    side = tr.side.to_numpy(float)
    e, x, sl = tr.entry.to_numpy(), tr.exit.to_numpy(), tr.sl.to_numpy()
    risk = np.abs(e - sl) / e
    notional = np.minimum(risk_pct / risk, max_lev)
    days = np.asarray((idx[tr.exit_i.to_numpy()] - idx[tr.entry_i.to_numpy()]).total_seconds()) / 86400.0
    swap_rate = np.where(side > 0, costs.swap_long_bps_day, costs.swap_short_bps_day) / 1e4
    net = side * (x / e - 1) * notional - 2 * costs.per_side * notional + notional * swap_rate * days
    out_df = tr.copy()
    out_df["notional"], out_df["hold_days"] = notional, days
    out_df["net"] = np.maximum(net, -1.0)
    out_df["R"] = net / (notional * risk)
    out_df["entry_time"], out_df["exit_time"] = idx[tr.entry_i.to_numpy()], idx[tr.exit_i.to_numpy()]
    return out_df


def stats(tr: pd.DataFrame, first, last) -> dict:
    years = max((last - first).days / 365.25, 1e-6)
    if tr.empty:
        return dict(n=0, years=years, total=0.0, cagr=0.0, sharpe=0.0, maxdd=0.0, win=0.0, exp_R=0.0, pf=0.0,
                    avg_win_R=0.0, avg_loss_R=0.0, be_win=float("nan"), streak=0, hold_days=0.0)
    net = tr.net.to_numpy()
    eq = np.cumprod(1 + net)
    peak = np.maximum.accumulate(np.r_[1.0, eq])[1:]
    daily = tr.groupby(tr.exit_time.dt.normalize()).net.sum()
    cal = pd.date_range(first.normalize(), last.normalize(), freq="B")
    daily = daily.reindex(daily.index.union(cal)).fillna(0.0)
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0.0
    wins, losses = tr.R[tr.R > 0], tr.R[tr.R <= 0]
    aw = float(wins.mean()) if len(wins) else 0.0
    al = float(-losses.mean()) if len(losses) else 0.0
    streak = cur = 0
    for v in net <= 0:
        cur = cur + 1 if v else 0
        streak = max(streak, cur)
    pos, neg = net[net > 0].sum(), -net[net <= 0].sum()
    return dict(n=len(tr), years=years, total=float(eq[-1] - 1),
                cagr=float(eq[-1] ** (1 / years) - 1) if eq[-1] > 0 else -1.0, sharpe=float(sharpe),
                maxdd=float((eq / peak - 1).min()), win=float((tr.R > 0).mean()), exp_R=float(tr.R.mean()),
                pf=float(pos / neg) if neg > 0 else float("inf"), avg_win_R=aw, avg_loss_R=al,
                be_win=al / (aw + al) if (aw + al) > 0 else float("nan"), streak=streak,
                hold_days=float(tr.hold_days.mean()))


def line(tag, s):
    out(f"  {tag:<14} trades {s['n']:>5}  win {s['win'] * 100:5.1f}%  (needs {s['be_win'] * 100:5.1f}%)  "
        f"avg R {s['exp_R']:+.3f}  PF {s['pf']:.2f}  return {pct(s['total'])}  CAGR {pct(s['cagr'])}  "
        f"maxDD {pct(s['maxdd'])}  Sharpe {s['sharpe']:+.2f}")


def monte_carlo(net: np.ndarray, rng, n_sims=2000):
    T = len(net)
    n_sims = int(min(n_sims, max(200, 2e7 // max(T, 1))))
    paths = np.cumprod(1 + rng.choice(net, size=(n_sims, T), replace=True), axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((n_sims, 1)), paths], 1), 1)[:, 1:]
    return paths, paths[:, -1] - 1, (paths / peak - 1).min(1)


def random_timing_null(real: pd.DataFrame, o, h, l, idx, costs, risk_pct, max_lev, rng, n_perm=100):
    n = len(o)
    side = real.side.to_numpy()
    d_sl = (np.abs(real.entry - real.sl) / real.entry).to_numpy()
    d_tp = (np.abs(real.tp - real.entry) / real.entry).to_numpy()
    lo, hi = 1, n - 2
    means = []
    for _ in range(n_perm):
        t = rng.integers(lo, hi, len(real))
        e = o[t]
        sl = e * (1 - side * d_sl)
        tp = e * (1 + side * d_tp)
        rows = []
        for i in range(len(t)):
            xi, xp, _ = scan_exit(o, h, l, int(t[i]), int(side[i]), tp[i], sl[i])
            if xi is None:
                continue
            rows.append((int(t[i]), xi, int(side[i]), e[i], xp, sl[i]))
        ev = evaluate(pd.DataFrame(rows, columns=["entry_i", "exit_i", "side", "entry", "exit", "sl"]),
                      idx, costs, risk_pct, max_lev)
        means.append(float(ev.R.mean()))
    return np.array(means)


# =====================================================================================
# MAIN RUNNER & BENCHMARK SUITE
# =====================================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="XAUUSDM15.csv")
    ap.add_argument("--demo", choices=["gold"])
    ap.add_argument("--left", type=int, default=3, help="Swing pivot left window")
    ap.add_argument("--right", type=int, default=3, help="Swing pivot right window")
    ap.add_argument("--buffer", type=float, default=0.50, help="SL buffer in USD")
    ap.add_argument("--rr", type=float, default=3.0, help="Take-profit Risk-to-Reward ratio (default 3.0)")
    ap.add_argument("--risk-pct", type=float, default=1.0, help="percent of equity risked per trade")
    ap.add_argument("--max-leverage", type=float, default=10.0)
    ap.add_argument("--spread-bps", type=float, default=1.5)
    ap.add_argument("--slippage-bps", type=float, default=0.5)
    ap.add_argument("--swap-long-bps", type=float, default=-0.8)
    ap.add_argument("--swap-short-bps", type=float, default=-0.3)
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--outdir", default="results_smc")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    if a.csv:
        df, src = load_csv(a.csv), a.csv
    elif a.demo:
        df, src = demo_gold(rng), "DEMO random-walk gold"
    else:
        raise SystemExit("Give me --csv FILE or --demo gold")

    costs = Costs(a.spread_bps, a.slippage_bps, a.swap_long_bps, a.swap_short_bps)
    risk, maxlev = a.risk_pct / 100.0, a.max_leverage
    os.makedirs(a.outdir, exist_ok=True)
    idx = df.index
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n, first, last = len(df), idx[0], idx[-1]
    cut = int(n * 0.7)

    out("=" * 100)
    out(f"DATA   {src}")
    out(f"       {n:,} bars, {first:%Y-%m-%d} to {last:%Y-%m-%d}")
    out(f"RULE   SMC Continuation: Sweep -> BOS 1 -> Fib 50% Retrace -> BOS 2 -> Demand/Supply Zone (1:{a.rr:.1f} RR)")
    out(f"       Swing pivot: L={a.left}, R={a.right} | SL buffer: ${a.buffer:.2f} | Risk: {a.risk_pct}% per trade, max {maxlev}x")
    out(f"COSTS  spread {costs.spread_bps} bps, slippage {costs.slippage_bps} bps/side, swap {costs.swap_long_bps_day}/"
        f"{costs.swap_short_bps_day} bps/day (long/short)")
    out("=" * 100)

    raw = find_smc_trades(df, left_bars=a.left, right_bars=a.right, sl_buffer_usd=a.buffer, rr_ratio=a.rr)
    tr = evaluate(raw, idx, costs, risk, maxlev)

    if len(tr) == 0:
        raise SystemExit("The SMC continuation rule produced no trades on this dataset.")

    S = stats(tr, first, last)
    be_win_ideal = 1.0 / (1.0 + a.rr)

    out("\n[0] WHAT TO EXPECT")
    out(f"    Target is {a.rr:.1f}x the stop distance -> reward/risk ratio {a.rr:.2f}:1.")
    out(f"    Theoretical break-even win rate before costs: {be_win_ideal * 100:.1f}%.")
    out("    On a pure random walk, win rate is ~that number; expectancy is ~0 before costs and negative after.")
    out("    Any real edge has to come from structural demand/supply zone liquidity sweeps, and must beat costs.")

    out("\n[1] THE RULE, FULL SAMPLE (no parameters were tuned for this result)")
    line("all trades", S)
    for nm, sd in (("longs", 1), ("shorts", -1)):
        sub = tr[tr.side == sd]
        if len(sub):
            line(nm, stats(sub, first, last))
    bh = c[-1] / o[0] - 1
    out(f"    Buy & hold over the same period: {pct(bh)}.  Average trade lasts {S['hold_days']:.1f} days; "
        f"longest losing streak {S['streak']}.")
    out(f"    Exits: " + ", ".join(f"{k} {v}" for k, v in tr.why.value_counts().items()))

    out("\n[2] CHRONOLOGICAL HALVES (rule fixed in advance; first 70% vs last 30%)")
    for nm, m in (("first 70%", tr.entry_i < cut), ("last 30%", tr.entry_i >= cut)):
        sub = tr[m]
        if len(sub):
            line(nm, stats(sub, idx[0] if nm.startswith("first") else idx[cut], idx[cut] if nm.startswith("first") else last))

    out("\n[3] BY YEAR (by entry date)")
    ylist = []
    for y, g in tr.groupby(tr.entry_time.dt.year):
        ylist.append((y, len(g), (g.R > 0).mean(), g.R.mean(), (1 + g.net).prod() - 1))
        out(f"    {y}: {len(g):>4} trades  win {ylist[-1][2] * 100:5.1f}%  avg R {ylist[-1][3]:+.3f}  return {pct(ylist[-1][4])}")
    pos_years = sum(1 for r in ylist if r[3] > 0)

    out("\n[4] SENSITIVITY (varying pivot window L, R and SL buffer) - avg R per trade after costs")
    pivot_settings = [(2, 2), (3, 3), (4, 4), (5, 5)]
    buffer_settings = [0.25, 0.50, 1.00]
    out("    Pivots (L, R) \\ SL Buffer  " + "  ".join(f"${b:.2f}".rjust(12) for b in buffer_settings))
    grid = {}
    for (pl, pr) in pivot_settings:
        cells = []
        for b in buffer_settings:
            raw_sens = find_smc_trades(df, left_bars=pl, right_bars=pr, sl_buffer_usd=b, rr_ratio=a.rr)
            ev = evaluate(raw_sens, idx, costs, risk, maxlev)
            avg_r = float(ev.R.mean()) if len(ev) else np.nan
            grid[(pl, pr, b)] = (avg_r, len(ev))
            cells.append(f"{avg_r:+.3f} ({len(ev)})".rjust(12) if not np.isnan(avg_r) else "  N/A (0)".rjust(12))
        out(f"    L={pl}, R={pr}".ljust(28) + "  ".join(cells))

    gv = np.array([v[0] for v in grid.values() if not np.isnan(v[0])])
    pos_share = float((gv > 0).mean()) if len(gv) else 0.0
    out(f"    {pos_share * 100:.0f}% of these settings have positive average R.")

    out("\n[5] COST STRESS")
    stress = {}
    for mult in (0, 1, 2, 3):
        c2 = replace(costs, spread_bps=costs.spread_bps * mult, slippage_bps=costs.slippage_bps * mult,
                     swap_long_bps_day=costs.swap_long_bps_day * mult, swap_short_bps_day=costs.swap_short_bps_day * mult)
        s2 = stats(evaluate(raw, idx, c2, risk, maxlev), first, last)
        stress[mult] = s2
        out(f"    {mult}x costs: avg R {s2['exp_R']:+.3f}   return {pct(s2['total'])}   maxDD {pct(s2['maxdd'])}")

    out("\n[6] MONTE CARLO (trade order reshuffled)")
    flag_dd = None
    if len(tr) >= 10:
        paths, fin, dd = monte_carlo(tr.net.to_numpy(), rng)
        p5, p50, p95 = np.percentile(fin, [5, 50, 95])
        d5, d50 = np.percentile(dd, [5, 50])
        flag_dd = float((dd < -0.30).mean())
        out(f"    Final return 5th {pct(p5)}  median {pct(p50)}  95th {pct(p95)}")
        out(f"    Max drawdown worst-5% {pct(d5)}  median {pct(d50)}   P(drawdown > 30%) = {flag_dd * 100:.0f}%   "
            f"P(ends down) = {(fin < 0).mean() * 100:.0f}%")
        fig, ax = plt.subplots(figsize=(8, 4.6))
        for p in paths[:120]:
            ax.plot(p, color=GRAY, alpha=0.10, lw=0.8)
        ax.plot(np.cumprod(1 + tr.net.to_numpy()), color=BLUE, lw=1.8)
        ax.set_yscale("log")
        ax.set_title("Reshuffled-trade equity paths (blue = actual order)", fontsize=11)
        ax.set_xlabel("trade #", color=MUTED)
        style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(a.outdir, "monte_carlo.png"), dpi=120)
        plt.close(fig)
    else:
        out(f"    Only {len(tr)} trades - insufficient for Monte Carlo (needs >= 10).")

    out(f"\n[7] RANDOM-TIMING TEST ({a.perms} runs: same side and same stop/target distances, random entry times)")
    null = random_timing_null(raw, o, h, l, idx, costs, risk, maxlev, rng, a.perms)
    pval = float((np.sum(null >= S["exp_R"]) + 1) / (len(null) + 1))
    out(f"    Actual avg R {S['exp_R']:+.3f}  vs random timing: mean {null.mean():+.3f}, 95th pct {np.percentile(null, 95):+.3f}, p = {pval:.3f}")
    out("    Small p = SMC entry timing beats random entries with identical reward/risk.")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null, bins=20, color=GRAY, alpha=0.85)
    ax.axvline(S["exp_R"], color=BLUE, lw=2)
    ax.set_title("Avg R per trade: random timing (gray) vs SMC rule (blue line)", fontsize=11)
    ax.set_xlabel("average R per trade, after costs", color=MUTED)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "null_test.png"), dpi=120)
    plt.close(fig)

    flags = []
    if S["exp_R"] <= 0:
        flags.append(f"average R after costs is {S['exp_R']:+.3f} (not positive)")
    if S["win"] < S["be_win"]:
        flags.append(f"win rate {S['win'] * 100:.1f}% is below the {S['be_win'] * 100:.1f}% needed to break even")
    if S["n"] < 100:
        flags.append(f"only {S['n']} trades - small sample size")
    if len(ylist) >= 3 and pos_years < len(ylist) / 2:
        flags.append(f"profitable (avg R > 0) in only {pos_years} of {len(ylist)} years")
    lo, sh = tr[tr.side == 1], tr[tr.side == -1]
    if len(lo) and len(sh) and (lo.R.mean() > 0) != (sh.R.mean() > 0):
        flags.append("longs and shorts disagree (one side carries it)")
    if pval > 0.05:
        flags.append(f"not distinguishable from random entry timing (p={pval:.2f})")
    if stress[2]["exp_R"] <= 0:
        flags.append("falls to zero or below if costs are 2x")
    if pos_share < 0.5:
        flags.append("fewer than half of nearby settings are profitable")
    if flag_dd is not None and flag_dd > 0.10:
        flags.append(f"{flag_dd * 100:.0f}% chance of a >30% drawdown at this risk size")

    out("\n" + "=" * 100)
    if flags:
        out("RED FLAGS:")
        for f in flags:
            out(f"  - {f}")
    else:
        out("No automatic red flags. That justifies forward testing on demo.")
    out("=" * 100)

    # Equity chart
    eq = pd.Series(np.cumprod(1 + tr.net.to_numpy()), index=tr.exit_time)
    bh_s = pd.Series(c / o[0], index=idx)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    ax1.plot(bh_s.index, bh_s.values, color=GRAY, lw=1.2)
    ax1.plot(eq.index, eq.values, color=BLUE, lw=1.8)
    ax1.set_yscale("log")
    ax1.set_title("Growth of 1: SMC Rule (blue) vs Buy & Hold (gray), after costs", fontsize=11)
    ax1.text(eq.index[-1], eq.values[-1], " SMC rule", color=INK, va="center", fontsize=9)
    ax1.text(bh_s.index[-1], bh_s.values[-1], " buy & hold", color=INK, va="center", fontsize=9)
    style(ax1)
    ax2.fill_between(eq.index, eq.values / np.maximum.accumulate(eq.values) - 1, 0, color=BLUE, alpha=0.35, lw=0)
    ax2.set_title("Drawdown of SMC Rule", fontsize=10)
    style(ax2)
    fig.tight_layout()
    fig.savefig(os.path.join(a.outdir, "equity.png"), dpi=120)
    plt.close(fig)

    tr.to_csv(os.path.join(a.outdir, "trades.csv"), index=False)
    with open(os.path.join(a.outdir, "report.txt"), "w") as fh:
        fh.write("\n".join(LINES))
    print(f"\nSaved report, trades and charts to ./{a.outdir}/")


if __name__ == "__main__":
    main()
