#!/usr/bin/env python3
"""
smc_variations_backtest.py - Tests two realistic, institutional SMC variations on Gold:

VARIATION 1: Liquidity Sweep Reversal (Turtle Soup / Stop-Hunt Reversal)
  - Price sweeps beyond a key confirmed swing high or low (trapping breakout traders).
  - Candle rejects and closes back inside the range.
  - Enter immediately on reversal back inside.
  - SL placed just beyond the sweep wick.
  - TP targeted at 1:2 or 1:3 RR (or the opposite swing extreme).

VARIATION 2: Active Leg Trend Pullback (Smart Money Continuation in Discount/Premium)
  - Market structure trend is established (Bullish: HH/HL, Bearish: LH/LL).
  - Break of Structure (BOS) confirms active trend expansion.
  - Instead of waiting for a 100% crash back to ancient history, enter when price
    pulls back into the 50%-61.8% Discount zone of the ACTIVE impulse leg.
  - SL placed below the active swing low (for Bullish) or above active swing high (for Bearish).
  - TP targeted at 1:2 or 1:3 RR (or prior swing high/low extension).
"""
from __future__ import annotations

import argparse
import os
from dataclasses import replace
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fibo_backtest import load_csv, Costs
from market_structure import find_swings, classify_structure, SwingPoint
from smc_backtest import scan_exit, evaluate, stats, line, monte_carlo, random_timing_null, pct, style

INK, MUTED, GRID, BLUE, GRAY, GREEN, RED = "#2b2b2b", "#6b6b6b", "#e6e6e6", "#2a6fbb", "#9aa0a6", "#27ae60", "#c0392b"


# =====================================================================================
# VARIATION 1: LIQUIDITY SWEEP REVERSAL (TURTLE SOUP)
# =====================================================================================

def find_sweep_reversal_trades(
    df: pd.DataFrame,
    left_bars: int = 3,
    right_bars: int = 3,
    sl_buffer_usd: float = 0.50,
    rr_ratio: float = 2.0
) -> pd.DataFrame:
    """
    Identifies false breakouts / liquidity sweeps:
    - Bullish: low < active_low, but close > active_low (sweep & reclaim).
    - Bearish: high > active_high, but close < active_high (sweep & reclaim).
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

    rows = []
    busy_until = -1
    last_swept_low_idx = -1
    last_swept_high_idx = -1

    for i in range(n - 1):
        if i in conf_swings:
            for s in conf_swings[i]:
                if s.swing_type == "HIGH":
                    last_h = s
                else:
                    last_l = s

        # 1. Bullish Sweep Reversal (Sweep Low and Reclaim)
        if last_l is not None and i > busy_until and last_l.index != last_swept_low_idx:
            # Did current bar sweep below the low and close back above?
            if l[i] < last_l.price and c[i] > last_l.price:
                entry_i = i + 1
                fill = o[entry_i]
                sl = min(l[i], l[max(0, i-1)]) - sl_buffer_usd
                risk = fill - sl

                if risk > 0.50:  # Valid risk
                    tp = fill + rr_ratio * risk
                    xi, xp, why = scan_exit(o, h, l, entry_i, 1, tp, sl)
                    if xi is None:
                        xi, xp, why = n - 1, c[n - 1], "open"
                    rows.append((entry_i, xi, 1, fill, xp, sl, tp, why))
                    busy_until = xi
                    last_swept_low_idx = last_l.index

        # 2. Bearish Sweep Reversal (Sweep High and Reclaim)
        if last_h is not None and i > busy_until and last_h.index != last_swept_high_idx:
            if h[i] > last_h.price and c[i] < last_h.price:
                entry_i = i + 1
                fill = o[entry_i]
                sl = max(h[i], h[max(0, i-1)]) + sl_buffer_usd
                risk = sl - fill

                if risk > 0.50:
                    tp = fill - rr_ratio * risk
                    xi, xp, why = scan_exit(o, h, l, entry_i, -1, tp, sl)
                    if xi is None:
                        xi, xp, why = n - 1, c[n - 1], "open"
                    rows.append((entry_i, xi, -1, fill, xp, sl, tp, why))
                    busy_until = xi
                    last_swept_high_idx = last_h.index

    return pd.DataFrame(rows, columns=["entry_i", "exit_i", "side", "entry", "exit", "sl", "tp", "why"])


# =====================================================================================
# VARIATION 2: ACTIVE LEG TREND PULLBACK (DISCOUNT / PREMIUM CONTINUATION)
# =====================================================================================

def find_active_leg_trades(
    df: pd.DataFrame,
    left_bars: int = 3,
    right_bars: int = 3,
    sl_buffer_usd: float = 0.50,
    rr_ratio: float = 2.0,
    fib_entry: float = 0.50
) -> pd.DataFrame:
    """
    Trades pullbacks to the active trend impulse leg:
    - Bullish: in uptrend, price broke previous high (BOS). As it pulls back to >= 50%
      of the active leg [active_low -> active_high], enter Long.
    - SL placed at active_low - buffer.
    - TP targeted at 1:2 or 1:3 RR.
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

    # Track active impulse leg
    leg_low = np.nan
    leg_high = np.nan
    bos_active = False
    bos_direction = 0  # +1 for Bullish, -1 for Bearish
    last_bos_swing_idx = -1

    rows = []
    busy_until = -1

    for i in range(n - 1):
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

        # Check for Break of Structure (BOS)
        if trend == "BULLISH" and last_h is not None and last_l is not None:
            if c[i] > last_h.price and last_h.index != last_bos_swing_idx:
                bos_active = True
                bos_direction = 1
                leg_low = last_l.price
                leg_high = h[i]
                last_bos_swing_idx = last_h.index
        elif trend == "BEARISH" and last_l is not None and last_h is not None:
            if c[i] < last_l.price and last_l.index != last_bos_swing_idx:
                bos_active = True
                bos_direction = -1
                leg_high = last_h.price
                leg_low = l[i]
                last_bos_swing_idx = last_l.index

        # If BOS active, track peak/trough of the impulse
        if bos_active:
            if bos_direction == 1:
                if h[i] > leg_high:
                    leg_high = h[i]

                # Check pullback into discount
                fib_level = leg_low + fib_entry * (leg_high - leg_low)
                if l[i] <= fib_level and c[i] > leg_low:
                    entry_i = i + 1
                    fill = o[entry_i]
                    sl = leg_low - sl_buffer_usd
                    risk = fill - sl

                    if risk > 0.50 and entry_i > busy_until:
                        tp = fill + rr_ratio * risk
                        xi, xp, why = scan_exit(o, h, l, entry_i, 1, tp, sl)
                        if xi is None:
                            xi, xp, why = n - 1, c[n - 1], "open"
                        rows.append((entry_i, xi, 1, fill, xp, sl, tp, why))
                        busy_until = xi
                    bos_active = False  # Consumed

                elif l[i] <= leg_low:  # Invalidated
                    bos_active = False

            else:
                if l[i] < leg_low:
                    leg_low = l[i]

                fib_level = leg_high - fib_entry * (leg_high - leg_low)
                if h[i] >= fib_level and c[i] < leg_high:
                    entry_i = i + 1
                    fill = o[entry_i]
                    sl = leg_high + sl_buffer_usd
                    risk = sl - fill

                    if risk > 0.50 and entry_i > busy_until:
                        tp = fill - rr_ratio * risk
                        xi, xp, why = scan_exit(o, h, l, entry_i, -1, tp, sl)
                        if xi is None:
                            xi, xp, why = n - 1, c[n - 1], "open"
                        rows.append((entry_i, xi, -1, fill, xp, sl, tp, why))
                        busy_until = xi
                    bos_active = False

                elif h[i] >= leg_high:
                    bos_active = False

    return pd.DataFrame(rows, columns=["entry_i", "exit_i", "side", "entry", "exit", "sl", "tp", "why"])


# =====================================================================================
# RUNNER & COMPARISON SUITE
# =====================================================================================

def run_strategy_suite(
    name: str,
    raw_df: pd.DataFrame,
    df: pd.DataFrame,
    costs: Costs,
    risk: float,
    maxlev: float,
    outdir: str,
    rr_ratio: float = 2.0
):
    idx = df.index
    first, last = idx[0], idx[-1]
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    cut = int(len(df) * 0.7)

    os.makedirs(outdir, exist_ok=True)
    report_lines = []

    def rout(s=""):
        print(s)
        report_lines.append(s)

    rout("=" * 100)
    rout(f"STRATEGY: {name}")
    rout(f"DATA:     {len(df):,} bars, {first:%Y-%m-%d} to {last:%Y-%m-%d}")
    rout(f"RULES:    Risk-to-Reward {rr_ratio:.1f}:1  |  Risk {risk*100:.1f}% per trade, max {maxlev}x")
    rout(f"COSTS:    spread {costs.spread_bps} bps, slippage {costs.slippage_bps} bps/side, swap {costs.swap_long_bps_day}/{costs.swap_short_bps_day} bps/day")
    rout("=" * 100)

    tr = evaluate(raw_df, idx, costs, risk, maxlev)
    if tr.empty:
        rout("No trades produced.")
        return

    S = stats(tr, first, last)
    be_win_ideal = 1.0 / (1.0 + rr_ratio)

    rout(f"\n[0] WHAT TO EXPECT")
    rout(f"    Target is {rr_ratio:.1f}x stop -> Break-even win rate before costs: {be_win_ideal * 100:.1f}%.")

    rout(f"\n[1] FULL SAMPLE METRICS")
    rout(f"  all trades     trades {S['n']:>5}  win {S['win'] * 100:5.1f}%  (needs {S['be_win'] * 100:5.1f}%)  "
         f"avg R {S['exp_R']:+.3f}  PF {S['pf']:.2f}  return {pct(S['total'])}  CAGR {pct(S['cagr'])}  "
         f"maxDD {pct(S['maxdd'])}  Sharpe {S['sharpe']:+.2f}")

    for nm, sd in (("longs", 1), ("shorts", -1)):
        sub = tr[tr.side == sd]
        if len(sub):
            s_sub = stats(sub, first, last)
            rout(f"  {nm:<14} trades {s_sub['n']:>5}  win {s_sub['win'] * 100:5.1f}%  (needs {s_sub['be_win'] * 100:5.1f}%)  "
                 f"avg R {s_sub['exp_R']:+.3f}  PF {s_sub['pf']:.2f}  return {pct(s_sub['total'])}  CAGR {pct(s_sub['cagr'])}  "
                 f"maxDD {pct(s_sub['maxdd'])}  Sharpe {s_sub['sharpe']:+.2f}")

    bh = c[-1] / o[0] - 1
    rout(f"    Buy & hold over period: {pct(bh)}. Avg trade hold: {S['hold_days']:.1f} days; longest losing streak: {S['streak']}.")
    rout(f"    Exits: " + ", ".join(f"{k} {v}" for k, v in tr.why.value_counts().items()))

    rout(f"\n[2] CHRONOLOGICAL HALVES (first 70% vs last 30%)")
    for nm, m in (("first 70%", tr.entry_i < cut), ("last 30%", tr.entry_i >= cut)):
        sub = tr[m]
        if len(sub):
            s_half = stats(sub, idx[0] if nm.startswith("first") else idx[cut], idx[cut] if nm.startswith("first") else last)
            rout(f"  {nm:<14} trades {s_half['n']:>5}  win {s_half['win'] * 100:5.1f}%  avg R {s_half['exp_R']:+.3f}  "
                 f"PF {s_half['pf']:.2f}  return {pct(s_half['total'])}  maxDD {pct(s_half['maxdd'])}  Sharpe {s_half['sharpe']:+.2f}")

    rout(f"\n[3] COST STRESS")
    for mult in (0, 1, 2):
        c2 = replace(costs, spread_bps=costs.spread_bps * mult, slippage_bps=costs.slippage_bps * mult,
                     swap_long_bps_day=costs.swap_long_bps_day * mult, swap_short_bps_day=costs.swap_short_bps_day * mult)
        s2 = stats(evaluate(raw_df, idx, c2, risk, maxlev), first, last)
        rout(f"    {mult}x costs: avg R {s2['exp_R']:+.3f}   return {pct(s2['total'])}   maxDD {pct(s2['maxdd'])}")

    # Plot Equity
    eq = pd.Series(np.cumprod(1 + tr.net.to_numpy()), index=tr.exit_time)
    bh_s = pd.Series(c / o[0], index=idx)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    ax1.plot(bh_s.index, bh_s.values, color=GRAY, lw=1.2)
    ax1.plot(eq.index, eq.values, color=BLUE, lw=1.8)
    ax1.set_yscale("log")
    ax1.set_title(f"Growth of 1: {name} (blue) vs Buy & Hold (gray)", fontsize=11)
    ax1.text(eq.index[-1], eq.values[-1], f" {name}", color=INK, va="center", fontsize=9)
    ax1.text(bh_s.index[-1], bh_s.values[-1], " buy & hold", color=INK, va="center", fontsize=9)
    style(ax1)
    ax2.fill_between(eq.index, eq.values / np.maximum.accumulate(eq.values) - 1, 0, color=BLUE, alpha=0.35, lw=0)
    ax2.set_title(f"Drawdown: {name}", fontsize=10)
    style(ax2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "equity.png"), dpi=120)
    plt.close(fig)

    tr.to_csv(os.path.join(outdir, "trades.csv"), index=False)
    with open(os.path.join(outdir, "report.txt"), "w") as fh:
        fh.write("\n".join(report_lines))


def main():
    parser = argparse.ArgumentParser(description="SMC Variations Backtester")
    parser.add_argument("--csv", default="XAUUSDM15.csv")
    parser.add_argument("--left", type=int, default=3)
    parser.add_argument("--right", type=int, default=3)
    parser.add_argument("--buffer", type=float, default=0.50)
    parser.add_argument("--rr", type=float, default=2.0, help="Risk-to-Reward ratio (default 2.0)")
    parser.add_argument("--risk-pct", type=float, default=1.0)
    parser.add_argument("--max-leverage", type=float, default=10.0)
    parser.add_argument("--spread-bps", type=float, default=1.5)
    parser.add_argument("--slippage-bps", type=float, default=0.5)
    parser.add_argument("--swap-long-bps", type=float, default=-0.8)
    parser.add_argument("--swap-short-bps", type=float, default=-0.3)
    args = parser.parse_args()

    df = load_csv(args.csv)
    costs = Costs(args.spread_bps, args.slippage_bps, args.swap_long_bps, args.swap_short_bps)
    risk = args.risk_pct / 100.0

    print("\nRunning Variation 1: Liquidity Sweep Reversal (Turtle Soup)...")
    v1_raw = find_sweep_reversal_trades(df, left_bars=args.left, right_bars=args.right, sl_buffer_usd=args.buffer, rr_ratio=args.rr)
    run_strategy_suite("Variation 1 - Liquidity Sweep Reversal", v1_raw, df, costs, risk, args.max_leverage, "results_v1_sweep", rr_ratio=args.rr)

    print("\n\nRunning Variation 2: Active Leg Trend Pullback (Discount/Premium)...")
    v2_raw = find_active_leg_trades(df, left_bars=args.left, right_bars=args.right, sl_buffer_usd=args.buffer, rr_ratio=args.rr)
    run_strategy_suite("Variation 2 - Active Leg Pullback", v2_raw, df, costs, risk, args.max_leverage, "results_v2_active_leg", rr_ratio=args.rr)


if __name__ == "__main__":
    main()
