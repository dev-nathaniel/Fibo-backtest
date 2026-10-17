#!/usr/bin/env python3
"""
validate_deep.py - Deep validation of the SMC Active Leg Pullback (Session 08:00-16:00) on Gold:
1. Session Window Sensitivity (plateau check)
2. Fibonacci Entry Level Sensitivity (38.2%, 50%, 61.8%)
3. Pivot Window Sensitivity (L=2..5, R=2..5)
4. Cost Stress Test (0x, 1x, 2x, 3x spread/slippage)
5. Monte Carlo Reshuffling (2,000 paths)
6. Random-Timing Null Test (p-value)
"""
from __future__ import annotations

import os
from dataclasses import replace
import numpy as np
import pandas as pd

from fibo_backtest import load_csv, Costs
from explore_filters import find_active_leg_trades_with_filters
from smc_backtest import evaluate, stats, scan_exit, monte_carlo, random_timing_null


def main():
    df = load_csv("XAUUSDM15.csv")
    costs = Costs(1.5, 0.5, -0.8, -0.3)
    risk = 0.01
    maxlev = 10.0
    first, last = df.index[0], df.index[-1]

    print("=" * 85)
    print("      DEEP QUANTITATIVE VALIDATION SUITE (GOLD M15 - SESSION FILTER)")
    print("=" * 85)

    # 1. Base Strategy Performance
    raw_base = find_active_leg_trades_with_filters(df, session_start_hour=8, session_end_hour=16, rr_ratio=2.0)
    ev_base = evaluate(raw_base, df.index, costs, risk, maxlev)
    s_base = stats(ev_base, first, last)

    print(f"\n[1] BASELINE TEST (08:00 - 16:00, RR 2:1, Fib 50%)")
    print(f"    Trades: {s_base['n']} | Win Rate: {s_base['win']*100:.1f}% (Needs: {s_base['be_win']*100:.1f}%)")
    print(f"    Avg R: {s_base['exp_R']:+.3f} | Profit Factor: {s_base['pf']:.2f} | Net Return: {s_base['total']*100:+.1f}%")
    print(f"    Max Drawdown: {s_base['maxdd']*100:.1f}% | Sharpe: {s_base['sharpe']:+.2f}")

    # 2. Session Window Sensitivity (Plateau Check)
    print(f"\n[2] SESSION WINDOW SENSITIVITY (Is 08:00-16:00 an isolated spike or a broad plateau?)")
    print(f"    {'Session Window':18} | {'Trades':6} | {'Win %':6} | {'Avg R':8} | {'PF':5} | {'Return':8} | {'MaxDD':6}")
    print("    " + "-" * 70)
    for s_start, s_end in [(7, 15), (7, 16), (7, 17), (7, 18), (8, 15), (8, 16), (8, 17), (9, 16), (9, 17)]:
        raw = find_active_leg_trades_with_filters(df, session_start_hour=s_start, session_end_hour=s_end, rr_ratio=2.0)
        ev = evaluate(raw, df.index, costs, risk, maxlev)
        s = stats(ev, first, last)
        print(f"    {s_start:02d}:00 - {s_end:02d}:00 UTC    | {s['n']:>6} | {s['win']*100:5.1f}% | {s['exp_R']:>+7.3f} | {s['pf']:>4.2f} | {s['total']*100:>+7.1f}% | {s['maxdd']*100:>5.1f}%")

    # 3. Fibonacci Retracement Depth Sensitivity
    print(f"\n[3] FIBONACCI RETRACEMENT DEPTH SENSITIVITY")
    print(f"    {'Fib Level':18} | {'Trades':6} | {'Win %':6} | {'Avg R':8} | {'PF':5} | {'Return':8} | {'MaxDD':6}")
    print("    " + "-" * 70)
    for fib in [0.382, 0.500, 0.618]:
        raw = find_active_leg_trades_with_filters(df, session_start_hour=8, session_end_hour=16, fib_entry=fib, rr_ratio=2.0)
        ev = evaluate(raw, df.index, costs, risk, maxlev)
        s = stats(ev, first, last)
        fib_label = "38.2% (Shallow)" if fib == 0.382 else ("50.0% (Equilibrium)" if fib == 0.500 else "61.8% (Golden)")
        print(f"    {fib_label:18} | {s['n']:>6} | {s['win']*100:5.1f}% | {s['exp_R']:>+7.3f} | {s['pf']:>4.2f} | {s['total']*100:>+7.1f}% | {s['maxdd']*100:>5.1f}%")

    # 4. Pivot Lookback Sensitivity
    print(f"\n[4] PIVOT LOOKBACK (L, R) SENSITIVITY")
    print(f"    {'Pivots (L, R)':18} | {'Trades':6} | {'Win %':6} | {'Avg R':8} | {'PF':5} | {'Return':8} | {'MaxDD':6}")
    print("    " + "-" * 70)
    for p in [2, 3, 4, 5]:
        raw = find_active_leg_trades_with_filters(df, left_bars=p, right_bars=p, session_start_hour=8, session_end_hour=16, rr_ratio=2.0)
        ev = evaluate(raw, df.index, costs, risk, maxlev)
        s = stats(ev, first, last)
        print(f"    L={p}, R={p} ({2*p+1} bars)    | {s['n']:>6} | {s['win']*100:5.1f}% | {s['exp_R']:>+7.3f} | {s['pf']:>4.2f} | {s['total']*100:>+7.1f}% | {s['maxdd']*100:>5.1f}%")

    # 5. Cost Stress Test
    print(f"\n[5] COST STRESS TEST")
    for mult in [0, 1, 2, 3]:
        c2 = replace(costs, spread_bps=costs.spread_bps * mult, slippage_bps=costs.slippage_bps * mult,
                     swap_long_bps_day=costs.swap_long_bps_day * mult, swap_short_bps_day=costs.swap_short_bps_day * mult)
        s2 = stats(evaluate(raw_base, df.index, c2, risk, maxlev), first, last)
        print(f"    {mult}x Costs: Avg R {s2['exp_R']:+.3f} | Return {s2['total']*100:+.1f}% | MaxDD {s2['maxdd']*100:.1f}% | PF {s2['pf']:.2f}")

    # 6. Monte Carlo Reshuffling (2,000 paths)
    rng = np.random.default_rng(7)
    paths, fin, dd = monte_carlo(ev_base.net.to_numpy(), rng, n_sims=2000)
    p5, p50, p95 = np.percentile(fin, [5, 50, 95])
    d5, d50 = np.percentile(dd, [5, 50])
    p_dd10 = float((dd < -0.10).mean()) * 100
    p_down = float((fin < 0).mean()) * 100

    print(f"\n[6] MONTE CARLO SIMULATION (2,000 Reshuffled Paths)")
    print(f"    Return Distribution: 5th pct: {p5*100:+.1f}% | Median: {p50*100:+.1f}% | 95th pct: {p95*100:+.1f}%")
    print(f"    Max Drawdown:        Worst-5%: {d5*100:.1f}% | Median: {d50*100:.1f}%")
    print(f"    Probability of Drawdown > 10%: {p_dd10:.1f}%")
    print(f"    Probability of Ending in Loss:  {p_down:.1f}%")

    # 7. Random-Timing Null Test
    null = random_timing_null(raw_base, df["open"].values, df["high"].values, df["low"].values, df.index, costs, risk, maxlev, rng, 100)
    pval = float((np.sum(null >= s_base["exp_R"]) + 1) / (len(null) + 1))
    print(f"\n[7] RANDOM-TIMING NULL TEST (100 Permutations)")
    print(f"    Actual Avg R: {s_base['exp_R']:+.3f} vs Random Timing: Mean {null.mean():+.3f}, 95th pct {np.percentile(null, 95):+.3f}")
    print(f"    Statistical Significance (p-value): p = {pval:.3f} {'(Statistically Significant: p < 0.05)' if pval < 0.05 else '(Not significant)'}")


if __name__ == "__main__":
    main()
