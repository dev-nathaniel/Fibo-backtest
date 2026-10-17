#!/usr/bin/env python3
"""
run_3year_backtest.py - Comprehensive 3-Year / 4-Year (2023 - 2026) Backtest on Gold:
Testing SMC strategies across 89,481 bars on XAUUSD_M15_History.csv.

Strategies evaluated:
  1. Original SMC Rule (100% Retrace to Sweep Origin)
  2. Variation 1: Liquidity Sweep Reversal (Turtle Soup)
  3. Variation 2: Active Leg Pullback (No Filters)
  4. Variation 2 + Session Filter (08:00 - 16:00 London/NY Peak)
  5. Variation 2 + Combined (H1 Trend Alignment + Session Filter)
"""
from __future__ import annotations

import os
import time
from dataclasses import replace
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fibo_backtest import load_csv, Costs
from smc_backtest import find_smc_trades, evaluate, stats, line, pct, style
from smc_variations_backtest import find_sweep_reversal_trades, find_active_leg_trades
from explore_filters import get_h1_trend_series, find_active_leg_trades_with_filters

INK, MUTED, GRID, BLUE, GRAY, GREEN, RED, ORANGE, PURPLE = (
    "#2b2b2b", "#6b6b6b", "#e6e6e6", "#2a6fbb", "#9aa0a6", "#27ae60", "#c0392b", "#e67e22", "#8e44ad"
)

OUTDIR = "results_3year"
REPORT_LINES = []


def rout(s=""):
    print(s)
    REPORT_LINES.append(s)


def print_strategy_report(name: str, tr: pd.DataFrame, df: pd.DataFrame, costs: Costs, risk: float, maxlev: float, cut: int):
    first, last = df.index[0], df.index[-1]
    bh = df["close"].iloc[-1] / df["open"].iloc[0] - 1

    rout("\n" + "=" * 100)
    rout(f"STRATEGY: {name}")
    rout(f"DATA:     {len(df):,} bars, {first:%Y-%m-%d} to {last:%Y-%m-%d}")
    rout("=" * 100)

    if tr.empty:
        rout("No trades produced.")
        return None

    S = stats(tr, first, last)
    rout("\n[1] FULL SAMPLE METRICS (2023 - 2026)")
    line("all trades", S)
    for nm, sd in (("longs", 1), ("shorts", -1)):
        sub = tr[tr.side == sd]
        if len(sub):
            line(nm, stats(sub, first, last))
    rout(f"    Buy & hold over period: {pct(bh)}. Avg trade hold: {S['hold_days']:.1f} days; longest losing streak: {S['streak']}.")
    rout(f"    Exits: " + ", ".join(f"{k} {v}" for k, v in tr.why.value_counts().items()))

    rout("\n[2] CHRONOLOGICAL HALVES (first 70% vs last 30%)")
    for nm, m in (("first 70%", tr.entry_i < cut), ("last 30%", tr.entry_i >= cut)):
        sub = tr[m]
        if len(sub):
            s_half = stats(sub, df.index[0] if nm.startswith("first") else df.index[cut], df.index[cut] if nm.startswith("first") else last)
            line(nm, s_half)

    rout("\n[3] BY YEAR BREAKDOWN")
    ylist = []
    for y, g in tr.groupby(tr.entry_time.dt.year):
        ylist.append((y, len(g), (g.R > 0).mean(), g.R.mean(), (1 + g.net).prod() - 1))
        rout(f"    {y}: {len(g):>4} trades  win {ylist[-1][2] * 100:5.1f}%  avg R {ylist[-1][3]:+.3f}  return {pct(ylist[-1][4])}")

    rout("\n[4] COST STRESS")
    for mult in (0, 1, 2, 3):
        c2 = replace(costs, spread_bps=costs.spread_bps * mult, slippage_bps=costs.slippage_bps * mult,
                     swap_long_bps_day=costs.swap_long_bps_day * mult, swap_short_bps_day=costs.swap_short_bps_day * mult)
        raw_tr = tr[["entry_i", "exit_i", "side", "entry", "exit", "sl", "tp", "why"]]
        s2 = stats(evaluate(raw_tr, df.index, c2, risk, maxlev), first, last)
        rout(f"    {mult}x costs: avg R {s2['exp_R']:+.3f}   return {pct(s2['total'])}   maxDD {pct(s2['maxdd'])}   PF {s2['pf']:.2f}")

    return S


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    t0 = time.time()
    rout("Loading 3-year Gold M15 dataset (XAUUSD_M15_History.csv)...")
    df_m15 = load_csv("XAUUSD_M15_History.csv")
    rout(f"Loaded {len(df_m15):,} bars from {df_m15.index[0]} to {df_m15.index[-1]} ({time.time()-t0:.1f}s)")

    # Resample to H1 for HTF filter
    rout("Building H1 market structure from 3-year history...")
    df_h1 = df_m15.resample("1h").agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    h1_trend = get_h1_trend_series(df_m15, df_h1)
    rout(f"Built {len(df_h1):,} H1 bars with zero lookahead.")

    costs = Costs(spread_bps=1.5, slippage_bps=0.5, swap_long_bps_day=-0.8, swap_short_bps_day=-0.3)
    risk = 0.01  # 1% per trade
    maxlev = 10.0
    cut = int(len(df_m15) * 0.7)

    # 1. Strategy 1: Original SMC Rule (Deep 100% Retracement)
    rout("\n[Executing Strategy 1: Original SMC 100% Retracement...]")
    t1 = time.time()
    raw_s1 = find_smc_trades(df_m15, left_bars=3, right_bars=3, sl_buffer_usd=0.50, rr_ratio=3.0)
    tr_s1 = evaluate(raw_s1, df_m15.index, costs, risk, maxlev)
    rout(f"Done in {time.time()-t1:.1f}s ({len(tr_s1)} trades)")
    s1 = print_strategy_report("1. Original SMC Rule (100% Retrace to Sweep Origin, 1:3 RR)", tr_s1, df_m15, costs, risk, maxlev, cut)

    # 2. Strategy 2: Liquidity Sweep Reversal
    rout("\n[Executing Strategy 2: Liquidity Sweep Reversal...]")
    t2 = time.time()
    raw_s2 = find_sweep_reversal_trades(df_m15, left_bars=3, right_bars=3, sl_buffer_usd=0.50, rr_ratio=2.0)
    tr_s2 = evaluate(raw_s2, df_m15.index, costs, risk, maxlev)
    rout(f"Done in {time.time()-t2:.1f}s ({len(tr_s2)} trades)")
    s2 = print_strategy_report("2. Variation 1: Liquidity Sweep Reversal (Turtle Soup, 1:2 RR)", tr_s2, df_m15, costs, risk, maxlev, cut)

    # 3. Strategy 3: Active Leg Pullback (Baseline, all hours)
    rout("\n[Executing Strategy 3: Active Leg Pullback (No Filters)...]")
    t3 = time.time()
    raw_s3 = find_active_leg_trades(df_m15, left_bars=3, right_bars=3, sl_buffer_usd=0.50, rr_ratio=2.0)
    tr_s3 = evaluate(raw_s3, df_m15.index, costs, risk, maxlev)
    rout(f"Done in {time.time()-t3:.1f}s ({len(tr_s3)} trades)")
    s3 = print_strategy_report("3. Variation 2: Active Leg Pullback (Baseline, All Hours, 1:2 RR)", tr_s3, df_m15, costs, risk, maxlev, cut)

    # 4. Strategy 4: Active Leg Pullback + Session Filter (08:00 - 16:00)
    rout("\n[Executing Strategy 4: Active Leg Pullback + Session Filter...]")
    t4 = time.time()
    raw_s4 = find_active_leg_trades_with_filters(df_m15, session_start_hour=8, session_end_hour=16, rr_ratio=2.0)
    tr_s4 = evaluate(raw_s4, df_m15.index, costs, risk, maxlev)
    rout(f"Done in {time.time()-t4:.1f}s ({len(tr_s4)} trades)")
    s4 = print_strategy_report("4. Variation 2 + Peak Session Filter (08:00 - 16:00 London/NY)", tr_s4, df_m15, costs, risk, maxlev, cut)

    # 5. Strategy 5: Active Leg Pullback + H1 Trend + Session Filter
    rout("\n[Executing Strategy 5: Active Leg Pullback + H1 Trend + Session Filter...]")
    t5 = time.time()
    raw_s5 = find_active_leg_trades_with_filters(df_m15, h1_trend_series=h1_trend, use_htf_filter=True, session_start_hour=8, session_end_hour=16, rr_ratio=2.0)
    tr_s5 = evaluate(raw_s5, df_m15.index, costs, risk, maxlev)
    rout(f"Done in {time.time()-t5:.1f}s ({len(tr_s5)} trades)")
    s5 = print_strategy_report("5. Variation 2 + Combined (H1 Trend + 08:00-16:00 Session)", tr_s5, df_m15, costs, risk, maxlev, cut)

    # Save Trades CSV
    tr_s1.to_csv(os.path.join(OUTDIR, "trades_s1_original.csv"), index=False)
    tr_s2.to_csv(os.path.join(OUTDIR, "trades_s2_sweep_reversal.csv"), index=False)
    tr_s3.to_csv(os.path.join(OUTDIR, "trades_s3_active_leg.csv"), index=False)
    tr_s4.to_csv(os.path.join(OUTDIR, "trades_s4_session_filter.csv"), index=False)
    tr_s5.to_csv(os.path.join(OUTDIR, "trades_s5_combined.csv"), index=False)

    # Plot Multi-Year Equity Comparison
    dates = df_m15.index
    bh = df_m15["close"] / df_m15["open"].iloc[0]

    def get_eq_series(tr):
        if tr.empty:
            return pd.Series(1.0, index=dates)
        s = pd.Series(1 + tr.net.to_numpy(), index=tr.exit_time).groupby(level=0).prod()
        return s.reindex(dates).fillna(1.0).cumprod()

    eq_s1 = get_eq_series(tr_s1)
    eq_s2 = get_eq_series(tr_s2)
    eq_s3 = get_eq_series(tr_s3)
    eq_s4 = get_eq_series(tr_s4)
    eq_s5 = get_eq_series(tr_s5)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}, dpi=150)
    fig.patch.set_facecolor("#1e222d")
    ax1.set_facecolor("#1e222d")
    ax2.set_facecolor("#1e222d")

    ax1.plot(dates, bh.values, color=GRAY, lw=1.2, label=f"Buy & Hold Gold ({bh.iloc[-1]*100-100:+.1f}%)", alpha=0.6)
    if not tr_s1.empty:
        ax1.plot(dates, eq_s1.values, color=RED, lw=1.5, label=f"1. Original SMC Rule: {s1['total']*100:+.1f}% ({s1['n']} tr, PF {s1['pf']:.2f})")
    if not tr_s2.empty:
        ax1.plot(dates, eq_s2.values, color=ORANGE, lw=1.5, label=f"2. Sweep Reversal: {s2['total']*100:+.1f}% ({s2['n']} tr, PF {s2['pf']:.2f})")
    if not tr_s3.empty:
        ax1.plot(dates, eq_s3.values, color=BLUE, lw=1.5, label=f"3. Active Leg (Baseline): {s3['total']*100:+.1f}% ({s3['n']} tr, PF {s3['pf']:.2f})")
    if not tr_s4.empty:
        ax1.plot(dates, eq_s4.values, color=GREEN, lw=2.0, label=f"4. Active Leg + Session: {s4['total']*100:+.1f}% ({s4['n']} tr, PF {s4['pf']:.2f})")
    if not tr_s5.empty:
        ax1.plot(dates, eq_s5.values, color=PURPLE, lw=2.0, label=f"5. Combined (H1+Session): {s5['total']*100:+.1f}% ({s5['n']} tr, PF {s5['pf']:.2f})")

    ax1.set_yscale("log")
    ax1.set_title("3-Year Gold Backtest (2023 - 2026, 89,481 M15 Bars) - Net of Costs", fontsize=14, color="#ffffff", fontweight="bold", pad=12)
    ax1.set_ylabel("Growth of $1.00 (Log)", color="#d1d4dc", fontsize=11)
    ax1.tick_params(colors="#d1d4dc", labelsize=9)
    ax1.grid(True, linestyle=":", color="#363c4e", alpha=0.6)
    ax1.legend(loc="upper left", facecolor="#2a2e39", edgecolor="#363c4e", labelcolor="#ffffff", fontsize=9)
    for spine in ax1.spines.values():
        spine.set_color("#363c4e")

    # Drawdown pane for top strategy (s4)
    dd_s4 = eq_s4 / np.maximum.accumulate(eq_s4) - 1
    ax2.fill_between(dates, dd_s4.values, 0, color=GREEN, alpha=0.35, label="Strategy 4 Drawdown")
    ax2.set_title("Drawdown: Strategy 4 (Active Leg + Session Filter)", fontsize=10, color="#ffffff")
    ax2.set_ylabel("Drawdown", color="#d1d4dc", fontsize=9)
    ax2.tick_params(colors="#d1d4dc", labelsize=9)
    ax2.grid(True, linestyle=":", color="#363c4e", alpha=0.6)
    for spine in ax2.spines.values():
        spine.set_color("#363c4e")

    fig.tight_layout()
    chart_path = os.path.join(OUTDIR, "equity_3year.png")
    fig.savefig(chart_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    rout(f"\nSaved comparison chart to {chart_path}")

    report_path = os.path.join(OUTDIR, "report_3year.txt")
    with open(report_path, "w") as fh:
        fh.write("\n".join(REPORT_LINES))
    rout(f"Saved full report to {report_path}")


if __name__ == "__main__":
    main()
