#!/usr/bin/env python3
"""
explore_filters.py - Explores Higher Timeframe (H1) and Session (Killzone) filters
on Variation 2 (Active Leg Trend Pullback) on Gold M15.
"""
from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fibo_backtest import load_csv, Costs
from market_structure import find_swings, classify_structure
from smc_backtest import scan_exit, evaluate, stats, line, pct, style


def get_h1_trend_series(df_m15: pd.DataFrame, df_h1: pd.DataFrame) -> pd.Series:
    """
    Computes H1 market structure trend and maps it to M15 timestamps with zero lookahead.
    An H1 bar ending at 12:00 is only known to M15 at 12:00 or after.
    """
    h1_swings = classify_structure(find_swings(df_h1, left_bars=3, right_bars=3, enforce_alternation=True))
    conf_map = {}
    for s in h1_swings:
        conf_map.setdefault(s.confirmed_index, []).append(s)

    h1_trends = []
    curr_trend = "NEUTRAL"
    last_h, last_l = None, None

    for i in range(len(df_h1)):
        if i in conf_map:
            for s in conf_map[i]:
                if s.swing_type == "HIGH":
                    last_h = s
                    if s.label == "HH":
                        curr_trend = "BULLISH"
                    elif s.label == "LH" and last_l and last_l.label == "LL":
                        curr_trend = "BEARISH"
                else:
                    last_l = s
                    if s.label == "LL":
                        curr_trend = "BEARISH"
                    elif s.label == "HL" and last_h and last_h.label == "HH":
                        curr_trend = "BULLISH"
        h1_trends.append(curr_trend)

    h1_df = pd.DataFrame({"h1_trend": h1_trends}, index=df_h1.index)
    # Forward-fill onto M15 index with merge_asof (backward direction ensures no lookahead)
    merged = pd.merge_asof(
        df_m15,
        h1_df,
        left_index=True,
        right_index=True,
        direction="backward"
    )

    return merged["h1_trend"].fillna("NEUTRAL")


def find_active_leg_trades_with_filters(
    df: pd.DataFrame,
    h1_trend_series: Optional[pd.Series] = None,
    use_htf_filter: bool = False,
    session_start_hour: Optional[int] = None,
    session_end_hour: Optional[int] = None,
    left_bars: int = 3,
    right_bars: int = 3,
    sl_buffer_usd: float = 0.50,
    rr_ratio: float = 2.0,
    fib_entry: float = 0.50
) -> pd.DataFrame:
    swings = classify_structure(find_swings(df, left_bars=left_bars, right_bars=right_bars, enforce_alternation=True))
    conf_swings = {}
    for s in swings:
        conf_swings.setdefault(s.confirmed_index, []).append(s)

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    times = df.index
    n = len(df)

    last_h = None
    last_l = None
    trend = "NEUTRAL"

    leg_low = np.nan
    leg_high = np.nan
    bos_active = False
    bos_direction = 0
    last_bos_swing_idx = -1

    rows = []
    busy_until = -1

    for i in range(n - 1):
        t_bar = times[i]

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

        # Check BOS
        if trend == "BULLISH" and last_h and last_l:
            if c[i] > last_h.price and last_h.index != last_bos_swing_idx:
                bos_active = True
                bos_direction = 1
                leg_low = last_l.price
                leg_high = h[i]
                last_bos_swing_idx = last_h.index
        elif trend == "BEARISH" and last_l and last_h:
            if c[i] < last_l.price and last_l.index != last_bos_swing_idx:
                bos_active = True
                bos_direction = -1
                leg_high = last_h.price
                leg_low = l[i]
                last_bos_swing_idx = last_l.index

        if bos_active:
            if bos_direction == 1:
                if h[i] > leg_high:
                    leg_high = h[i]
                fib = leg_low + fib_entry * (leg_high - leg_low)
                if l[i] <= fib and c[i] > leg_low:
                    entry_i = i + 1
                    entry_time = times[entry_i]

                    # HTF Filter
                    allow_htf = True
                    if use_htf_filter and h1_trend_series is not None:
                        allow_htf = (h1_trend_series.iloc[entry_i] == "BULLISH")

                    # Session Filter
                    allow_session = True
                    if session_start_hour is not None and session_end_hour is not None:
                        allow_session = (session_start_hour <= entry_time.hour < session_end_hour)

                    if allow_htf and allow_session:
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
                    bos_active = False
                elif l[i] <= leg_low:
                    bos_active = False

            else:
                if l[i] < leg_low:
                    leg_low = l[i]
                fib = leg_high - fib_entry * (leg_high - leg_low)
                if h[i] >= fib and c[i] < leg_high:
                    entry_i = i + 1
                    entry_time = times[entry_i]

                    # HTF Filter
                    allow_htf = True
                    if use_htf_filter and h1_trend_series is not None:
                        allow_htf = (h1_trend_series.iloc[entry_i] == "BEARISH")

                    # Session Filter
                    allow_session = True
                    if session_start_hour is not None and session_end_hour is not None:
                        allow_session = (session_start_hour <= entry_time.hour < session_end_hour)

                    if allow_htf and allow_session:
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


def main():
    df_m15 = load_csv("XAUUSDM15.csv")
    df_h1 = load_csv("XAUUSDH1.csv")
    costs = Costs(1.5, 0.5, -0.8, -0.3)
    risk = 0.01
    maxlev = 10.0

    print("Mapping H1 market structure to M15 bars...")
    h1_trend = get_h1_trend_series(df_m15, df_h1)

    print("\n" + "=" * 90)
    print("      SMC VARIATION 2 (ACTIVE LEG PULLBACK) - FILTER EXPLORATION MATRIX")
    print("=" * 90)

    configs = [
        ("1. Baseline (No Filters)", False, None, None, 2.0),
        ("2. Session Filter Only (07:00-18:00)", False, 7, 18, 2.0),
        ("3. Session Filter (08:00-16:00 London/NY Peak)", False, 8, 16, 2.0),
        ("4. H1 Market Structure Filter Only", True, None, None, 2.0),
        ("5. Combined: H1 Trend + Session (07:00-18:00)", True, 7, 18, 2.0),
        ("6. Combined: H1 Trend + Peak Session (08:00-16:00)", True, 8, 16, 2.0),
        ("7. Combined + RR 1.5:1", True, 7, 18, 1.5),
        ("8. Combined + RR 2.5:1", True, 7, 18, 2.5),
    ]

    results = []

    for name, use_htf, s_start, s_end, rr in configs:
        raw = find_active_leg_trades_with_filters(
            df_m15,
            h1_trend_series=h1_trend,
            use_htf_filter=use_htf,
            session_start_hour=s_start,
            session_end_hour=s_end,
            rr_ratio=rr
        )
        ev = evaluate(raw, df_m15.index, costs, risk, maxlev)
        s = stats(ev, df_m15.index[0], df_m15.index[-1])
        results.append((name, s, ev))

        print(f"\nConfiguration: {name}")
        print(f"  Trades: {s['n']:2d} | Win: {s['win']*100:5.1f}% (Needs: {s['be_win']*100:5.1f}%) | "
              f"Avg R: {s['exp_R']:+.3f} | PF: {s['pf']:.2f} | Return: {pct(s['total'])} | "
              f"MaxDD: {pct(s['maxdd'])} | Sharpe: {s['sharpe']:+.2f}")

        # Long vs Short
        lo, sh = ev[ev.side == 1], ev[ev.side == -1]
        lo_s = stats(lo, df_m15.index[0], df_m15.index[-1]) if len(lo) else None
        sh_s = stats(sh, df_m15.index[0], df_m15.index[-1]) if len(sh) else None
        if lo_s and sh_s:
            print(f"    Longs:  {lo_s['n']:2d} trades, win {lo_s['win']*100:4.1f}%, avg R {lo_s['exp_R']:+.3f}, ret {pct(lo_s['total'])}")
            print(f"    Shorts: {sh_s['n']:2d} trades, win {sh_s['win']*100:4.1f}%, avg R {sh_s['exp_R']:+.3f}, ret {pct(sh_s['total'])}")

    # Plot Comparison of Top Configurations
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    fig.patch.set_facecolor("#1e222d")
    ax.set_facecolor("#1e222d")

    dates = df_m15.index
    bh = df_m15["close"] / df_m15["open"].iloc[0]
    ax.plot(dates, bh.values, color="#9aa0a6", lw=1.2, label="Buy & Hold Gold (-4.3%)", alpha=0.6)

    colors = ["#e74c3c", "#f39c12", "#3498db", "#9b59b6", "#2ecc71", "#1abc9c", "#00e676", "#e91e63"]
    for i, (name, s, ev) in enumerate(results[:6]):
        if ev.empty:
            continue
        series = pd.Series(1 + ev.net.to_numpy(), index=ev.exit_time).groupby(level=0).prod()
        eq = series.reindex(dates).fillna(1.0).cumprod()
        ax.plot(dates, eq.values, color=colors[i], lw=1.6, label=f"{name[:32]}: {s['total']*100:+.1f}% (PF {s['pf']:.2f})")

    ax.set_yscale("log")
    ax.set_title("Filter Exploration: Active Leg Pullbacks on Gold M15 (Net of Costs)", fontsize=13, color="#ffffff", fontweight="bold", pad=12)
    ax.set_ylabel("Growth of $1.00 (Log)", color="#d1d4dc", fontsize=10)
    ax.tick_params(colors="#d1d4dc", labelsize=9)
    ax.grid(True, linestyle=":", color="#363c4e", alpha=0.6)
    ax.legend(loc="lower left", facecolor="#2a2e39", edgecolor="#363c4e", labelcolor="#ffffff", fontsize=8.5)
    for spine in ax.spines.values():
        spine.set_color("#363c4e")

    fig.tight_layout()
    fig.savefig("filter_exploration_chart.png", facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    print("\nSaved filter exploration chart to filter_exploration_chart.png")


if __name__ == "__main__":
    main()
