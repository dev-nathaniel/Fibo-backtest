#!/usr/bin/env python3
"""
market_structure.py - Identifies Swing Highs, Swing Lows, and classifies Market Structure:
  - Higher Highs (HH)
  - Higher Lows (HL)
  - Lower Highs (LH)
  - Lower Lows (LL)
  - Break of Structure (BOS) & Change of Character (CHoCH)

Designed for zero look-ahead bias in backtesting:
A swing at candle t is only confirmed and visible to the strategy at candle t + right_bars.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from fibo_backtest import load_csv


@dataclass
class SwingPoint:
    index: int             # Bar index where extreme occurred
    timestamp: pd.Timestamp
    price: float           # Price level (high or low)
    swing_type: str        # 'HIGH' or 'LOW'
    label: str = ""        # 'HH', 'LH', 'HL', 'LL'
    confirmed_index: int = 0  # Bar index where swing was confirmed (index + right_bars)
    confirmed_ts: Optional[pd.Timestamp] = None


@dataclass
class StructureBreak:
    bar_index: int
    timestamp: pd.Timestamp
    break_type: str        # 'BOS_BULL', 'BOS_BEAR', 'CHOCH_BULL', 'CHOCH_BEAR'
    level_broken: float
    swing_ref: SwingPoint


def find_swings(
    df: pd.DataFrame,
    left_bars: int = 3,
    right_bars: int = 3,
    enforce_alternation: bool = True
) -> List[SwingPoint]:
    """
    Identifies pivot highs and lows with strictly defined left and right windows.
    A swing high at bar i requires High[i] > High[i-L .. i-1] and High[i] >= High[i+1 .. i+R].
    A swing low at bar i requires Low[i] < Low[i-L .. i-1] and Low[i] <= Low[i+1 .. i+R].

    Confirmed at bar i + right_bars.
    If enforce_alternation is True, consecutive swings of the same type are resolved
    to the most extreme one (highest of consecutive highs, lowest of consecutive lows).
    """
    highs = df["high"].values
    lows = df["low"].values
    times = df.index
    n = len(df)

    raw_swings: List[SwingPoint] = []

    for i in range(left_bars, n - right_bars):
        curr_h = highs[i]
        left_h = highs[i - left_bars:i]
        right_h = highs[i + 1:i + right_bars + 1]
        is_pivot_high = (curr_h > np.max(left_h)) and (curr_h >= np.max(right_h))

        curr_l = lows[i]
        left_l = lows[i - left_bars:i]
        right_l = lows[i + 1:i + right_bars + 1]
        is_pivot_low = (curr_l < np.min(left_l)) and (curr_l <= np.min(right_l))

        conf_idx = i + right_bars
        conf_ts = times[conf_idx]

        if is_pivot_high:
            raw_swings.append(SwingPoint(
                index=i,
                timestamp=times[i],
                price=float(curr_h),
                swing_type="HIGH",
                confirmed_index=conf_idx,
                confirmed_ts=conf_ts
            ))
        if is_pivot_low:
            raw_swings.append(SwingPoint(
                index=i,
                timestamp=times[i],
                price=float(curr_l),
                swing_type="LOW",
                confirmed_index=conf_idx,
                confirmed_ts=conf_ts
            ))

    raw_swings.sort(key=lambda s: s.index)

    if not enforce_alternation:
        return raw_swings

    # Filter to alternate HIGH -> LOW -> HIGH -> LOW
    alternated: List[SwingPoint] = []
    for sp in raw_swings:
        if not alternated:
            alternated.append(sp)
            continue

        prev = alternated[-1]
        if sp.swing_type == prev.swing_type:
            if sp.swing_type == "HIGH":
                if sp.price > prev.price:
                    alternated[-1] = sp
            else:
                if sp.price < prev.price:
                    alternated[-1] = sp
        else:
            alternated.append(sp)

    return alternated


def classify_structure(swings: List[SwingPoint]) -> List[SwingPoint]:
    """
    Labels each swing point:
      Highs:
        - Higher High (HH): price > previous swing high
        - Lower High (LH):  price <= previous swing high
      Lows:
        - Higher Low (HL):  price >= previous swing low
        - Lower Low (LL):   price < previous swing low
    """
    last_high: Optional[SwingPoint] = None
    last_low: Optional[SwingPoint] = None

    for sp in swings:
        if sp.swing_type == "HIGH":
            if last_high is None:
                sp.label = "H"
            elif sp.price > last_high.price:
                sp.label = "HH"
            else:
                sp.label = "LH"
            last_high = sp
        elif sp.swing_type == "LOW":
            if last_low is None:
                sp.label = "L"
            elif sp.price < last_low.price:
                sp.label = "LL"
            else:
                sp.label = "HL"
            last_low = sp

    return swings


def detect_structure_breaks(
    df: pd.DataFrame,
    swings: List[SwingPoint],
    break_on_close: bool = True
) -> Tuple[pd.DataFrame, List[StructureBreak]]:
    """
    Simulates market structure step-by-step with zero look-ahead bias.
    Tracks:
      - Current Trend: BULLISH, BEARISH, NEUTRAL
      - Active Swing High & Active Swing Low
      - Break of Structure (BOS): Trend continuation break
      - Change of Character (CHoCH): Trend reversal break
    """
    out = df.copy()
    n = len(df)

    out["trend"] = "NEUTRAL"
    out["active_high"] = np.nan
    out["active_low"] = np.nan
    out["active_high_label"] = ""
    out["active_low_label"] = ""
    out["structure_event"] = ""

    # Map swings by their confirmation index
    conf_swings = {}
    for s in swings:
        conf_swings.setdefault(s.confirmed_index, []).append(s)

    breaks: List[StructureBreak] = []

    last_high_swing: Optional[SwingPoint] = None
    last_low_swing: Optional[SwingPoint] = None
    current_trend = "NEUTRAL"

    trend_col = []
    ah_col = []
    al_col = []
    ahl_col = []
    all_col = []
    ev_col = []

    high_vals = df["high"].values
    low_vals = df["low"].values
    close_vals = df["close"].values
    times = df.index

    for i in range(n):
        # 1. Update confirmed swings at bar i
        if i in conf_swings:
            for s in conf_swings[i]:
                if s.swing_type == "HIGH":
                    last_high_swing = s
                else:
                    last_low_swing = s

        # 2. Check for structural breaks
        event = ""
        c = close_vals[i] if break_on_close else high_vals[i]
        c_low = close_vals[i] if break_on_close else low_vals[i]

        if last_high_swing is not None and c > last_high_swing.price:
            if current_trend == "BULLISH":
                event = "BOS_BULL"
                breaks.append(StructureBreak(i, times[i], "BOS_BULL", last_high_swing.price, last_high_swing))
            elif current_trend in ("BEARISH", "NEUTRAL"):
                event = "CHOCH_BULL"
                current_trend = "BULLISH"
                breaks.append(StructureBreak(i, times[i], "CHOCH_BULL", last_high_swing.price, last_high_swing))

        elif last_low_swing is not None and c_low < last_low_swing.price:
            if current_trend == "BEARISH":
                event = "BOS_BEAR"
                breaks.append(StructureBreak(i, times[i], "BOS_BEAR", last_low_swing.price, last_low_swing))
            elif current_trend in ("BULLISH", "NEUTRAL"):
                event = "CHOCH_BEAR"
                current_trend = "BEARISH"
                breaks.append(StructureBreak(i, times[i], "CHOCH_BEAR", last_low_swing.price, last_low_swing))

        trend_col.append(current_trend)
        ah_col.append(last_high_swing.price if last_high_swing else np.nan)
        al_col.append(last_low_swing.price if last_low_swing else np.nan)
        ahl_col.append(last_high_swing.label if last_high_swing else "")
        all_col.append(last_low_swing.label if last_low_swing else "")
        ev_col.append(event)

    out["trend"] = trend_col
    out["active_high"] = ah_col
    out["active_low"] = al_col
    out["active_high_label"] = ahl_col
    out["active_low_label"] = all_col
    out["structure_event"] = ev_col

    return out, breaks


def plot_market_structure(
    df: pd.DataFrame,
    swings: List[SwingPoint],
    breaks: Optional[List[StructureBreak]] = None,
    output_png: str = "market_structure_chart.png",
    num_bars: int = 150
):
    """
    Renders candlesticks, zigzags, and HH/HL/LH/LL labels.
    """
    sub_df = df.iloc[-num_bars:].copy()
    start_idx = len(df) - num_bars
    sub_swings = [s for s in swings if s.index >= start_idx]

    fig, ax = plt.subplots(figsize=(18, 9), dpi=150)
    fig.patch.set_facecolor("#1e222d")
    ax.set_facecolor("#1e222d")

    # Draw Candlesticks
    dates = mdates.date2num(sub_df.index)
    width = 0.02 * (dates[-1] - dates[0]) / num_bars

    up = sub_df[sub_df["close"] >= sub_df["open"]]
    down = sub_df[sub_df["close"] < sub_df["open"]]

    up_dates = mdates.date2num(up.index)
    down_dates = mdates.date2num(down.index)

    # Bullish candles (Teal/Green)
    ax.vlines(up_dates, up["low"], up["high"], color="#26a69a", linewidth=1.0, alpha=0.8)
    ax.vlines(down_dates, down["low"], down["high"], color="#ef5350", linewidth=1.0, alpha=0.8)

    candle_width = (dates[1] - dates[0]) * 0.7 if len(dates) > 1 else 0.02
    ax.bar(up_dates, up["close"] - up["open"], candle_width, bottom=up["open"], color="#26a69a", edgecolor="#26a69a")
    ax.bar(down_dates, down["open"] - down["close"], candle_width, bottom=down["close"], color="#ef5350", edgecolor="#ef5350")

    # Connect Swings with Zigzag Line
    if sub_swings:
        swing_dates = mdates.date2num([s.timestamp for s in sub_swings])
        swing_prices = [s.price for s in sub_swings]
        ax.plot(swing_dates, swing_prices, color="#f1c40f", linestyle="--", linewidth=1.8, alpha=0.85, label="Market Structure Zigzag")

    # Styling colors for swing labels
    colors = {
        "HH": "#00e676",  # Bright Neon Green
        "LH": "#ff9100",  # Bright Amber Orange
        "HL": "#00b0ff",  # Bright Sky Blue
        "LL": "#ff1744",  # Bright Crimson Red
        "H":  "#b0bec5",
        "L":  "#b0bec5"
    }

    price_range = sub_df["high"].max() - sub_df["low"].min()
    offset = price_range * 0.025

    for s in sub_swings:
        c = colors.get(s.label, "#ffffff")
        is_high = s.swing_type == "HIGH"
        y_pos = s.price + (offset if is_high else -offset)
        va = "bottom" if is_high else "top"
        s_date = mdates.date2num(s.timestamp)

        ax.scatter(s_date, s.price, color=c, s=45, zorder=6, edgecolors="#1e222d", linewidth=1.2)
        ax.annotate(
            f"{s.label}\n{s.price:.2f}",
            (s_date, y_pos),
            color=c,
            fontsize=8.5,
            fontweight="bold",
            ha="center",
            va=va,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#2a2e39", edgecolor=c, alpha=0.9, linewidth=1.2),
            zorder=7
        )

    # Plot Structure Breaks (BOS / CHoCH) if within range
    if breaks:
        sub_breaks = [b for b in breaks if b.bar_index >= start_idx]
        for b in sub_breaks:
            b_date = mdates.date2num(b.timestamp)
            is_bull = "BULL" in b.break_type
            tag = "BOS" if "BOS" in b.break_type else "CHoCH"
            b_color = "#00e676" if is_bull else "#ff1744"
            ax.scatter(b_date, b.level_broken, color=b_color, marker="^" if is_bull else "v", s=70, zorder=8)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()

    ax.grid(True, linestyle=":", color="#363c4e", alpha=0.6)
    ax.tick_params(colors="#d1d4dc", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#363c4e")

    ax.set_title(f"Market Structure (HH / HL / LH / LL) — XAUUSD (Last {num_bars} Bars)", fontsize=14, color="#ffffff", fontweight="bold", pad=15)
    ax.set_ylabel("Price (USD)", color="#d1d4dc", fontsize=11)
    ax.legend(loc="upper left", facecolor="#2a2e39", edgecolor="#363c4e", labelcolor="#ffffff")

    plt.tight_layout()
    plt.savefig(output_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Chart successfully saved to {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Market Structure (HH, HL, LH, LL) Detector & Backtest Foundation")
    parser.add_argument("--csv", default="XAUUSDH1.csv", help="Path to OHLC CSV file")
    parser.add_argument("--left", type=int, default=3, help="Left window bars for pivot")
    parser.add_argument("--right", type=int, default=3, help="Right window bars for pivot")
    parser.add_argument("--bars", type=int, default=150, help="Number of bars to display in plot")
    parser.add_argument("--out", default="market_structure_candlestick.png", help="Output plot filename")
    args = parser.parse_args()

    print(f"Loading {args.csv}...")
    df = load_csv(args.csv)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}.")

    print(f"Detecting swings (L={args.left}, R={args.right})...")
    swings = find_swings(df, left_bars=args.left, right_bars=args.right, enforce_alternation=True)
    swings = classify_structure(swings)

    df_struct, breaks = detect_structure_breaks(df, swings)

    print(f"\nTotal swings identified: {len(swings)}")
    print(f"Total structure breaks: {len(breaks)}")

    print("\n--- Last 10 Market Structure Swings ---")
    for s in swings[-10:]:
        print(f"Bar {s.index:4d} | {s.timestamp} | {s.swing_type:4s} | Price: {s.price:8.2f} | Label: {s.label:3s} | Confirmed Bar: {s.confirmed_index:4d} ({s.confirmed_ts})")

    plot_market_structure(df, swings, breaks, output_png=args.out, num_bars=args.bars)


if __name__ == "__main__":
    main()
