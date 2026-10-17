#!/usr/bin/env python3
"""
smc_strategy.py - Smart Money Concepts (SMC) Continuation Strategy Backtest.

Strategy Rules:
1. Trend Identification:
   - Bullish trend: Series of Higher Highs (HH) and Higher Lows (HL).
   - Bearish trend: Series of Lower Highs (LH) and Lower Lows (LL).

2. Bullish Continuation Setup:
   a. Liquidity Sweep: Price takes out (breaks below) a confirmed Higher Low (HL).
      Demand Zone is marked at the extreme (last bearish candle before the sweep reversal).
   b. BOS 1: Price rallies from the sweep low and takes out the previous Higher High (HH1),
      forming a new Higher High (HH2).
   c. Retracement (>= 50% Fibo): From HH2, price pulls back for another Higher Low (HL2).
      This pullback MUST reach >= 50% of the Fibonacci range measured from Sweep Low to HH2.
   d. BOS 2: From HL2, price rallies and takes out HH2 (breaking the previous higher high again).
   e. Trade Armed & Entry: The trade is armed. When/if price pulls back to the original Demand Zone,
      a BUY trade is entered.
      - Entry: Top of Demand Zone (candle high)
      - Stop Loss (SL): Below Sweep Low minus buffer
      - Take Profit (TP): 1:3 Risk-to-Reward ratio (Entry + 3 * Risk)

3. Bearish Continuation Setup (Mirror):
   a. Liquidity Sweep: Price takes out (breaks above) a confirmed Lower High (LH).
      Supply Zone is marked at the extreme (last bullish candle before the sweep reversal).
   b. BOS 1: Price drops from the sweep high and takes out the previous Lower Low (LL1),
      forming a new Lower Low (LL2).
   c. Retracement (>= 50% Fibo): From LL2, price pulls back for another Lower High (LH2).
      This pullback MUST reach >= 50% of the Fibonacci range measured from Sweep High to LL2.
   d. BOS 2: From LH2, price drops and takes out LL2 (breaking the previous lower low again).
   e. Trade Armed & Entry: When/if price pulls back to the original Supply Zone,
      a SELL trade is entered.
      - Entry: Bottom of Supply Zone (candle low)
      - Stop Loss (SL): Above Sweep High plus buffer
      - Take Profit (TP): 1:3 Risk-to-Reward ratio (Entry - 3 * Risk)
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from fibo_backtest import load_csv
from market_structure import find_swings, classify_structure, SwingPoint


class SetupStage(Enum):
    IDLE = 0
    SWEPT = 1                # HL swept (Bull) or LH swept (Bear)
    BOS_1 = 2                # First HH taken out (Bull) or LL taken out (Bear)
    FIB_50_REACHED = 3       # Pullback >= 50% Fib reached
    BOS_2_ARMED = 4          # Second HH/LL taken out, setup is ARMED waiting for zone mitigation
    IN_TRADE = 5             # Entry filled, managing position
    COMPLETED = 6            # Trade hit TP or SL
    INVALIDATED = 7          # Setup violated before entry


@dataclass
class Candle:
    index: int
    timestamp: pd.Timestamp
    open: float
    high: float
    low: float
    close: float

    @property
    def is_bullish(self) -> bool:
        return self.close >= self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open


@dataclass
class Zone:
    candle_idx: int
    timestamp: pd.Timestamp
    top: float
    bottom: float
    zone_type: str  # 'DEMAND' or 'SUPPLY'


@dataclass
class Trade:
    id: int
    direction: str          # 'BUY' or 'SELL'
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    sl_price: float
    tp_price: float
    risk: float
    exit_idx: Optional[int] = None
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    pnl_r: Optional[float] = None  # in R multiples (+3.0 or -1.0)
    result: str = "OPEN"           # 'WIN', 'LOSS', 'OPEN'


@dataclass
class SMCSetup:
    id: int
    direction: str                     # 'BULLISH' or 'BEARISH'
    stage: SetupStage = SetupStage.IDLE

    # Sweep info
    swept_swing: Optional[SwingPoint] = None
    sweep_extreme: float = 0.0          # Low for Bull, High for Bear
    sweep_bar_idx: int = 0
    zone: Optional[Zone] = None

    # BOS 1 info
    target_swing_1: Optional[SwingPoint] = None  # HH1 (Bull) or LL1 (Bear)
    bos_1_bar_idx: int = 0
    leg_extreme_1: float = 0.0         # HH2 (Bull) or LL2 (Bear)

    # Pullback info
    fib_50_level: float = 0.0
    pullback_extreme: float = 0.0      # HL2 (Bull) or LH2 (Bear)
    pullback_bar_idx: int = 0

    # BOS 2 info
    bos_2_bar_idx: int = 0

    # Associated trade
    trade: Optional[Trade] = None


def find_last_candle_of_type(df: pd.DataFrame, end_idx: int, lookback: int = 10, want_bearish: bool = True) -> Optional[Zone]:
    """
    Finds the last candle of specified type (bearish for Demand, bullish for Supply)
    up to end_idx.
    """
    start_idx = max(0, end_idx - lookback)
    for i in range(end_idx, start_idx - 1, -1):
        row = df.iloc[i]
        c = Candle(i, df.index[i], row["open"], row["high"], row["low"], row["close"])
        if want_bearish and c.is_bearish:
            return Zone(i, c.timestamp, c.high, c.low, "DEMAND")
        elif not want_bearish and c.is_bullish:
            return Zone(i, c.timestamp, c.high, c.low, "SUPPLY")

    # Fallback to candle at end_idx if none matched
    row = df.iloc[end_idx]
    z_type = "DEMAND" if want_bearish else "SUPPLY"
    return Zone(end_idx, df.index[end_idx], row["high"], row["low"], z_type)


def run_smc_backtest(
    df: pd.DataFrame,
    left_bars: int = 3,
    right_bars: int = 3,
    sl_buffer_usd: float = 0.50,
    rr_ratio: float = 3.0,
    break_on_close: bool = True
) -> Tuple[List[Trade], List[SMCSetup]]:
    """
    Simulates the SMC continuation strategy bar-by-bar with zero look-ahead bias.
    """
    swings = find_swings(df, left_bars=left_bars, right_bars=right_bars, enforce_alternation=True)
    swings = classify_structure(swings)

    # Map swings by confirmed bar index
    conf_swings = {}
    for s in swings:
        conf_swings.setdefault(s.confirmed_index, []).append(s)

    trades: List[Trade] = []
    setups: List[SMCSetup] = []

    last_high: Optional[SwingPoint] = None
    last_low: Optional[SwingPoint] = None
    trend = "NEUTRAL"

    active_bull_setup: Optional[SMCSetup] = None
    active_bear_setup: Optional[SMCSetup] = None
    open_trade: Optional[Trade] = None
    setup_counter = 0

    highs = df["high"].values
    lows = df["low"].values
    opens = df["open"].values
    closes = df["close"].values
    times = df.index
    n = len(df)

    for i in range(n):
        o_bar = opens[i]
        h_bar = highs[i]
        l_bar = lows[i]
        c_bar = closes[i]
        t_bar = times[i]

        check_h = c_bar if break_on_close else h_bar
        check_l = c_bar if break_on_close else l_bar

        # -------------------------------------------------------------
        # 1. Update Market Structure from confirmed swings at bar i
        # -------------------------------------------------------------
        if i in conf_swings:
            for s in conf_swings[i]:
                if s.swing_type == "HIGH":
                    last_high = s
                    if s.label == "HH":
                        trend = "BULLISH"
                    elif s.label == "LH" and last_low and last_low.label == "LL":
                        trend = "BEARISH"
                else:
                    last_low = s
                    if s.label == "LL":
                        trend = "BEARISH"
                    elif s.label == "HL" and last_high and last_high.label == "HH":
                        trend = "BULLISH"

        # -------------------------------------------------------------
        # 2. Manage Open Trade (if any)
        # -------------------------------------------------------------
        if open_trade is not None:
            if open_trade.direction == "BUY":
                # Check SL first
                if l_bar <= open_trade.sl_price:
                    open_trade.exit_idx = i
                    open_trade.exit_time = t_bar
                    open_trade.exit_price = open_trade.sl_price
                    open_trade.pnl_r = -1.0
                    open_trade.result = "LOSS"
                    trades.append(open_trade)
                    open_trade = None
                elif h_bar >= open_trade.tp_price:
                    open_trade.exit_idx = i
                    open_trade.exit_time = t_bar
                    open_trade.exit_price = open_trade.tp_price
                    open_trade.pnl_r = rr_ratio
                    open_trade.result = "WIN"
                    trades.append(open_trade)
                    open_trade = None
            elif open_trade.direction == "SELL":
                # Check SL first
                if h_bar >= open_trade.sl_price:
                    open_trade.exit_idx = i
                    open_trade.exit_time = t_bar
                    open_trade.exit_price = open_trade.sl_price
                    open_trade.pnl_r = -1.0
                    open_trade.result = "LOSS"
                    trades.append(open_trade)
                    open_trade = None
                elif l_bar <= open_trade.tp_price:
                    open_trade.exit_idx = i
                    open_trade.exit_time = t_bar
                    open_trade.exit_price = open_trade.tp_price
                    open_trade.pnl_r = rr_ratio
                    open_trade.result = "WIN"
                    trades.append(open_trade)
                    open_trade = None

        # -------------------------------------------------------------
        # 3. BULLISH Setup State Machine
        # -------------------------------------------------------------
        # Step A: Initiate Sweep if in Bullish trend and HL taken out
        if trend == "BULLISH" and last_low is not None and last_low.label == "HL":
            if active_bull_setup is None or active_bull_setup.stage == SetupStage.IDLE:
                if l_bar < last_low.price:
                    setup_counter += 1
                    # Find last bearish candle up to this bar
                    d_zone = find_last_candle_of_type(df, i, lookback=5, want_bearish=True)
                    active_bull_setup = SMCSetup(
                        id=setup_counter,
                        direction="BULLISH",
                        stage=SetupStage.SWEPT,
                        swept_swing=last_low,
                        sweep_extreme=l_bar,
                        sweep_bar_idx=i,
                        zone=d_zone,
                        target_swing_1=last_high
                    )
                    setups.append(active_bull_setup)

        if active_bull_setup is not None and active_bull_setup.stage != SetupStage.IDLE:
            st = active_bull_setup.stage

            # Track sweep low expansion while still in SWEPT
            if st == SetupStage.SWEPT:
                if l_bar < active_bull_setup.sweep_extreme:
                    active_bull_setup.sweep_extreme = l_bar
                    active_bull_setup.sweep_bar_idx = i
                    # Update demand zone to lowest bearish candle
                    active_bull_setup.zone = find_last_candle_of_type(df, i, lookback=5, want_bearish=True)

                # Check Step B: BOS 1 (Take out previous HH)
                if active_bull_setup.target_swing_1 is not None:
                    if check_h > active_bull_setup.target_swing_1.price:
                        active_bull_setup.stage = SetupStage.BOS_1
                        active_bull_setup.bos_1_bar_idx = i
                        active_bull_setup.leg_extreme_1 = h_bar

            elif st == SetupStage.BOS_1:
                # Update HH2 peak as price moves higher
                if h_bar > active_bull_setup.leg_extreme_1:
                    active_bull_setup.leg_extreme_1 = h_bar

                # Check Step C: Pullback >= 50% Fibo
                sweep_low = active_bull_setup.sweep_extreme
                hh2 = active_bull_setup.leg_extreme_1
                fib_50 = sweep_low + 0.50 * (hh2 - sweep_low)
                active_bull_setup.fib_50_level = fib_50

                if check_l <= fib_50:
                    active_bull_setup.stage = SetupStage.FIB_50_REACHED
                    active_bull_setup.pullback_extreme = l_bar
                    active_bull_setup.pullback_bar_idx = i

            elif st == SetupStage.FIB_50_REACHED:
                # Track lowest point of pullback
                if l_bar < active_bull_setup.pullback_extreme:
                    active_bull_setup.pullback_extreme = l_bar
                    active_bull_setup.pullback_bar_idx = i

                # Invalidation: if pullback breaks below sweep_low, setup fails
                if l_bar < active_bull_setup.sweep_extreme:
                    active_bull_setup.stage = SetupStage.INVALIDATED
                    active_bull_setup = None
                # Check Step D: BOS 2 (Rally takes out HH2)
                elif check_h > active_bull_setup.leg_extreme_1:
                    active_bull_setup.stage = SetupStage.BOS_2_ARMED
                    active_bull_setup.bos_2_bar_idx = i

            elif st == SetupStage.BOS_2_ARMED:
                # Check Step E: Entry at Demand Zone
                if active_bull_setup.zone is not None:
                    z_top = active_bull_setup.zone.top
                    sl = active_bull_setup.sweep_extreme - sl_buffer_usd
                    risk = z_top - sl

                    if risk > 0 and l_bar <= z_top:
                        # Entry triggered!
                        entry_p = min(o_bar, z_top) if o_bar < z_top else z_top
                        tp = entry_p + (rr_ratio * risk)
                        t = Trade(
                            id=len(trades) + 1,
                            direction="BUY",
                            entry_idx=i,
                            entry_time=t_bar,
                            entry_price=entry_p,
                            sl_price=sl,
                            tp_price=tp,
                            risk=risk
                        )
                        active_bull_setup.trade = t
                        active_bull_setup.stage = SetupStage.IN_TRADE
                        if open_trade is None:
                            open_trade = t
                        active_bull_setup = None

                # Invalidation if price breaks below sweep_extreme without filling
                if active_bull_setup is not None and l_bar < active_bull_setup.sweep_extreme:
                    active_bull_setup.stage = SetupStage.INVALIDATED
                    active_bull_setup = None

        # -------------------------------------------------------------
        # 4. BEARISH Setup State Machine
        # -------------------------------------------------------------
        if trend == "BEARISH" and last_high is not None and last_high.label == "LH":
            if active_bear_setup is None or active_bear_setup.stage == SetupStage.IDLE:
                if h_bar > last_high.price:
                    setup_counter += 1
                    s_zone = find_last_candle_of_type(df, i, lookback=5, want_bearish=False)
                    active_bear_setup = SMCSetup(
                        id=setup_counter,
                        direction="BEARISH",
                        stage=SetupStage.SWEPT,
                        swept_swing=last_high,
                        sweep_extreme=h_bar,
                        sweep_bar_idx=i,
                        zone=s_zone,
                        target_swing_1=last_low
                    )
                    setups.append(active_bear_setup)

        if active_bear_setup is not None and active_bear_setup.stage != SetupStage.IDLE:
            st = active_bear_setup.stage

            if st == SetupStage.SWEPT:
                if h_bar > active_bear_setup.sweep_extreme:
                    active_bear_setup.sweep_extreme = h_bar
                    active_bear_setup.sweep_bar_idx = i
                    active_bear_setup.zone = find_last_candle_of_type(df, i, lookback=5, want_bearish=False)

                if active_bear_setup.target_swing_1 is not None:
                    if check_l < active_bear_setup.target_swing_1.price:
                        active_bear_setup.stage = SetupStage.BOS_1
                        active_bear_setup.bos_1_bar_idx = i
                        active_bear_setup.leg_extreme_1 = l_bar

            elif st == SetupStage.BOS_1:
                if l_bar < active_bear_setup.leg_extreme_1:
                    active_bear_setup.leg_extreme_1 = l_bar

                sweep_high = active_bear_setup.sweep_extreme
                ll2 = active_bear_setup.leg_extreme_1
                fib_50 = sweep_high - 0.50 * (sweep_high - ll2)
                active_bear_setup.fib_50_level = fib_50

                if check_h >= fib_50:
                    active_bear_setup.stage = SetupStage.FIB_50_REACHED
                    active_bear_setup.pullback_extreme = h_bar
                    active_bear_setup.pullback_bar_idx = i

            elif st == SetupStage.FIB_50_REACHED:
                if h_bar > active_bear_setup.pullback_extreme:
                    active_bear_setup.pullback_extreme = h_bar
                    active_bear_setup.pullback_bar_idx = i

                if h_bar > active_bear_setup.sweep_extreme:
                    active_bear_setup.stage = SetupStage.INVALIDATED
                    active_bear_setup = None
                elif check_l < active_bear_setup.leg_extreme_1:
                    active_bear_setup.stage = SetupStage.BOS_2_ARMED
                    active_bear_setup.bos_2_bar_idx = i

            elif st == SetupStage.BOS_2_ARMED:
                if active_bear_setup.zone is not None:
                    z_bottom = active_bear_setup.zone.bottom
                    sl = active_bear_setup.sweep_extreme + sl_buffer_usd
                    risk = sl - z_bottom

                    if risk > 0 and h_bar >= z_bottom:
                        entry_p = max(o_bar, z_bottom) if o_bar > z_bottom else z_bottom
                        tp = entry_p - (rr_ratio * risk)
                        t = Trade(
                            id=len(trades) + 1,
                            direction="SELL",
                            entry_idx=i,
                            entry_time=t_bar,
                            entry_price=entry_p,
                            sl_price=sl,
                            tp_price=tp,
                            risk=risk
                        )
                        active_bear_setup.trade = t
                        active_bear_setup.stage = SetupStage.IN_TRADE
                        if open_trade is None:
                            open_trade = t
                        active_bear_setup = None

                if active_bear_setup is not None and h_bar > active_bear_setup.sweep_extreme:
                    active_bear_setup.stage = SetupStage.INVALIDATED
                    active_bear_setup = None

    return trades, setups


def print_backtest_report(trades: List[Trade], setups: List[SMCSetup]):
    print("\n" + "=" * 65)
    print("        SMC CONTINUATION STRATEGY BACKTEST RESULTS")
    print("=" * 65)
    print(f"Total Setups Detected: {len(setups)}")

    stage_counts = {}
    for s in setups:
        stage_counts[s.stage.name] = stage_counts.get(s.stage.name, 0) + 1

    print("\nSetup Stages Breakdown:")
    for k, v in stage_counts.items():
        print(f"  - {k:16s}: {v:3d}")

    total_trades = len(trades)
    print(f"\nTotal Executed Trades: {total_trades}")

    if total_trades == 0:
        print("No trades were triggered with the current parameter set.")
        return

    wins = [t for t in trades if t.result == "WIN"]
    losses = [t for t in trades if t.result == "LOSS"]
    open_tr = [t for t in trades if t.result == "OPEN"]

    win_rate = (len(wins) / (len(wins) + len(losses)) * 100) if (wins or losses) else 0.0
    total_r = sum(t.pnl_r for t in trades if t.pnl_r is not None)

    print(f"Wins: {len(wins)} | Losses: {len(losses)} | Open: {len(open_tr)}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Total Return: {total_r:+.2f} R (with 1:3 RR, each win = +3R, loss = -1R)")

    print("\n--- Trade Log ---")
    print("-" * 75)
    for t in trades:
        res_str = t.result
        pnl_str = f"{t.pnl_r:+.1f}R" if t.pnl_r is not None else "0.0R"
        t_time_str = str(t.entry_time)[:16]
        print(f"{t.id:3d} | {t.direction:4s} | {t_time_str:16s} | {t.entry_price:8.2f} | {t.sl_price:8.2f} | {t.tp_price:8.2f} | {res_str:6s} | {pnl_str:>7s}")


def plot_smc_setup(
    df: pd.DataFrame,
    setup: SMCSetup,
    output_png: str = "smc_trade_setup.png",
    pad_bars: int = 25
):
    """
    Renders an in-depth chart visualizing the complete lifecycle of an SMC setup:
    1. HL sweep & Demand Zone creation
    2. BOS 1 (Take out previous HH)
    3. Retracement >= 50% Fib
    4. BOS 2 (Take out HH2)
    5. Entry at Demand Zone, SL, and TP
    """
    start_idx = max(0, setup.sweep_bar_idx - pad_bars)
    end_idx = min(len(df), (setup.trade.exit_idx if setup.trade and setup.trade.exit_idx else setup.bos_2_bar_idx) + pad_bars)

    sub_df = df.iloc[start_idx:end_idx].copy()
    dates = mdates.date2num(sub_df.index)

    fig, ax = plt.subplots(figsize=(18, 9), dpi=150)
    fig.patch.set_facecolor("#1e222d")
    ax.set_facecolor("#1e222d")

    # Draw Candlesticks
    up = sub_df[sub_df["close"] >= sub_df["open"]]
    down = sub_df[sub_df["close"] < sub_df["open"]]

    up_dates = mdates.date2num(up.index)
    down_dates = mdates.date2num(down.index)

    candle_width = (dates[1] - dates[0]) * 0.7 if len(dates) > 1 else 0.02
    ax.vlines(up_dates, up["low"], up["high"], color="#26a69a", linewidth=1.0, alpha=0.8)
    ax.vlines(down_dates, down["low"], down["high"], color="#ef5350", linewidth=1.0, alpha=0.8)
    ax.bar(up_dates, up["close"] - up["open"], candle_width, bottom=up["open"], color="#26a69a", edgecolor="#26a69a")
    ax.bar(down_dates, down["open"] - down["close"], candle_width, bottom=down["close"], color="#ef5350", edgecolor="#ef5350")

    # 1. Plot Swept HL level
    if setup.swept_swing:
        hl_p = setup.swept_swing.price
        ax.axhline(hl_p, color="#f39c12", linestyle=":", linewidth=1.5, alpha=0.8)
        ax.text(dates[0], hl_p, f" Swept HL: {hl_p:.2f}", color="#f39c12", fontsize=9, fontweight="bold", va="bottom")

    # 2. Plot Demand/Supply Zone
    if setup.zone:
        z = setup.zone
        z_start = mdates.date2num(z.timestamp)
        z_end = dates[-1]
        z_color = "#00e676" if z.zone_type == "DEMAND" else "#ff1744"
        ax.fill_between([z_start, z_end], [z.bottom, z.bottom], [z.top, z.top], color=z_color, alpha=0.25, label=f"{z.zone_type} Zone")
        ax.text(z_start, z.top, f" {z.zone_type} [{z.bottom:.2f} - {z.top:.2f}]", color=z_color, fontsize=9, fontweight="bold", va="bottom")

    # 3. Plot Sweep Extreme Point
    sw_date = mdates.date2num(df.index[setup.sweep_bar_idx])
    ax.scatter(sw_date, setup.sweep_extreme, color="#e74c3c", s=80, marker="v", zorder=7)
    ax.annotate("1. Liquidity Sweep", (sw_date, setup.sweep_extreme), textcoords="offset points", xytext=(0, -25),
                ha="center", color="#e74c3c", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="#2a2e39", edgecolor="#e74c3c", alpha=0.9))

    # 4. Plot BOS 1
    if setup.bos_1_bar_idx > 0:
        b1_date = mdates.date2num(df.index[setup.bos_1_bar_idx])
        ax.scatter(b1_date, setup.leg_extreme_1, color="#2ecc71", s=80, marker="^", zorder=7)
        ax.annotate(f"2. BOS 1 (HH Taken Out)\nPeak: {setup.leg_extreme_1:.2f}", (b1_date, setup.leg_extreme_1),
                    textcoords="offset points", xytext=(0, 15), ha="center", color="#2ecc71", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#2a2e39", edgecolor="#2ecc71", alpha=0.9))

    # 5. Plot 50% Fib level & Pullback
    if setup.fib_50_level > 0:
        ax.axhline(setup.fib_50_level, color="#00b0ff", linestyle="--", linewidth=1.2, alpha=0.7)
        ax.text(dates[0], setup.fib_50_level, f" 50% Fib: {setup.fib_50_level:.2f}", color="#00b0ff", fontsize=9, va="bottom")

    if setup.pullback_bar_idx > 0:
        pb_date = mdates.date2num(df.index[setup.pullback_bar_idx])
        ax.scatter(pb_date, setup.pullback_extreme, color="#00b0ff", s=70, marker="o", zorder=7)
        ax.annotate(f"3. Pullback >= 50% Fib\nLow: {setup.pullback_extreme:.2f}", (pb_date, setup.pullback_extreme),
                    textcoords="offset points", xytext=(0, -28), ha="center", color="#00b0ff", fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#2a2e39", edgecolor="#00b0ff", alpha=0.9))

    # 6. Plot BOS 2
    if setup.bos_2_bar_idx > 0:
        b2_date = mdates.date2num(df.index[setup.bos_2_bar_idx])
        b2_price = df.iloc[setup.bos_2_bar_idx]["high"]
        ax.scatter(b2_date, b2_price, color="#f1c40f", s=80, marker="^", zorder=7)
        ax.annotate("4. BOS 2 (HH2 Taken Out)\nTRADE ARMED", (b2_date, b2_price),
                    textcoords="offset points", xytext=(0, 15), ha="center", color="#f1c40f", fontsize=8.5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#2a2e39", edgecolor="#f1c40f", alpha=0.9))

    # 7. Plot Trade Execution
    if setup.trade:
        t = setup.trade
        t_entry_date = mdates.date2num(t.entry_time)
        ax.scatter(t_entry_date, t.entry_price, color="#ffffff", s=100, marker="*", zorder=8, label="5. Entry Filled")
        ax.annotate(f"5. Entry @ {t.entry_price:.2f}", (t_entry_date, t.entry_price),
                    textcoords="offset points", xytext=(20, 0), va="center", color="#ffffff", fontsize=9, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="#2a2e39", edgecolor="#ffffff", alpha=0.9))

        # SL and TP lines
        ax.hlines(t.sl_price, t_entry_date, dates[-1], color="#ff1744", linestyle="-", linewidth=1.5, label=f"SL: {t.sl_price:.2f}")
        ax.hlines(t.tp_price, t_entry_date, dates[-1], color="#00e676", linestyle="-", linewidth=1.5, label=f"TP (1:3): {t.tp_price:.2f}")

        if t.exit_idx:
            t_exit_date = mdates.date2num(t.exit_time)
            ex_color = "#00e676" if t.result == "WIN" else "#ff1744"
            ax.scatter(t_exit_date, t.exit_price, color=ex_color, s=90, marker="X", zorder=9)
            ax.annotate(f"Exit ({t.result})", (t_exit_date, t.exit_price),
                        textcoords="offset points", xytext=(15, -10 if t.result == 'LOSS' else 10),
                        color=ex_color, fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="#2a2e39", edgecolor=ex_color, alpha=0.9))

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    fig.autofmt_xdate()

    ax.grid(True, linestyle=":", color="#363c4e", alpha=0.6)
    ax.tick_params(colors="#d1d4dc", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("#363c4e")

    ax.set_title(f"SMC Continuation Setup Lifecycle ({setup.direction}) — XAUUSD", fontsize=14, color="#ffffff", fontweight="bold", pad=15)
    ax.set_ylabel("Price (USD)", color="#d1d4dc", fontsize=11)
    ax.legend(loc="upper left", facecolor="#2a2e39", edgecolor="#363c4e", labelcolor="#ffffff")

    plt.tight_layout()
    plt.savefig(output_png, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"Lifecycle chart successfully saved to {output_png}")


def main():
    parser = argparse.ArgumentParser(description="SMC Continuation Strategy Backtester")
    parser.add_argument("--csv", default="XAUUSDH1.csv", help="CSV data file")
    parser.add_argument("--left", type=int, default=3, help="Swing left window")
    parser.add_argument("--right", type=int, default=3, help="Swing right window")
    parser.add_argument("--buffer", type=float, default=0.50, help="SL buffer in USD")
    parser.add_argument("--rr", type=float, default=3.0, help="Risk to Reward ratio")
    parser.add_argument("--plot", action="store_true", help="Plot first executed trade lifecycle")
    args = parser.parse_args()

    df = load_csv(args.csv)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}.")

    trades, setups = run_smc_backtest(
        df,
        left_bars=args.left,
        right_bars=args.right,
        sl_buffer_usd=args.buffer,
        rr_ratio=args.rr
    )

    print_backtest_report(trades, setups)

    if args.plot or True:
        # Plot executed trade setup if available, or first armed setup
        tr_setups = [s for s in setups if s.trade is not None]
        if tr_setups:
            plot_smc_setup(df, tr_setups[0], "smc_trade_setup.png")
        else:
            armed = [s for s in setups if s.stage in (SetupStage.BOS_2_ARMED, SetupStage.IN_TRADE)]
            if armed:
                plot_smc_setup(df, armed[0], "smc_trade_setup.png")


if __name__ == "__main__":
    main()

