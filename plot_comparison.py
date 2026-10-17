#!/usr/bin/env python3
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fibo_backtest import load_csv, Costs
from smc_backtest import find_smc_trades, evaluate as eval_orig
from smc_variations_backtest import find_sweep_reversal_trades, find_active_leg_trades, evaluate, style

INK, MUTED, GRID, BLUE, GRAY, GREEN, RED, ORANGE = "#2b2b2b", "#6b6b6b", "#e6e6e6", "#2a6fbb", "#9aa0a6", "#27ae60", "#c0392b", "#e67e22"

df = load_csv("XAUUSDM15.csv")
costs = Costs(1.5, 0.5, -0.8, -0.3)
risk = 0.01
maxlev = 10.0
idx = df.index

# 1. Original SMC Rule (Deep 100% Retracement)
tr_orig = eval_orig(find_smc_trades(df, 3, 3, 0.50, 3.0), idx, costs, risk, maxlev)

# 2. Variation 1: Sweep Reversal
tr_v1 = evaluate(find_sweep_reversal_trades(df, 3, 3, 0.50, 2.0), idx, costs, risk, maxlev)

# 3. Variation 2: Active Leg Pullback
tr_v2 = evaluate(find_active_leg_trades(df, 3, 3, 0.50, 2.0), idx, costs, risk, maxlev)

# Build continuous equity curves indexed by date
dates = df.index
bh = df["close"] / df["open"].iloc[0]

def make_curve(tr):
    if tr.empty:
        return pd.Series(1.0, index=dates)
    s = pd.Series(1 + tr.net.to_numpy(), index=tr.exit_time)
    s = s.groupby(s.index).prod()
    eq = s.reindex(dates).fillna(1.0).cumprod()
    return eq

eq_orig = make_curve(tr_orig)
eq_v1 = make_curve(tr_v1)
eq_v2 = make_curve(tr_v2)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}, dpi=150)
fig.patch.set_facecolor("#1e222d")
ax1.set_facecolor("#1e222d")
ax2.set_facecolor("#1e222d")

# Equity pane
ax1.plot(dates, bh.values, color=GRAY, lw=1.2, label="Buy & Hold Gold (-4.3%)", alpha=0.7)
ax1.plot(dates, eq_orig.values, color=RED, lw=1.6, label=f"Original 100% Retrace: 17 trades (-13.7%)")
ax1.plot(dates, eq_v1.values, color=ORANGE, lw=1.6, label=f"Var 1: Sweep Reversal: 165 trades (-38.9%)")
ax1.plot(dates, eq_v2.values, color=GREEN, lw=2.0, label=f"Var 2: Active Leg Pullback: 61 trades (-2.6% / PF 0.95)")

ax1.set_yscale("log")
ax1.set_title("SMC Strategy Comparison on Gold M15 (Net of Costs)", fontsize=13, color="#ffffff", fontweight="bold", pad=12)
ax1.set_ylabel("Growth of $1.00 (Log)", color="#d1d4dc", fontsize=10)
ax1.tick_params(colors="#d1d4dc", labelsize=9)
ax1.grid(True, linestyle=":", color="#363c4e", alpha=0.6)
ax1.legend(loc="lower left", facecolor="#2a2e39", edgecolor="#363c4e", labelcolor="#ffffff", fontsize=9)
for spine in ax1.spines.values():
    spine.set_color("#363c4e")

# Drawdown pane
dd_v2 = eq_v2 / np.maximum.accumulate(eq_v2) - 1
ax2.fill_between(dates, dd_v2.values, 0, color=GREEN, alpha=0.35, label="Var 2 Drawdown")
ax2.set_title("Drawdown: Variation 2 (Active Leg Pullback)", fontsize=10, color="#ffffff")
ax2.set_ylabel("Drawdown", color="#d1d4dc", fontsize=9)
ax2.tick_params(colors="#d1d4dc", labelsize=9)
ax2.grid(True, linestyle=":", color="#363c4e", alpha=0.6)
for spine in ax2.spines.values():
    spine.set_color("#363c4e")

fig.tight_layout()
fig.savefig("smc_variations_comparison.png", facecolor=fig.get_facecolor(), edgecolor="none")
plt.close(fig)
print("Saved comparison plot to smc_variations_comparison.png")
