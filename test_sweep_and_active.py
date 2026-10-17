#!/usr/bin/env python3
import pandas as pd
from fibo_backtest import load_csv, Costs
from smc_variations_backtest import find_active_leg_trades, find_sweep_reversal_trades, evaluate, stats

df = load_csv("XAUUSDM15.csv")
costs = Costs(1.5, 0.5, -0.8, -0.3)
risk = 0.01

print("--- Variation 2 (Active Leg Pullback) RR sweep on M15 ---")
for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
    raw = find_active_leg_trades(df, left_bars=3, right_bars=3, sl_buffer_usd=0.50, rr_ratio=rr)
    ev = evaluate(raw, df.index, costs, risk, 10.0)
    s = stats(ev, df.index[0], df.index[-1])
    print(f"RR {rr:.1f}:1 | trades: {s['n']:2d} | win: {s['win']*100:4.1f}% (needs {s['be_win']*100:4.1f}%) | avg R: {s['exp_R']:+.3f} | return: {s['total']*100:+5.1f}% | PF: {s['pf']:.2f}")

print("\n--- Variation 1 (Liquidity Sweep Reversal) RR sweep on M15 ---")
for rr in [1.0, 1.5, 2.0, 2.5, 3.0]:
    raw = find_sweep_reversal_trades(df, left_bars=3, right_bars=3, sl_buffer_usd=0.50, rr_ratio=rr)
    ev = evaluate(raw, df.index, costs, risk, 10.0)
    s = stats(ev, df.index[0], df.index[-1])
    print(f"RR {rr:.1f}:1 | trades: {s['n']:2d} | win: {s['win']*100:4.1f}% (needs {s['be_win']*100:4.1f}%) | avg R: {s['exp_R']:+.3f} | return: {s['total']*100:+5.1f}% | PF: {s['pf']:.2f}")
