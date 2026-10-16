#!/usr/bin/env python3
"""
fibo_backtest.py - tests the "Auto Fibonacci Retracement" rule (with SL/TP) on XAUUSD or any CFD.

THE RULE (as you described it)
  Levels come from the highest high / lowest low of the last N bars (default N=100).
  "Down fibo"  = the lowest low is more recent than the highest high (price fell, then bounced).
      levels run up from the low:   F0 = low,  F236 = low + 0.236*range,  F500 = low + 0.5*range
      -> BUY when price rises to F236, target F500, stop-loss at F0.
  "Up fibo"    = the highest high is more recent (price rose, then pulled back).
      levels run down from the high: F0 = high, F236 = high - 0.236*range, F500 = high - 0.5*range
      -> SELL when price falls to F236, target F500, stop-loss at F0.

HOW IT IS SIMULATED (so the result is not flattered)
  * Levels at bar t use only COMPLETED bars t-N..t-1 (the TradingView indicator includes the live bar,
    which repaints; a backtest must not).
  * Entry triggers when price touches F236 (default), or when a candle closes beyond F236
    if --entry-on-close is set (entering at the next bar's open).
  * SL / TP are frozen at entry. One position at a time.
  * If a bar touches both SL and TP, the STOP is assumed to hit first. If the entry bar itself touches
    the stop, the trade is counted as stopped out. Gaps through a level fill at the open.
  * Position size = risk_pct of equity per trade (distance to SL), capped at max_leverage.
  * Costs: spread + slippage on entry and exit, plus overnight swap per calendar day held.
  Use M1/M5 data if you can: on coarse bars (H1+) the same-bar SL/TP ambiguity matters more.

USAGE
  python fibo_backtest.py --selftest                       # checks the engine on hand-built cases
  python fibo_backtest.py --demo gold                      # pipeline check on random-walk "gold"
  python fibo_backtest.py --csv XAUUSD_M15.csv --spread-bps 1.5 --risk-pct 1
  python fibo_backtest.py --csv XAUUSD_M15.csv --entry-on-close

NOT FINANCIAL ADVICE. A backtest does not guarantee future results.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from numpy.lib.stride_tricks import sliding_window_view  # noqa: E402

INK, MUTED, GRID, BLUE, GRAY = "#2b2b2b", "#6b6b6b", "#e6e6e6", "#2a6fbb", "#9aa0a6"


# =====================================================================================
# DATA
# =====================================================================================

def load_csv(path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-16", "utf-8", "latin1"]
    df = None
    chosen_enc = "utf-8-sig"
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=None, engine="python", encoding=enc)
            chosen_enc = enc
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    if df is None:
        raise SystemExit(f"Unable to decode CSV file '{path}'.")

    cols_str = " ".join(str(c).lower() for c in df.columns)
    if not any(k in cols_str for k in ("open", "close", "high", "low", "price")):
        df = pd.read_csv(path, sep=None, engine="python", encoding=chosen_enc, header=None)
        if len(df.columns) >= 5:
            sample_0 = str(df.iloc[0, 0])
            if " " in sample_0 and any(c.isdigit() for c in sample_0):
                df = df.rename(columns={0: "datetime", 1: "open", 2: "high", 3: "low", 4: "close"})
            elif len(df.columns) >= 6:
                df = df.rename(columns={0: "date", 1: "time", 2: "open", 3: "high", 4: "low", 5: "close"})

    df.columns = [str(c).strip().strip("<>").lower() for c in df.columns]
    if "date" in df.columns and "time" in df.columns:
        ts = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), format="mixed", errors="coerce")
    else:
        col = next((c for c in ("datetime", "timestamp", "gmt time", "local time", "time", "date") if c in df.columns), None)
        if col is None:
            raise SystemExit("Could not find a time column in the CSV.")
        ts = pd.to_datetime(df[col], format="mixed", errors="coerce")
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise SystemExit(f"CSV is missing the '{c}' column.")
    out = df[["open", "high", "low", "close"]].astype(float)
    out.index = ts
    out = out[~out.index.isna()].sort_index()
    return out[~out.index.duplicated()]


def demo_gold(rng: np.random.Generator, years: int = 8, drift_annual: float = 0.0) -> pd.DataFrame:
    """Hourly random-walk 'gold': fat tails, weekends, NO edge by construction (pipeline check only)."""
    idx = pd.date_range("2018-01-01", periods=years * 365 * 24, freq="h")
    idx = idx[idx.dayofweek < 5]
    n = len(idx)
    bpy = n / years
    ret = rng.standard_t(4, n) / np.sqrt(2) * 0.0020 + drift_annual / bpy
    close = 1300 * np.exp(np.cumsum(ret))
    open_ = np.r_[1300.0, close[:-1]]
    wick = np.abs(rng.normal(0, 0.0008, (2, n))) * close
    return pd.DataFrame({"open": open_, "high": np.maximum(open_, close) + wick[0],
                         "low": np.minimum(open_, close) - wick[1], "close": close}, index=idx)


# =====================================================================================
# FIBONACCI LEVELS (no look-ahead)
# =====================================================================================

def fib_levels(h: np.ndarray, l: np.ndarray, period: int, chunk: int = 100_000):
    """For each bar t, use bars t-period..t-1 only. Returns (Fhigh, Flow, down).
    down = lowest low is more recent than highest high (Pine: highestbars < lowestbars)."""
    n = len(h)
    Fh, Fl = np.full(n, np.nan), np.full(n, np.nan)
    down = np.zeros(n, bool)
    m = n - period
    if m <= 0:
        return Fh, Fl, down
    wh, wl = sliding_window_view(h, period), sliding_window_view(l, period)
    for s in range(0, m, chunk):
        e = min(s + chunk, m)
        a, b = wh[s:e], wl[s:e]
        Fh[period + s: period + e] = a.max(1)
        Fl[period + s: period + e] = b.min(1)
        # bars-ago of the most recent extreme (0 = latest completed bar)
        down[period + s: period + e] = np.argmax(a[:, ::-1], 1) > np.argmin(b[:, ::-1], 1)
    return Fh, Fl, down


def level(Fh, Fl, down, x):
    rng = Fh - Fl
    return np.where(down, Fl + rng * x, Fh - rng * x)


# =====================================================================================
# TRADE ENGINE
# =====================================================================================

def scan_exit(o, h, l, t, side, tp, sl):
    """Walk forward from the entry bar t. Returns (exit_bar, exit_price, reason) or (None, None, None)."""
    n = len(o)
    if side == 1:
        if l[t] <= sl:
            return t, sl, "sl"            # conservative: a stop touch on the entry bar counts
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
                    return i, o[i], "sl"   # gapped through the stop
                if o[i] >= tp:
                    return i, o[i], "tp"
            else:
                if o[i] >= sl:
                    return i, o[i], "sl"
                if o[i] <= tp:
                    return i, o[i], "tp"
            return (i, sl, "sl") if hit_sl[i - j] else (i, tp, "tp")   # both touched -> stop first
        j, step = k, step * 2
    return None, None, None


def find_trades(o, h, l, c, period=100, entry_lvl=0.236, tp_lvl=0.5, sl_lvl=0.0, levels=None,
                entry_on_close: bool = False) -> pd.DataFrame:
    n = len(o)
    Fh, Fl, down = levels if levels is not None else fib_levels(h, l, period)
    E, TP, SL = (level(Fh, Fl, down, x) for x in (entry_lvl, tp_lvl, sl_lvl))
    prev_c = np.r_[np.nan, c[:-1]]
    valid = ~np.isnan(Fh) & (Fh > Fl)
    rows, busy = [], -1

    if not entry_on_close:
        trig = valid & ((down & (prev_c < E) & (h >= E)) | (~down & (prev_c > E) & (l <= E)))
        for t in np.flatnonzero(trig):
            if t <= busy:
                continue
            side = 1 if down[t] else -1
            e, tp, sl = E[t], TP[t], SL[t]
            fill = max(e, o[t]) if side == 1 else min(e, o[t])
            if (side == 1 and (fill >= tp or fill <= sl)) or (side == -1 and (fill <= tp or fill >= sl)):
                continue                                   # gapped past the target/stop: no trade
            xi, xp, why = scan_exit(o, h, l, t, side, tp, sl)
            if xi is None:                                 # still open at the end of the data
                xi, xp, why = n - 1, c[n - 1], "open"
            rows.append((t, xi, side, fill, xp, sl, tp, why))
            busy = xi
    else:
        # Require candle close confirmation: above F236 for downFibo (long), below F236 for upFibo (short)
        trig = valid & ((down & (prev_c <= E) & (c > E)) | (~down & (prev_c >= E) & (c < E)))
        for t in np.flatnonzero(trig):
            entry_i = t + 1
            if entry_i >= n or entry_i <= busy:
                continue
            side = 1 if down[t] else -1
            tp, sl = TP[t], SL[t]
            # If the confirming bar itself touched TP or SL, the setup was already consumed
            if side == 1:
                if h[t] >= tp or l[t] <= sl:
                    continue
            else:
                if l[t] <= tp or h[t] >= sl:
                    continue
            fill = o[entry_i]
            if (side == 1 and (fill >= tp or fill <= sl)) or (side == -1 and (fill <= tp or fill >= sl)):
                continue                                   # gapped past target or stop at open
            xi, xp, why = scan_exit(o, h, l, entry_i, side, tp, sl)
            if xi is None:
                xi, xp, why = n - 1, c[n - 1], "open"
            rows.append((entry_i, xi, side, fill, xp, sl, tp, why))
            busy = xi

    return pd.DataFrame(rows, columns=["entry_i", "exit_i", "side", "entry", "exit", "sl", "tp", "why"])


# =====================================================================================
# COSTS, SIZING, METRICS
# =====================================================================================

@dataclass(frozen=True)
class Costs:
    spread_bps: float = 1.5            # full spread, basis points of price
    slippage_bps: float = 0.5          # per side
    swap_long_bps_day: float = -0.8    # per calendar day held; negative = you pay
    swap_short_bps_day: float = -0.3

    @property
    def per_side(self) -> float:
        return (self.spread_bps / 2 + self.slippage_bps) / 1e4


def evaluate(tr: pd.DataFrame, idx, costs: Costs, risk_pct: float, max_lev: float) -> pd.DataFrame:
    """Turn raw trades into equity returns. risk_pct is a fraction (0.01 = 1% of equity at the stop)."""
    if tr.empty:
        return tr.assign(notional=[], net=[], R=[], hold_days=[], entry_time=[], exit_time=[])
    side = tr.side.to_numpy(float)
    e, x, sl = tr.entry.to_numpy(), tr.exit.to_numpy(), tr.sl.to_numpy()
    risk = np.abs(e - sl) / e
    notional = np.minimum(risk_pct / risk, max_lev)
    days = np.asarray((idx[tr.exit_i.to_numpy()] - idx[tr.entry_i.to_numpy()]).total_seconds()) / 86400.0
    swap_rate = np.where(side > 0, costs.swap_long_bps_day, costs.swap_short_bps_day) / 1e4
    net = side * (x / e - 1) * notional - 2 * costs.per_side * notional + notional * swap_rate * days
    out = tr.copy()
    out["notional"], out["hold_days"] = notional, days
    out["net"] = np.maximum(net, -1.0)
    out["R"] = net / (notional * risk)
    out["entry_time"], out["exit_time"] = idx[tr.entry_i.to_numpy()], idx[tr.exit_i.to_numpy()]
    return out


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


# =====================================================================================
# ROBUSTNESS TOOLS
# =====================================================================================

def monte_carlo(net: np.ndarray, rng, n_sims=2000):
    T = len(net)
    n_sims = int(min(n_sims, max(200, 2e7 // max(T, 1))))
    paths = np.cumprod(1 + rng.choice(net, size=(n_sims, T), replace=True), axis=1)
    peak = np.maximum.accumulate(np.concatenate([np.ones((n_sims, 1)), paths], 1), 1)[:, 1:]
    return paths, paths[:, -1] - 1, (paths / peak - 1).min(1)


def random_timing_null(real: pd.DataFrame, o, h, l, idx, costs, risk_pct, max_lev, rng, n_perm=100):
    """Same side and same stop/target distances (as % of entry), but entered at random times.
    Asks: does timing entries at the Fibonacci level add anything beyond the reward/risk geometry + drift?"""
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
# SELF-TEST (hand-built cases)
# =====================================================================================

def selftest():
    z = Costs(0, 0, 0, 0)

    def mk(rows):
        a = np.array(rows, float)   # open, high, low, close
        return a[:, 0], a[:, 1], a[:, 2], a[:, 3]

    # five falling bars: high 110 oldest, low 100 latest -> DOWN fibo; F0=100, F236=102.36, F500=105
    down_bars = [(109.5, 110, 108, 109), (108.5, 108.5, 106, 107), (107, 107, 104, 105),
                 (105, 105, 102, 103), (103, 103, 100, 101)]
    # 1) long hits TP
    o, h, l, c = mk(down_bars + [(101.5, 103, 101.2, 102.9), (103, 105.5, 102.8, 105.2)])
    tr = find_trades(o, h, l, c, period=5)
    assert len(tr) == 1 and tr.side[0] == 1 and tr.why[0] == "tp", tr
    assert abs(tr.entry[0] - 102.36) < 1e-9 and abs(tr.exit[0] - 105) < 1e-9
    ev = evaluate(tr, pd.date_range("2024-01-01", periods=7, freq="h"), z, 0.01, 99)
    assert abs(ev.R[0] - (105 - 102.36) / (102.36 - 100)) < 1e-9, ev.R[0]
    # 2) long hits SL
    o, h, l, c = mk(down_bars + [(101.5, 103, 101.2, 102.9), (102, 103, 99.5, 100)])
    tr = find_trades(o, h, l, c, period=5)
    assert tr.why[0] == "sl" and abs(tr.exit[0] - 100) < 1e-9
    assert abs(evaluate(tr, pd.date_range("2024-01-01", periods=7, freq="h"), z, 0.01, 99).R[0] + 1) < 1e-9
    # 3) both touched in one bar -> stop first
    o, h, l, c = mk(down_bars + [(101.5, 103, 101.2, 102.9), (102, 106, 99.5, 104)])
    assert find_trades(o, h, l, c, period=5).why[0] == "sl"
    # 4) gap down through the stop fills at the open
    o, h, l, c = mk(down_bars + [(101.5, 103, 101.2, 102.9), (99, 99.5, 98, 98.5)])
    tr = find_trades(o, h, l, c, period=5)
    assert tr.why[0] == "sl" and abs(tr.exit[0] - 99) < 1e-9
    # 5) SHORT mirror: rising bars -> UP fibo; F0=100(high), F236=97.64, F500=95
    up_bars = [(90.5, 92, 90, 91), (91.5, 94, 91.5, 93), (93, 96, 93, 95), (95, 98, 95, 97), (97, 100, 97, 99)]
    o, h, l, c = mk(up_bars + [(98.5, 98.7, 97.2, 97.3), (97.3, 97.4, 94.8, 95.1)])
    tr = find_trades(o, h, l, c, period=5)
    assert len(tr) == 1 and tr.side[0] == -1 and tr.why[0] == "tp", tr
    assert abs(tr.entry[0] - 97.64) < 1e-9 and abs(tr.exit[0] - 95) < 1e-9
    # 6) no look-ahead: changing bar t's own high/low must not change the levels used at bar t
    rng = np.random.default_rng(1)
    hh = 100 + rng.normal(0, 1, 300).cumsum()
    ll = hh - 0.5
    a = fib_levels(hh, ll, 20)
    hh2, ll2 = hh.copy(), ll.copy()
    hh2[150], ll2[150] = hh[150] + 50, ll[150] - 50
    b = fib_levels(hh2, ll2, 20)
    assert np.allclose(a[0][:151], b[0][:151], equal_nan=True) and np.allclose(a[1][:151], b[1][:151], equal_nan=True)
    # 7) entry on candle close:
    # Bar 5 closes above F236 (close 102.9 > 102.36), trade enters at Bar 6 open (103.0) and hits TP (105.0)
    o, h, l, c = mk(down_bars + [(101.5, 103, 101.2, 102.9), (103, 105.5, 102.8, 105.2)])
    tr_close = find_trades(o, h, l, c, period=5, entry_on_close=True)
    assert len(tr_close) == 1 and tr_close.entry_i[0] == 6 and abs(tr_close.entry[0] - 103.0) < 1e-9 and tr_close.why[0] == "tp"
    # But if bar 5 touched F236 (high 103) yet closed BELOW F236 (close 102.0), entry_on_close does not trigger:
    o, h, l, c = mk(down_bars + [(101.5, 103, 101.2, 102.0), (102, 103, 101, 102)])
    assert len(find_trades(o, h, l, c, period=5, entry_on_close=False)) == 1   # touch triggers
    assert len(find_trades(o, h, l, c, period=5, entry_on_close=True)) == 0    # close does not trigger
    print("selftest: all 7 engine checks passed")


# =====================================================================================
# REPORT
# =====================================================================================

LINES: list[str] = []


def out(s=""):
    print(s)
    LINES.append(s)


def pct(x):
    return f"{x * 100:7.1f}%"


def line(tag, s):
    out(f"  {tag:<14} trades {s['n']:>5}  win {s['win'] * 100:5.1f}%  (needs {s['be_win'] * 100:5.1f}%)  "
        f"avg R {s['exp_R']:+.3f}  PF {s['pf']:.2f}  return {pct(s['total'])}  CAGR {pct(s['cagr'])}  "
        f"maxDD {pct(s['maxdd'])}  Sharpe {s['sharpe']:+.2f}")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.title.set_color(INK)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv")
    ap.add_argument("--demo", choices=["gold"])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--period", type=int, default=100)
    ap.add_argument("--entry", type=float, default=0.236)
    ap.add_argument("--target", type=float, default=0.5)
    ap.add_argument("--entry-on-close", action="store_true",
                    help="enter at next bar open only after a candle closes beyond entry level (above for downFibo, below for upFibo)")
    ap.add_argument("--risk-pct", type=float, default=1.0, help="percent of equity risked per trade")
    ap.add_argument("--max-leverage", type=float, default=10.0)
    ap.add_argument("--spread-bps", type=float, default=1.5)
    ap.add_argument("--slippage-bps", type=float, default=0.5)
    ap.add_argument("--swap-long-bps", type=float, default=-0.8)
    ap.add_argument("--swap-short-bps", type=float, default=-0.3)
    ap.add_argument("--perms", type=int, default=100)
    ap.add_argument("--outdir", default="results_fibo")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    if a.selftest:
        selftest()
        return
    rng = np.random.default_rng(a.seed)
    if a.csv:
        df, src = load_csv(a.csv), a.csv
    elif a.demo:
        df, src = demo_gold(rng), "DEMO random-walk gold (no edge by construction - pipeline check only)"
    else:
        raise SystemExit("Give me --csv FILE (XAUUSD export) or --demo gold, or --selftest")

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
    mode_str = f"enter at {a.entry} (on candle close)" if a.entry_on_close else f"enter at {a.entry}"
    out(f"RULE   period {a.period}, {mode_str}, target {a.target}, stop at 0.0  |  risk {a.risk_pct}% per trade, "
        f"max {maxlev}x")
    out(f"COSTS  spread {costs.spread_bps} bps, slippage {costs.slippage_bps} bps/side, swap {costs.swap_long_bps_day}/"
        f"{costs.swap_short_bps_day} bps/day (long/short)")
    out("=" * 100)

    levels = fib_levels(h, l, a.period)
    raw = find_trades(o, h, l, c, a.period, a.entry, a.target, 0.0, levels, entry_on_close=a.entry_on_close)
    tr = evaluate(raw, idx, costs, risk, maxlev)
    if len(tr) == 0:
        raise SystemExit("The rule produced no trades on this data.")
    S = stats(tr, first, last)

    geo_win = a.entry / (a.entry + (a.target - a.entry))
    out("\n[0] WHAT TO EXPECT")
    if a.entry_on_close:
        out("    Entry requires candle close confirmation (above F236 for downFibo, below F236 for upFibo).")
        out(f"    Break-even win rate before costs at theoretical level: {geo_win * 100:.1f}% "
            f"(actual depends on bar-close execution price).")
    else:
        out(f"    Stop is {a.entry:.3f} of the range away, target {a.target - a.entry:.3f} away -> reward/risk "
            f"{(a.target - a.entry) / a.entry:.2f}. Break-even win rate before costs: {geo_win * 100:.1f}%.")
    out("    On a pure random walk the win rate is also ~that number, so expectancy is ~0 before costs and negative after.")
    out("    Any real edge has to come from price behaving differently at these levels, and it must beat costs.")

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

    out("\n[4] SENSITIVITY (is this a plateau or a lucky spike?) - avg R per trade after costs; not for picking settings")
    periods = sorted(set([34, 55, 89, 100, 144, 200, 300, a.period]))
    combos = [(0.236, 0.5), (0.236, 0.618), (0.382, 0.618), (0.382, 0.786)]
    grid = {}
    out("    period  " + "  ".join(f"E{e}/T{t}".rjust(14) for e, t in combos))
    for p in periods:
        lv = fib_levels(h, l, p)
        cells = []
        for e, t in combos:
            ev = evaluate(find_trades(o, h, l, c, p, e, t, 0.0, lv, entry_on_close=a.entry_on_close), idx, costs, risk, maxlev)
            grid[(p, e, t)] = (float(ev.R.mean()) if len(ev) else np.nan, len(ev))
            cells.append(f"{grid[(p, e, t)][0]:+.3f} ({len(ev)})".rjust(14))
        out(f"    {p:>6}  " + "  ".join(cells))
    gv = np.array([v[0] for v in grid.values() if not np.isnan(v[0])])
    pos_share = float((gv > 0).mean()) if len(gv) else 0.0
    out(f"    {pos_share * 100:.0f}% of these settings have positive average R. A real edge usually shows up across most of the grid.")

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
    if len(tr) >= 20:
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
        out("    Too few trades.")

    out(f"\n[7] RANDOM-TIMING TEST ({a.perms} runs: same side and same stop/target distances, random entry times)")
    null = random_timing_null(raw, o, h, l, idx, costs, risk, maxlev, rng, a.perms)
    pval = float((np.sum(null >= S["exp_R"]) + 1) / (len(null) + 1))
    out(f"    Actual avg R {S['exp_R']:+.3f}  vs random timing: mean {null.mean():+.3f}, 95th pct {np.percentile(null, 95):+.3f}, p = {pval:.3f}")
    out("    Small p = the Fibonacci timing beats random entries with identical reward/risk (and identical long/short mix).")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(null, bins=24, color=GRAY, alpha=0.85)
    ax.axvline(S["exp_R"], color=BLUE, lw=2)
    ax.set_title("Avg R per trade: random timing (gray) vs the rule (blue line)", fontsize=11)
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
        flags.append(f"only {S['n']} trades - too few to conclude")
    if len(ylist) >= 3 and pos_years < len(ylist) / 2:
        flags.append(f"profitable (avg R > 0) in only {pos_years} of {len(ylist)} years")
    lo, sh = tr[tr.side == 1], tr[tr.side == -1]
    if len(lo) and len(sh) and (lo.R.mean() > 0) != (sh.R.mean() > 0):
        flags.append("longs and shorts disagree (one side carries it) - could just be market drift, not the rule")
    if pval > 0.05:
        flags.append(f"not distinguishable from random entry timing (p={pval:.2f})")
    if stress[2]["exp_R"] <= 0:
        flags.append("falls to zero or below if costs are 2x")
    if pos_share < 0.5:
        flags.append("fewer than half of nearby settings are profitable (no plateau)")
    if flag_dd is not None and flag_dd > 0.10:
        flags.append(f"{flag_dd * 100:.0f}% chance of a >30% drawdown at this risk size")
    out("\n" + "=" * 100)
    if flags:
        out("RED FLAGS:")
        for f in flags:
            out(f"  - {f}")
    else:
        out("No automatic red flags. That justifies a demo-account forward test, not live money.")
    out("=" * 100)

    # equity chart
    eq = pd.Series(np.cumprod(1 + tr.net.to_numpy()), index=tr.exit_time)
    bh_s = pd.Series(c / o[0], index=idx)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    ax1.plot(bh_s.index, bh_s.values, color=GRAY, lw=1.2)
    ax1.plot(eq.index, eq.values, color=BLUE, lw=1.8)
    ax1.set_yscale("log")
    ax1.set_title("Growth of 1: the rule (blue) vs buy & hold (gray), after costs", fontsize=11)
    ax1.text(eq.index[-1], eq.values[-1], " rule", color=INK, va="center", fontsize=9)
    ax1.text(bh_s.index[-1], bh_s.values[-1], " buy & hold", color=INK, va="center", fontsize=9)
    style(ax1)
    ax2.fill_between(eq.index, eq.values / np.maximum.accumulate(eq.values) - 1, 0, color=BLUE, alpha=0.35, lw=0)
    ax2.set_title("Drawdown of the rule", fontsize=10)
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
