# 
# Crypto bot optimisation pipeline (crypto-ready)
# ------------------------------------------------
# 
# - Bar-level backtest, next-bar execution, fees/slippage
# - Purged & embargoed walk-forward
# - Unimodality sweep + ternary search
# - Optuna single- & multi-objective
#
# Author: QuantJourney (Jakub Polec) (jakub@quantjourney.pro)
# Date: 2025-08-13
# URL: https://github.com/QuantJourneyOrg
# 
# ------------------------------------------------

import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from dataclasses import dataclass

# Config toggles ------------------------------------------------
RUN_SWEEP_PLOTS = True
RUN_SO_OPTUNA   = True     # single-objective
RUN_MO_OPTUNA   = True     # multi-objective (can be slower)

# Symbols & data ------------------------------------------------
SYMBOL   = "BTC-USD"   # e.g., "ETH-USD"
INTERVAL = "1h"        # "1d", "1h", or "15m" (YF intraday has limited history)
PERIOD   = "730d"

# Cost & scoring ------------------------------------------------
FEE_BPS    = 6          # realistic starter defaults
SLIP_BPS   = 2
PENALTY    = 0.001      # penalty per missing trade
MIN_TRADES = 30

os.makedirs("studies", exist_ok=True)

# Helpers ------------------------------------------------
def bars_per_year(interval: str) -> int:
    """
    Calculate the number of bars per year based on the interval.
    
    Args:
        interval: The interval of the data.
        
    Returns:
        The number of bars per year.
    """
    if interval.endswith("m"):
        m = int(interval[:-1]);  return int((60/m)*24*365)
    if interval.endswith("h"):
        h = int(interval[:-1]);  return int((24/h)*365)
    return 365  # daily

BARS_PER_YEAR = bars_per_year(INTERVAL)

def make_purged_embargo_splits(
    n: int,
    n_splits: int=6,
    min_train_frac: float=0.5,
    test_len: int=None,
    purge: int=0,
    embargo: int=24
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Walk-forward splits with purge (train tail removed) and embargo (gap before test).
    Returns list of (train_idx, test_idx).

    Args:
        n: The number of bars.
        n_splits: The number of splits.
        min_train_frac: The minimum fraction of the data to use for training.
        test_len: The length of the test set.
        purge: The number of bars to purge.
        embargo: The number of bars to embargo.
        
    Returns:
        The list of (train_idx, test_idx).
    """
    assert 0 < min_train_frac < 1 and n > 0
    min_train = int(n * min_train_frac)
    rem = max(0, n - min_train)
    fold = test_len if test_len is not None else max(1, rem // n_splits)
    splits = []
    for k in range(n_splits):
        tr_end   = min_train + k * fold
        te_start = tr_end + embargo
        te_end   = min(n, te_start + fold)
        if te_end <= te_start: break
        tr_end_purged = max(0, tr_end - purge)
        train = np.arange(0, tr_end_purged, dtype=int)
        test  = np.arange(te_start, te_end, dtype=int)
        if train.size and test.size:
            splits.append((train, test))
    return splits

def extract_close_column(df: pd.DataFrame) -> pd.Series:
    """
    Return a 1-D numeric close-price Series regardless of yfinance column shape:
    - Plain columns ("Close"/"Adj Close"/"close")
    - MultiIndex (('BTC-USD','Close') / ('Close','BTC-USD'))
    - Flattened names (e.g., "BTC-USD_Close")

    Args:
        df: The DataFrame to extract the close column from.
        
    Returns:
        The 1-D numeric close-price Series.
    """
    # 1) Direct hits ------------------------
    for key in ("close", "Close", "Adj Close", "adj close", "adj_close"):
        if key in df.columns:
            s = df[key]
            if isinstance(s, pd.DataFrame) and s.shape[1] == 1:
                s = s.iloc[:, 0]
            return pd.to_numeric(s, errors="coerce")

    # 2) MultiIndex search ------------------------
    if isinstance(df.columns, pd.MultiIndex):
        # prefer fields named 'Close' or 'Adj Close'
        candidates = []
        for col in df.columns:
            parts = [str(p).lower() for p in (col if isinstance(col, tuple) else (col,))]
            if any(p in ("close", "adj close", "adj_close") for p in parts):
                candidates.append(col)
        if candidates:
            s = df[candidates[0]]
            if isinstance(s, pd.DataFrame) and s.shape[1] == 1:
                s = s.iloc[:, 0]
            return pd.to_numeric(s, errors="coerce")

    # 3) Flattened names containing 'close' ------------------------
    for c in df.columns:
        if isinstance(c, str) and "close" in c.lower():
            s = df[c]
            if isinstance(s, pd.DataFrame) and s.shape[1] == 1:
                s = s.iloc[:, 0]
            return pd.to_numeric(s, errors="coerce")

    # 4) Last resort: single-column frame
    if df.shape[1] == 1:
        return pd.to_numeric(df.iloc[:, 0], errors="coerce")

    raise RuntimeError("Could not locate a 'close' price column in the downloaded data.")

# Data ------------------------------------------------
raw = yf.download(SYMBOL, period=PERIOD, interval=INTERVAL,
                  auto_adjust=True, progress=False)
if raw is None or raw.empty:
    raise RuntimeError("No data returned — adjust SYMBOL/INTERVAL/PERIOD.")

# Build a minimal df with a guaranteed 1-D 'close' ------------------------
close = extract_close_column(raw).dropna()
df = pd.DataFrame({"close": close}).dropna()
df.index = pd.to_datetime(df.index)

SPLITS = make_purged_embargo_splits(len(df), n_splits=6, min_train_frac=0.5,
                                    test_len=None, purge=0, embargo=24)

# Backtest (bar-level, next-bar) ------------------------------------------------
@dataclass
class BTConfig:
    ema_fast: int
    ema_slow: int
    sl_pct: float
    tp_pct: float
    fee_bps: float = FEE_BPS
    slip_bps: float = SLIP_BPS
    min_bars_between: int = 0

def _apply_cost(
    equity: float,
    fee_bps: float,
    slip_bps: float
) -> float:
    """
    Apply cost at entry/exit.
    
    Args:
        equity: The equity.
        fee_bps: The fee in basis points.
        slip_bps: The slippage in basis points.
        
    Returns:
        The equity after applying the cost.
    """
    # multiplicative one-shot cost at entry/exit
    cost = (fee_bps + slip_bps) * 1e-4
    return equity * (1.0 - cost)

def run_backtest_ema_barlevel(
    df: pd.DataFrame, 
    cfg: BTConfig, 
    bars_per_year: int,
    return_equity: bool=False
) -> tuple[dict, np.ndarray]:
    """
    Long-only EMA crossover; signals on bar t-1, execute on bar t (next-bar).
    SL/TP on close vs entry; MDD on equity; Sharpe on bar-level returns.

    Args:
        df: The DataFrame to backtest.
        cfg: The configuration.
        bars_per_year: The number of bars per year.
        return_equity: Whether to return the equity.
        
    Returns:
        The results and the equity.
    """
    close = df["close"]
    # force 1-D numeric
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = pd.to_numeric(close, errors="coerce").dropna()

    px = close.to_numpy(dtype=float)  # shape (n,)
    n  = len(px)
    if n < max(cfg.ema_fast, cfg.ema_slow) + 5:
        res = {"sharpe":0.0,"mdd":1.0,"trades":0}
        return (res, None) if return_equity else res

    ema_fast = close.ewm(span=cfg.ema_fast, adjust=False).mean().to_numpy()
    ema_slow = close.ewm(span=cfg.ema_slow, adjust=False).mean().to_numpy()

    equity, position = 1.0, 0
    last_bar, entry_px = -10**9, np.nan
    trades_completed = 0
    eq_series = [equity]

    start = max(cfg.ema_fast, cfg.ema_slow) + 1
    for i in range(start, n-1):
        # signals computed on i-1 vs i-2
        long_up   = (ema_fast[i-2] <= ema_slow[i-2]) and (ema_fast[i-1] > ema_slow[i-1])
        long_down = (ema_fast[i-2] >= ema_slow[i-2]) and (ema_fast[i-1] < ema_slow[i-1])

        # SL/TP vs entry
        if position == 1 and not np.isnan(entry_px):
            r_from_entry = (px[i-1] - entry_px) / entry_px
            stop = r_from_entry <= -cfg.sl_pct
            take = r_from_entry >=  cfg.tp_pct
        else:
            stop = take = False

        exit_now  = (position == 1) and (long_down or stop or take)
        enter_now = (position == 0) and long_up and (i - last_bar) >= cfg.min_bars_between

        if exit_now:
            equity = _apply_cost(equity, cfg.fee_bps, cfg.slip_bps)  # exit costs
            position, last_bar, entry_px = 0, i, np.nan
            trades_completed += 1

        if enter_now:
            equity = _apply_cost(equity, cfg.fee_bps, cfg.slip_bps)  # entry costs
            position, last_bar, entry_px = 1, i, px[i-1]

        r_bar = (px[i] / px[i-1] - 1.0) if position == 1 else 0.0
        equity *= (1.0 + r_bar)
        eq_series.append(equity)

    eq = np.array(eq_series, float)
    if len(eq) < 5 or trades_completed < 1:
        res = {"sharpe":0.0,"mdd":1.0,"trades":trades_completed}
        return (res, eq) if return_equity else res

    r = eq[1:] / eq[:-1] - 1.0
    mu = float(np.mean(r)); sd = float(np.std(r, ddof=1) or 1e-12)
    sharpe = float(mu / sd * np.sqrt(bars_per_year))
    roll_max = np.maximum.accumulate(eq)
    mdd = float(1.0 - np.min(eq / roll_max))
    res = {"sharpe": sharpe, "mdd": mdd, "trades": int(trades_completed)}
    return (res, eq) if return_equity else res

# Scoring & unimodality helpers ------------------------------------------------
def score_dict_to_scalar(m):
    """
    Score the results.
    
    Args:
        m: The results.
        
    Returns:
        The score.
    """
    return m["sharpe"] - 0.5*m["mdd"] - PENALTY * max(0, MIN_TRADES - m["trades"])

def sweep_ema_fast(
    df: pd.DataFrame,
    ema_slow: int=50,
    sl: float=0.03,
    tp: float=0.06,
    rng: range=range(5, 41),
    bpy: int=BARS_PER_YEAR
) -> tuple[int, list[float]]:
    """
    Sweep the ema_fast parameter.
    
    Args:
        df: The DataFrame to sweep.
        ema_slow: The ema_slow parameter.
        sl: The sl parameter.
        tp: The tp parameter.
        rng: The range to sweep.
        bpy: The number of bars per year.
        
    Returns:
        The best ema_fast and the scores.
    """
    scores = []
    for fast in rng:
        if fast >= ema_slow:
            scores.append(-1e9); continue
        m = run_backtest_ema_barlevel(df, BTConfig(fast, ema_slow, sl, tp), bpy)
        scores.append(score_dict_to_scalar(m))
    best_fast = rng[int(np.argmax(scores))]
    return best_fast, scores

def ternary_search_int(
    df: pd.DataFrame,
    name: str,
    lo: int,
    hi: int,
    base_cfg: BTConfig,
    iters: int=18,
    bpy: int=BARS_PER_YEAR
) -> int:
    """
    Ternary search for the best ema_slow parameter.
    
    Args:
        df: The DataFrame to search.
        name: The name of the parameter to search.
        lo: The lower bound of the search.
        hi: The upper bound of the search.
        base_cfg: The base configuration.
        iters: The number of iterations.
        bpy: The number of bars per year.
        
    Returns:
        The best ema_slow.
    """
    def eval_with(v):
        params = {**base_cfg.__dict__}; params[name] = int(round(v))
        cfg = base_cfg.__class__(**params)
        if cfg.ema_fast >= cfg.ema_slow: return -1e9
        return score_dict_to_scalar(run_backtest_ema_barlevel(df, cfg, bpy))
    a, b = float(lo), float(hi)
    for _ in range(iters):
        m1, m2 = a + (b-a)/3, b - (b-a)/3
        if eval_with(m1) < eval_with(m2): a = m1
        else:                              b = m2
    return int(round((a+b)/2))

def optimize_on_train_ema(
    train_df: pd.DataFrame,
    base_cfg: BTConfig,
    bpy: int=BARS_PER_YEAR
) -> BTConfig:
    """
    Optimize the ema_fast and ema_slow parameters on the training data.
    
    Args:
        train_df: The training DataFrame.
        base_cfg: The base configuration.
        bpy: The number of bars per year.
        
    Returns:
        The optimized configuration.
    """
    best_fast, _ = sweep_ema_fast(train_df, ema_slow=base_cfg.ema_slow,
                                  sl=base_cfg.sl_pct, tp=base_cfg.tp_pct, bpy=bpy)
    tmp = BTConfig(best_fast, base_cfg.ema_slow, base_cfg.sl_pct, base_cfg.tp_pct,
                   base_cfg.fee_bps, base_cfg.slip_bps, base_cfg.min_bars_between)
    ema_slow = ternary_search_int(train_df, "ema_slow",
                                  lo=max(tmp.ema_fast+5, 30), hi=160, base_cfg=tmp, bpy=bpy)
    return BTConfig(tmp.ema_fast, ema_slow, tmp.sl_pct, tmp.tp_pct,
                    tmp.fee_bps, tmp.slip_bps, tmp.min_bars_between)

def walkforward_score_ema(
    df: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    base_cfg: BTConfig,
    bpy: int=BARS_PER_YEAR,
    reoptimize: bool=True
) -> float:
    """
    Walk-forward score the ema_fast and ema_slow parameters.
    
    Args:
        df: The DataFrame to score.
        splits: The splits.
        base_cfg: The base configuration.
        bpy: The number of bars per year.
        reoptimize: Whether to reoptimize.
        
    Returns:
        The score.
    """
    scores = []
    for tr_idx, te_idx in splits:
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        cfg = optimize_on_train_ema(tr, base_cfg, bpy) if reoptimize else base_cfg
        m = run_backtest_ema_barlevel(te, cfg, bpy)
        scores.append(score_dict_to_scalar(m))
    return float(np.mean(scores))

# Base run, sweep, and plots ------------------------------------------------
base = BTConfig(ema_fast=12, ema_slow=50, sl_pct=0.03, tp_pct=0.06, min_bars_between=5)
base_metrics, base_eq = run_backtest_ema_barlevel(df, base, BARS_PER_YEAR, return_equity=True)
print("Base EMA metrics:", base_metrics)

best_fast, sweep_scores = sweep_ema_fast(df, ema_slow=50, sl=0.03, tp=0.06)
tmp_cfg = BTConfig(best_fast, 50, 0.03, 0.06, base.fee_bps, base.slip_bps, base.min_bars_between)
best_slow = ternary_search_int(df, "ema_slow", lo=max(best_fast+5, 30), hi=160, base_cfg=tmp_cfg)
tuned = BTConfig(best_fast, best_slow, 0.03, 0.06, base.fee_bps, base.slip_bps, base.min_bars_between)
tuned_metrics, tuned_eq = run_backtest_ema_barlevel(df, tuned, BARS_PER_YEAR, return_equity=True)
print("Tuned EMA metrics:", tuned_metrics)

wfo_score = walkforward_score_ema(df, SPLITS, base, reoptimize=True)
print("Walk-forward score (re-opt=True):", wfo_score)

# --- Plot 1: sweep curve (ema_fast vs score)
if RUN_SWEEP_PLOTS:
    rng = list(range(5, 41))
    plt.figure()
    plt.title("Unimodality sweep: ema_fast vs score")
    plt.plot(rng, sweep_scores)
    plt.xlabel("ema_fast")
    plt.ylabel("score (Sharpe - 0.5*MDD - penalty)")
    plt.grid(True)
    plt.show()

# --- Plot 2: equity curves (base vs tuned)
plt.figure()
plt.title("Equity curve (base vs tuned)")
plt.plot(base_eq, label="Base")
plt.plot(tuned_eq, label="Tuned")
plt.xlabel("Bars")
plt.ylabel("Equity")
plt.legend()
plt.grid(True)
plt.show()

# Optuna: single-objective ------------------------------------------------
if RUN_SO_OPTUNA:
    import optuna
    from optuna.importance import get_param_importances

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=1)
    study = optuna.create_study(direction="maximize",
                                sampler=sampler, pruner=pruner,
                                storage="sqlite:///studies/btc_ema_single.db",
                                study_name=f"{SYMBOL.lower()}_{INTERVAL}_ema_single",
                                load_if_exists=True)
    def objective(trial: optuna.Trial):
        ema_fast = trial.suggest_int("ema_fast", 5, 40)
        ema_slow = trial.suggest_int("ema_slow", 30, 160)
        if ema_fast >= ema_slow: raise optuna.TrialPruned()
        sl_pct   = trial.suggest_float("sl_pct", 0.003, 0.03, log=True)
        tp_pct   = trial.suggest_float("tp_pct", 0.005, 0.05, log=True)
        cooldown = trial.suggest_int("min_bars_between", 0, 12)
        base_cfg = BTConfig(ema_fast, ema_slow, sl_pct, tp_pct, FEE_BPS, SLIP_BPS, cooldown)

        fold_scores, trades_acc = [], []
        for i, (tr_idx, te_idx) in enumerate(SPLITS):
            tr, te = df.iloc[tr_idx], df.iloc[te_idx]
            tuned_cfg = optimize_on_train_ema(tr, base_cfg, BARS_PER_YEAR)
            m = run_backtest_ema_barlevel(te, tuned_cfg, BARS_PER_YEAR)
            s = score_dict_to_scalar(m)
            fold_scores.append(s); trades_acc.append(m["trades"])
            trial.report(float(np.mean(fold_scores)), step=i)
            if trial.should_prune(): raise optuna.TrialPruned()
        trial.set_user_attr("trades", float(np.mean(trades_acc)))
        return float(np.mean(fold_scores))

    study.optimize(objective, n_trials=80, timeout=1800)  # adjust as you like
    print("SO best value:", study.best_value)
    print("SO best params:", study.best_params)

    # --- Plot 3: parameter importances (matplotlib)
    try:
        imps = get_param_importances(study)
        names = list(imps.keys()); vals = [imps[k] for k in names]
        plt.figure()
        plt.title("Optuna parameter importances")
        plt.barh(names, vals)
        plt.xlabel("Importance")
        plt.ylabel("Parameter")
        plt.grid(True, axis="x")
        plt.show()
    except Exception as e:
        print("Could not compute/import importances:", e)

# Optuna: multi-objective (Pareto) ------------------------------------------------
if RUN_MO_OPTUNA:
    import optuna
    mstudy = optuna.create_study(directions=["maximize","minimize"],
                                 sampler=optuna.samplers.NSGAIISampler(seed=42),
                                 pruner=optuna.pruners.MedianPruner(),
                                 storage="sqlite:///studies/btc_ema_mo.db",
                                 study_name=f"{SYMBOL.lower()}_{INTERVAL}_ema_mo",
                                 load_if_exists=True)
    def objective_mo(trial: optuna.Trial):
        ema_fast = trial.suggest_int("ema_fast", 5, 40)
        ema_slow = trial.suggest_int("ema_slow", 30, 160)
        if ema_fast >= ema_slow: raise optuna.TrialPruned()
        sl_pct   = trial.suggest_float("sl_pct", 0.003, 0.03, log=True)
        tp_pct   = trial.suggest_float("tp_pct", 0.005, 0.05, log=True)
        cooldown = trial.suggest_int("min_bars_between", 0, 12)
        base_cfg = BTConfig(ema_fast, ema_slow, sl_pct, tp_pct, FEE_BPS, SLIP_BPS, cooldown)
        sh_list, mdd_list = [], []
        for tr_idx, te_idx in SPLITS:
            tr, te = df.iloc[tr_idx], df.iloc[te_idx]
            tuned_cfg = optimize_on_train_ema(tr, base_cfg, BARS_PER_YEAR)
            m = run_backtest_ema_barlevel(te, tuned_cfg, BARS_PER_YEAR)
            sh_list.append(m["sharpe"]); mdd_list.append(m["mdd"])
        return float(np.mean(sh_list)), float(np.mean(mdd_list))

    mstudy.optimize(objective_mo, n_trials=120)  # adjust as needed
    print("MO Pareto count:", len(mstudy.best_trials))

    # --- Plot 4: Pareto front (MDD vs Sharpe)
    xs, ys = [], []
    for t in mstudy.best_trials:
        sh, mdd = t.values[0], t.values[1]
        xs.append(mdd); ys.append(sh)
    plt.figure()
    plt.title("Pareto front (MDD vs Sharpe)")
    plt.scatter(xs, ys)
    plt.xlabel("Max Drawdown (lower is better)")
    plt.ylabel("Sharpe (higher is better)")
    plt.grid(True)
    plt.show()

