"""metrics.py
============================================================
Performance metrics for in-sample and out-of-sample evaluation.

In-sample (per-rebalancing-date checks):
   - ex-ante Sharpe
   - constraint compliance (budget violation, sector violation, ...)
   - weight deviation between classical and SA solutions

Out-of-sample (whole-backtest-period):
   - Sharpe, Sortino, Information Ratio
   - max drawdown, Calmar
   - average turnover, total transaction cost paid
"""

from __future__ import annotations
import numpy as np
import pandas as pd


WEEKS_PER_YEAR = 52


# =====================================================================
# In-sample metrics
# =====================================================================

def ex_ante_sharpe(omega: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
                   rf: float = 0.0) -> float:
    """Ex-ante (model-implied) Sharpe ratio."""
    sigma = float(np.sqrt(omega @ Sigma @ omega))
    if sigma < 1e-12:
        return 0.0
    return float((mu @ omega - rf) / sigma)


def budget_violation(omega: np.ndarray) -> float:
    return abs(float(omega.sum()) - 1.0)


def sector_violations(omega: np.ndarray, sectors: dict) -> dict[str, float]:
    """Per-sector violation magnitudes (0 if compliant)."""
    out = {}
    for name, info in sectors.items():
        s = float(omega[info["members"]].sum())
        if info.get("U") is not None and s > info["U"]:
            out[f"{name}_over"] = s - info["U"]
        elif info.get("L") is not None and s < info["L"]:
            out[f"{name}_under"] = info["L"] - s
    return out


def weight_deviation(omega_a: np.ndarray, omega_b: np.ndarray) -> dict:
    """L2 and L_inf distance between two weight vectors."""
    d = omega_a - omega_b
    return {"l2": float(np.linalg.norm(d, 2)),
            "linf": float(np.linalg.norm(d, np.inf))}


# =====================================================================
# OOS metrics (work on a return series r_t)
# =====================================================================

def sharpe_ratio(returns: pd.Series, rf: float = 0.0,
                 periods_per_year: int = WEEKS_PER_YEAR) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - rf
    sigma = excess.std()
    if sigma < 1e-12:
        return 0.0
    return float(excess.mean() / sigma * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, rf: float = 0.0,
                  periods_per_year: int = WEEKS_PER_YEAR) -> float:
    excess = returns - rf
    downside = excess[excess < 0]
    if len(downside) < 2 or downside.std() < 1e-12:
        return float("nan")
    return float(excess.mean() / downside.std() * np.sqrt(periods_per_year))


def information_ratio(returns: pd.Series, bench: pd.Series,
                      periods_per_year: int = WEEKS_PER_YEAR) -> float:
    active = returns - bench
    sigma = active.std()
    if sigma < 1e-12:
        return float("nan")
    return float(active.mean() / sigma * np.sqrt(periods_per_year))


def max_drawdown(wealth: pd.Series) -> float:
    cummax = wealth.cummax()
    dd = (wealth - cummax) / cummax
    return float(dd.min())


def calmar_ratio(returns: pd.Series, wealth: pd.Series,
                 periods_per_year: int = WEEKS_PER_YEAR) -> float:
    ann_return = (1 + returns.mean()) ** periods_per_year - 1
    mdd = abs(max_drawdown(wealth))
    if mdd < 1e-12:
        return float("inf")
    return float(ann_return / mdd)


def turnover(weights_history: pd.DataFrame) -> float:
    """Average per-period weight change (one-way)."""
    diff = weights_history.diff().abs().sum(axis=1).iloc[1:]
    return float(diff.mean() / 2.0)   # one-way: half of round-trip


# =====================================================================
# Benchmarks
# =====================================================================

def equally_weighted(n: int) -> np.ndarray:
    """1/N portfolio of size n."""
    return np.full(n, 1.0 / n)


def buy_and_hold_weights(initial: np.ndarray,
                         returns: pd.DataFrame) -> pd.DataFrame:
    """Weight trajectory of a buy-and-hold portfolio.

    Starting weights `initial` drift with the realized returns.
    Returns a DataFrame of weights aligned with the return index.
    """
    cum_growth = (1 + returns).cumprod()
    raw = cum_growth * initial
    return raw.div(raw.sum(axis=1), axis=0)


def index_returns_from_constituents(returns: pd.DataFrame,
                                    weights: np.ndarray) -> pd.Series:
    """Reconstruct an index series from constituent returns + weights."""
    return (returns * weights).sum(axis=1)
