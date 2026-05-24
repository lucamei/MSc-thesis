"""backtest.py
============================================================
Walk-forward backtest engine.

Algorithm (mirrors Section 3.8 of the thesis):

  for each rebalancing date t:
      1. Estimate (mu_t, Sigma_t) on the rolling/EWMA window
         ending at t-1.
      2. Solve the optimization (classical or QUBO/SA) given the
         CURRENT portfolio omega_{t-} and constraints active at t.
      3. Pay transaction costs on the rebalancing |omega_t* - omega_{t-}|.
      4. Hold the new portfolio omega_t* for one week.
      5. Compute the next-week realized return r_p(t+1).
      6. Drift omega_t* with realized returns to get omega_{t+1,-}.

We return a complete record (wealth path, weights, returns,
turnover, costs) suitable for plotting and reporting.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd

from data_utils import ewma_inputs, sample_inputs


# ---------------------------------------------------------------------
@dataclass
class BacktestResult:
    name: str
    weights: pd.DataFrame              # T x n
    returns: pd.Series                 # T
    costs: pd.Series                   # T (in pct of wealth)
    wealth: pd.Series                  # T (starts at 1.0)
    turnover_path: pd.Series           # T (one-way)
    diagnostics: list[dict] = field(default_factory=list)
# ---------------------------------------------------------------------


def drift_weights(omega: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Drift portfolio weights by one period of asset returns."""
    new = omega * (1 + r)
    s = new.sum()
    return new / s if s > 0 else omega


def walk_forward(returns: pd.DataFrame,
                 rebalance_dates: pd.DatetimeIndex,
                 solver: Callable,
                 *,
                 estimator: str = "ewma",
                 window_weeks: int = 104,
                 alpha_mu: float = 0.99,
                 alpha_sigma: float = 0.97,
                 nu_bps: float = 20.0,
                 initial_weights: Optional[np.ndarray] = None,
                 name: str = "strategy") -> BacktestResult:
    """Run a walk-forward backtest.

    Parameters
    ----------
    returns : DataFrame of asset returns, dates x tickers.
    rebalance_dates : DatetimeIndex of the dates at which to rebalance.
    solver : callable that takes (mu, Sigma, omega_0, **kwargs) and
             returns the new optimal weights as a 1-D numpy array.
             This is the only "plug" point: pass a classical solver
             for the benchmark, the QUBO/SA solver for the candidate.
    """
    n = returns.shape[1]
    if initial_weights is None:
        initial_weights = np.full(n, 1.0 / n)

    weights_records = []
    return_records = []
    cost_records = []
    turnover_records = []
    diagnostics = []

    omega_current = initial_weights.copy()
    wealth = 1.0
    wealth_records = []

    for i, date in enumerate(rebalance_dates):
        # 1. Estimate inputs
        try:
            if estimator == "ewma":
                inp = ewma_inputs(returns, date,
                                  alpha_mu=alpha_mu,
                                  alpha_sigma=alpha_sigma)
            elif estimator == "rolling":
                inp = sample_inputs(returns, date, window_weeks=window_weeks)
            else:
                raise ValueError(f"Unknown estimator {estimator!r}")
        except ValueError:
            # Not enough history yet -- skip, keep current weights
            continue

        # 2. Solve the optimization
        try:
            omega_new = solver(mu=inp.mu, Sigma=inp.Sigma,
                               omega_0=omega_current, date=date)
        except Exception as e:
            diagnostics.append({"date": date, "error": str(e)})
            omega_new = omega_current.copy()

        # 3. Transaction cost on the rebalancing
        turnover_t = float(np.abs(omega_new - omega_current).sum() / 2.0)
        cost_t = nu_bps * 1e-4 * turnover_t * 2.0   # round-trip
        wealth *= (1 - cost_t)

        # 4. Hold until next date
        # Find next date in `returns` index after the current one
        try:
            next_date = rebalance_dates[i + 1]
        except IndexError:
            break
        r_period = returns.loc[date:next_date].iloc[1:].sum(axis=0).to_numpy()
        # Realized portfolio log-return over the holding period:
        r_p = float((np.exp(r_period) - 1) @ omega_new)
        wealth *= (1 + r_p)
        omega_current = drift_weights(omega_new, np.exp(r_period) - 1)

        # 5. Record
        weights_records.append(pd.Series(omega_new, index=returns.columns,
                                         name=date))
        return_records.append((date, r_p))
        cost_records.append((date, cost_t))
        turnover_records.append((date, turnover_t))
        wealth_records.append((date, wealth))

    weights_df = pd.DataFrame(weights_records)
    ret_s = pd.Series(dict(return_records), name="ret")
    cost_s = pd.Series(dict(cost_records), name="cost")
    turn_s = pd.Series(dict(turnover_records), name="turnover")
    wealth_s = pd.Series(dict(wealth_records), name="wealth")

    return BacktestResult(
        name=name,
        weights=weights_df,
        returns=ret_s,
        costs=cost_s,
        wealth=wealth_s,
        turnover_path=turn_s,
        diagnostics=diagnostics,
    )
