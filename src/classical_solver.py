"""classical_solver.py
============================================================
Classical reference solver for the constrained Markowitz problem.

Two flavours:

  - solve_qp_continuous(...)  -> continuous-relaxation QP using cvxpy
  - solve_miqp_roundlot(...)  -> exact MIQP using Gurobi (or SCIP)

For the OOS backtest we call solve_qp_continuous at each
rebalancing date and treat its output as ground truth. The MIQP
solver is used for the in-sample correctness check of Section 3.3
(round-lot) where the QUBO must converge to the exact integer
optimum, not just the continuous relaxation.
"""

from __future__ import annotations
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import cvxpy as cp


# ---------------------------------------------------------------------
@dataclass
class ClassicalResult:
    omega: np.ndarray
    objective: float
    sharpe: float
    wall_time_s: float
    solver_name: str
    status: str
# ---------------------------------------------------------------------


def _sharpe(omega: np.ndarray, mu: np.ndarray, Sigma: np.ndarray,
            rf: float = 0.0) -> float:
    """Ex-ante Sharpe ratio (weekly)."""
    sigma = float(np.sqrt(omega @ Sigma @ omega))
    if sigma < 1e-12:
        return 0.0
    return float((mu @ omega - rf) / sigma)


# =====================================================================
# Continuous QP solver (used for C1 baseline + every backtest step)
# =====================================================================

def solve_qp_continuous(mu: np.ndarray,
                        Sigma: np.ndarray,
                        theta: float = 0.5,
                        long_only: bool = True,
                        sectors: Optional[dict] = None,
                        R_min: Optional[float] = None,
                        omega_0: Optional[np.ndarray] = None,
                        nu_bps: Optional[float] = None,
                        d_turnover: Optional[float] = None,
                        verbose: bool = False) -> ClassicalResult:
    """Solve the constrained Markowitz QP in continuous weights.

    Builds the constraints incrementally so the same function covers
    C1..C6:
      - long_only=True              -> C1 baseline (budget + non-negativity)
      - R_min                       -> C3
      - sectors                     -> C4
      - omega_0, nu_bps             -> C5 (transaction cost penalty)
      - omega_0, d_turnover         -> C6 (hard turnover constraint)
    """
    n = len(mu)
    omega = cp.Variable(n)

    # ----- Objective ------------------------------------------------
    obj = theta * mu @ omega - (1 - theta) * cp.quad_form(omega, cp.psd_wrap(Sigma))
    if nu_bps is not None and omega_0 is not None:
        lam_tc = nu_bps * 1e-4   # bps -> decimal
        obj = obj - lam_tc * cp.sum_squares(omega - omega_0)
    objective = cp.Maximize(obj)

    # ----- Constraints ----------------------------------------------
    constraints = [cp.sum(omega) == 1]
    if long_only:
        constraints.append(omega >= 0)
    if R_min is not None:
        constraints.append(mu @ omega >= R_min)
    if sectors is not None:
        for sec_name, info in sectors.items():
            members = info["members"]
            sec_sum = cp.sum(omega[members])
            if info.get("L") is not None:
                constraints.append(sec_sum >= info["L"])
            if info.get("U") is not None:
                constraints.append(sec_sum <= info["U"])
    if d_turnover is not None and omega_0 is not None:
        constraints.append(cp.sum_squares(omega - omega_0) <= d_turnover)

    # ----- Solve ----------------------------------------------------
    prob = cp.Problem(objective, constraints)
    t0 = time.time()
    try:
        prob.solve(solver=cp.ECOS, verbose=verbose)
    except cp.error.SolverError:
        prob.solve(solver=cp.SCS, verbose=verbose)
    dt = time.time() - t0

    if omega.value is None:
        raise RuntimeError(f"Solver failed: status {prob.status}")

    omega_star = np.asarray(omega.value).flatten()
    return ClassicalResult(
        omega=omega_star,
        objective=float(prob.value),
        sharpe=_sharpe(omega_star, mu, Sigma),
        wall_time_s=dt,
        solver_name=prob.solver_stats.solver_name if prob.solver_stats else "ECOS",
        status=prob.status,
    )


# =====================================================================
# MIQP solver for the exact round-lot reference (C2 onwards)
# =====================================================================

def solve_miqp_roundlot(mu: np.ndarray,
                        Sigma: np.ndarray,
                        lambdas: np.ndarray,
                        Z_max: np.ndarray,
                        theta: float = 0.5,
                        verbose: bool = False) -> ClassicalResult:
    """Solve the exact MIQP with integer z_i in [0, Z_max[i]].

    Tries Gurobi first (academic license required). If not
    available, falls back to a CVXPY model with a mixed-integer
    open-source backend if one is installed (SCIP). If none of those
    is available, raises a clear error pointing the user to README.
    """
    try:
        import gurobipy as gp
        return _solve_miqp_gurobi(mu, Sigma, lambdas, Z_max, theta, verbose)
    except ImportError:
        warnings.warn("Gurobi not available; trying SCIP via cvxpy.")

    n = len(mu)
    z = cp.Variable(n, integer=True)
    omega = cp.multiply(lambdas, z)
    obj = theta * mu @ omega - (1 - theta) * cp.quad_form(omega, cp.psd_wrap(Sigma))
    constraints = [
        cp.sum(omega) == 1,
        z >= 0,
        z <= Z_max,
    ]
    prob = cp.Problem(cp.Maximize(obj), constraints)
    t0 = time.time()
    try:
        prob.solve(solver=cp.SCIP, verbose=verbose)
    except Exception as e:
        raise RuntimeError(
            "No MIQP solver available. Install Gurobi (free academic "
            "license; see README) or PySCIPOpt."
        ) from e
    dt = time.time() - t0
    omega_star = np.asarray(omega.value).flatten()
    return ClassicalResult(
        omega=omega_star,
        objective=float(prob.value),
        sharpe=_sharpe(omega_star, mu, Sigma),
        wall_time_s=dt,
        solver_name="SCIP",
        status=prob.status,
    )


def _solve_miqp_gurobi(mu, Sigma, lambdas, Z_max, theta, verbose):
    """Direct Gurobi formulation (avoids cvxpy MIQP overhead)."""
    import gurobipy as gp
    from gurobipy import GRB
    n = len(mu)
    m = gp.Model("roundlot_miqp")
    m.Params.OutputFlag = 1 if verbose else 0
    z = m.addVars(n, vtype=GRB.INTEGER, lb=0, ub=Z_max, name="z")
    omega = [lambdas[i] * z[i] for i in range(n)]
    # Objective: maximize theta * mu' omega - (1-theta) * omega' Sigma omega
    obj = gp.quicksum(theta * mu[i] * omega[i] for i in range(n))
    obj -= gp.quicksum((1 - theta) * Sigma[i, j] * omega[i] * omega[j]
                       for i in range(n) for j in range(n))
    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstr(gp.quicksum(omega) == 1, "budget")
    t0 = time.time()
    m.optimize()
    dt = time.time() - t0
    omega_star = np.array([lambdas[i] * z[i].X for i in range(n)])
    return ClassicalResult(
        omega=omega_star,
        objective=m.objVal,
        sharpe=_sharpe(omega_star, mu, Sigma),
        wall_time_s=dt,
        solver_name="Gurobi",
        status="optimal" if m.Status == GRB.OPTIMAL else str(m.Status),
    )
