"""experiments.py
============================================================
Section-by-section runner for Chapter 3 of the thesis.

Usage:
    python -m experiments              # runs everything
    python -m experiments c1           # runs only Section 3.2 (C1)
    python -m experiments c1 c2 c3     # runs C1, C2, C3

Each run_C<k> function:
  - builds the appropriate QUBO
  - solves it with classical + SA
  - prints the in-sample comparison
  - saves a CSV with the numbers used in the thesis tables.
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from data_utils import (load_csv, load_yfinance, to_log_returns,
                        ewma_inputs, sample_inputs, subset_inputs,
                        UNIVERSE_5, UNIVERSE_10, UNIVERSE_30)
from encoding import uniform_encoding, roundlot_encoding
from qubo_builder import IncrementalQUBOBuilder
from classical_solver import solve_qp_continuous, solve_miqp_roundlot
from sa_solver import solve_sa
from metrics import (ex_ante_sharpe, budget_violation, weight_deviation,
                     sharpe_ratio, sortino_ratio, information_ratio,
                     calmar_ratio, max_drawdown, turnover,
                     equally_weighted, index_returns_from_constituents)
from backtest import walk_forward


OUTPUTS = Path("./results")
OUTPUTS.mkdir(exist_ok=True)


# =====================================================================
# Data setup (replace with Caporin's CSV when you have it)
# =====================================================================
def load_dataset():
    """Load the EuroStoxx 50 dataset and return weekly log-returns."""
    csv_path = Path("./data/eurostoxx50_weekly.csv")
    if csv_path.exists():
        print(f"Loading from {csv_path}")
        prices = load_csv(csv_path)
    else:
        print("CSV not found, falling back to yfinance.")
        prices = load_yfinance(UNIVERSE_30)
    returns = to_log_returns(prices)
    return returns


# =====================================================================
# Section 3.2 -- C1: budget-only baseline
# =====================================================================
def run_C1(returns: pd.DataFrame, theta: float = 0.5,
           universe: list[str] = UNIVERSE_5) -> dict:
    print("\n" + "=" * 60)
    print(f"Section 3.2 -- C1: Budget-only baseline ({len(universe)} stocks)")
    print("=" * 60)

    # Use a fixed reference date for in-sample tests
    ref_date = pd.Timestamp("2020-01-03")  # Friday
    inp = ewma_inputs(returns[universe], ref_date)
    mu, Sigma = inp.mu, inp.Sigma

    # ----- Classical -------------------------------------------------
    res_cl = solve_qp_continuous(mu, Sigma, theta=theta, long_only=True)
    print(f"  Classical: Sharpe={res_cl.sharpe:.4f}, "
          f"time={res_cl.wall_time_s:.4f}s")

    # ----- QUBO + SA -------------------------------------------------
    enc = uniform_encoding(n=len(mu), Q=4)
    builder = IncrementalQUBOBuilder(mu, Sigma, enc)
    builder.add_objective(theta=theta)
    builder.add_budget(rho_bud=5)
    print(f"  QUBO: {builder.n_total_qubits} qubits")
    res_sa = solve_sa(builder, num_reads=1000, num_sweeps=10000, seed=42)
    dev = weight_deviation(res_sa.omega, res_cl.omega)
    print(f"  SA:        Sharpe={ex_ante_sharpe(res_sa.omega, mu, Sigma):.4f}, "
          f"time={res_sa.wall_time_s:.2f}s, "
          f"l2={dev['l2']:.4f}, "
          f"budget_viol={budget_violation(res_sa.omega):.4f}")

    out = {
        "config": "C1",
        "n_assets": len(universe),
        "n_qubits": builder.n_total_qubits,
        "sharpe_cl": res_cl.sharpe,
        "sharpe_sa": ex_ante_sharpe(res_sa.omega, mu, Sigma),
        "l2_deviation": dev["l2"],
        "budget_violation": budget_violation(res_sa.omega),
        "time_cl_s": res_cl.wall_time_s,
        "time_sa_s": res_sa.wall_time_s,
        "success_rate": res_sa.success_rate,
    }
    return out


# =====================================================================
# Section 3.3 -- C2: round-lot constraint
# =====================================================================
def run_C2(returns: pd.DataFrame, prices_at_date: np.ndarray,
           theta: float = 0.5,
           universe: list[str] = UNIVERSE_5) -> dict:
    print("\n" + "=" * 60)
    print(f"Section 3.3 -- C2: Round-lot constraint ({len(universe)} stocks)")
    print("=" * 60)

    ref_date = pd.Timestamp("2020-01-03")
    inp = ewma_inputs(returns[universe], ref_date)
    mu, Sigma = inp.mu, inp.Sigma
    budget = 100_000.0   # EUR

    enc = roundlot_encoding(prices=prices_at_date, budget=budget,
                            Z_max=15, lot_sizes=1)

    # Classical reference: MIQP (Gurobi or SCIP)
    try:
        res_cl = solve_miqp_roundlot(mu, Sigma, enc.lambdas,
                                     Z_max=np.full(len(mu), 15), theta=theta)
        print(f"  Classical MIQP: Sharpe={res_cl.sharpe:.4f}, "
              f"time={res_cl.wall_time_s:.4f}s ({res_cl.solver_name})")
    except RuntimeError as e:
        print(f"  WARNING: MIQP not available ({e}); using continuous fallback.")
        res_cl = solve_qp_continuous(mu, Sigma, theta=theta)

    # QUBO + SA
    builder = IncrementalQUBOBuilder(mu, Sigma, enc)
    builder.add_objective(theta=theta)
    builder.add_budget(rho_bud=5)
    res_sa = solve_sa(builder, num_reads=1000, num_sweeps=10000, seed=42)
    print(f"  SA: Sharpe={ex_ante_sharpe(res_sa.omega, mu, Sigma):.4f}, "
          f"budget_viol={budget_violation(res_sa.omega):.4f}")

    out = {
        "config": "C2",
        "n_assets": len(universe),
        "n_qubits": builder.n_total_qubits,
        "sharpe_cl": res_cl.sharpe,
        "sharpe_sa": ex_ante_sharpe(res_sa.omega, mu, Sigma),
        "l2_deviation": weight_deviation(res_sa.omega, res_cl.omega)["l2"],
        "budget_violation": budget_violation(res_sa.omega),
    }
    return out


# =====================================================================
# Section 3.4 -- C3: + minimum expected return
# =====================================================================
def run_C3(returns: pd.DataFrame, prices_at_date: np.ndarray,
           R_min: float = 1e-3, theta: float = 0.5,
           universe: list[str] = UNIVERSE_10) -> dict:
    print("\n" + "=" * 60)
    print(f"Section 3.4 -- C3: + min return R_min={R_min}")
    print("=" * 60)

    ref_date = pd.Timestamp("2020-01-03")
    inp = ewma_inputs(returns[universe], ref_date)
    mu, Sigma = inp.mu, inp.Sigma

    # Classical
    res_cl = solve_qp_continuous(mu, Sigma, theta=theta, R_min=R_min)
    print(f"  Classical: Sharpe={res_cl.sharpe:.4f}, "
          f"mu'w={mu @ res_cl.omega:.5f} (>={R_min})")

    # QUBO + SA
    enc = roundlot_encoding(prices=prices_at_date,
                            budget=100_000.0, Z_max=15)
    builder = IncrementalQUBOBuilder(mu, Sigma, enc)
    builder.add_objective(theta=theta)
    builder.add_budget(rho_bud=5)
    builder.add_min_return(R_min=R_min, Q_s=4, rho_min=3)
    print(f"  QUBO: {builder.n_total_qubits} qubits "
          f"({builder.enc.N} weight + {builder._n_slack} slack)")
    res_sa = solve_sa(builder, num_reads=1000, num_sweeps=10000, seed=42)
    achieved_R = float(mu @ res_sa.omega)
    print(f"  SA: Sharpe={ex_ante_sharpe(res_sa.omega, mu, Sigma):.4f}, "
          f"mu'w={achieved_R:.5f}, "
          f"viol={'OK' if achieved_R >= R_min - 1e-6 else f'-{R_min-achieved_R:.4f}'}")

    return {
        "config": "C3",
        "n_qubits": builder.n_total_qubits,
        "sharpe_cl": res_cl.sharpe,
        "sharpe_sa": ex_ante_sharpe(res_sa.omega, mu, Sigma),
        "min_return_target": R_min,
        "min_return_achieved": achieved_R,
    }


# =====================================================================
# Section 3.5 -- C4: + sector limits
# =====================================================================
def run_C4(returns: pd.DataFrame, prices_at_date: np.ndarray,
           theta: float = 0.5,
           universe: list[str] = UNIVERSE_10) -> dict:
    print("\n" + "=" * 60)
    print("Section 3.5 -- C4: + sector limits")
    print("=" * 60)

    sectors = {
        "Financials":  {"members": [0, 1], "U": 0.30, "L": None},
        "TechTelecom": {"members": [5],    "U": None, "L": 0.10},
    }
    ref_date = pd.Timestamp("2020-01-03")
    inp = ewma_inputs(returns[universe], ref_date)
    mu, Sigma = inp.mu, inp.Sigma
    R_min = 1e-3

    res_cl = solve_qp_continuous(mu, Sigma, theta=theta,
                                 R_min=R_min, sectors=sectors)
    print(f"  Classical: Sharpe={res_cl.sharpe:.4f}")
    for name, info in sectors.items():
        s = float(res_cl.omega[info["members"]].sum())
        print(f"    {name}: {s:.3f} (L={info['L']}, U={info['U']})")

    enc = roundlot_encoding(prices=prices_at_date, budget=100_000.0, Z_max=15)
    builder = IncrementalQUBOBuilder(mu, Sigma, enc)
    builder.add_objective(theta=theta)
    builder.add_budget(rho_bud=5)
    builder.add_min_return(R_min=R_min)
    builder.add_sector_limits(sectors, Q_s=4, rho_sec=3)
    print(f"  QUBO: {builder.n_total_qubits} qubits")
    res_sa = solve_sa(builder, num_reads=2000, num_sweeps=10000, seed=42)
    print(f"  SA: Sharpe={ex_ante_sharpe(res_sa.omega, mu, Sigma):.4f}")
    for name, info in sectors.items():
        s = float(res_sa.omega[info["members"]].sum())
        print(f"    {name}: {s:.3f}")

    return {"config": "C4", "n_qubits": builder.n_total_qubits,
            "sharpe_cl": res_cl.sharpe,
            "sharpe_sa": ex_ante_sharpe(res_sa.omega, mu, Sigma)}


# =====================================================================
# Section 3.6 -- C5: + transaction costs
# =====================================================================
def run_C5(returns: pd.DataFrame, prices_at_date: np.ndarray,
           nu_bps: float = 20.0, theta: float = 0.5,
           universe: list[str] = UNIVERSE_10) -> dict:
    print("\n" + "=" * 60)
    print(f"Section 3.6 -- C5: + transaction costs (nu={nu_bps} bps)")
    print("=" * 60)

    ref_date = pd.Timestamp("2020-01-03")
    inp = ewma_inputs(returns[universe], ref_date)
    mu, Sigma = inp.mu, inp.Sigma
    omega_0 = equally_weighted(len(mu))
    R_min = 1e-3
    sectors = {"Financials": {"members": [0, 1], "U": 0.30}}

    res_cl = solve_qp_continuous(mu, Sigma, theta=theta, R_min=R_min,
                                 sectors=sectors, omega_0=omega_0,
                                 nu_bps=nu_bps)
    print(f"  Classical: Sharpe={res_cl.sharpe:.4f}, "
          f"turnover={float(np.abs(res_cl.omega-omega_0).sum()/2):.3f}")

    enc = roundlot_encoding(prices=prices_at_date, budget=100_000.0, Z_max=15)
    builder = IncrementalQUBOBuilder(mu, Sigma, enc)
    builder.add_objective(theta=theta)
    builder.add_budget(rho_bud=5)
    builder.add_min_return(R_min=R_min)
    builder.add_sector_limits(sectors)
    builder.add_transaction_costs(omega_0=omega_0, lambda_tc=nu_bps * 1e-4)
    print(f"  QUBO: {builder.n_total_qubits} qubits (no new slacks for TC)")
    res_sa = solve_sa(builder, num_reads=2000, num_sweeps=10000, seed=42)
    print(f"  SA: Sharpe={ex_ante_sharpe(res_sa.omega, mu, Sigma):.4f}")

    return {"config": "C5", "n_qubits": builder.n_total_qubits,
            "sharpe_cl": res_cl.sharpe,
            "sharpe_sa": ex_ante_sharpe(res_sa.omega, mu, Sigma)}


# =====================================================================
# Section 3.7 -- C6: + turnover constraint (original contribution)
# =====================================================================
def run_C6(returns: pd.DataFrame, prices_at_date: np.ndarray,
           d_turnover: float = 0.1, theta: float = 0.5,
           universe: list[str] = UNIVERSE_10) -> dict:
    print("\n" + "=" * 60)
    print(f"Section 3.7 -- C6: + turnover constraint d={d_turnover}")
    print("=" * 60)

    ref_date = pd.Timestamp("2020-01-03")
    inp = ewma_inputs(returns[universe], ref_date)
    mu, Sigma = inp.mu, inp.Sigma
    omega_0 = equally_weighted(len(mu))

    res_cl = solve_qp_continuous(mu, Sigma, theta=theta,
                                 omega_0=omega_0,
                                 d_turnover=d_turnover, nu_bps=20.0)
    print(f"  Classical: Sharpe={res_cl.sharpe:.4f}, "
          f"||w-w0||^2={float(np.sum((res_cl.omega-omega_0)**2)):.4f}")

    enc = roundlot_encoding(prices=prices_at_date, budget=100_000.0, Z_max=15)
    builder = IncrementalQUBOBuilder(mu, Sigma, enc)
    builder.add_objective(theta=theta)
    builder.add_budget(rho_bud=5)
    builder.add_transaction_costs(omega_0=omega_0, lambda_tc=20e-4)
    builder.add_turnover(omega_0=omega_0, d_max=d_turnover,
                         Q_t=4, rho_t=4)
    print(f"  QUBO: {builder.n_total_qubits} qubits")
    res_sa = solve_sa(builder, num_reads=2000, num_sweeps=10000, seed=42)
    diff_sq = float(np.sum((res_sa.omega - omega_0) ** 2))
    print(f"  SA: Sharpe={ex_ante_sharpe(res_sa.omega, mu, Sigma):.4f}, "
          f"||w-w0||^2={diff_sq:.4f} (<= {d_turnover})")

    return {"config": "C6", "n_qubits": builder.n_total_qubits,
            "sharpe_cl": res_cl.sharpe,
            "sharpe_sa": ex_ante_sharpe(res_sa.omega, mu, Sigma)}


# =====================================================================
# Section 3.9 -- Out-of-sample walk-forward backtest
# =====================================================================
def run_oos_backtest(returns: pd.DataFrame,
                     universe: list[str] = UNIVERSE_30) -> None:
    print("\n" + "=" * 60)
    print(f"Section 3.9 -- OOS walk-forward backtest ({len(universe)} stocks)")
    print("=" * 60)

    sub = returns[universe].dropna()
    rebalance_dates = sub.loc["2010-01-01":"2024-12-31"].index[::1]  # weekly

    sectors = {
        "Financials": {"members": [0, 1], "U": 0.30},
    }

    # --- Strategy 1: Classical full-stack ---------------------------
    def classical_solver(mu, Sigma, omega_0, date, **kw):
        res = solve_qp_continuous(mu, Sigma, theta=0.5,
                                  R_min=1e-3, sectors=sectors,
                                  omega_0=omega_0, nu_bps=20.0,
                                  d_turnover=0.1)
        return res.omega

    bt_cl = walk_forward(sub, rebalance_dates, classical_solver,
                         nu_bps=20.0, name="classical_C6")

    # --- Strategy 2: QUBO/SA full-stack -----------------------------
    def sa_solver(mu, Sigma, omega_0, date, **kw):
        prices_at_date = np.full(len(mu), 100.0)  # placeholder
        enc = roundlot_encoding(prices=prices_at_date,
                                budget=100_000.0, Z_max=15)
        builder = IncrementalQUBOBuilder(mu, Sigma, enc)
        builder.add_objective(theta=0.5)
        builder.add_budget(rho_bud=5)
        builder.add_min_return(R_min=1e-3)
        builder.add_sector_limits(sectors)
        builder.add_transaction_costs(omega_0=omega_0, lambda_tc=20e-4)
        builder.add_turnover(omega_0=omega_0, d_max=0.1, rho_t=4)
        res = solve_sa(builder, num_reads=500, num_sweeps=5000)
        return res.omega

    bt_sa = walk_forward(sub, rebalance_dates, sa_solver,
                         nu_bps=20.0, name="qubo_sa_C6")

    # --- Strategy 3: 1/N --------------------------------------------
    def ew_solver(mu, Sigma, omega_0, date, **kw):
        return equally_weighted(len(mu))

    bt_ew = walk_forward(sub, rebalance_dates, ew_solver,
                         nu_bps=20.0, name="ew")

    # --- Strategy 4: Index (1/N as proxy with no rebalancing) -------
    bt_idx = walk_forward(sub, rebalance_dates[:1], ew_solver,
                          nu_bps=20.0, name="bah_index")

    # --- Print metrics ----------------------------------------------
    print("\nMetric           Classical  QUBO/SA    1/N        Index(BAH)")
    print("-" * 60)
    for label, bt in [("Sharpe", bt_cl), ("QUBO/SA", bt_sa),
                      ("1/N", bt_ew), ("BAH", bt_idx)]:
        print(f"{label:12} Sharpe={sharpe_ratio(bt.returns):.3f}  "
              f"Sortino={sortino_ratio(bt.returns):.3f}  "
              f"MDD={max_drawdown(bt.wealth):.3f}  "
              f"Calmar={calmar_ratio(bt.returns, bt.wealth):.3f}")


# =====================================================================
# Main entry point
# =====================================================================
def main():
    args = sys.argv[1:]
    if not args:
        args = ["c1", "c2", "c3", "c4", "c5", "c6", "oos"]

    returns = load_dataset()
    # Pretend prices for round-lot (replace with real prices at ref_date):
    prices_at_date = np.full(len(UNIVERSE_10), 100.0)

    results = []
    if "c1" in args:
        results.append(run_C1(returns, universe=UNIVERSE_5))
    if "c2" in args:
        results.append(run_C2(returns, prices_at_date[:5],
                              universe=UNIVERSE_5))
    if "c3" in args:
        results.append(run_C3(returns, prices_at_date, universe=UNIVERSE_10))
    if "c4" in args:
        results.append(run_C4(returns, prices_at_date, universe=UNIVERSE_10))
    if "c5" in args:
        results.append(run_C5(returns, prices_at_date, universe=UNIVERSE_10))
    if "c6" in args:
        results.append(run_C6(returns, prices_at_date, universe=UNIVERSE_10))
    if "oos" in args:
        run_oos_backtest(returns, universe=UNIVERSE_30)

    if results:
        df = pd.DataFrame(results)
        print("\n=== Summary ===")
        print(df.to_string(index=False))
        df.to_csv(OUTPUTS / "in_sample_summary.csv", index=False)
        print(f"\nSaved to {OUTPUTS / 'in_sample_summary.csv'}")


if __name__ == "__main__":
    main()
