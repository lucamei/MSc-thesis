# Portfolio Optimization via Quantum Annealing -- Python implementation

This is the experimental code that reproduces the results of
Chapter 3 of the master thesis "Quantum annealing for portfolio
optimization with extended constraints" (L. Meinardo, University
of Padova, 2026).

---

## 0. Quickstart

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the experiments
python -m experiments.run_all
```

## 1. Module structure

| File                              | What it does                                      |
|-----------------------------------|---------------------------------------------------|
| `data_utils.py`                   | Load EuroStoxx data, EWMA mean/cov estimators     |
| `encoding.py`                     | Binary encoding helpers (weights, slacks)         |
| `qubo_builder.py`                 | `IncrementalQUBOBuilder` -- THE main class        |
| `classical_solver.py`             | QP / MIQP via cvxpy + Gurobi                      |
| `sa_solver.py`                    | Simulated annealing via dwave-neal                |
| `metrics.py`                      | Sharpe, Sortino, IR, Calmar, MDD, turnover        |
| `benchmarks.py`                   | 1/N, buy-and-hold, index                          |
| `backtest.py`                     | Walk-forward backtest engine                      |
| `experiments/run_section_3_x.py`  | One script per section of Chapter 3               |
| `experiments/run_all.py`          | Run everything end-to-end                         |

## 2. About the classical solver -- Gurobi vs.\ free alternatives

The classical solver appears in two roles in this thesis:

| Problem type             | Used in        | Solver         |
|--------------------------|----------------|----------------|
| Convex QP (continuous w) | C1 baseline    | cvxpy + ECOS   |
| MIQP (round-lot, integer)| C2 onwards     | **Gurobi** (academic free) |
| QUBO (binary x)          | All sections   | dwave-neal     |

For the **continuous QP** you do not need anything special: cvxpy
ships with the ECOS and SCS solvers which are open-source and
free. The convex relaxation of every $C_k$ can be solved this way.

For the **MIQP reference** (round-lot constraint, $C_2$ onwards) you
need a solver that handles mixed-integer quadratic programs. Your
options, ranked by ease of use:

1. **Gurobi with academic license -- RECOMMENDED.**
   - Free for university students at https://www.gurobi.com/academia
   - Register with your `@studenti.unipd.it` email -> get the license
     file -> `grbgetkey <license-key>` -> done.
   - Installation: `pip install gurobipy`.
   - Used automatically by `cvxpy` if installed.

2. **SCIP via PySCIPOpt -- fully free, no registration.**
   - `pip install pyscipopt`
   - cvxpy needs SCIP $\ge$ 7 and the `cvxpy[SCIP]` extra.
   - Slower than Gurobi but no license barriers.

3. **Brute-force enumeration -- only for $\mathcal U_5$ with $Q=3$.**
   - With 5 assets and 3 bits each you have $2^{15}=32k$
     configurations: enumerable in seconds.
   - Useful as a sanity-check ground truth for both classical and
     QUBO results.

The code defaults to Gurobi when available and falls back to
brute force for $\mathcal U_5$ when not, with a clear warning.

## 3. About the simulated annealer

We use `dwave-neal`, the open-source simulated annealer maintained
by D-Wave. Installation:

```bash
pip install dwave-neal dimod
```

It does NOT require any D-Wave Leap account. If you later want to
run on a real D-Wave QPU, install `dwave-system` and authenticate
with your D-Wave Leap token; the same QUBO can be sent to either
sampler with a one-line change.

## 4. About the dataset

Two options for getting EuroStoxx 50 weekly data:

1. **Use the same dataset as Prof. Caporin's slides**. Ask the
   professor for the CSV file used in the QAA course (June 2001 --
   December 2024, weekly Friday close, 47 constituents). This is
   the right choice for the thesis since it ensures the empirical
   setup of Section 3.1 of the thesis reproduces Part V of the
   QAA course exactly.

2. **Re-download via yfinance**. The code has a `load_yfinance()`
   function that fetches the current 50 constituents from
   Yahoo Finance. Some names will have shorter histories than 2001
   (recent listings); you will have to drop them or splice in
   historical data manually.

The code is agnostic: you just pass a `pandas.DataFrame` with one
column per ticker, indexed by date, containing close prices.
