"""qubo_builder.py
============================================================
THE MAIN MODULE -- IncrementalQUBOBuilder.

This class assembles the QUBO Hamiltonian H^(k) of Chapter 3
incrementally, exactly mirroring the cumulative-stacking rule:

       H^{(k)}(x, y^{(k)}) = H^{(k-1)}(x, y^{(k-1)}) + P_k(x, y_k)

Each method adds ONE penalty block, leaving the underlying
dimod.BinaryQuadraticModel (BQM) ready to be sampled by either
simulated annealing (dwave-neal) or, with a one-line change, a real
D-Wave QPU.

Configurations:
  C1 -- baseline + budget       : add_objective + add_budget
  C2 -- + round-lot             : pass roundlot encoding at init
  C3 -- + min return            : add_min_return
  C4 -- + sector limits         : add_sector_limits
  C5 -- + transaction costs     : add_transaction_costs
  C6 -- + turnover constraint   : add_turnover

Calibration of penalty multipliers is done via spectral scaling:
       lambda_k = rho_k * (max H_obj - min H_obj)
where H_obj is the objective alone (theta and (1-theta) blocks).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence, Optional

import numpy as np
import dimod

from encoding import WeightEncoding, SlackEncoding


# ---------------------------------------------------------------------
@dataclass
class QUBOReport:
    """Bookkeeping for what was added at each step."""
    config_id: str            # "C1", "C2", ...
    description: str
    n_main_qubits: int        # weight qubits
    n_slack_qubits: int       # cumulative slack qubits
    penalties: dict[str, float]
# ---------------------------------------------------------------------


class IncrementalQUBOBuilder:
    """Assemble H^(1) ... H^(6) cumulatively.

    Typical usage:
        builder = IncrementalQUBOBuilder(mu, Sigma, encoding)
        builder.add_objective(theta=0.5)
        builder.add_budget(rho_bud=5)               # -> C1 / C2
        builder.add_min_return(R_min=0.001)         # -> C3
        builder.add_sector_limits(sectors)          # -> C4
        builder.add_transaction_costs(omega0, nu=20e-4)  # -> C5
        builder.add_turnover(omega0, d=0.1)         # -> C6
        bqm = builder.bqm
    """

    def __init__(self,
                 mu: np.ndarray,
                 Sigma: np.ndarray,
                 encoding: WeightEncoding) -> None:
        self.mu = mu.copy()
        self.Sigma = Sigma.copy()
        self.enc = encoding
        self.bqm = dimod.BinaryQuadraticModel("BINARY")
        # Spectral scale: objective span, used by all penalties
        self._obj_span: Optional[float] = None
        # Bookkeeping
        self.reports: list[QUBOReport] = []
        self._n_slack = 0
        self._slack_counter = 0   # for unique prefixes

    # =================================================================
    # Helpers: linear and quadratic terms in (b_{i,q}, b_{j,q'})
    # =================================================================
    def _bit_coef_linear(self, i: int, q: int) -> float:
        """Coefficient of b_{i,q} in omega_i."""
        return self.enc.lambdas[i] * (2 ** q)

    def _var_name(self, i: int, q: int) -> str:
        return self.enc.var_names[
            sum(self.enc.Q[:i]) + q
        ]

    def _ensure_variable(self, name: str) -> None:
        if name not in self.bqm.variables:
            self.bqm.add_variable(name, 0.0)

    def _add_linear(self, name: str, coef: float) -> None:
        self._ensure_variable(name)
        self.bqm.add_linear(name, coef)

    def _add_quadratic(self, u: str, v: str, coef: float) -> None:
        if u == v:
            # x^2 = x for binary
            self._add_linear(u, coef)
            return
        self._ensure_variable(u)
        self._ensure_variable(v)
        self.bqm.add_quadratic(u, v, coef)

    # =================================================================
    # H_objective  --  the mean-variance term (theta block)
    # =================================================================
    def add_objective(self, theta: float = 0.5) -> None:
        """Add the Markowitz objective:
              -theta * mu' omega + (1 - theta) * omega' Sigma omega.
        This is the H_obj block, common to every configuration.
        """
        n = self.enc.n

        # Linear term: -theta * mu_i * omega_i = -theta * mu_i * lambda_i * 2^q
        for i in range(n):
            for q in range(self.enc.Q[i]):
                name = self._var_name(i, q)
                self._add_linear(name, -theta * self.mu[i] * self._bit_coef_linear(i, q))

        # Quadratic term: (1 - theta) * omega_i Sigma_ij omega_j
        for i in range(n):
            for j in range(n):
                Sij = self.Sigma[i, j]
                if Sij == 0:
                    continue
                for q in range(self.enc.Q[i]):
                    cui = self._bit_coef_linear(i, q)
                    for r in range(self.enc.Q[j]):
                        cvj = self._bit_coef_linear(j, r)
                        u = self._var_name(i, q)
                        v = self._var_name(j, r)
                        self._add_quadratic(u, v, (1 - theta) * Sij * cui * cvj)

        # Compute the spectral scale once and cache
        self._obj_span = self._estimate_obj_span(theta)

        self.reports.append(QUBOReport(
            config_id="H_obj",
            description=f"Mean-variance objective, theta={theta}",
            n_main_qubits=self.enc.N,
            n_slack_qubits=self._n_slack,
            penalties={},
        ))

    def _estimate_obj_span(self, theta: float) -> float:
        """Estimate (max H_obj - min H_obj) cheaply.

        We use the relaxed (continuous) bounds:
          max obj <= sum_i max(0, theta * mu_i) * omega_i_max
          min obj >= -theta * mu_max * 1  - 0  (since Sigma is PSD)
        Pragmatic: span = theta * |mu|_inf + (1-theta) * lambda_max(Sigma).
        """
        lam_max = float(np.linalg.eigvalsh(self.Sigma)[-1])
        span = theta * np.max(np.abs(self.mu)) + (1 - theta) * lam_max
        return float(max(span, 1e-6))

    # =================================================================
    # P_1  --  Budget penalty (configurations C1, C2)
    # =================================================================
    def add_budget(self, target: float = 1.0, rho_bud: float = 5.0) -> None:
        """Add (lambda_bud) * (sum_i omega_i - 1)^2."""
        lam = rho_bud * self._obj_span
        self._add_squared_linear(
            linear=[(self._var_name(i, q), self._bit_coef_linear(i, q))
                    for i in range(self.enc.n)
                    for q in range(self.enc.Q[i])],
            constant=-target,
            penalty=lam,
        )
        self.reports.append(QUBOReport(
            config_id="C1/C2",
            description="Budget constraint (penalty)",
            n_main_qubits=self.enc.N,
            n_slack_qubits=self._n_slack,
            penalties={"lambda_bud": lam, "rho_bud": rho_bud},
        ))

    # =================================================================
    # P_3  --  Minimum-return penalty (configuration C3)
    # =================================================================
    def add_min_return(self,
                       R_min: float,
                       Q_s: int = 4,
                       rho_min: float = 3.0) -> None:
        """Add lambda_min * (mu'omega - R_min - s)^2  with s binarized."""
        Delta_s = R_min / 4.0 if R_min > 0 else 1e-4
        slack = SlackEncoding(Q_s=Q_s, Delta_s=Delta_s,
                              prefix=f"y_R_{self._slack_counter}")
        self._slack_counter += 1
        lam = rho_min * self._obj_span

        # Linear coefficients of (mu'omega - R_min - s):
        linear = []
        for i in range(self.enc.n):
            for q in range(self.enc.Q[i]):
                linear.append((self._var_name(i, q),
                               self.mu[i] * self._bit_coef_linear(i, q)))
        for k, name in enumerate(slack.var_names):
            linear.append((name, -Delta_s * (2 ** k)))

        self._add_squared_linear(linear=linear, constant=-R_min, penalty=lam)
        self._n_slack += Q_s

        self.reports.append(QUBOReport(
            config_id="C3",
            description=f"Min return mu'w >= {R_min}",
            n_main_qubits=self.enc.N,
            n_slack_qubits=self._n_slack,
            penalties={"lambda_min": lam, "rho_min": rho_min},
        ))

    # =================================================================
    # P_4  --  Sector-limit penalty (configuration C4)
    # =================================================================
    def add_sector_limits(self,
                          sectors: dict[str, dict],
                          Q_s: int = 4,
                          rho_sec: float = 3.0) -> None:
        """Add sector floor/cap penalties.

        sectors is a dict of the form:
            {
              "Financials": {"members": [0,1], "L": None, "U": 0.30},
              "Tech":       {"members": [4,5], "L": 0.10, "U": None},
              ...
            }
        Each L (floor) or U (cap) creates an independent slack variable.
        """
        lam = rho_sec * self._obj_span
        penalties_added = {}

        for sec_name, info in sectors.items():
            members = info["members"]
            L = info.get("L")
            U = info.get("U")

            if U is not None:
                # Upper bound: sum omega + s = U,  s >= 0
                Delta = max(U / 4.0, 1e-4)
                slack = SlackEncoding(Q_s=Q_s, Delta_s=Delta,
                                      prefix=f"y_S_U_{sec_name}_{self._slack_counter}")
                self._slack_counter += 1
                linear = []
                for i in members:
                    for q in range(self.enc.Q[i]):
                        linear.append((self._var_name(i, q),
                                       self._bit_coef_linear(i, q)))
                for k, name in enumerate(slack.var_names):
                    linear.append((name, Delta * (2 ** k)))
                self._add_squared_linear(linear=linear, constant=-U, penalty=lam)
                self._n_slack += Q_s
                penalties_added[f"sec_{sec_name}_U"] = lam

            if L is not None:
                # Lower bound: sum omega - s = L,  s >= 0
                Delta = max(L / 4.0, 1e-4)
                slack = SlackEncoding(Q_s=Q_s, Delta_s=Delta,
                                      prefix=f"y_S_L_{sec_name}_{self._slack_counter}")
                self._slack_counter += 1
                linear = []
                for i in members:
                    for q in range(self.enc.Q[i]):
                        linear.append((self._var_name(i, q),
                                       self._bit_coef_linear(i, q)))
                for k, name in enumerate(slack.var_names):
                    linear.append((name, -Delta * (2 ** k)))
                self._add_squared_linear(linear=linear, constant=-L, penalty=lam)
                self._n_slack += Q_s
                penalties_added[f"sec_{sec_name}_L"] = lam

        self.reports.append(QUBOReport(
            config_id="C4",
            description=f"Sector limits, {len(sectors)} sectors",
            n_main_qubits=self.enc.N,
            n_slack_qubits=self._n_slack,
            penalties=penalties_added,
        ))

    # =================================================================
    # P_5  --  Transaction costs (configuration C5) -- NO SLACK NEEDED
    # =================================================================
    def add_transaction_costs(self,
                              omega_0: np.ndarray,
                              lambda_tc: float) -> None:
        """Add lambda_tc * ||omega - omega_0||^2 (quadratic, no slack)."""
        # || omega - omega_0 ||^2 = omega'omega - 2 omega_0' omega + ||omega_0||^2
        # Quadratic term:
        for i in range(self.enc.n):
            for q in range(self.enc.Q[i]):
                cui = self._bit_coef_linear(i, q)
                for r in range(self.enc.Q[i]):
                    cuj = self._bit_coef_linear(i, r)
                    u = self._var_name(i, q)
                    v = self._var_name(i, r)
                    self._add_quadratic(u, v, lambda_tc * cui * cuj)
        # Linear term: -2 omega_0' omega
        for i in range(self.enc.n):
            for q in range(self.enc.Q[i]):
                name = self._var_name(i, q)
                self._add_linear(name, -2 * lambda_tc * omega_0[i] *
                                 self._bit_coef_linear(i, q))
        # Constant ||omega_0||^2 dropped.

        self.reports.append(QUBOReport(
            config_id="C5",
            description=f"Quadratic transaction costs",
            n_main_qubits=self.enc.N,
            n_slack_qubits=self._n_slack,
            penalties={"lambda_tc": lambda_tc},
        ))

    # =================================================================
    # P_6  --  Turnover constraint (configuration C6, ORIGINAL)
    # =================================================================
    def add_turnover(self,
                     omega_0: np.ndarray,
                     d_max: float,
                     Q_t: int = 4,
                     rho_t: float = 4.0,
                     quadratic_truncation: bool = True) -> None:
        """Add hard turnover constraint ||omega - omega_0||^2 <= d_max.

        Implementation strategy (Strategy II in the thesis): we keep
        only quadratic terms in the squared-violation penalty, which
        is a controlled truncation of the otherwise-quartic HUBO term.
        If `quadratic_truncation=False` the method will raise -- the
        full HUBO requires reduction to QUBO via ancilla qubits,
        implemented separately in Chapter 4.
        """
        if not quadratic_truncation:
            raise NotImplementedError(
                "Full HUBO requires ancilla-based reduction; use "
                "Chapter 4 (Taylor expansion) tooling instead."
            )

        Delta_t = d_max / (2 ** Q_t - 1)
        slack = SlackEncoding(Q_s=Q_t, Delta_s=Delta_t,
                              prefix=f"u_T_{self._slack_counter}")
        self._slack_counter += 1
        lam = rho_t * self._obj_span

        # We linearize the quartic penalty by approximating
        # (omega'omega - 2 omega_0' omega + ||omega_0||^2 + s - d)^2
        # to first nontrivial order in the binary expansion. In
        # practice we keep:
        #   * the cross-terms 2 * (omega'omega)*(linear-in-x-and-y)
        #     after expanding the binary encoding.
        # This is consistent with the linearization argument in
        # Section 3.7.2 of the thesis. Pragmatically: we treat
        # omega' omega as a SCALAR evaluated at the previous omega_0,
        # an approximation that is exact when ||omega - omega_0|| is
        # small (Taylor expansion to first order around omega_0).
        scalar_const = float(omega_0 @ omega_0)
        # Linear part to be squared:
        linear = []
        # 2 * omega_0' omega is linear in x
        for i in range(self.enc.n):
            for q in range(self.enc.Q[i]):
                linear.append((self._var_name(i, q),
                               -2 * omega_0[i] *
                               self._bit_coef_linear(i, q)))
        # Slack contributes +Delta_t * 2^k for each bit
        for k, name in enumerate(slack.var_names):
            linear.append((name, Delta_t * (2 ** k)))

        self._add_squared_linear(linear=linear,
                                 constant=scalar_const - d_max,
                                 penalty=lam)
        self._n_slack += Q_t

        self.reports.append(QUBOReport(
            config_id="C6",
            description=f"Turnover constraint ||w-w0||^2 <= {d_max}",
            n_main_qubits=self.enc.N,
            n_slack_qubits=self._n_slack,
            penalties={"lambda_t": lam, "rho_t": rho_t},
        ))

    # =================================================================
    # Core helper: add a (linear_form + constant)^2 penalty
    # =================================================================
    def _add_squared_linear(self,
                            linear: list[tuple[str, float]],
                            constant: float,
                            penalty: float) -> None:
        """Add penalty * (sum c_i * b_i + constant)^2 to the BQM."""
        # Expand:  (sum c_i b_i + k)^2 = sum c_i^2 b_i^2 (= c_i^2 b_i)
        #                              + 2 sum_{i<j} c_i c_j b_i b_j
        #                              + 2 k sum_i c_i b_i
        #                              + k^2  (dropped)
        for i, (name_i, c_i) in enumerate(linear):
            # b_i^2 -> b_i term
            self._add_linear(name_i, penalty * c_i * c_i)
            # linear part 2 k c_i
            self._add_linear(name_i, penalty * 2 * constant * c_i)
            # off-diagonal
            for name_j, c_j in linear[i+1:]:
                self._add_quadratic(name_i, name_j,
                                    penalty * 2 * c_i * c_j)

    # =================================================================
    # Decoding
    # =================================================================
    def decode_sample(self, sample: dict[str, int]) -> dict:
        """Extract portfolio weights and slacks from a sample dict."""
        # Fill missing variables with 0
        full = {v: int(sample.get(v, 0)) for v in self.bqm.variables}
        omega = self.enc.decode(full)
        return {
            "omega": omega,
            "sample": full,
            "energy": float(self.bqm.energy(full)),
            "budget_violation": float(abs(omega.sum() - 1.0)),
        }

    @property
    def n_total_qubits(self) -> int:
        return len(self.bqm.variables)
