"""sa_solver.py
============================================================
Simulated annealing solver wrapper for QUBO.

We use D-Wave's `neal` simulated annealer with conservative
parameter choices (num_reads=1000, num_sweeps=10000) that match
the recommendations of Cohen-Khan-Alexander (2020) and
Phillipson-Bhatia (2020). Section 3.1.4 of the thesis.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import neal


# ---------------------------------------------------------------------
@dataclass
class SAResult:
    omega: np.ndarray
    sample_best: dict[str, int]
    energy_best: float
    energies: np.ndarray         # full distribution of energies
    n_reads: int
    wall_time_s: float
    success_rate: float          # fraction of reads within 1% of best
# ---------------------------------------------------------------------


def solve_sa(builder,
             num_reads: int = 1000,
             num_sweeps: int = 10000,
             beta_range: tuple[float, float] = (1e-6, 10.0),
             seed: Optional[int] = None,
             verbose: bool = False) -> SAResult:
    """Run simulated annealing on the BQM produced by the builder.

    The builder must have its objective and all desired penalty
    blocks added already.
    """
    sampler = neal.SimulatedAnnealingSampler()
    t0 = time.time()
    sampleset = sampler.sample(
        builder.bqm,
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        beta_range=beta_range,
        beta_schedule_type="geometric",
        seed=seed,
    )
    dt = time.time() - t0

    energies = np.array([rec.energy for rec in sampleset.record])
    best_idx = int(np.argmin(energies))
    best_sample = {v: int(sampleset.record[best_idx].sample[i])
                   for i, v in enumerate(sampleset.variables)}
    decoded = builder.decode_sample(best_sample)

    # Success rate: reads within 1% of best
    energy_best = energies.min()
    if energy_best == 0:
        success_rate = float((energies == 0).mean())
    else:
        tol = abs(energy_best) * 0.01
        success_rate = float((energies <= energy_best + tol).mean())

    if verbose:
        print(f"  SA: best energy={energy_best:.4f}, "
              f"omega.sum()={decoded['omega'].sum():.4f}, "
              f"success_rate={success_rate:.2%}, "
              f"wall_time={dt:.2f}s")

    return SAResult(
        omega=decoded["omega"],
        sample_best=best_sample,
        energy_best=float(energy_best),
        energies=energies,
        n_reads=num_reads,
        wall_time_s=dt,
        success_rate=success_rate,
    )
