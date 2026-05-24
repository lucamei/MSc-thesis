"""encoding.py
============================================================
Binary encoding helpers.

Two encoding paradigms are used in this thesis:

1. Continuous weight encoding (C1 baseline, Section 3.2):
       omega_i = lambda * sum_q 2^q * b_{i,q}
   with lambda = 1/(2^Q - 1) for normalized weights.

2. Round-lot encoding (C2 onwards, Section 3.3):
       omega_i = lambda_i * z_i,
       z_i = sum_q 2^q * b_{i,q},
   with lambda_i = L_i * pi_i / B (asset-specific resolution).

Slack variables (sections 3.4, 3.5, 3.7) follow the same positional
binary scheme:
       s = Delta_s * sum_k 2^k * y_k.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Iterator, List

import numpy as np


# ---------------------------------------------------------------------
#  Encoding configuration objects
# ---------------------------------------------------------------------

@dataclass
class WeightEncoding:
    """Per-asset binary encoding for portfolio weights.

    Each asset i contributes Q_i bits b_{i,0}, ..., b_{i,Q_i-1}.
    The weight is omega_i = lambda_i * sum_q 2^q * b_{i,q}.
    """
    Q: list[int]                   # bits per asset
    lambdas: np.ndarray            # resolution per asset, shape (n,)
    n: int = field(init=False)
    N: int = field(init=False)
    var_names: list[str] = field(init=False)

    def __post_init__(self) -> None:
        assert len(self.Q) == len(self.lambdas), \
            "Q and lambdas must have the same length."
        self.n = len(self.Q)
        self.N = int(sum(self.Q))
        self.var_names = [f"b_{i}_{q}" for i in range(self.n)
                          for q in range(self.Q[i])]

    # ---- helpers for iteration --------------------------------------

    def iter_bits(self) -> Iterator[tuple[int, int, str]]:
        """Yield (asset_index, bit_index, var_name) for every bit."""
        k = 0
        for i in range(self.n):
            for q in range(self.Q[i]):
                yield i, q, self.var_names[k]
                k += 1

    # ---- decoding ----------------------------------------------------

    def decode(self, sample: dict[str, int]) -> np.ndarray:
        """Convert a {var_name: 0/1} sample dict into weights omega."""
        omega = np.zeros(self.n)
        for i, q, name in self.iter_bits():
            omega[i] += self.lambdas[i] * (2 ** q) * sample[name]
        return omega


@dataclass
class SlackEncoding:
    """Binary encoding for a non-negative slack variable."""
    Q_s: int
    Delta_s: float
    prefix: str  # to make variable names unique, e.g. "y_R"
    var_names: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self.var_names = [f"{self.prefix}_{k}" for k in range(self.Q_s)]

    def decode(self, sample: dict[str, int]) -> float:
        return self.Delta_s * sum(
            (2 ** k) * sample[name] for k, name in enumerate(self.var_names)
        )


# ---------------------------------------------------------------------
#  Builders
# ---------------------------------------------------------------------

def uniform_encoding(n: int, Q: int = 4) -> WeightEncoding:
    """C1 baseline: uniform resolution lambda = 1/(2^Q - 1) across assets."""
    lam = 1.0 / (2**Q - 1)
    return WeightEncoding(Q=[Q] * n, lambdas=np.full(n, lam))


def roundlot_encoding(prices: np.ndarray,
                      budget: float,
                      Z_max: int | Sequence[int] = 15,
                      lot_sizes: int | Sequence[int] = 1) -> WeightEncoding:
    """C2 onwards: per-asset resolution lambda_i = L_i * pi_i / B."""
    n = len(prices)
    if np.isscalar(Z_max):
        Z_max = [Z_max] * n
    if np.isscalar(lot_sizes):
        lot_sizes = [lot_sizes] * n
    Q = [int(np.ceil(np.log2(z + 1))) for z in Z_max]
    lambdas = np.array([l * p / budget for l, p in zip(lot_sizes, prices)])
    return WeightEncoding(Q=Q, lambdas=lambdas)
