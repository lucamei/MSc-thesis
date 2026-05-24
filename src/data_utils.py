"""data_utils.py
============================================================
Data loading and Markowitz-input estimation utilities.

Conventions:
* Prices are pandas.DataFrames indexed by date, columns are
  tickers.
* Returns are weekly log-returns by default.
* All "moments" are weekly. Annualization is done downstream in
  metrics.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
#  Data loading
# ---------------------------------------------------------------------

def load_csv(path: str | Path,
             date_col: str = "Date",
             freq: str = "W-FRI") -> pd.DataFrame:
    """Load a wide-format price CSV (Date, Ticker1, Ticker2, ...).

    Resamples to the requested frequency (weekly Friday close by
    default) using the last observation of each window.
    """
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.set_index(date_col).sort_index()
    df = df.resample(freq).last().dropna(how="all")
    return df


def load_yfinance(tickers: Sequence[str],
                  start: str = "2001-06-01",
                  end: str = "2024-12-31",
                  freq: str = "W-FRI") -> pd.DataFrame:
    """Download adjusted close prices via yfinance.

    Note: some EuroStoxx 50 tickers (e.g. .DE, .PA suffixes) may
    have shorter histories than 2001. Missing rows are forward-
    filled then dropped if still missing.
    """
    import yfinance as yf
    px = yf.download(list(tickers), start=start, end=end,
                     auto_adjust=True, progress=False)["Close"]
    px = px.resample(freq).last().ffill().dropna(how="all")
    return px


def to_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Weekly log-returns from prices."""
    return np.log(prices / prices.shift(1)).dropna(how="any")


# ---------------------------------------------------------------------
#  Markowitz-input estimators
# ---------------------------------------------------------------------

@dataclass
class MarkowitzInputs:
    """Container for mu (n,) and Sigma (n,n) at a given date."""
    mu: np.ndarray
    Sigma: np.ndarray
    tickers: list[str]


def sample_inputs(returns: pd.DataFrame,
                  date: pd.Timestamp,
                  window_weeks: int = 104) -> MarkowitzInputs:
    """Rolling-window sample mean and covariance.

    Uses the W most recent observations strictly before `date`.
    Equivalent to QAA_part3 "rolling" estimator.
    """
    r = returns.loc[:date].iloc[-window_weeks-1:-1]
    if len(r) < window_weeks // 2:
        raise ValueError(f"Not enough data before {date}.")
    mu = r.mean(axis=0).to_numpy()
    Sigma = r.cov().to_numpy()
    return MarkowitzInputs(mu, Sigma, list(r.columns))


def ewma_inputs(returns: pd.DataFrame,
                date: pd.Timestamp,
                alpha_mu: float = 0.99,
                alpha_sigma: float = 0.97,
                lookback_weeks: int = 520) -> MarkowitzInputs:
    """EWMA mean and covariance with decay parameters alpha.

    Convention: alpha close to 1 -> long memory, alpha close to 0 ->
    short memory. We use Caporin's default (0.99 / 0.97).
    """
    r = returns.loc[:date].iloc[-lookback_weeks-1:-1]
    if len(r) < lookback_weeks // 2:
        raise ValueError(f"Not enough data before {date}.")

    # EWMA mean
    weights_mu = (1 - alpha_mu) * alpha_mu ** np.arange(len(r) - 1, -1, -1)
    weights_mu = weights_mu / weights_mu.sum()
    mu = (r.to_numpy() * weights_mu[:, None]).sum(axis=0)

    # EWMA covariance (de-meaned with EWMA mean)
    R = r.to_numpy() - mu
    weights_sigma = (1 - alpha_sigma) * alpha_sigma ** np.arange(len(r) - 1, -1, -1)
    weights_sigma = weights_sigma / weights_sigma.sum()
    Sigma = R.T @ np.diag(weights_sigma) @ R
    # Symmetrize against numerical noise
    Sigma = 0.5 * (Sigma + Sigma.T)
    return MarkowitzInputs(mu, Sigma, list(r.columns))


# ---------------------------------------------------------------------
#  Sub-universe selection
# ---------------------------------------------------------------------

def subset_inputs(inp: MarkowitzInputs,
                  tickers: Sequence[str]) -> MarkowitzInputs:
    """Restrict Markowitz inputs to a sub-universe of tickers."""
    idx = [inp.tickers.index(t) for t in tickers]
    return MarkowitzInputs(
        mu=inp.mu[idx],
        Sigma=inp.Sigma[np.ix_(idx, idx)],
        tickers=list(tickers),
    )


# Universes used in Chapter 3 (Section 3.1.2 of the thesis)
UNIVERSE_5 = ["ALV.DE", "INGA.AS", "AIR.PA", "MC.PA", "IBE.MC"]
UNIVERSE_10 = UNIVERSE_5 + [
    "ASML.AS", "TTE.PA", "SAN.PA", "BAS.DE", "OR.PA"
]
UNIVERSE_30 = UNIVERSE_10 + [
    "SAP.DE", "SIE.DE", "MUV2.DE", "BAYN.DE", "BMW.DE",
    "DTE.DE", "ENEL.MI", "ISP.MI", "STLAM.MI", "G.MI",
    "BNP.PA", "ENGI.PA", "DG.PA", "RI.PA", "EL.PA",
    "ITX.MC", "TEF.MC", "BBVA.MC", "ABI.BR", "AD.AS",
]
