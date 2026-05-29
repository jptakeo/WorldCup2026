"""Dixon–Coles model utility functions (score probability matrices, home advantage)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import poisson

from src.constants import MAX_GOALS


def effective_home_gamma(home_effect: float, neutral: bool, home_boost: float) -> float:
    """Multiplier applied to the home team's expected goals."""
    if not neutral:
        return home_effect
    if home_boost > 0:
        return 1.0 + (home_effect - 1.0) * home_boost
    return 1.0


def effective_home_gamma_vec(
    home_effect: NDArray[np.floating],
    neutral: bool,
    home_boost: float,
) -> NDArray[np.floating]:
    """Broadcast version of :func:`effective_home_gamma` for shape ``(B,)``."""
    he = np.asarray(home_effect, dtype=float)
    if not neutral:
        return he
    if home_boost > 0:
        return 1.0 + (he - 1.0) * home_boost
    return np.ones_like(he, dtype=float)


def score_probability_matrix(
    hl: float,
    al: float,
    rho: float,
    max_goals: int = MAX_GOALS,
) -> NDArray[np.floating]:
    """Dixon–Coles adjusted score matrix given Poisson intensities ``hl``, ``al``."""
    goals = np.arange(max_goals + 1)
    prob = np.outer(poisson.pmf(goals, hl), poisson.pmf(goals, al))

    prob[0, 0] *= 1 - hl * al * rho
    prob[1, 0] *= 1 + al * rho
    prob[0, 1] *= 1 + hl * rho
    prob[1, 1] *= 1 - rho
    prob /= prob.sum()
    return prob


def score_probability_matrix_batched(
    hl: NDArray[np.floating],
    al: NDArray[np.floating],
    rho: NDArray[np.floating],
    max_goals: int = MAX_GOALS,
) -> NDArray[np.floating]:
    """Dixon–Coles matrices for many matches at once. Shape ``(B, M, M)``."""
    hl = np.asarray(hl, dtype=float)
    al = np.asarray(al, dtype=float)
    rho = np.asarray(rho, dtype=float)
    if hl.shape != al.shape or hl.ndim != 1:
        raise ValueError("hl and al must be 1-D arrays of equal length.")
    b = hl.shape[0]
    if rho.shape not in ((b,), ()):
        raise ValueError("rho must be scalar or shape (B,) matching hl.")
    if rho.ndim == 0:
        rho = np.broadcast_to(float(rho), (b,))
    goals = np.arange(max_goals + 1, dtype=float)
    ph = poisson.pmf(goals, hl[:, None])
    pa = poisson.pmf(goals, al[:, None])
    prob = ph[:, :, None] * pa[:, None, :]

    prob[:, 0, 0] *= 1 - hl * al * rho
    prob[:, 1, 0] *= 1 + al * rho
    prob[:, 0, 1] *= 1 + hl * rho
    prob[:, 1, 1] *= 1 - rho
    s = prob.sum(axis=(1, 2), keepdims=True)
    s = np.where(s > 0, s, 1.0)
    prob /= s
    return prob
