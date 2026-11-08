"""Constant-acceleration Kalman filter + RTS smoother for longitudinal position.

State ``s = [y, v, a]``, discrete white-noise jerk model with spectral density ``q``::

    F = [[1, dt, dt^2/2], [0, 1, dt], [0, 0, 1]]
    Q = q * [[dt^5/20, dt^4/8, dt^3/6], [dt^4/8, dt^3/3, dt^2/2], [dt^3/6, dt^2/2, dt]]
    H = [1, 0, 0],  R = sigma^2

The forward pass is the Kalman filter; the backward Rauch-Tung-Striebel pass gives the
smoothed estimate ``E[s_k | y_1..y_N]``, which is what offline trajectory reconstruction needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SmoothResult:
    position: np.ndarray
    speed: np.ndarray
    accel: np.ndarray
    filtered_speed: np.ndarray
    log_likelihood: float


def rts_smooth(t: np.ndarray, y: np.ndarray, *, sigma_m: float, q: float = 1.0,
               v0: float | None = None) -> SmoothResult:
    """Filter and smooth one vehicle's longitudinal positions (irregular ``t`` allowed)."""
    n = y.size
    if n < 2:
        raise ValueError("need at least two observations")
    xs = np.zeros((n, 3))
    ps = np.zeros((n, 3, 3))
    xp = np.zeros((n, 3))
    pp = np.zeros((n, 3, 3))
    fs = np.zeros((n, 3, 3))
    r = sigma_m ** 2
    v_init = v0 if v0 is not None else (y[min(3, n - 1)] - y[0]) / max(t[min(3, n - 1)] - t[0], 1e-6)
    x = np.array([y[0], v_init, 0.0])
    p = np.diag([r, 25.0, 4.0])
    ll = 0.0
    for k in range(n):
        if k > 0:
            dt = float(t[k] - t[k - 1])
            f = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
            qm = q * np.array([
                [dt ** 5 / 20, dt ** 4 / 8, dt ** 3 / 6],
                [dt ** 4 / 8, dt ** 3 / 3, dt ** 2 / 2],
                [dt ** 3 / 6, dt ** 2 / 2, dt],
            ])
            x = f @ x
            p = f @ p @ f.T + qm
            fs[k] = f
        xp[k], pp[k] = x, p
        s = p[0, 0] + r
        innov = y[k] - x[0]
        gain = p[:, 0] / s
        x = x + gain * innov
        p = p - np.outer(gain, p[0, :])
        ll += -0.5 * (np.log(2 * np.pi * s) + innov * innov / s)
        xs[k], ps[k] = x, p
    sm = xs.copy()
    for k in range(n - 2, -1, -1):
        c = ps[k] @ fs[k + 1].T @ np.linalg.inv(pp[k + 1])
        sm[k] = xs[k] + c @ (sm[k + 1] - xp[k + 1])
    return SmoothResult(sm[:, 0], sm[:, 1], sm[:, 2], xs[:, 1], float(ll))


def tune_q(t: np.ndarray, y: np.ndarray, sigma_m: float, grid: np.ndarray | None = None) -> float:
    """Pick the jerk density ``q`` that maximises the innovation log-likelihood."""
    grid = grid if grid is not None else np.logspace(-2, 2, 13)
    lls = [rts_smooth(t, y, sigma_m=sigma_m, q=float(g)).log_likelihood for g in grid]
    return float(grid[int(np.argmax(lls))])
