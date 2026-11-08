"""Lane-level HMM map matching (Newson & Krumm 2009, adapted from road links to lanes).

Hidden state: the lane a probe vehicle is in. Observation: its noisy lateral position.

* **Emission** ``p(x | lane = l) = N(x; c_l, sigma^2)``, where ``c_l`` is the lane centreline
  offset estimated from the data and ``sigma`` is the positioning noise.
* **Transition** per observation interval: stay with probability ``1 - p_lc``, move to an
  adjacent lane with ``p_lc / 2`` each, jump two or more lanes with a small ``eps``. ``p_lc`` is
  learned by maximum likelihood from ground-truth training vehicles.
* **Lane availability**: a lane only exists over its observed longitudinal extent (for
  example, the I-80 on-ramp auxiliary lane ends); outside it the emission is zero.

Decoding is log-space Viterbi, vectorised over lanes: O(T * L^2) per vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LOG_EPS = -30.0


@dataclass(frozen=True)
class LaneMap:
    lanes: np.ndarray  # lane ids (sorted)
    centres: np.ndarray  # lateral centreline offset (m) per lane
    y_min: np.ndarray  # longitudinal extent where the lane exists
    y_max: np.ndarray

    @classmethod
    def from_trajectories(cls, df: pd.DataFrame, q: float = 0.01) -> "LaneMap":
        g = df.groupby("lane_id")
        lanes = np.array(sorted(g.groups), dtype=np.int64)
        return cls(
            lanes,
            g["x"].median().reindex(lanes).to_numpy(float),
            g["y"].quantile(q).reindex(lanes).to_numpy(float),
            g["y"].quantile(1 - q).reindex(lanes).to_numpy(float),
        )

    def nearest(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Baseline: nearest available centreline."""
        d = np.abs(x[:, None] - self.centres[None, :])
        d = np.where((y[:, None] >= self.y_min) & (y[:, None] <= self.y_max), d, np.inf)
        return self.lanes[np.argmin(d, axis=1)]


@dataclass
class LaneHMM:
    lane_map: LaneMap
    sigma_m: float
    p_lane_change: float = 0.02
    eps: float = 1e-6

    @staticmethod
    def learn_lane_change_rate(df: pd.DataFrame, step_s: float) -> float:
        """MLE of the per-observation lane-change probability from ground truth sampled every ``step_s``."""
        d = df.sort_values(["vehicle_id", "t"])
        d = d[np.isclose(d["t"] % step_s, 0.0, atol=0.051) | np.isclose(d["t"] % step_s, step_s, atol=0.051)]
        change = d.groupby("vehicle_id")["lane_id"].diff().dropna()
        return float(np.clip((change != 0).mean(), 1e-4, 0.5))

    def _log_transition(self) -> np.ndarray:
        n = self.lane_map.lanes.size
        idx = np.arange(n)
        dist = np.abs(idx[:, None] - idx[None, :])
        p = np.where(dist == 0, 1.0 - self.p_lane_change, np.where(dist == 1, self.p_lane_change / 2.0, self.eps))
        p = p / p.sum(axis=1, keepdims=True)
        return np.log(p)

    def _log_emission(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        lm = self.lane_map
        z = (x[:, None] - lm.centres[None, :]) / self.sigma_m
        le = -0.5 * z * z
        avail = (y[:, None] >= lm.y_min - 5.0) & (y[:, None] <= lm.y_max + 5.0)
        return np.where(avail, le, LOG_EPS * 10)

    def decode(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Most likely lane sequence for one vehicle's observations."""
        t_len = x.size
        if t_len == 0:
            return np.empty(0, dtype=np.int64)
        lt = self._log_transition()
        le = self._log_emission(x, y)
        n = lt.shape[0]
        score = le[0].copy()
        back = np.zeros((t_len, n), dtype=np.int64)
        for k in range(1, t_len):
            cand = score[:, None] + lt  # from i to j
            back[k] = np.argmax(cand, axis=0)
            score = cand[back[k], np.arange(n)] + le[k]
        path = np.empty(t_len, dtype=np.int64)
        path[-1] = int(np.argmax(score))
        for k in range(t_len - 1, 0, -1):
            path[k - 1] = back[k, path[k]]
        return self.lane_map.lanes[path]

    def decode_frame(self, probes: pd.DataFrame) -> pd.Series:
        """Decode every vehicle in a probe table (columns vehicle_id, t, x_obs, y_obs)."""
        out = pd.Series(index=probes.index, dtype="int64")
        for _, g in probes.sort_values(["vehicle_id", "t"]).groupby("vehicle_id", sort=False):
            out.loc[g.index] = self.decode(g["x_obs"].to_numpy(float), g["y_obs"].to_numpy(float))
        return out


def degrade_to_probe(df: pd.DataFrame, *, rate_hz: float, sigma_m: float, seed: int) -> pd.DataFrame:
    """Turn 10-Hz ground truth into probe data: resample to ``rate_hz`` and add Gaussian noise."""
    step = 1.0 / rate_hz
    keep = np.isclose((df["t"] / step) - np.round(df["t"] / step), 0.0, atol=1e-3)
    p = df.loc[keep, ["vehicle_id", "t", "x", "y", "lane_id", "speed"]].copy()
    rng = np.random.default_rng(seed)
    p["x_obs"] = p["x"] + rng.normal(0.0, sigma_m, len(p))
    p["y_obs"] = p["y"] + rng.normal(0.0, sigma_m, len(p))
    return p.reset_index(drop=True)
