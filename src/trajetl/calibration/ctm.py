"""Cell Transmission Model (Daganzo 1994) with a triangular fundamental diagram.

Per lane: free-flow speed ``vf``, backward wave speed ``w``, jam density ``kj``; capacity
``q_max = cap_factor * vf * w * kj / (vf + w)``. With ``N_i`` lanes in cell ``i`` (density ``k_i``
in veh/m across all lanes)::

    sending   S_i = min(vf * k_i, q_max * N_i)
    receiving R_i = min(q_max * N_i, w * (kj * N_i - k_i))
    flow      y_i = min(S_i, R_{i+1})

Boundary and initial conditions (all measured from the trajectories):

* upstream: observed count at the first station, fed through a point queue;
* on-ramp: observed entering count, injected into the merge cell;
* downstream: the exit can accept ``min(q_max N, w (kj N - k_obs))``, where ``k_obs`` is the
  observed density in the last cell. This is a congested supply boundary, so a queue from a
  bottleneck beyond the study area propagates in, as it does in the data. Without density
  data, the outflow is capped at the observed downstream count;
* initial state: observed cell densities at the start of the window (the section is already
  congested when recording begins).

Time step and cell length satisfy CFL: ``vf * dt <= dx``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.exceptions import ConfigurationError


@dataclass(frozen=True)
class FundamentalDiagram:
    vf: float = 29.0  # m/s (65 mph)
    w: float = 5.4  # m/s
    kj: float = 0.125  # veh/m/lane (~200 veh/mi/lane)
    cap_factor: float = 1.0

    @property
    def q_max(self) -> float:
        """Capacity per lane (veh/s)."""
        return self.cap_factor * self.vf * self.w * self.kj / (self.vf + self.w)


@dataclass(frozen=True)
class CTMResult:
    station_counts: np.ndarray  # (n_stations, n_intervals)
    station_speed: np.ndarray  # (n_stations, n_intervals) space-mean speed, m/s
    density: np.ndarray  # (n_intervals, n_cells) veh/m, interval mean


def run_ctm(
    fd: FundamentalDiagram,
    *,
    length_m: float,
    lanes_at: np.ndarray,  # lanes per cell
    inflow_counts: np.ndarray,  # observed counts per interval at the upstream boundary
    outflow_cap_counts: np.ndarray,  # observed counts per interval at the downstream boundary
    station_y: np.ndarray,
    interval_s: float,
    dx: float = 30.0,
    dt: float = 0.8,
    initial_density: np.ndarray | None = None,  # veh/m per cell
    downstream_density: np.ndarray | None = None,  # veh/m in the last cell, per interval
    ramp_counts: np.ndarray | None = None,  # entering vehicles per interval
    ramp_cell: int = 0,
) -> CTMResult:
    """Simulate and aggregate flows/speeds at stations per interval."""
    if fd.vf * dt > dx + 1e-9:
        raise ConfigurationError("CFL violated: vf * dt must not exceed dx", context={"vf": fd.vf, "dt": dt, "dx": dx})
    n_cells = lanes_at.size
    if abs(n_cells * dx - length_m) > dx:
        raise ConfigurationError("lanes_at does not match the section length")
    n_int = inflow_counts.size
    steps_per = int(round(interval_s / dt))
    cap = fd.q_max * lanes_at
    jam = fd.kj * lanes_at
    k = np.zeros(n_cells) if initial_density is None else np.minimum(np.asarray(initial_density, float), jam)
    queue = 0.0
    st_idx = np.clip(np.round(station_y / dx).astype(int) - 1, 0, n_cells - 2)  # boundary nearest each station
    counts = np.zeros((station_y.size, n_int))
    speed_num = np.zeros((station_y.size, n_int))
    speed_den = np.zeros((station_y.size, n_int))
    dens = np.zeros((n_int, n_cells))
    for j in range(n_int):
        demand_rate = inflow_counts[j] / interval_s
        if downstream_density is not None:
            exit_supply = min(cap[-1], max(fd.w * (jam[-1] - downstream_density[j]), 0.0))
        else:
            exit_supply = outflow_cap_counts[j] / interval_s
        ramp_rate = 0.0 if ramp_counts is None else ramp_counts[j] / interval_s
        for _ in range(steps_per):
            queue += demand_rate * dt
            send = np.minimum(fd.vf * k, cap)
            recv = np.minimum(cap, np.maximum(fd.w * (jam - k), 0.0))
            inner = np.minimum(send[:-1], recv[1:])
            q_in = min(queue / dt, recv[0])
            q_out = min(send[-1], exit_supply)
            queue -= q_in * dt
            k[0] += (q_in - inner[0]) * dt / dx
            k[1:-1] += (inner[:-1] - inner[1:]) * dt / dx
            k[-1] += (inner[-1] - q_out) * dt / dx
            if ramp_rate:
                k[ramp_cell] += ramp_rate * dt / dx
            np.clip(k, 0.0, jam, out=k)
            flow_st = inner[st_idx]
            counts[:, j] += flow_st * dt
            # space-mean speed over the interval = time-integrated flow / time-integrated density
            speed_num[:, j] += flow_st * dt
            speed_den[:, j] += k[st_idx] * dt
            dens[j] += k / steps_per
    with np.errstate(invalid="ignore", divide="ignore"):
        sms = np.where(speed_den > 1e-9, speed_num / np.where(speed_den > 1e-9, speed_den, 1.0), fd.vf)
    return CTMResult(counts, np.minimum(sms, fd.vf), dens)


def geh(model_vph: np.ndarray, obs_vph: np.ndarray) -> np.ndarray:
    m = np.asarray(model_vph, float)
    c = np.asarray(obs_vph, float)
    s = m + c
    return np.where(s > 0, np.sqrt(2.0 * (m - c) ** 2 / np.where(s > 0, s, 1.0)), 0.0)
