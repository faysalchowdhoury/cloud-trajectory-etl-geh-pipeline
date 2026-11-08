"""Optuna calibration of the CTM fundamental diagram against virtual-detector data.

Objective (minimised, on the **training** intervals only)::

    J = mean_s RMSE(speed_model, speed_obs) / 10 m/s  +  0.1 * mean GEH(flow_model, flow_obs)

FHWA guidance calibrates on volumes **and** speeds/travel times. Here volumes are largely
pinned by conservation and the boundary conditions, so GEH alone cannot discriminate
between parameter sets; speeds carry most of the signal. The GEH acceptance test
(>= 85% of station-intervals below 5) is reported separately on held-out intervals.
Flows use 60-s counts expressed as hourly rates, which is stricter than hourly GEH.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import optuna

from .ctm import FundamentalDiagram, geh, run_ctm

BOUNDS = {"vf": (20.0, 37.0), "w": (3.0, 8.0), "kj": (0.10, 0.20), "cap_factor": (0.75, 1.15)}


@dataclass(frozen=True)
class DetectorData:
    """Cross-section aggregates: arrays shaped (n_stations, n_intervals)."""

    station_y: np.ndarray
    counts: np.ndarray
    speed: np.ndarray
    interval_s: float
    length_m: float
    lanes_at: np.ndarray
    density: np.ndarray | None = None  # (n_intervals, n_cells) observed veh/m
    ramp_counts: np.ndarray | None = None
    ramp_cell: int = 0

    def split(self, train_frac: float = 0.5) -> tuple[slice, slice]:
        n = self.counts.shape[1]
        cut = int(round(n * train_frac))
        return slice(0, cut), slice(cut, n)


@dataclass
class Evaluation:
    objective: float
    speed_rmse: float
    geh_mean: float
    geh_pass: float
    geh_window_pass: float  # stations with GEH < 5 on window totals (hourly rate)
    model_counts: np.ndarray
    model_speed: np.ndarray


@dataclass
class CalibrationResult:
    sampler: str
    best_params: dict[str, float]
    best_value: float
    history: list[float] = field(default_factory=list)


def evaluate(fd: FundamentalDiagram, d: DetectorData, window: slice) -> Evaluation:
    res = run_ctm(fd, length_m=d.length_m, lanes_at=d.lanes_at, inflow_counts=d.counts[0],
                  outflow_cap_counts=d.counts[-1], station_y=d.station_y, interval_s=d.interval_s,
                  initial_density=None if d.density is None else d.density[0],
                  downstream_density=None if d.density is None else d.density[:, -1],
                  ramp_counts=d.ramp_counts, ramp_cell=d.ramp_cell)
    inner = slice(1, d.counts.shape[0] - 1)  # boundary stations are inputs, not targets
    mc = res.station_counts[inner, window]
    oc = d.counts[inner, window]
    ms = res.station_speed[inner, window]
    os_ = d.speed[inner, window]
    ok = np.isfinite(os_)
    rmse = float(np.sqrt(np.mean((ms[ok] - os_[ok]) ** 2)))
    to_vph = 3600.0 / d.interval_s
    g = geh(mc * to_vph, oc * to_vph)
    span_h = mc.shape[1] * d.interval_s / 3600.0
    gw = geh(mc.sum(axis=1) / span_h, oc.sum(axis=1) / span_h)
    return Evaluation(rmse / 10.0 + 0.1 * float(g.mean()), rmse, float(g.mean()), float((g < 5.0).mean()),
                      float((gw < 5.0).mean()), res.station_counts, res.station_speed)


def calibrate(d: DetectorData, *, sampler: Literal["tpe", "random"], trials: int, seed: int = 0) -> CalibrationResult:
    train, _ = d.split()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    smp = optuna.samplers.TPESampler(seed=seed, n_startup_trials=15) if sampler == "tpe" else optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=smp)

    def objective(trial: optuna.Trial) -> float:
        fd = FundamentalDiagram(**{k: trial.suggest_float(k, *b) for k, b in BOUNDS.items()})
        return evaluate(fd, d, train).objective

    study.optimize(objective, n_trials=trials)
    vals = [t.value for t in study.trials if t.value is not None]
    return CalibrationResult(sampler, dict(study.best_params), float(study.best_value),
                             list(np.minimum.accumulate(vals)))
