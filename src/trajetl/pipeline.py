"""End-to-end pipeline: ETL -> map matching -> smoothing -> calibration -> drift -> report."""

from __future__ import annotations

import json
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .calibration.ctm import FundamentalDiagram
from .calibration.optimize import DetectorData, calibrate, evaluate
from .core.logging import get_logger
from .mapmatch.hmm import LaneHMM, LaneMap, degrade_to_probe
from .monitoring.drift import monitor
from .smoothing.kalman import rts_smooth, tune_q

log = get_logger(__name__)


@dataclass(frozen=True)
class PipelineConfig:
    raw_dir: Path
    out_dir: Path
    stations_m: tuple[float, ...] = (30.0, 120.0, 210.0, 300.0, 390.0, 450.0)
    detector_interval_s: float = 60.0
    section_length_m: float = 480.0
    cell_m: float = 30.0
    sigmas_m: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
    probe_rate_hz: float = 1.0
    smoothing_vehicles: int = 300
    trials: int = 150
    seed: int = 7
    spark_master: str = "local[2]"
    timings: dict[str, float] = field(default_factory=dict)


def _is_test(vehicle_id: int, frac: float = 0.3) -> bool:
    return (zlib.crc32(str(vehicle_id).encode()) % 1000) < frac * 1000


def run(cfg: PipelineConfig) -> dict[str, Any]:
    from .spark.etl import build_session, clean, detectors, load_raw, quality_report, write_table

    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"timings_s": {}}

    # ---------------------------------------------------------------- 1. Spark ETL
    t0 = time.perf_counter()
    spark = build_session(master=cfg.spark_master)
    raw = load_raw(spark, cfg.raw_dir)
    cleaned = clean(raw).cache()
    qa = quality_report(raw, cleaned)
    det = detectors(cleaned, list(cfg.stations_m), cfg.detector_interval_s).cache()
    write_table(cleaned.drop("t_prev", "y_prev"), out / "parquet" / "trajectories", ["lane_id"])
    write_table(det, out / "parquet" / "detectors")
    traj = cleaned.select("vehicle_id", "frame_id", "t", "x", "y", "lane_id", "speed", "length").toPandas()
    det_pd = det.toPandas()
    spark.stop()
    summary["timings_s"]["spark_etl"] = time.perf_counter() - t0
    summary["quality"] = qa.as_dict()
    summary["data"] = {"vehicles": int(traj["vehicle_id"].nunique()), "rows": len(traj),
                       "duration_s": float(traj["t"].max() - traj["t"].min()),
                       "lanes": sorted(int(x) for x in traj["lane_id"].unique())}
    log.info("pipeline.etl", extra=summary["quality"])

    # ---------------------------------------------------------------- 2. map matching
    t0 = time.perf_counter()
    test_mask = traj["vehicle_id"].map(_is_test)
    train, test = traj[~test_mask], traj[test_mask]
    lane_map = LaneMap.from_trajectories(train)
    p_lc = LaneHMM.learn_lane_change_rate(train, 1.0 / cfg.probe_rate_hz)
    mm_rows = []
    for sigma in cfg.sigmas_m:
        probes = degrade_to_probe(test, rate_hz=cfg.probe_rate_hz, sigma_m=sigma, seed=cfg.seed)
        hmm = LaneHMM(lane_map, sigma_m=sigma, p_lane_change=p_lc)
        tt = time.perf_counter()
        probes["hmm"] = hmm.decode_frame(probes)
        dec_s = time.perf_counter() - tt
        probes["nearest"] = lane_map.nearest(probes["x_obs"].to_numpy(), probes["y_obs"].to_numpy())
        true_changes = probes.groupby("vehicle_id")["lane_id"].diff().fillna(0).ne(0)
        hmm_changes = probes.groupby("vehicle_id")["hmm"].diff().fillna(0).ne(0)
        near_changes = probes.groupby("vehicle_id")["nearest"].diff().fillna(0).ne(0)
        mm_rows.append({
            "sigma_m": sigma, "points": len(probes), "vehicles": int(probes["vehicle_id"].nunique()),
            "nearest_accuracy": float((probes["nearest"] == probes["lane_id"]).mean()),
            "hmm_accuracy": float((probes["hmm"] == probes["lane_id"]).mean()),
            "true_lane_changes": int(true_changes.sum()),
            "nearest_lane_changes": int(near_changes.sum()),
            "hmm_lane_changes": int(hmm_changes.sum()),
            "decode_ms_per_vehicle": 1000.0 * dec_s / max(probes["vehicle_id"].nunique(), 1),
        })
    summary["map_matching"] = {"p_lane_change_per_obs": p_lc, "lane_centres_m": dict(zip(lane_map.lanes.tolist(),
                               np.round(lane_map.centres, 2).tolist())), "results": mm_rows}
    summary["timings_s"]["map_matching"] = time.perf_counter() - t0

    # ---------------------------------------------------------------- 3. smoothing
    t0 = time.perf_counter()
    sigma_long = 1.0
    probes = degrade_to_probe(test, rate_hz=cfg.probe_rate_hz, sigma_m=sigma_long, seed=cfg.seed + 1)
    vids = probes.groupby("vehicle_id").size()
    vids = vids[vids >= 15].index.to_numpy()
    rng = np.random.default_rng(cfg.seed)
    chosen = rng.choice(vids, size=min(cfg.smoothing_vehicles, vids.size), replace=False)
    tune_ids = chosen[: max(20, chosen.size // 5)]
    eval_ids = chosen[max(20, chosen.size // 5):]
    qs = []
    for v in tune_ids:
        g = probes[probes["vehicle_id"] == v].sort_values("t")
        qs.append(tune_q(g["t"].to_numpy(), g["y_obs"].to_numpy(), sigma_long))
    q = float(np.median(qs))
    err_fd, err_kf, err_rts = [], [], []
    for v in eval_ids:
        g = probes[probes["vehicle_id"] == v].sort_values("t")
        t, y, truth = g["t"].to_numpy(), g["y_obs"].to_numpy(), g["speed"].to_numpy()
        sm = rts_smooth(t, y, sigma_m=sigma_long, q=q)
        fd = np.gradient(y, t)
        err_fd.append(fd - truth)
        err_kf.append(sm.filtered_speed - truth)
        err_rts.append(sm.speed - truth)
    rmse = lambda e: float(np.sqrt(np.mean(np.concatenate(e) ** 2)))  # noqa: E731
    summary["smoothing"] = {"probe_rate_hz": cfg.probe_rate_hz, "sigma_m": sigma_long, "q_jerk": q,
                            "vehicles_evaluated": int(eval_ids.size),
                            "speed_rmse_mps": {"finite_difference": rmse(err_fd), "kalman_filter": rmse(err_kf),
                                               "rts_smoother": rmse(err_rts)}}
    summary["timings_s"]["smoothing"] = time.perf_counter() - t0

    # ---------------------------------------------------------------- 4. calibration
    t0 = time.perf_counter()
    dd = detector_arrays(det_pd, traj, cfg, lane_map)
    train_w, val_w = dd.split()
    default_fd = FundamentalDiagram()
    cal = {s: calibrate(dd, sampler=s, trials=cfg.trials, seed=cfg.seed) for s in ("random", "tpe")}
    best = cal["tpe"] if cal["tpe"].best_value <= cal["random"].best_value else cal["random"]
    fd_best = FundamentalDiagram(**best.best_params)
    ev = {name: {"train": evaluate(fd, dd, train_w), "validation": evaluate(fd, dd, val_w)}
          for name, fd in (("default_hcm_like", default_fd), ("calibrated", fd_best))}
    summary["calibration"] = {
        "stations_m": list(cfg.stations_m), "interval_s": cfg.detector_interval_s,
        "intervals": int(dd.counts.shape[1]), "lanes_per_cell": dd.lanes_at.tolist(),
        "best_sampler": best.sampler, "best_params": best.best_params,
        "q_max_vphpl": fd_best.q_max * 3600.0, "default_q_max_vphpl": default_fd.q_max * 3600.0,
        "convergence": {s: cal[s].history for s in cal},
        "best_value": {s: cal[s].best_value for s in cal},
        "ramp_counts": dd.ramp_counts.tolist() if dd.ramp_counts is not None else None, "ramp_cell": dd.ramp_cell,
        "metrics": {name: {split: {"speed_rmse_mps": e.speed_rmse, "geh_mean": e.geh_mean, "geh_pass": e.geh_pass,
                                   "geh_window_pass": e.geh_window_pass}
                           for split, e in d.items()} for name, d in ev.items()},
    }
    summary["timings_s"]["calibration"] = time.perf_counter() - t0
    _plot_calibration(dd, ev, cal, out / "figures")

    # ---------------------------------------------------------------- 5. drift
    st = traj[(traj["y"] > 200) & (traj["y"] < 260)]
    drift = monitor(st[["t", "speed"]])
    drift.to_csv(out / "drift_windows.csv", index=False)
    summary["drift"] = {"windows": len(drift), "flagged": int(drift["drift"].sum()),
                        "first_flag_s": float(drift.loc[drift["drift"], "window_start_s"].min()) if drift["drift"].any() else None,
                        "reference_median_mps": float(st.loc[st["t"] < 120, "speed"].median())}
    summary["mm_table"] = mm_rows
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    pd.DataFrame(mm_rows).to_csv(out / "map_matching.csv", index=False)
    _plot_mapmatching(pd.DataFrame(mm_rows), out / "figures")
    return summary


def detector_arrays(det: pd.DataFrame, traj: pd.DataFrame, cfg: PipelineConfig, lane_map: LaneMap) -> DetectorData:
    """Aggregate lane-level detector rows to cross-section arrays and derive lanes per cell."""
    traj = traj.assign(lane_id=traj["lane_id"].astype(np.int64))  # partitioned Parquet returns a categorical
    d = det.copy()
    d["inv"] = d["count"] / d["space_mean_speed"]
    g = d.groupby(["station_y", "interval"]).agg(count=("count", "sum"), inv=("inv", "sum")).reset_index()
    g["sms"] = g["count"] / g["inv"]
    stations = np.array(sorted(g["station_y"].unique()))
    intervals = np.arange(int(g["interval"].min()), int(g["interval"].max()) + 1)
    # drop the partial first/last minute (vehicles still entering/leaving the recording)
    intervals = intervals[1:-1]
    counts = g.pivot(index="station_y", columns="interval", values="count").reindex(index=stations, columns=intervals).fillna(0.0)
    sms = g.pivot(index="station_y", columns="interval", values="sms").reindex(index=stations, columns=intervals)
    lanes_at = lanes_per_cell(traj, cfg.section_length_m, cfg.cell_m)
    n_cells = lanes_at.size
    # observed density per (interval, cell): samples are 10 Hz, so rows / (10 * interval) = mean vehicles present
    tt = traj[(traj["y"] >= 0) & (traj["y"] < cfg.section_length_m)]
    iv = np.floor(tt["t"].to_numpy() / cfg.detector_interval_s).astype(int)
    cc = (tt["y"].to_numpy() // cfg.cell_m).astype(int)
    occ = pd.crosstab(iv, cc).reindex(index=intervals, columns=range(n_cells), fill_value=0).to_numpy(float)
    density = occ / (10.0 * cfg.detector_interval_s) / cfg.cell_m
    # on-ramp entries: vehicles first seen in the auxiliary lane downstream of the first station
    first = traj.sort_values("t").groupby("vehicle_id").first()
    ramp = first[(first["lane_id"] == first["lane_id"].max()) & (first["y"] > stations[0] + 15.0)]
    ramp_iv = np.floor(ramp["t"].to_numpy() / cfg.detector_interval_s).astype(int)
    ramp_counts = pd.Series(ramp_iv).value_counts().reindex(intervals, fill_value=0).to_numpy(float)
    ramp_cell = int(np.median(ramp["y"].to_numpy() // cfg.cell_m)) if len(ramp) else 0
    return DetectorData(stations.astype(float), counts.to_numpy(float), sms.to_numpy(float), cfg.detector_interval_s,
                        cfg.section_length_m, lanes_at, density, ramp_counts, min(ramp_cell, n_cells - 1))


def lanes_per_cell(traj: pd.DataFrame, length_m: float, cell_m: float, min_share: float = 0.02) -> np.ndarray:
    """Lanes present in each cell, from where vehicles actually drive.

    A lane counts in a cell if it carries at least ``min_share`` of that cell's samples. Cells
    at the recording edges (where video coverage begins/ends and samples are sparse) inherit
    the lane count of their nearest well-observed neighbour, so the model is not given a
    phantom bottleneck at the boundary.
    """
    n_cells = int(round(length_m / cell_m))
    cell = np.clip((traj["y"].to_numpy() // cell_m).astype(int), 0, n_cells - 1)
    tab = pd.crosstab(cell, traj["lane_id"].to_numpy()).reindex(range(n_cells), fill_value=0)
    totals = tab.sum(axis=1)
    lanes = (tab.div(totals.replace(0, np.nan), axis=0) >= min_share).sum(axis=1).to_numpy(float)
    well = totals.to_numpy() >= 0.5 * np.median(totals.to_numpy())
    idx = np.flatnonzero(well)
    for i in range(n_cells):
        if not well[i] and idx.size:
            lanes[i] = lanes[idx[np.argmin(np.abs(idx - i))]]
    return np.maximum(lanes, 1.0)


def _plot_calibration(dd: DetectorData, ev: dict, cal: dict, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    inner = list(range(1, dd.station_y.size - 1))
    fig, axes = plt.subplots(1, len(inner) + 1, figsize=(4.2 * (len(inner) + 1), 3.6))
    t = np.arange(dd.counts.shape[1]) * dd.interval_s / 60.0
    cut = dd.split()[0].stop * dd.interval_s / 60.0
    for ax, s in zip(axes[:-1], inner):
        ax.plot(t, dd.speed[s] * 2.23694, "k.-", label="NGSIM (observed)")
        ax.plot(t, ev["default_hcm_like"]["validation"].model_speed[s] * 2.23694, color="#adb5bd", label="default FD")
        ax.plot(t, ev["calibrated"]["validation"].model_speed[s] * 2.23694, color="#1c7ed6", label="calibrated CTM")
        ax.axvline(cut, color="#e03131", ls=":", lw=1)
        ax.set_title(f"station y = {dd.station_y[s]:.0f} m", fontsize=9)
        ax.set_xlabel("minute")
        ax.set_ylabel("space-mean speed (mph)")
    axes[0].legend(fontsize=7)
    for s, r in cal.items():
        axes[-1].plot(np.arange(1, len(r.history) + 1), r.history, label=s.upper())
    axes[-1].set_title("calibration convergence (train)", fontsize=9)
    axes[-1].set_xlabel("trial")
    axes[-1].set_ylabel("best objective")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration.png", dpi=130)
    plt.close(fig)


def _plot_mapmatching(df: pd.DataFrame, fig_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.plot(df["sigma_m"], 100 * df["nearest_accuracy"], "o-", color="#adb5bd", label="nearest lane")
    ax.plot(df["sigma_m"], 100 * df["hmm_accuracy"], "o-", color="#1c7ed6", label="HMM (Viterbi)")
    ax.set_xlabel("GPS noise sigma (m)")
    ax.set_ylabel("lane accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "map_matching.png", dpi=130)
    plt.close(fig)
