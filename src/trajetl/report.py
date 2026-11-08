"""Markdown report for a pipeline run."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def write_report(s: dict[str, Any], out_dir: Path) -> Path:
    q, d, mm, sm, c, dr = s["quality"], s["data"], s["map_matching"], s["smoothing"], s["calibration"], s["drift"]
    L = ["# Pipeline results: NGSIM I-80, 4:00 pm window", ""]
    L += ["## Data and ETL quality", "", "| Item | Value |", "|---|---|",
          f"| Raw rows downloaded | {q['raw_rows']:,} |", f"| Duplicate (vehicle, frame) rows removed | {q['duplicate_rows']:,} |",
          f"| Clean rows | {q['clean_rows']:,} |", f"| Vehicles | {q['vehicles']:,} |",
          f"| Rows with implausible finite-difference acceleration (>10 m/s²) | {q['implausible_accel_rows']:,} |",
          f"| Rows moving backwards (> 0.5 m/s) | {q['negative_speed_rows']:,} |",
          f"| Duration covered | {d['duration_s'] / 60:.1f} min |", f"| Lanes | {d['lanes']} |", ""]
    L += ["## Lane-level map matching (held-out 30% of vehicles, 1 Hz probes)", "",
          f"Lane-change probability per observation learned on the training vehicles: {mm['p_lane_change_per_obs']:.4f}.", "",
          "| GPS sigma | Points | Nearest-lane accuracy | HMM accuracy | True lane changes | Nearest-lane changes | HMM changes | Decode time |",
          "|---|---|---|---|---|---|---|---|"]
    for r in mm["results"]:
        L.append(f"| {r['sigma_m']:.1f} m | {r['points']:,} | {100 * r['nearest_accuracy']:.1f}% | {100 * r['hmm_accuracy']:.1f}% | "
                 f"{r['true_lane_changes']} | {r['nearest_lane_changes']} | {r['hmm_lane_changes']} | {r['decode_ms_per_vehicle']:.2f} ms/veh |")
    rm = sm["speed_rmse_mps"]
    L += ["", f"## Speed reconstruction ({sm['vehicles_evaluated']} held-out vehicles, 1 Hz, sigma {sm['sigma_m']} m)", "",
          f"Jerk spectral density q = {sm['q_jerk']:.3g} selected by innovation log-likelihood on separate tuning vehicles.", "",
          "| Estimator | Speed RMSE vs NGSIM speed |", "|---|---|",
          f"| Finite difference | {rm['finite_difference']:.2f} m/s |", f"| Kalman filter (causal) | {rm['kalman_filter']:.2f} m/s |",
          f"| RTS smoother | {rm['rts_smoother']:.2f} m/s |", ""]
    L += ["## CTM calibration (train = first half, validation = second half of the window)", "",
          f"Stations {c['stations_m']} m, {c['interval_s']:.0f}-s intervals ({c['intervals']} intervals). Lanes per 30-m cell: "
          f"{[int(x) for x in c['lanes_per_cell']]}. Best sampler: **{c['best_sampler'].upper()}**.", "",
          "| Parameter | Default (HCM-like) | Calibrated |", "|---|---|---|"]
    bp = c["best_params"]
    L += [f"| Free-flow speed vf | 29.0 m/s | {bp['vf']:.1f} m/s |", f"| Wave speed w | 5.4 m/s | {bp['w']:.2f} m/s |",
          f"| Jam density kj | 0.125 veh/m/lane | {bp['kj']:.3f} veh/m/lane |", f"| Capacity factor | 1.00 | {bp['cap_factor']:.2f} |",
          f"| Capacity q_max | {c['default_q_max_vphpl']:.0f} vphpl | {c['q_max_vphpl']:.0f} vphpl |", "",
          "| Model | Split | Speed RMSE | Mean GEH (1-min rates) | 1-min intervals GEH < 5 | Stations GEH < 5 (window totals) |",
          "|---|---|---|---|---|---|"]
    for name, label in (("default_hcm_like", "Default FD"), ("calibrated", "Calibrated")):
        for split in ("train", "validation"):
            m = c["metrics"][name][split]
            L.append(f"| {label} | {split} | {m['speed_rmse_mps']:.2f} m/s ({m['speed_rmse_mps'] * 2.23694:.1f} mph) | "
                     f"{m['geh_mean']:.2f} | {100 * m['geh_pass']:.0f}% | {100 * m['geh_window_pass']:.0f}% |")
    bv = c["best_value"]
    L += ["", f"Best training objective after {len(c['convergence']['tpe'])} trials: TPE {bv['tpe']:.4f}, random {bv['random']:.4f}.", ""]
    L += ["## Drift monitoring (speeds at y = 200-260 m, 30-s windows vs. first 2 min)", "",
          f"Reference median speed {dr['reference_median_mps']:.1f} m/s; {dr['flagged']} of {dr['windows']} windows flagged "
          f"(KS p < 0.01 after Bonferroni and PSI > 0.25); first flag at t = {dr['first_flag_s']} s.", ""]
    t = s["timings_s"]
    L += ["## Runtime", "", "| Stage | Seconds |", "|---|---|"] + [f"| {k} | {v:.1f} |" for k, v in t.items()]
    L += ["", "![Calibration](figures/calibration.png)", "", "![Map matching](figures/map_matching.png)", ""]
    md = out_dir / "RESULTS.md"
    md.write_text("\n".join(L), encoding="utf-8")
    return md
