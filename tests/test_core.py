from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from trajetl.api.app import create_app
from trajetl.calibration.ctm import FundamentalDiagram, geh, run_ctm
from trajetl.calibration.optimize import DetectorData, calibrate, evaluate
from trajetl.core.exceptions import ConfigurationError
from trajetl.ingest.ngsim import build_query
from trajetl.mapmatch.hmm import LaneHMM, LaneMap, degrade_to_probe
from trajetl.monitoring.drift import monitor, psi
from trajetl.smoothing.kalman import rts_smooth, tune_q


def synthetic_lanes(n_veh: int = 60, seed: int = 0) -> pd.DataFrame:
    """Vehicles at 10 Hz on 3 lanes (centres 1.8, 5.4, 9.0 m) with occasional lane changes."""
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_veh):
        lane = int(rng.integers(1, 4))
        speed = rng.uniform(8, 20)
        y0 = rng.uniform(0, 20)
        change_at = rng.uniform(5, 25) if rng.random() < 0.4 else np.inf
        for i in range(300):
            t = round(i * 0.1, 1)
            if t >= change_at:
                lane = int(np.clip(lane + rng.choice([-1, 1]), 1, 3))
                change_at = np.inf
            rows.append((v, t, 1.8 + 3.6 * (lane - 1), y0 + speed * t, lane, speed))
    return pd.DataFrame(rows, columns=["vehicle_id", "t", "x", "y", "lane_id", "speed"])


def test_query_is_ordered_and_filtered() -> None:
    q = build_query("i-80", 1, 2, 50000, 100000)
    assert "%24order=global_time%2Cvehicle_id" in q and "%24offset=100000" in q and "i-80" in q


def test_hmm_beats_nearest_lane_under_noise() -> None:
    df = synthetic_lanes()
    lm = LaneMap.from_trajectories(df)
    np.testing.assert_allclose(lm.centres, [1.8, 5.4, 9.0], atol=1e-9)
    p_lc = LaneHMM.learn_lane_change_rate(df, 1.0)
    assert 0 < p_lc < 0.1
    probes = degrade_to_probe(df, rate_hz=1.0, sigma_m=1.5, seed=1)
    assert len(probes) == 60 * 30
    hmm = LaneHMM(lm, sigma_m=1.5, p_lane_change=p_lc)
    pred = hmm.decode_frame(probes)
    near = lm.nearest(probes["x_obs"].to_numpy(), probes["y_obs"].to_numpy())
    acc_hmm = (pred == probes["lane_id"]).mean()
    acc_near = (near == probes["lane_id"]).mean()
    assert acc_hmm > acc_near + 0.1 and acc_hmm > 0.9


def test_viterbi_exact_on_clean_data() -> None:
    lm = LaneMap(np.array([1, 2, 3]), np.array([0.0, 3.6, 7.2]), np.full(3, -1e9), np.full(3, 1e9))
    x = np.array([0.0, 0.1, 3.5, 3.7, 7.1])
    assert LaneHMM(lm, sigma_m=0.5).decode(x, np.zeros(5)).tolist() == [1, 1, 2, 2, 3]


def test_rts_recovers_constant_acceleration() -> None:
    t = np.arange(0, 30, 1.0)
    y_true = 5 + 10 * t + 0.25 * t**2
    v_true = 10 + 0.5 * t
    rng = np.random.default_rng(3)
    y = y_true + rng.normal(0, 1.0, t.size)
    q = tune_q(t, y, 1.0)
    sm = rts_smooth(t, y, sigma_m=1.0, q=q)
    fd = np.gradient(y, t)
    assert np.sqrt(np.mean((sm.speed - v_true) ** 2)) < 0.5 * np.sqrt(np.mean((fd - v_true) ** 2))
    with pytest.raises(ValueError):
        rts_smooth(t[:1], y[:1], sigma_m=1.0)


def test_ctm_conservation_and_free_flow() -> None:
    fd = FundamentalDiagram()
    n_int = 20
    inflow = np.full(n_int, 40.0)  # 40 veh/min = 2400 vph on 3 lanes: uncongested
    res = run_ctm(fd, length_m=300, lanes_at=np.full(10, 3.0), inflow_counts=inflow, outflow_cap_counts=np.full(n_int, 1e6),
                  station_y=np.array([90.0, 210.0]), interval_s=60.0)
    assert res.station_counts[:, 5:].mean() == pytest.approx(40.0, rel=0.02)
    assert res.station_speed[:, 5:].mean() == pytest.approx(fd.vf, rel=0.02)
    jam = run_ctm(fd, length_m=300, lanes_at=np.full(10, 3.0), inflow_counts=inflow, outflow_cap_counts=np.full(n_int, 15.0),
                  station_y=np.array([90.0, 210.0]), interval_s=60.0)
    assert jam.station_speed[:, -5:].mean() < 0.5 * fd.vf  # downstream bottleneck -> queue spills back
    with pytest.raises(ConfigurationError):
        run_ctm(FundamentalDiagram(vf=40), length_m=300, lanes_at=np.full(10, 3.0), inflow_counts=inflow,
                outflow_cap_counts=inflow, station_y=np.array([90.0]), interval_s=60.0)


def test_calibration_recovers_known_parameters() -> None:
    truth = FundamentalDiagram(vf=24.0, w=4.5, kj=0.15, cap_factor=0.9)
    n_int = 16
    inflow = np.r_[np.full(8, 70.0), np.full(8, 90.0)]
    cap = np.r_[np.full(8, 80.0), np.full(8, 60.0)]
    stations = np.array([30.0, 150.0, 270.0, 390.0])
    lanes = np.full(14, 3.0)
    r = run_ctm(truth, length_m=420, lanes_at=lanes, inflow_counts=inflow, outflow_cap_counts=cap, station_y=stations,
                interval_s=60.0)
    counts = r.station_counts.copy()
    counts[0], counts[-1] = inflow, cap
    dd = DetectorData(stations, counts, r.station_speed, 60.0, 420.0, lanes)
    res = calibrate(dd, sampler="tpe", trials=60, seed=1)
    fd_hat = FundamentalDiagram(**res.best_params)
    assert evaluate(fd_hat, dd, slice(0, n_int)).speed_rmse < evaluate(FundamentalDiagram(), dd, slice(0, n_int)).speed_rmse
    assert res.history == sorted(res.history, reverse=True)


def test_lanes_per_cell_ignores_sparse_edges() -> None:
    from trajetl.pipeline import lanes_per_cell

    rng = np.random.default_rng(0)
    rows = []
    for lane in range(1, 4):
        y = rng.uniform(25, 300, 3000)  # recording starts at ~25 m: first cell is sparse
        rows.append(pd.DataFrame({"y": y, "lane_id": lane}))
    rows.append(pd.DataFrame({"y": rng.uniform(25, 120, 1500), "lane_id": 4}))  # auxiliary lane ends at 120 m
    lanes = lanes_per_cell(pd.concat(rows), 300.0, 30.0)
    assert lanes[0] == 4 and lanes[1] == 4 and lanes[-1] == 3


def test_geh_values() -> None:
    assert geh(np.array([1000.0]), np.array([900.0]))[0] == pytest.approx(np.sqrt(2 * 100**2 / 1900))
    assert geh(np.array([0.0]), np.array([0.0]))[0] == 0.0


def test_drift_detects_regime_change() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(0, 600, 0.5)
    v = np.where(t < 300, rng.normal(25, 2, t.size), rng.normal(8, 3, t.size))
    out = monitor(pd.DataFrame({"t": t, "speed": v}))
    assert not out.loc[out["window_start_s"] < 300, "drift"].any()
    assert out.loc[out["window_start_s"] >= 300, "drift"].all()
    assert psi(v[:200], v[:200]) == pytest.approx(0.0, abs=1e-9)


def test_api_smooth_and_validation() -> None:
    c = TestClient(create_app(None))
    assert c.get("/health").json()["summary_loaded"] is False
    t = list(range(20))
    r = c.post("/v1/smooth", json={"t": t, "x": [0.0] * 20, "y": [10.0 * i for i in t]})
    assert r.status_code == 200 and abs(np.mean(r.json()["speed_mps"]) - 10.0) < 0.5
    assert c.post("/v1/smooth", json={"t": [0, 0], "x": [0, 0], "y": [0, 1]}).status_code == 422
    assert c.post("/v1/lanes/match", json={"t": t, "x": [0.0] * 20, "y": t}).status_code == 503
