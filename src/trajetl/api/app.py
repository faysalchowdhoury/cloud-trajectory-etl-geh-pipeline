"""API: online lane matching of probe traces and access to the calibrated model.

``POST /v1/lanes/match``  probe trace (t, x, y) -> lane sequence, HMM with the trained lane map
``POST /v1/smooth``       probe trace -> RTS-smoothed speed
``GET  /v1/calibration``  calibrated fundamental diagram + GEH/speed metrics
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, model_validator

from .. import __version__
from ..mapmatch.hmm import LaneHMM, LaneMap
from ..smoothing.kalman import rts_smooth


class Trace(BaseModel):
    t: list[float] = Field(min_length=2, max_length=20_000)
    x: list[float]
    y: list[float]
    sigma_m: float = Field(1.0, gt=0.0, le=20.0)

    @model_validator(mode="after")
    def _same_len(self) -> "Trace":
        if not (len(self.t) == len(self.x) == len(self.y)):
            raise ValueError("t, x, y must have equal length")
        if any(b <= a for a, b in zip(self.t, self.t[1:])):
            raise ValueError("t must be strictly increasing")
        return self


def create_app(summary_path: Path | None = None) -> FastAPI:
    app = FastAPI(title="trajetl", version=__version__)
    summary = json.loads(summary_path.read_text()) if summary_path and summary_path.is_file() else None

    def lane_map() -> tuple[LaneMap, float]:
        if summary is None:
            raise HTTPException(503, "no pipeline summary loaded; run `trajetl run` first")
        mm = summary["map_matching"]
        lanes = np.array(sorted(int(k) for k in mm["lane_centres_m"]))
        centres = np.array([mm["lane_centres_m"][str(k)] if str(k) in mm["lane_centres_m"] else mm["lane_centres_m"][k]
                            for k in lanes], float)
        big = np.full(lanes.size, 1e9)
        return LaneMap(lanes, centres, -big, big), float(mm["p_lane_change_per_obs"])

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "version": __version__, "summary_loaded": summary is not None}

    @app.post("/v1/lanes/match")
    def match(trace: Trace) -> dict[str, list[int]]:
        lm, p = lane_map()
        hmm = LaneHMM(lm, sigma_m=trace.sigma_m, p_lane_change=p)
        return {"lanes": hmm.decode(np.asarray(trace.x), np.asarray(trace.y)).tolist()}

    @app.post("/v1/smooth")
    def smooth(trace: Trace) -> dict[str, list[float]]:
        r = rts_smooth(np.asarray(trace.t), np.asarray(trace.y), sigma_m=trace.sigma_m)
        return {"speed_mps": np.round(r.speed, 3).tolist(), "position_m": np.round(r.position, 3).tolist()}

    @app.get("/v1/calibration")
    def calibration() -> dict[str, object]:
        if summary is None:
            raise HTTPException(503, "no pipeline summary loaded")
        c = summary["calibration"]
        return {"params": c["best_params"], "q_max_vphpl": c["q_max_vphpl"], "metrics": c["metrics"]}

    return app
