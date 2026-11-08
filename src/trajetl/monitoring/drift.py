"""Distribution-drift monitor for streaming detector or probe speeds.

Each window is compared with a reference window using the two-sample Kolmogorov-Smirnov
test (Bonferroni-corrected across windows) and the Population Stability Index (PSI)
on reference-decile bins. A window is flagged when both agree (KS significant **and**
PSI > 0.25, the common "major shift" rule), which suppresses alarms from sample-size effects alone.
In this pipeline a flag means the calibrated model's operating regime has changed
(e.g. free flow -> congestion) and recalibration should be triggered.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass(frozen=True)
class DriftConfig:
    window_s: float = 30.0
    reference_s: float = 120.0
    alpha: float = 0.01
    psi_threshold: float = 0.25


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if edges.size < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(reference, edges)[0] / max(reference.size, 1)
    c = np.histogram(current, edges)[0] / max(current.size, 1)
    r = np.clip(r, 1e-4, None)
    c = np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def monitor(samples: pd.DataFrame, value: str = "speed", time: str = "t", cfg: DriftConfig | None = None) -> pd.DataFrame:
    """Return one row per window with KS statistic, adjusted p-value, PSI and drift flag."""
    cfg = cfg or DriftConfig()
    ref = samples.loc[samples[time] < cfg.reference_s, value].to_numpy(float)
    rest = samples[samples[time] >= cfg.reference_s]
    win = np.floor(rest[time] / cfg.window_s).astype(int)
    groups = list(rest.groupby(win))
    m = max(len(groups), 1)
    rows = []
    for w, g in groups:
        cur = g[value].to_numpy(float)
        ks = stats.ks_2samp(ref, cur)
        p_adj = min(1.0, float(ks.pvalue) * m)
        s = psi(ref, cur)
        rows.append({"window_start_s": w * cfg.window_s, "n": cur.size, "median": float(np.median(cur)),
                     "ks_D": float(ks.statistic), "p_bonferroni": p_adj, "psi": s,
                     "drift": bool(p_adj < cfg.alpha and s > cfg.psi_threshold)})
    return pd.DataFrame(rows)
