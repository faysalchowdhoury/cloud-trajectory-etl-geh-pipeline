"""cloud-trajectory-etl-geh-pipeline.

Real NGSIM trajectories -> PySpark ETL (clean, de-duplicate, SI units, virtual loop
detectors) -> lane-level HMM map matching of GPS-degraded probes -> Kalman/RTS smoothing ->
Optuna calibration of a cell transmission model against detector counts (FHWA GEH) ->
KS-test drift monitoring.
"""

__version__ = "1.0.0"
