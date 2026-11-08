My traffic model passed the GEH check at 100%. So did the uncalibrated one.

I built a pipeline on real USDOT NGSIM trajectories (I-80, 703,742 rows, 1,314 vehicles): PySpark ETL → virtual loop detectors → lane-level map matching → Kalman smoothing → Optuna calibration of a cell transmission model → drift monitoring.

What the data showed:

1️⃣ GEH alone can't validate a model. On window totals, both the default and the calibrated model passed GEH at 100%, because conservation and boundary counts pin the volumes. Speeds told them apart: default 20.1 m/s error, calibrated 3.0 m/s on held-out minutes. That's why FHWA asks for volumes and speeds.

2️⃣ The calibrated parameters make physical sense: a 13 km/h congestion wave and a queue-discharge rate around 1,350 veh/h/lane, consistent with the capacity drop. The defaults predicted 65 mph on a road moving at 20.

3️⃣ My model still fails the 1-minute GEH test on validation (44% vs. the 85% target). A first-order CTM can't produce stop-and-go waves, and the drift monitor flagged 10 of 16 windows as regime shifts. That's a model-class limit, and I'd rather report it than tune it away.

4️⃣ Probe data needs map matching. With 2 m GPS noise, nearest-lane assignment got 68% of points right and invented 9,257 lane changes (171 real ones). An HMM with a learned lane-change rate held 98.3% and found 197.

5️⃣ Real-time ≠ offline. A causal Kalman filter was worse than finite differences in stop-and-go traffic (1.28 vs 0.99 m/s). The RTS smoother cut error to 0.86.

Stack: PySpark 3.5, Optuna, NumPy, FastAPI, Docker, and 11 pytest cases including a real local Spark job. On Windows, Spark needs Python 3.11, a distutils shim and an Arrow-based Parquet writer.

Repo: [link]

Modelers: for congested freeways, what's your go-to when first-order models flatten the oscillations?

#TrafficEngineering #TransportationData #PySpark #MachineLearning #DataEngineering #NGSIM #Calibration #ITS
