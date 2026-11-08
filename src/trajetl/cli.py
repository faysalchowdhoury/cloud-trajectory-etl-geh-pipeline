"""``trajetl`` command line.

download   fetch an NGSIM window from data.transportation.gov (resumable)
run        full pipeline (Spark ETL, map matching, smoothing, calibration, drift, report)
serve      map-matching / calibration API
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.exceptions import PipelineError
from .core.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="trajetl", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--out", default="data/raw/i80_1600")
    d.add_argument("--location", default="i-80")
    d.add_argument("--t-start-ms", type=int, default=1113433135300)
    d.add_argument("--minutes", type=float, default=10.0)
    r = sub.add_parser("run")
    r.add_argument("--raw", default="data/raw/i80_1600")
    r.add_argument("--out", default="results/i80_1600")
    r.add_argument("--trials", type=int, default=150)
    r.add_argument("--spark-master", default="local[2]")
    s = sub.add_parser("serve")
    s.add_argument("--summary", default="results/i80_1600/summary.json")
    s.add_argument("--port", type=int, default=8000)
    a = p.parse_args(argv)
    configure_logging(a.log_level)
    try:
        if a.cmd == "download":
            from .ingest.ngsim import download

            pages = download(a.out, location=a.location, t_start_ms=a.t_start_ms, minutes=a.minutes)
            print(json.dumps({"pages": len(pages), "dir": a.out}))
        elif a.cmd == "run":
            from .pipeline import PipelineConfig, run
            from .report import write_report

            cfg = PipelineConfig(raw_dir=Path(a.raw), out_dir=Path(a.out), trials=a.trials, spark_master=a.spark_master)
            summary = run(cfg)
            md = write_report(summary, Path(a.out))
            print(md.read_text(encoding="utf-8"))
        else:
            import uvicorn

            from .api.app import create_app

            uvicorn.run(create_app(Path(a.summary)), host="127.0.0.1", port=a.port, log_level="warning")
    except PipelineError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
