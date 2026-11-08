"""Spark ETL tests on a tiny synthetic NGSIM-format page (requires Java)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("pyspark")
if shutil.which("java") is None:
    pytest.skip("Java runtime not available", allow_module_level=True)

from trajetl.spark.etl import FT, build_session, clean, detectors, load_raw, quality_report  # noqa: E402

HEADER = "vehicle_id,frame_id,global_time,local_x,local_y,v_length,v_width,v_class,v_vel,v_acc,lane_id,preceding,following,space_headway,location\n"


@pytest.fixture(scope="module")
def spark():  # type: ignore[no-untyped-def]
    s = build_session(master="local[1]", driver_memory="1g")
    yield s
    s.stop()


def write_page(tmp: Path) -> Path:
    lines = []
    for v, lane, speed_ftps in ((1, 1, 50.0), (2, 2, 30.0)):
        for f in range(100):
            y = 10.0 + speed_ftps * f * 0.1
            lines.append(f"{v},{f + 1},{1_000_000 + f * 100},{6.0 + 12 * (lane - 1)},{y},15,6,2,{speed_ftps},0,{lane},0,0,0,i-80\n")
    lines.append(lines[5])  # exact duplicate row, as found in the public NGSIM files
    d = tmp / "raw"
    d.mkdir()
    (d / "page_00000.csv").write_text(HEADER + "".join(lines))
    return d


def test_clean_dedup_units_and_detectors(spark, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    raw = load_raw(spark, write_page(tmp_path))
    cl = clean(raw)
    qa = quality_report(raw, cl)
    assert qa.raw_rows == 201 and qa.duplicate_rows == 1 and qa.clean_rows == 200 and qa.vehicles == 2
    assert qa.implausible_accel_rows == 0
    pdf = cl.toPandas()
    v1 = pdf[pdf.vehicle_id == 1].sort_values("frame_id")
    assert v1["speed"].iloc[0] == pytest.approx(50 * FT)
    assert v1["speed_fd"].dropna().mean() == pytest.approx(50 * FT, rel=1e-6)
    assert v1["t"].iloc[1] == pytest.approx(0.1)
    det = detectors(cl, [100 * FT], interval_s=30.0).toPandas()
    assert det["count"].sum() == 2  # both vehicles cross the station once
    assert set(det["lane_id"]) == {1, 2}
