"""Resumable, paginated download of USDOT NGSIM trajectories from data.transportation.gov.

Dataset: *Next Generation Simulation (NGSIM) Vehicle Trajectories and Supporting Data*,
Socrata id ``8ect-6jqj`` (US DOT, public domain). Pages are requested as CSV with an
explicit ``$order`` so offsets are stable, written as ``page_00000.csv`` and skipped on
re-run, so an interrupted download resumes where it stopped.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from ..core.exceptions import IngestError
from ..core.logging import get_logger

log = get_logger(__name__)

ENDPOINT = "https://data.transportation.gov/resource/8ect-6jqj.csv"
COLUMNS = (
    "vehicle_id", "frame_id", "global_time", "local_x", "local_y", "v_length", "v_width", "v_class",
    "v_vel", "v_acc", "lane_id", "preceding", "following", "space_headway", "location",
)
USER_AGENT = "trajetl/1.0 (research pipeline)"


def build_query(location: str, t_start_ms: int, t_end_ms: int, limit: int, offset: int) -> str:
    params = {
        "$select": ",".join(COLUMNS),
        "$where": f"location='{location}' AND global_time >= {t_start_ms} AND global_time < {t_end_ms}",
        "$order": "global_time,vehicle_id",
        "$limit": str(limit),
        "$offset": str(offset),
    }
    return ENDPOINT + "?" + urllib.parse.urlencode(params)


def _fetch(url: str, retries: int = 4, timeout_s: float = 180.0) -> bytes:
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return resp.read()
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == retries:
                raise IngestError(f"download failed: {exc}", context={"url": url[:120]}) from exc
            wait = 2.0 ** attempt
            log.warning("ngsim.retry", extra={"attempt": attempt + 1, "wait_s": wait, "error": str(exc)})
            time.sleep(wait)
    raise IngestError("unreachable")


def download(
    out_dir: str | Path,
    *,
    location: str = "i-80",
    t_start_ms: int = 1113433135300,
    minutes: float = 10.0,
    page_size: int = 50_000,
    max_pages: int = 200,
) -> list[Path]:
    """Download one time window; returns the page files (existing pages are reused)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t_end_ms = int(t_start_ms + minutes * 60_000)
    pages: list[Path] = []
    for i in range(max_pages):
        path = out / f"page_{i:05d}.csv"
        if path.is_file() and path.stat().st_size > 0:
            pages.append(path)
            n_lines = sum(1 for _ in path.open("rb")) - 1
            if n_lines < page_size:
                break
            continue
        body = _fetch(build_query(location, t_start_ms, t_end_ms, page_size, i * page_size))
        n_lines = body.count(b"\n") - 1
        if n_lines <= 0:
            break
        tmp = path.with_suffix(".part")
        tmp.write_bytes(body)
        tmp.replace(path)
        pages.append(path)
        log.info("ngsim.page", extra={"page": i, "rows": n_lines, "bytes": len(body)})
        if n_lines < page_size:
            break
    else:
        raise IngestError("max_pages reached; increase max_pages or shorten the window")
    return pages
