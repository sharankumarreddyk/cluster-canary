"""DVC entrypoint: scrape Prometheus → parquet, then run label generator.

Env-driven (so DVC parametrization works):
- PROMETHEUS_URL    (default: http://localhost:9090)
- SCRAPE_START      (default: now - 24h)
- SCRAPE_END        (default: now)
- SCRAPE_STEP       (default: 15s)
- LEAD_MINUTES      (default: 30)
- COOLDOWN_MINUTES  (default: 5)
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from cluster_canary.data.label_generator import generate_labels
from cluster_canary.data.schema import (
    DEFAULT_COOLDOWN_MINUTES,
    DEFAULT_LEAD_MINUTES,
    DEFAULT_SCRAPE_STEP_SEC,
)
from cluster_canary.data.scraper import (
    _configure_logging,
    _parse_dt,
    scrape_window,
)


def main() -> int:
    """Run a 24 h Prometheus scrape and emit a labeled parquet."""
    _configure_logging()
    import pandas as pd
    import structlog

    log = structlog.get_logger(__name__)

    prom_url = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
    step = os.environ.get("SCRAPE_STEP", f"{DEFAULT_SCRAPE_STEP_SEC}s")
    lead_min = int(os.environ.get("LEAD_MINUTES", str(DEFAULT_LEAD_MINUTES)))
    cool_min = int(os.environ.get("COOLDOWN_MINUTES", str(DEFAULT_COOLDOWN_MINUTES)))

    end = (
        _parse_dt(os.environ["SCRAPE_END"]) if "SCRAPE_END" in os.environ else datetime.now(tz=UTC)
    )
    start = (
        _parse_dt(os.environ["SCRAPE_START"])
        if "SCRAPE_START" in os.environ
        else end - timedelta(hours=24)
    )

    raw_dir = Path("data/raw/scrape")
    labeled_dir = Path("data/interim/labeled")

    written = scrape_window(
        prom_url=prom_url,
        start=start,
        end=end,
        step=step,
        out_dir=raw_dir,
    )
    if not written:
        log.error("pipeline.scrape.empty")
        return 2

    parquets = sorted(raw_dir.rglob("*.parquet"))
    long_df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    wide = generate_labels(long_df, lead_minutes=lead_min, cooldown_minutes=cool_min)

    labeled_dir.mkdir(parents=True, exist_ok=True)
    out = labeled_dir / "labeled.parquet"
    wide.to_parquet(out, index=False)
    log.info(
        "pipeline.done",
        scrape_partitions=len(written),
        labeled_rows=len(wide),
        labeled_path=str(out),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
