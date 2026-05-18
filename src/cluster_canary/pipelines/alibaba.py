"""DVC entrypoint: download → harmonize → label the Alibaba 2018 trace.

Env-driven so DVC parametrization works:
- ALIBABA_RAW_DIR        (default: data/raw/alibaba)
- ALIBABA_INTERIM_DIR    (default: data/interim/alibaba)
- ALIBABA_LABELED_DIR    (default: data/interim/alibaba_labeled)
- LEAD_MINUTES           (default: 30)
- COOLDOWN_MINUTES       (default: 5)
- ALIBABA_FILES          (default: machine_meta,container_meta,machine_usage,container_usage,batch_task)
- ALIBABA_SKIP_DOWNLOAD  (default: 0; set to 1 to skip download if CSVs already extracted)

Skips download if already present + size-band-valid (see alibaba_ingest).
"""

from __future__ import annotations

import os
from pathlib import Path

from cluster_canary.data.alibaba_harmonize import harmonize_all
from cluster_canary.data.alibaba_ingest import (
    DEFAULT_FILES,
    FILE_BY_NAME,
    fetch,
)
from cluster_canary.data.alibaba_oom_detector import (
    DEFAULT_LABELED_DIR,
    generate_labels,
)
from cluster_canary.data.schema import (
    DEFAULT_COOLDOWN_MINUTES,
    DEFAULT_LEAD_MINUTES,
)


def main() -> int:
    """Run the full Alibaba ingest → harmonize → label pipeline."""
    import pandas as pd
    import structlog

    from cluster_canary.data.scraper import _configure_logging

    _configure_logging()
    log = structlog.get_logger(__name__)

    raw_dir = Path(os.environ.get("ALIBABA_RAW_DIR", "data/raw/alibaba"))
    interim_dir = Path(os.environ.get("ALIBABA_INTERIM_DIR", "data/interim/alibaba"))
    labeled_dir = Path(os.environ.get("ALIBABA_LABELED_DIR", str(DEFAULT_LABELED_DIR)))
    lead_min = int(os.environ.get("LEAD_MINUTES", str(DEFAULT_LEAD_MINUTES)))
    cool_min = int(os.environ.get("COOLDOWN_MINUTES", str(DEFAULT_COOLDOWN_MINUTES)))
    skip_download = os.environ.get("ALIBABA_SKIP_DOWNLOAD", "0") == "1"

    file_env = os.environ.get("ALIBABA_FILES", "").strip()
    files = tuple(FILE_BY_NAME[n] for n in file_env.split(",") if n) if file_env else DEFAULT_FILES

    if not skip_download:
        log.info("alibaba.pipeline.fetch.start", files=[f.name for f in files])
        fetch(files=files, raw_dir=raw_dir, extract=True)

    log.info("alibaba.pipeline.harmonize.start", raw_dir=str(raw_dir))
    harmonize_all(raw_dir=raw_dir, out_dir=interim_dir)

    log.info("alibaba.pipeline.label.start", interim_dir=str(interim_dir))
    parquets = sorted(interim_dir.glob("*.parquet"))
    long_df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    wide = generate_labels(long_df, lead_minutes=lead_min, cooldown_minutes=cool_min)

    labeled_dir.mkdir(parents=True, exist_ok=True)
    out = labeled_dir / "labeled.parquet"
    wide.to_parquet(out, index=False)
    log.info(
        "alibaba.pipeline.done",
        n_labeled_rows=len(wide),
        labeled_path=str(out),
        n_positive=int(wide["event_within_30min"].sum()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
