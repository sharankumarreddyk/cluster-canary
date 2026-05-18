"""Download + extract the Alibaba Cluster Trace 2018 sample.

The trace is hosted as 6 tar.gz files on Alibaba's OSS bucket. For the
cluster-canary OOM use case we fetch 5 of them (skipping `batch_instance`
which is 20 GB and outside Phase 2's scope):

    machine_meta     (~92 KB)   machine state events
    container_meta   (~2.4 MB)  container state events — CRITICAL for OOM signal
    machine_usage    (~1.7 GB)  per-machine resource usage
    container_usage  (~28 GB)   per-container metrics — main feature source
    batch_task       (~125 MB)  job/task hierarchy

Total ~30 GB compressed, ~180 GB extracted. The download is bandwidth-bound;
budget a couple of hours on a home connection.

This module is built to be:
- **Idempotent**: re-running skips files that are already on disk and within the
  expected size band (configurable per-file).
- **Resumable**: a partial `<file>.tar.gz.part` is left on disk if download fails;
  the next run picks up via HTTP Range from the existing byte offset.
- **Sanity-bounded**: each file has a (min_mib, max_mib) bracket; out-of-band
  sizes fail loudly so we don't silently train on a truncated download.

Usage:
    python -m cluster_canary.data.alibaba_ingest
    python -m cluster_canary.data.alibaba_ingest --only container_meta machine_usage
    ALIBABA_RAW_DIR=/path/to/raw python -m cluster_canary.data.alibaba_ingest
"""

from __future__ import annotations

import argparse
import os
import shutil
import tarfile
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import structlog
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = structlog.get_logger(__name__)

OSS_BASE_URL = "https://aliopentrace.oss-cn-beijing.aliyuncs.com/v2018Traces"
DEFAULT_RAW_DIR = Path("data/raw/alibaba")

_DOWNLOAD_TIMEOUT = 600  # seconds — large files; this is per-read, not total
_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MiB streaming chunks


class AlibabaIngestError(RuntimeError):
    """Raised when an Alibaba data fetch cannot complete after retries."""


@dataclass(frozen=True)
class TraceFile:
    """One downloadable file in the Alibaba trace bundle."""

    name: str                                 # e.g. "container_meta"
    size_mib_band: tuple[float, float]        # (min, max) expected compressed size in MiB
    description: str

    @property
    def url(self) -> str:
        return f"{OSS_BASE_URL}/{self.name}.tar.gz"

    @property
    def archive_name(self) -> str:
        return f"{self.name}.tar.gz"


# Default fetch set for the OOM use case. `batch_instance` (~20 GB) excluded
# until we start modeling batch failures.
#
# Size bands are ±10 % of the live-verified Content-Length on the OSS bucket
# (HEAD-checked 2026-05-18 against re-uploaded 2023-02 objects). Out-of-band
# triggers AlibabaIngestError so we never silently train on a truncated file.
DEFAULT_FILES: tuple[TraceFile, ...] = (
    TraceFile(
        "machine_meta", (0.07, 0.12),
        "Machine state events (ADD/REMOVE/UPDATE). ~90 KB.",
    ),
    TraceFile(
        "container_meta", (2.0, 3.0),
        "Container state events — only container-level status signal. ~2.4 MB.",
    ),
    TraceFile(
        "machine_usage", (1500.0, 1850.0),
        "Per-machine resource usage; 10-60s non-uniform sampling. ~1.65 GB.",
    ),
    TraceFile(
        "container_usage", (27_000.0, 28_500.0),
        "Per-container metrics — primary feature source. ~27.2 GB.",
    ),
    TraceFile(
        "batch_task", (110.0, 140.0),
        "Job/task hierarchy. ~124 MB.",
    ),
)
FILE_BY_NAME: dict[str, TraceFile] = {f.name: f for f in DEFAULT_FILES}


def _size_mib(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def _looks_complete(archive: Path, expected_band: tuple[float, float]) -> bool:
    """True if `archive` exists and its size falls inside the expected band."""
    if not archive.exists():
        return False
    size = _size_mib(archive)
    lo, hi = expected_band
    return lo <= size <= hi


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    retry=retry_if_exception_type((urllib.error.URLError, TimeoutError, OSError)),
    reraise=True,
    before_sleep=before_sleep_log(log, "warning"),  # type: ignore[arg-type]
)
def _stream_download(url: str, dest: Path, resume_from: int = 0) -> int:
    """Stream `url` to `dest.part`, resuming from `resume_from` bytes.

    Returns total bytes on disk after this call (sum of resume_from + bytes written).
    Raises after exhausting retries.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)

    request = urllib.request.Request(url)
    mode = "ab" if resume_from > 0 else "wb"
    if resume_from > 0:
        request.add_header("Range", f"bytes={resume_from}-")
        log.info("download.resume", url=url, resume_from_mib=round(resume_from / 1024 / 1024, 1))
    else:
        log.info("download.start", url=url, dest=str(dest))

    with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT) as response:  # noqa: S310 — pinned http URL
        if response.status not in (200, 206):
            raise AlibabaIngestError(f"HTTP {response.status} for {url}")
        with part.open(mode) as fh:
            shutil.copyfileobj(response, fh, length=_CHUNK_SIZE)

    final_bytes = part.stat().st_size
    log.info(
        "download.done",
        url=url,
        size_mib=round(final_bytes / 1024 / 1024, 1),
    )
    return final_bytes


def download_one(spec: TraceFile, raw_dir: Path, *, force: bool = False) -> Path:
    """Download one TraceFile if not already present (idempotent + resumable).

    Returns the path to the on-disk archive. Raises `AlibabaIngestError` on
    sanity-bound violations or after retries are exhausted.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / spec.archive_name

    if archive.exists() and not force and _looks_complete(archive, spec.size_mib_band):
        log.info(
            "download.skip",
            file=spec.name,
            reason="already_present",
            size_mib=round(_size_mib(archive), 1),
        )
        return archive

    # Resume from any existing .part file.
    part = archive.with_suffix(archive.suffix + ".part")
    resume_from = part.stat().st_size if part.exists() else 0
    _stream_download(spec.url, archive, resume_from=resume_from)
    part.replace(archive)

    size = _size_mib(archive)
    lo, hi = spec.size_mib_band
    if not lo <= size <= hi:
        raise AlibabaIngestError(
            f"{spec.name}: downloaded size {size:.1f} MiB outside expected band "
            f"[{lo}, {hi}] — possible corruption or schema change"
        )
    return archive


def extract_one(archive: Path, *, into: Path) -> list[Path]:
    """Extract a `.tar.gz` archive into `into/`. Returns the list of extracted files.

    Idempotent: if every member already exists at the target size, skips.
    """
    into.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, mode="r:gz") as tar:
        members = [m for m in tar.getmembers() if m.isfile()]
        targets = [into / m.name for m in members]

        if all(
            t.exists() and t.stat().st_size == m.size
            for t, m in zip(targets, members, strict=True)
        ):
            log.info("extract.skip", archive=archive.name, reason="all_targets_present")
            return targets

        # Use `data` filter (PEP 706) — safe extraction, no symlink/abs-path attacks.
        tar.extractall(into, filter="data")
        log.info("extract.done", archive=archive.name, n_members=len(members))
        return targets


def fetch(
    files: Iterable[TraceFile] = DEFAULT_FILES,
    raw_dir: Path = DEFAULT_RAW_DIR,
    *,
    force: bool = False,
    extract: bool = True,
) -> dict[str, list[Path]]:
    """Download + (optionally) extract every file in `files`.

    Returns `{trace_name: [extracted_paths]}` for downstream consumers.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[Path]] = {}
    for spec in files:
        archive = download_one(spec, raw_dir, force=force)
        if extract:
            result[spec.name] = extract_one(archive, into=raw_dir)
        else:
            result[spec.name] = [archive]
    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cluster_canary.data.alibaba_ingest", description=__doc__)
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=Path(os.environ.get("ALIBABA_RAW_DIR", str(DEFAULT_RAW_DIR))),
    )
    p.add_argument(
        "--only",
        nargs="+",
        choices=sorted(FILE_BY_NAME),
        help=f"Subset of files to fetch. Default: {[f.name for f in DEFAULT_FILES]}",
    )
    p.add_argument("--force", action="store_true", help="Re-download even if archive looks complete")
    p.add_argument(
        "--no-extract",
        action="store_true",
        help="Skip extraction (just download the tar.gz archives)",
    )
    return p


def _configure_logging() -> None:
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ]
    )


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    args = _build_parser().parse_args(argv)

    files = (
        tuple(FILE_BY_NAME[name] for name in args.only) if args.only else DEFAULT_FILES
    )
    result = fetch(files=files, raw_dir=args.raw_dir, force=args.force, extract=not args.no_extract)
    total_files = sum(len(v) for v in result.values())
    log.info("ingest.complete", n_archives=len(result), n_files=total_files, raw_dir=str(args.raw_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
