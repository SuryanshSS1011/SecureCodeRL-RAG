"""Build data/nvdlib_cwe_medians.json from the NVD CVE feed.

This is run once (or on each NVD snapshot refresh), not at training time.
Output is committed to the repo so the reward function is reproducible
without network access at training time.

Usage:
    python scripts/build_nvdlib_medians.py \\
        --output data/nvdlib_cwe_medians.json \\
        --window-years 5 \\
        --cwes data/ictai_cwe_list.txt

Dependencies:
    pip install nvdlib

Notes:
    - Requires an NVD API key for reasonable throughput. Set NVD_API_KEY in
      the environment (or in a .env file the launcher sources). Without a
      key, throttled to ~1 req / 6s.
    - The NVD API rejects date ranges > 120 days, so we chunk the window
      into 120-day slices and aggregate.
    - CVSS v3.1/v3.0 base score is preferred; falls back to the generic
      `score` tuple (which may be v2.0, v3.x, or v4.0). v2-only CVEs that
      survive the fallback are included with a `v2_used` count in the meta.
    - Only CVEs whose CWE assignment is exactly one of the ICTAI CWE list
      contribute. Multi-CWE CVEs are currently counted under their first
      matching CWE only; if this proves limiting, revisit and distribute
      to all matching buckets.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("build_nvdlib_medians")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)


# NVD API rejects pubStartDate/pubEndDate ranges greater than 120 days.
_MAX_RANGE_DAYS = 120


def _parse_published(raw) -> Optional[datetime]:
    """nvdlib returns `cve.published` as a string; parse to UTC datetime."""
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    # Try ISO formats with and without fractional seconds.
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
    ):
        try:
            return datetime.strptime(str(raw), fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    logger.warning("could not parse published timestamp: %r", raw)
    return None


def _extract_score(cve) -> tuple[Optional[float], str]:
    """Return (score, version_label) for a CVE, or (None, "") if unscored.

    Order:
        1. v31score (CVSS v3.1)
        2. v30score (CVSS v3.0)
        3. score tuple (any version NVD ships; first element is version label)
    """
    v31 = getattr(cve, "v31score", None)
    if v31 is not None:
        return float(v31), "v31"
    v30 = getattr(cve, "v30score", None)
    if v30 is not None:
        return float(v30), "v30"
    raw = getattr(cve, "score", None)
    # `score` is [version_label, value, severity]
    if isinstance(raw, (list, tuple)) and len(raw) >= 2:
        try:
            return float(raw[1]), str(raw[0]).lower()
        except (TypeError, ValueError):
            return None, ""
    return None, ""


def _date_chunks(start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
    """Yield (chunk_start, chunk_end) pairs of at most `_MAX_RANGE_DAYS` days.

    NVD's `pubEndDate` is inclusive; we offset the next chunk's start by
    one second to avoid double-counting boundary CVEs.
    """
    cur = start
    delta = timedelta(days=_MAX_RANGE_DAYS - 1)
    while cur < end:
        chunk_end = min(cur + delta, end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(seconds=1)


def _fmt_nvd_date(dt: datetime) -> str:
    """NVD's pubStartDate / pubEndDate format: 'YYYY-MM-DD HH:MM'."""
    return dt.strftime("%Y-%m-%d %H:%M")


def build_medians(cwe_list: list[str], window_years: int, api_key: Optional[str]) -> dict:
    try:
        import nvdlib  # type: ignore
    except ImportError:
        sys.exit("nvdlib is required: pip install nvdlib")

    delay = 0.6 if api_key else 6.0
    now = datetime.now(timezone.utc).replace(microsecond=0)
    window_start = now - timedelta(days=365 * window_years)

    scores_by_cwe: dict[str, list[float]] = {cwe: [] for cwe in cwe_list}
    n_total = 0
    n_in_window = 0
    n_v31 = 0
    n_v30 = 0
    n_v40 = 0
    n_v2 = 0

    chunks = list(_date_chunks(window_start, now))
    logger.info(
        "querying %d CWEs across %d date chunks of <=%d days each "
        "(window: %s to %s)",
        len(cwe_list), len(chunks), _MAX_RANGE_DAYS,
        window_start.isoformat(), now.isoformat(),
    )

    for cwe in cwe_list:
        cwe_start = time.monotonic()
        cwe_count = 0
        for chunk_start, chunk_end in chunks:
            try:
                results = nvdlib.searchCVE(
                    cweId=cwe,
                    key=api_key,
                    delay=delay,
                    pubStartDate=_fmt_nvd_date(chunk_start),
                    pubEndDate=_fmt_nvd_date(chunk_end),
                )
            except Exception as exc:
                logger.warning(
                    "  %s [%s -> %s]: query failed (%s); skipping chunk",
                    cwe, chunk_start.date(), chunk_end.date(), exc,
                )
                continue
            for cve in results:
                n_total += 1
                pub = _parse_published(getattr(cve, "published", None))
                if pub is None or pub < window_start:
                    continue
                score, version = _extract_score(cve)
                if score is None:
                    continue
                n_in_window += 1
                if "v31" in version:
                    n_v31 += 1
                elif "v30" in version:
                    n_v30 += 1
                elif "v40" in version:
                    n_v40 += 1
                elif "v2" in version:
                    n_v2 += 1
                scores_by_cwe[cwe].append(float(score))
                cwe_count += 1
        elapsed = time.monotonic() - cwe_start
        logger.info(
            "  %s: %d CVEs scored (%.1fs)", cwe, cwe_count, elapsed,
        )

    medians: dict[str, dict] = {}
    for cwe, scores in scores_by_cwe.items():
        if not scores:
            logger.warning("  %s: no scored CVEs; will fall through to tier midpoint", cwe)
            continue
        medians[cwe] = {
            "median": round(statistics.median(scores), 2),
            "mean": round(statistics.mean(scores), 2),
            "n_cves": len(scores),
            "min": round(min(scores), 2),
            "max": round(max(scores), 2),
        }

    return {
        "_meta": {
            "status": "populated",
            "description": "Per-CWE empirical median CVSS base score, computed from NVD CVEs.",
            "method": "scripts/build_nvdlib_medians.py",
            "window_years": window_years,
            "window_start": window_start.isoformat(),
            "window_end": now.isoformat(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "nvd_snapshot_date": datetime.now(timezone.utc).date().isoformat(),
            "n_cwes_covered": len(medians),
            "n_cwes_queried": len(cwe_list),
            "n_cves_total_queried": n_total,
            "n_cves_in_window": n_in_window,
            "by_score_version": {
                "v31": n_v31, "v30": n_v30, "v40": n_v40, "v2": n_v2,
            },
            "spec_version": "reward_spec.md v0.1",
        },
        "medians": medians,
    }


def _read_cwe_list(path: Path) -> list[str]:
    """Read CWE list, stripping comments and blank lines."""
    cwes = []
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()  # drop inline + full comments
        if not line:
            continue
        cwes.append(line)
    return cwes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-years", type=int, default=5)
    parser.add_argument(
        "--cwes",
        type=Path,
        required=True,
        help="Path to a text file listing one CWE per line (e.g. 'CWE-787'). "
        "Lines starting with '#' or blank lines are ignored.",
    )
    args = parser.parse_args()

    cwes = _read_cwe_list(args.cwes)
    if not cwes:
        sys.exit("--cwes file is empty after stripping comments and blanks")
    bad = [c for c in cwes if not c.startswith("CWE-")]
    if bad:
        sys.exit(f"CWE entries must be of the form 'CWE-NNN'; got {bad[:3]}")

    api_key = os.environ.get("NVD_API_KEY")
    if not api_key:
        logger.warning(
            "NVD_API_KEY not set; throttled to one request per 6s. "
            "Get a key at https://nvd.nist.gov/developers/request-an-api-key"
        )

    logger.info("building medians for %d CWEs over %d years", len(cwes), args.window_years)
    result = build_medians(cwes, args.window_years, api_key)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n")
    logger.info(
        "wrote %s (covered %d / %d CWEs)",
        args.output,
        result["_meta"]["n_cwes_covered"],
        result["_meta"]["n_cwes_queried"],
    )


if __name__ == "__main__":
    main()
