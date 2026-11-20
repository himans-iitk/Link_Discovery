#!/usr/bin/env python3
"""Build the small notebook samples from the cleaned corpus.

The notebooks run against three derived files that are cheap to regenerate and
so are never distributed:

    apt_rows.csv              all appointments, minus free-text blobs
    don_20_22_0.5M.csv        <=500k donation rows from 2020-2022
    lob_results_mv_rows.csv   lobbying registration rows, minus blobs

Everything is streamed in chunks: don_mv.csv alone is ~4 GB.

    python scripts/make_samples.py            # build whatever is missing
    python scripts/make_samples.py --force    # rebuild all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import paths as cfg  # noqa: E402

# Free-text / full-text-search columns: never read by the notebooks, and they
# dominate on-disk size.
DROP_COLS = {"body", "fts"}

CHUNK = 250_000
DON_YEARS = (2020, 2022)
DON_TARGET_ROWS = 500_000
SEED = 0


def _drop(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c in DROP_COLS], errors="ignore")


def _stream_copy(src: Path, dest: Path) -> int:
    """Copy every row of `src` to `dest`, dropping blob columns."""
    rows = 0
    header = True
    with pd.option_context("mode.chained_assignment", None):
        for chunk in pd.read_csv(src, chunksize=CHUNK, dtype=str, low_memory=False):
            out = _drop(chunk)
            out.to_csv(dest, mode="w" if header else "a", header=header, index=False)
            header = False
            rows += len(out)
    return rows


def build_apt(force: bool) -> None:
    dest = cfg.APT_SAMPLE
    if dest.exists() and not force:
        print(f"  skip   {dest.name} (exists)")
        return
    src = cfg.require(cfg.CLEAN / "apt.csv")
    print(f"  build  {dest.name} <- {src.name}")
    rows = _stream_copy(src, dest)
    print(f"         {rows:,} rows, {dest.stat().st_size / 1e6:.1f} MB")


def build_lob(force: bool) -> None:
    dest = cfg.LOB_SAMPLE
    if dest.exists() and not force:
        print(f"  skip   {dest.name} (exists)")
        return
    src = cfg.require(cfg.CLEAN / "lob_rd_mv.csv")
    print(f"  build  {dest.name} <- {src.name}")
    rows = _stream_copy(src, dest)
    print(f"         {rows:,} rows, {dest.stat().st_size / 1e6:.1f} MB")


def build_don(force: bool) -> None:
    """Two passes over don_mv.csv: count in-range rows, then sample to target."""
    dest = cfg.DON_SAMPLE
    if dest.exists() and not force:
        print(f"  skip   {dest.name} (exists)")
        return
    src = cfg.require(cfg.CLEAN / "don_mv.csv")
    lo, hi = DON_YEARS
    print(f"  build  {dest.name} <- {src.name}  (years {lo}-{hi})")

    def in_range(chunk: pd.DataFrame) -> pd.DataFrame:
        yr = pd.to_numeric(chunk["donation_year"], errors="coerce")
        return chunk[(yr >= lo) & (yr <= hi)]

    print("         pass 1/2: counting in-range rows")
    total = 0
    for chunk in pd.read_csv(src, chunksize=CHUNK, dtype=str, low_memory=False):
        total += len(in_range(chunk))
    print(f"         {total:,} rows in range")

    if total == 0:
        raise SystemExit(f"No rows with donation_year in {lo}-{hi}; check {src}")

    keep_frac = min(1.0, DON_TARGET_ROWS / total)
    rng = np.random.default_rng(SEED)

    print(f"         pass 2/2: writing (keep {keep_frac:.1%})")
    rows = 0
    header = True
    for chunk in pd.read_csv(src, chunksize=CHUNK, dtype=str, low_memory=False):
        sel = _drop(in_range(chunk))
        if keep_frac < 1.0 and len(sel):
            sel = sel[rng.random(len(sel)) < keep_frac]
        if not len(sel):
            continue
        sel.to_csv(dest, mode="w" if header else "a", header=header, index=False)
        header = False
        rows += len(sel)
    print(f"         {rows:,} rows, {dest.stat().st_size / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="rebuild even if present")
    args = ap.parse_args()

    cfg.SAMPLES.mkdir(parents=True, exist_ok=True)
    print(f"samples -> {cfg.SAMPLES}")
    build_apt(args.force)
    build_lob(args.force)
    build_don(args.force)
    print("done.")


if __name__ == "__main__":
    main()
