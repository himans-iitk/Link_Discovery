"""Central data-path resolution for the IJF link-discovery notebooks.

The CSV corpus is far too large for git (~5.8G raw, ~8.2G cleaned), so it lives
outside the repository and is fetched separately (see README).  Every notebook
imports its paths from here instead of hardcoding relative ones.

Layout under the data root:

    <IJF_DATA_ROOT>/
        ijf-datasets/   raw source CSVs        <- downloaded from Zenodo
        ijf-clean/      cleaned CSVs           <- built by pipeline/clean_datasets.py
        datasets/       small samples + refs   <- built by scripts/make_samples.py

Override the whole tree with IJF_DATA_ROOT, or any single directory with
IJF_RAW_DIR / IJF_CLEAN_DIR / IJF_SAMPLES_DIR.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# Default keeps the existing sibling-directory layout working untouched.
DATA_ROOT = Path(os.environ.get("IJF_DATA_ROOT", REPO_ROOT.parent)).expanduser().resolve()


def _dir(env_var: str, default_name: str) -> Path:
    override = os.environ.get(env_var)
    return Path(override).expanduser().resolve() if override else DATA_ROOT / default_name


RAW = _dir("IJF_RAW_DIR", "ijf-datasets")
CLEAN = _dir("IJF_CLEAN_DIR", "ijf-clean")
SAMPLES = _dir("IJF_SAMPLES_DIR", "datasets")

# Small enough to be tracked in git alongside the code.
GROUND_TRUTH = REPO_ROOT / "ground_truth"
PROFILING_OUTPUT = REPO_ROOT / "profiling_output"
REFERENCE = REPO_ROOT / "reference"

# ---- Named files the notebooks use -------------------------------------

# Derived samples (scripts/make_samples.py)
DON_SAMPLE = SAMPLES / "don_20_22_0.5M.csv"
APT_SAMPLE = SAMPLES / "apt_rows.csv"
LOB_SAMPLE = SAMPLES / "lob_results_mv_rows.csv"

# Small hand-maintained reference table: tracked in git, not part of the
# Zenodo corpus, so it must live inside the repo.
RULING_PARTIES = REFERENCE / "canada_ruling_parties.csv"

#: Sample name -> (source table in CLEAN, columns kept, row filter)
SAMPLE_SOURCES = {
    "apt_rows.csv": "apt.csv",
    "don_20_22_0.5M.csv": "don_mv.csv",
    "lob_results_mv_rows.csv": "lob_rd_mv.csv",
}

#: The 15 tables that make up the distributed corpus.
TABLES = [
    "apt.csv",
    "cha_meta.csv",
    "don_mv.csv",
    "lob_firm.csv",
    "lob_lobbyist.csv",
    "lob_meta.csv",
    "lob_org.csv",
    "lob_rd_mv.csv",
    "lob_target.csv",
    "pro_award.csv",
    "pro_contact.csv",
    "pro_supplier.csv",
    "pro_tender.csv",
    "rep_meta.csv",
    "rep_target.csv",
]


def require(path: Path) -> Path:
    """Return `path`, or raise with a message pointing at the fix."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}\n"
            f"  data root : {DATA_ROOT}\n"
            f"  Fix       : set IJF_DATA_ROOT, or run\n"
            f"                python scripts/fetch_data.py      # download raw corpus\n"
            f"                python pipeline/clean_datasets.py # build ijf-clean/\n"
            f"                python scripts/make_samples.py    # build notebook samples\n"
            f"  See README.md > Getting the data."
        )
    return path


def status() -> str:
    """One-glance summary of which parts of the corpus are present."""
    lines = [f"data root : {DATA_ROOT}"]
    for label, d in (("raw", RAW), ("clean", CLEAN), ("samples", SAMPLES)):
        if d.is_dir():
            files = sorted(p for p in d.glob("*.csv"))
            size = sum(p.stat().st_size for p in files)
            lines.append(f"{label:<9}: {len(files):>2} csv, {size / 1e9:6.2f} GB  {d}")
        else:
            lines.append(f"{label:<9}: MISSING                  {d}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(status())
