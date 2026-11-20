"""Every path the project uses, resolved in one place.

Data lives outside git, next to the repo by default:

    <IJF_DATA_ROOT>/
        ijf-datasets/   raw corpus (Zenodo)
        ijf-clean/      cleaned corpus (pipeline/clean_datasets.py)
        datasets/       notebook samples (scripts/make_samples.py)
        work/           pipeline outputs

Override the root with IJF_DATA_ROOT, or one directory with
IJF_RAW_DIR / IJF_CLEAN_DIR / IJF_SAMPLES_DIR / IJF_WORK_DIR.

    python3 -m pipeline.paths      # show what is present
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("IJF_DATA_ROOT", REPO.parent)).expanduser().resolve()


def _dir(env_var, name):
    override = os.environ.get(env_var)
    return Path(override).expanduser().resolve() if override else DATA_ROOT / name


RAW = _dir("IJF_RAW_DIR", "ijf-datasets")
CLEAN = _dir("IJF_CLEAN_DIR", "ijf-clean")
SAMPLES = _dir("IJF_SAMPLES_DIR", "datasets")
WORK = _dir("IJF_WORK_DIR", "work")

# Small, tracked in git.
GROUND_TRUTH = REPO / "ground_truth"
REFERENCE = REPO / "reference"
DOCS = REPO / "docs"

STORIES = GROUND_TRUTH / "favoritism_ca.csv"
BODY_MAP = REFERENCE / "body_map.csv"
RULING_PARTIES = REFERENCE / "canada_ruling_parties.csv"

# Notebook samples (scripts/make_samples.py).
DON_SAMPLE = SAMPLES / "don_20_22_0.5M.csv"
APT_SAMPLE = SAMPLES / "apt_rows.csv"
LOB_SAMPLE = SAMPLES / "lob_results_mv_rows.csv"

# The 15 tables of the distributed corpus.
TABLES = [
    "apt.csv", "cha_meta.csv", "don_mv.csv",
    "lob_firm.csv", "lob_lobbyist.csv", "lob_meta.csv", "lob_org.csv",
    "lob_rd_mv.csv", "lob_target.csv",
    "pro_award.csv", "pro_contact.csv", "pro_supplier.csv", "pro_tender.csv",
    "rep_meta.csv", "rep_target.csv",
]


def require(path):
    """Return `path`, or fail with the command that creates it."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing data file: {path}\n"
            f"  data root : {DATA_ROOT}\n"
            f"  Fix       : set IJF_DATA_ROOT, or run\n"
            f"                python3 scripts/fetch_data.py       # raw corpus\n"
            f"                python3 -m pipeline.clean_datasets  # ijf-clean/\n"
            f"                python3 scripts/make_samples.py     # notebook samples"
        )
    return path


def status():
    """One line per data directory: present or missing, and its size."""
    lines = [f"data root : {DATA_ROOT}"]
    for label, d in (("raw", RAW), ("clean", CLEAN), ("samples", SAMPLES)):
        if d.is_dir():
            files = sorted(d.glob("*.csv"))
            size = sum(p.stat().st_size for p in files)
            lines.append(f"{label:<9}: {len(files):>2} csv, {size / 1e9:6.2f} GB  {d}")
        else:
            lines.append(f"{label:<9}: MISSING                  {d}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(status())
