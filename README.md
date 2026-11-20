# IJF Link Discovery

Find suspicious links between **lobbying** and **government contracts** in Canadian public records, and rank them as leads for investigative reporters.

> **Example lead:** "Company X registered to lobby Department Y, and 3 days later Y gave X a sole-source contract. By chance alone, a contract would follow lobbying of that department this closely about 1 time in 400."

The output is a ranked list of contracts to *check*, not proven wrongdoing. Every score breaks down into evidence a reporter can verify.

---

## Quick start

Needs **Python 3.11+** and nothing else. The pipeline uses only the standard library.

```bash
git clone https://github.com/himans-iitk/Link_Discovery.git
cd Link_Discovery

python3 scripts/fetch_data.py          # 1. download raw data (~6 GB), see "Getting the data"
python3 scripts/checksums.py verify    # 2. confirm your copy matches ours
python3 -m pipeline.clean_datasets     # 3. build the cleaned data (~9 GB, slow)
python3 -m pipeline.fav13              # 4. run the whole pipeline
```

Results go to `work/` (next to the repo) and reports to `docs/`.

---

## Getting the data

The data (~14 GB) is **not in git**. Code and data live side by side:

```
LinkDiscovery/
├── Link_Discovery/     ← this repo (git)
├── ijf-datasets/       ← raw data      (download)
├── ijf-clean/          ← cleaned data  (built by step 3)
└── work/               ← outputs       (built by step 4)
```

- **From Zenodo:** `scripts/fetch_data.py` downloads the raw data. Until the record is published, ask the maintainer for its ID and run `export IJF_ZENODO_RECORD=<id>`.
- **On a shared cluster:** point at the existing copy instead of downloading: `export IJF_DATA_ROOT=/path/to/shared/LinkDiscovery`.

All paths are set in [`pipeline/paths.py`](pipeline/paths.py). Never hard-code a path anywhere else.

---

## What the pipeline does

```
raw data ─► clean ─► resolve companies ─► score contracts ─► evaluate ─► reports
```

| Step | What happens | Main output |
|---|---|---|
| **clean** | Standardise names, dates and provinces. Adds columns, never changes or drops raw data. | `ijf-clean/` |
| **data** | Decide which raw names are the same company. Build each company's lobbying and contract history. | `entities.pkl`, `evidence.pkl` |
| **score** | Score every contract for how suspicious it looks (see below). | **`fav13_v2_scored.pkl`** |
| **evaluate** | Check where the published news stories land in our ranking. | rank tests, `hits.json` |
| **report** | Write the LaTeX and Excel summaries. | `docs/*.tex`, `docs/*.xlsx` |

### How a contract is scored (v2, the main scorer)

Each contract is tested on four questions. Each answer is a p-value: how likely this would happen by chance.

| Signal | Question |
|---|---|
| **Timing** | Did the company lobby *this exact department* shortly before the award? |
| **No competition** | Was it sole-sourced, and does this department rarely do that? |
| **Short tender** | Was the tender window unusually short? |
| **Dominance** | Does this company win more of the department's contracts than its fair share? |

The four are combined into one p-value (Brown's method), then:

```
score = identity confidence × (1 − p) × prominence
```

- **Identity confidence:** how sure we are that the records are the same company.
- **Prominence:** how newsworthy the company is. It is kept separate because big ≠ suspicious.

---

## Code layout

```
pipeline/
├── paths.py              all file locations
├── clean_datasets.py     step: clean
├── profile_clean.py      stats on the cleaned data
└── fav13/                the lobbying → contract pipeline
    ├── __main__.py       runs every stage in order
    ├── config.py         tunable parameters
    ├── common.py         shared helpers (file reading, dates, matching)
    ├── data/             entities.py, evidence.py, body_map.py, build.py
    ├── score/            v2.py (main), awards.py, company.py, bodycond.py, v3*.py
    ├── evaluate/         rank_v1/v2.py, hits.py, dossier.py, gt_motifs.py, negatives.py
    └── report/           render_*.py, models_xlsx.py, xlsx.py, latex.py
ground_truth/             published news stories + constructed negative examples
reference/                hand-curated tables (department name mapping, ruling parties)
docs/                     method write-ups and result workbooks
scripts/                  download, checksum, sampling
```

### Running parts of it

```bash
python3 -m pipeline.fav13 --list            # list stages
python3 -m pipeline.fav13 v2 rank_v2        # run only these stages
python3 -m pipeline.fav13.score.v2          # or run any module directly
```

Stage order: `build` → `company` → `awards` → `bodycond` → **`v2`** → `rank_v1` → `rank_v2` → `hits` → `dossier` → `gt_motifs` → `negatives` → `v3` → `v3_ablation` → `render_hits` → `render_dossier` → `models_xlsx`.

Everything after `v2` reads its output, so after the first full run you rarely need to rerun the early stages.

### Which scorer is which

| Version | Unit | Status |
|---|---|---|
| v1 `score/company.py` | whole company | Old baseline. It ranked big companies first. |
| `score/awards.py` | contract, weighted sum | Intermediate. |
| **v2 `score/v2.py`** | **contract, calibrated p-values** | **Current.** |
| v3 `score/v3.py`, `v3_ablation.py` | contract, learned model | Experiment. It lost to a simple rule, see the paper. |

---

## Things to know before changing anything

1. **Precision over recall.** A false positive here is a false accusation against a real company. When in doubt, miss the case rather than invent one.
2. **Entity matching is the only guesswork.** Everything after deciding "these names are one company" is plain arithmetic on records.
3. **Scores must stay explainable.** No black-box model in the main score.
4. **Do not tune `config.py` to improve a result.** The "load-bearing" settings there are fixed by reasoning, not by testing.
5. **The cleaning step only adds.** It never edits a raw value or drops a row.
6. **`reference/body_map.csv` is hand-edited.** Rebuilding it (`data.body_map`) overwrites the reviewed `approved` column.

### About the ground truth

`ground_truth/favoritism_ca.csv` lists **published news stories** (32 rows, only a few usable). It can test *where known cases rank*. It is too small and one-sided to *train* a model. Negative examples are constructed, and each row says how it was made.

---

## Contributing

- Keep data out of git. Only small CSVs in `ground_truth/` and `reference/` are committed; run `git status` before every commit.
- If you change scoring code, rerun the pipeline and compare `work/` outputs before and after, using `PYTHONHASHSEED=0` so the runs are reproducible.
- One module per stage. Put shared helpers in `fav13/common.py`, not copied into each file.
