# IJF Link Discovery

Finds typed, time-ordered influence links across Canadian civic databases
(donations, appointments, lobbying, procurement, charities) for investigative
reporters. Output is a ranked list of leads, not asserted fact.

The shipped pipeline targets one motif, **FAV 13**: a company lobbies a federal
department, and that department later awards it a contract. Every score
decomposes into named evidence a reporter can check.

Pure Python 3.11 standard library (runs on a cluster login node). Only
`scripts/make_samples.py` needs pandas/numpy.

## Layout

```
LinkDiscovery/
├── link_discovery_repo/          this repo
│   ├── pipeline/
│   │   ├── paths.py              every path, in one place
│   │   ├── clean_datasets.py     raw -> ijf-clean/
│   │   ├── profile_clean.py      profile of the cleaned layer
│   │   └── fav13/
│   │       ├── __main__.py       runs all stages in order
│   │       ├── config.py         parameters (load-bearing vs free)
│   │       ├── common.py         shared readers and helpers
│   │       ├── data/             entities, evidence, body map, cache build
│   │       ├── score/            v1 company, award-level, v2 (main), v3 learned
│   │       ├── evaluate/         rank tests, story hits, dossier, negatives
│   │       └── report/           LaTeX and xlsx into docs/
│   ├── ground_truth/             published cases, negatives, perturbations
│   ├── reference/                curated body_map.csv, ruling parties
│   ├── docs/                     method PDFs and result workbooks
│   └── scripts/                  fetch, checksum, sample
├── ijf-datasets/                 raw corpus     (~6 GB, not in git)
├── ijf-clean/                    cleaned corpus (~9 GB, not in git)
└── work/                         pipeline outputs (not in git)
```

Override the data location with `IJF_DATA_ROOT` (see `pipeline/paths.py`).

## Running

From this directory:

```bash
python3 scripts/fetch_data.py           # 1. raw corpus from Zenodo
python3 scripts/checksums.py verify     # 2. check the download
python3 -m pipeline.clean_datasets      # 3. build ijf-clean/ (slow)
python3 -m pipeline.fav13               # 4. every FAV 13 stage, in order
```

Run one stage with `python3 -m pipeline.fav13 v2 rank_v2`, list them with
`--list`, or run a module directly (`python3 -m pipeline.fav13.score.v2`).

## Flow

| Stage | Module | Writes (to `work/` unless noted) |
|---|---|---|
| **data** | `data.build` | `entities.pkl`, `evidence.pkl` |
| | `data.body_map` | `reference/body_map.csv` (rebuild only when re-curating) |
| **score** | `score.company` | v1 company-level baseline + Arm C (prints) |
| | `score.awards` | `fav13_awards_topk.csv` (award-level weighted sum) |
| | `score.bodycond` | `bodycond.pkl` (body-conditioned test) |
| | **`score.v2`** | **`fav13_v2_scored.pkl`**, `fav13_v2_top.csv`, the main scorer |
| | `score.v3`, `score.v3_ablation` | `v3_top.json`, `v3b_results.json` (learned; slow) |
| **evaluate** | `evaluate.rank_v1`, `evaluate.rank_v2` | rank tests vs published cases (prints) |
| | `evaluate.hits`, `evaluate.dossier` | `hits.json`, `dossier.json` |
| | `evaluate.gt_motifs` | `gt_motifs_positive.csv`, `docs/gt_motifs_and_negatives.xlsx` |
| | `evaluate.negatives [rules\|within\|permuted\|all]` | `favoritism_negatives*.csv` |
| **report** | `report.render_hits`, `report.render_dossier` | `docs/hits.tex`, `docs/ground_truth_cases.tex` |
| | `report.models_xlsx` | `docs/favouritism_models.xlsx` |

Everything after `score.v2` reads `fav13_v2_scored.pkl` (~386 MB, 1.5 M contracts).

### How v2 scores a contract

One row per contract. Only lobbying of the *buying* department counts. Four
p-values, each against its own null (timing, sole-source rate, tender window,
supplier dominance), combined with Brown's method, then
`score = identity gate × (1 − p) × salience`, with BH q-values.

## Design commitments

- **Entity resolution is the only inference.** Everything after it is
  arithmetic on observed records.
- **Precision over recall.** A false positive is a defamatory accusation.
- **No black box in the scored path.** Scores decompose into named components.
- **Monotone, non-negative aggregation**, the precondition for Fagin's
  threshold algorithm.
- **The cleaning layer is additive and deterministic**: never edits a raw
  value, never drops a row (one flagged exception, `recipient_kind`).

## Ground truth

`ground_truth/favoritism_ca.csv` holds published stories, not labels: positive
only and selected for newsworthiness, so it supports a rank test, not a
classifier. Negatives are constructed and each row says how
(`favoritism_negatives*.csv`, `gt_perturbations/T1..T6`).

## Data distribution

Only the raw corpus is distributed (Zenodo); `ijf-clean/` is rebuilt locally.
To publish a new version: upload to Zenodo, set `IJF_ZENODO_RECORD`, then
`python3 scripts/checksums.py write`.

## Working agreements

- Never commit CSVs outside `ground_truth/` and `reference/`.
- Never hardcode paths; use `pipeline.paths`.
- Never tune a load-bearing parameter to improve a result (`fav13/config.py`).
- `reference/body_map.csv` has a hand-curated `approved` column; rebuilding it
  overwrites those approvals.
