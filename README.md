# IJF Link Discovery

Discovering typed, temporally-ordered influence links across Canadian civic
databases: political donations, government appointments, lobbying
registrations, procurement records, and charity metadata.

The shipped pipeline targets one motif, **FAV 13**: a company lobbies a
federal department, and that same department later awards it a contract. Every
scored candidate carries the evidence that produced its score, so a journalist
can read the reasoning rather than trust a number.

**Code lives in git. Data does not.** The corpus is ~6 GB raw / ~9 GB cleaned,
so it is distributed via Zenodo and reassembled locally.

---

## Layout

```
LinkDiscovery/
├── link_discovery_repo/     <- this repo
│   ├── ijf_config.py            data paths for the cleaning layer
│   ├── pipeline/
│   │   ├── clean_datasets.py    raw -> cleaned layer
│   │   ├── profile_clean.py     profile of the cleaned layer
│   │   └── fav13/               the FAV 13 pipeline (see below)
│   ├── ground_truth/            published cases, negatives, perturbations
│   ├── reference/               curated lookup tables (body_map, parties)
│   ├── docs/                    method PDFs and result workbooks
│   └── scripts/                 fetch, sample, checksum
├── ijf-datasets/            raw corpus       (~6 GB, gitignored)
├── ijf-clean/               cleaned corpus   (~9 GB, gitignored)
└── work/                    pipeline artifacts (gitignored)
```

Paths resolve relative to the repo, so it runs from anywhere. Override the
data root with `IJF_DATA_ROOT`.

---

## Running it

Everything runs as a module from this directory. No third-party packages: the
pipeline is pure Python 3.11 standard library, by design, so it runs on a
cluster login node with no environment to build.

```bash
python3 scripts/fetch_data.py          # 1. raw corpus from Zenodo (~6 GB)
python3 scripts/checksums.py verify    # 2. confirm the download
python3 -m pipeline.clean_datasets     # 3. build ijf-clean/  (~9 GB, slow)

python3 -m pipeline.fav13.run --build  # 4. entity + evidence caches
python3 -m pipeline.fav13.score_v2     # 5. score every award-level motif
python3 -m pipeline.fav13.evaluate_v2  # 6. rank test against known cases
```

Step 5 writes `work/fav13_v2_scored.pkl` (~386 MB, 1.5 M motifs). Every module
after it reads that file.

---

## The `fav13` modules

**Data layer**

| Module | Role |
|---|---|
| `config.py` | load-bearing vs free parameters, kept separate on purpose |
| `entities.py` | name normalisation and entity resolution; the only inference in the pipeline |
| `evidence.py` | per-entity lobbying and award calendars, plus the national null reference |
| `body_map.py` | 253 curated lobbying-target to procurement-department mappings |

**Scoring**

| Module | Role |
|---|---|
| `features.py` | v1 company-level features and the Poisson-binomial null |
| `awards.py` | award-level motif construction (one award = one motif) |
| `score_v2.py` | the current scorer: four calibrated components, Brown-combined, BH-corrected |
| `bodycond.py` | body-conditioned temporal excess test |
| `rank.py` | Fagin threshold algorithm, brute force, Spearman, baselines |
| `run.py` | cache bootstrap; company-level baseline retained for Arm C |

**Learned variants**

| Module | Role |
|---|---|
| `v3.py` | hand-rolled monotone gradient boosting; monotonicity is what keeps TA valid |
| `v3b.py` | three-variant ablation, four baselines, leave-one-company-out |

**Evaluation and output**

| Module | Role |
|---|---|
| `eval_rank.py` | Mann-Whitney rank test (AUC) against the published cases |
| `evaluate_v2.py` | v2 hit positions against `favoritism_ca.csv` |
| `hits.py`, `dossier.py` | per-case evidence extraction |
| `gt_motifs.py` | ground-truth motif completion and the six perturbation sheets |
| `negatives.py` | rule-based, within-company, and permutation negatives |
| `render_hits.py`, `render_dossier.py` | LaTeX output into `docs/` |
| `report_xlsx.py`, `xlsx.py` | workbooks; `xlsx.py` is a minimal OOXML writer |

---

## Design commitments

These are constraints, not preferences. They shape what is in the repo.

- **Entity resolution is the only place inference happens.** All 16 link cases
  are `sameAs` assertions. Once entities are fixed, everything downstream is
  arithmetic on observed records.
- **Precision over recall.** A false positive here is a defamatory accusation.
- **No black box in the scored path.** Scores decompose into named components
  a journalist can be walked through. The learned variant in `v3.py` is
  monotone by construction and reports its stumps.
- **Monotone aggregation with non-negative weights.** This is the precondition
  for Fagin's threshold algorithm and the early termination it buys. A term
  that can go negative breaks the guarantee.
- **The cleaning layer is additive and deterministic.** It never modifies a raw
  value, never drops a row, and adds no inference. One exception,
  `recipient_kind`, is flagged in the code.

---

## Ground truth

`ground_truth/favoritism_ca.csv` holds published cases, not labels. It is
positive-only, and the positives were selected by *being newsworthy enough to
publish*, which breaks the SCAR assumption a PU learner needs. With three
complete cases it cannot train a classifier. It supports a rank test, which is
what `eval_rank.py` reports.

The negatives are constructed, and each row says how:

- `favoritism_negatives.csv` — rule-based, 7 categories
- `favoritism_negatives_within.csv` — within-company controls
- `favoritism_negatives_permuted.csv` — permutation nulls
- `gt_perturbations/T1..T6` — one sheet per perturbation of a true motif

---

## Data distribution

Only the raw corpus is distributed. `ijf-clean/` is derived, which halves the
download and makes cleaning reproducible rather than an opaque blob:

```
Zenodo -> ijf-datasets/ -> [clean_datasets.py] -> ijf-clean/
```

To publish a new corpus version: upload to Zenodo, note the record id from the
URL, set `ZENODO_RECORD` in `scripts/fetch_data.py` (or export
`IJF_ZENODO_RECORD`), then regenerate the manifest with
`python3 scripts/checksums.py write`.

Git LFS is the wrong tool at this size: GitHub's free tier gives 1 GB of
storage and 1 GB of monthly bandwidth, LFS stores every version of a rewritten
CSV whole, and every clone pulls the data whether or not it is wanted.

---

## Working agreements

- Never commit CSVs outside `ground_truth/` and `reference/`. `.gitignore`
  covers the data directories; check `git status` anyway.
- Never hardcode paths. Import from `pipeline.fav13.config` or `ijf_config`.
- Never tune a load-bearing parameter to improve a result. The split in
  `config.py` exists to make that visible.
