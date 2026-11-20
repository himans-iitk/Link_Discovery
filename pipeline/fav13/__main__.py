"""Run the FAV 13 pipeline in order.

    python3 -m pipeline.fav13                 # every stage
    python3 -m pipeline.fav13 v2 rank_v2      # just these
    python3 -m pipeline.fav13 --list

Prerequisite: ijf-clean/ built (python3 -m pipeline.clean_datasets).
"""

import importlib
import sys
import time

# (name, module, argv) in dependency order; argv None = main() takes no args.
# Every module also runs on its own.
STAGES = [
    # data
    ("build",          "data.build",          []),         # -> entities.pkl, evidence.pkl
    # scoring
    ("company",        "score.company",       []),         # v1 baseline (prints only)
    ("awards",         "score.awards",        None),      # -> fav13_awards_topk.csv
    ("bodycond",       "score.bodycond",      None),      # -> bodycond.pkl
    ("v2",             "score.v2",            None),      # -> fav13_v2_scored.pkl  (main scorer)
    # evaluation (all read fav13_v2_scored.pkl)
    ("rank_v1",        "evaluate.rank_v1",    None),
    ("rank_v2",        "evaluate.rank_v2",    None),
    ("hits",           "evaluate.hits",       None),      # -> hits.json
    ("dossier",        "evaluate.dossier",    None),      # -> dossier.json
    ("gt_motifs",      "evaluate.gt_motifs",  None),
    ("negatives",      "evaluate.negatives",  ["all"]),    # needs hits.json
    # learned variants (slow, pure Python)
    ("v3",             "score.v3",            None),      # -> v3_top.json
    ("v3_ablation",    "score.v3_ablation",   None),      # -> v3b_results.json
    # reports into docs/
    ("render_hits",    "report.render_hits",    None),
    ("render_dossier", "report.render_dossier", None),
    ("models_xlsx",    "report.models_xlsx",    None),
]


def main(argv):
    if "--list" in argv:
        for name, mod, _ in STAGES:
            print(f"{name:<16} pipeline.fav13.{mod}")
        return
    known = {s[0] for s in STAGES}
    unknown = set(argv) - known
    if unknown:
        raise SystemExit(f"unknown stage(s): {', '.join(sorted(unknown))}  (see --list)")
    for name, mod, args in STAGES:
        if argv and name not in argv:
            continue
        print(f"\n===== {name}", flush=True)
        t0 = time.time()
        m = importlib.import_module(f"pipeline.fav13.{mod}")
        m.main(args) if args is not None else m.main()
        print(f"===== {name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
