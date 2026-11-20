"""v3 ablation ("v3b"): does dropping prominence force the learner onto real signals?

In v3 the learner put ~half its weight on f_prominence (company size). This
retrains without it, then without channel too, and puts every variant next to
the rule baselines with leave-one-company-out.

    python3 -m pipeline.fav13.score.v3_ablation     # -> work/v3b_results.json
"""

import json
import random

from pipeline import paths
from .. import common
from ..common import NONCOMP
from .v3 import FEATURES, feature_matrix, order_by, train, weak_labels

VARIANTS = [("v3  all 9 features", set()),
            ("v3b without prominence", {"f_prominence"}),
            ("v3c anomaly signals only", {"f_prominence", "f_channel"})]
PARAMS = dict(rounds=45, lr=0.14)


def subset(X, drop):
    keep = [i for i, f in enumerate(FEATURES) if f not in drop]
    return [tuple(x[i] for i in keep) for x in X], [FEATURES[i] for i in keep]


def run_variant(label, X, names, y, comp, known, recs, pos_idx, neg_idx):
    total = len(X)
    loco = {}
    for held in sorted({c for c in comp if c}):
        m = train(X, y, comp, pos_idx, neg_idx, exclude=held, **PARAMS)
        loco[held] = common.best_ranks(order_by(m, X, recs)[1], {held}).get(held)

    m = train(X, y, comp, pos_idx, neg_idx, **PARAMS)
    best = common.best_ranks(order_by(m, X, recs)[1], known)
    return dict(label=label, auc=common.auc(best, total), ranks=best, loco=loco,
                importance=m.importance(names), n_features=len(names))


def baselines(recs, known, ent_awards):
    no_gap = 10 ** 9
    orders = {
        "BASELINE sole-source then recency":
            sorted(recs, key=lambda r: (0 if r["method"] in NONCOMP else 1,
                                        r["gap"] if r["gap"] is not None else no_gap)),
        "BASELINE raw nearest-gap":
            sorted(recs, key=lambda r: r["gap"] if r["gap"] is not None else no_gap),
        "v2 surprise only":
            sorted(recs, key=lambda r: r["p_combined"]),
        "BASELINE prominence":
            sorted(recs, key=lambda r: -ent_awards[r["entity"]]),
    }
    out = {}
    for k, o in orders.items():
        best = common.best_ranks([r["entity"] for r in o], known)
        out[k] = dict(label=k, auc=common.auc(best, len(recs)), ranks=best,
                      loco={}, importance=[])
    return out


def main():
    random.seed(0)
    print("building matrix ...", flush=True)
    recs = common.load_scored()
    ents = common.load_entities()
    display = ents.get("display", {})
    X, ent_awards = feature_matrix(recs)
    claims = common.story_claims(common.name_index(ents["raw2ent"]))
    y, comp = weak_labels(recs, claims)
    known = {e for e, _, _ in claims}

    total = len(recs)
    pos_idx = [i for i in range(total) if y[i]]
    neg_idx = [i for i in range(total) if not y[i]]
    random.shuffle(neg_idx)
    neg_idx = neg_idx[:25000]
    print(f"   {total:,} instances, {len(pos_idx)} weak positives, "
          f"{len(neg_idx):,} sampled unlabelled", flush=True)

    results = []
    for label, drop in VARIANTS:
        Xs, names = subset(X, drop)
        print(f"\ntraining {label}  ({len(names)} features) ...", flush=True)
        r = run_variant(label, Xs, names, y, comp, known, recs, pos_idx, neg_idx)
        results.append(r)
        print("   importance: " + ", ".join(f"{k} {v*100:.0f}%" for k, v in r["importance"]),
              flush=True)
    base = baselines(recs, known, ent_awards)

    def cell(v):
        return f"{(f'{v:,}' if v else '-'):>12}"

    print("\n\n" + "=" * 92)
    print(f"KNOWN-CASE RANKS  (of {total:,} contracts)")
    print("=" * 92)
    print(f"{'company':<26}" + "".join(f"{r['label'].split()[0][:10]:>12}" for r in results)
          + "".join(f"{k.split()[-1][:10]:>12}" for k in base))
    for e in sorted(known, key=lambda e: display.get(e, e)):
        print(f"{display.get(e,e)[:24]:<26}"
              + "".join(cell(r["ranks"].get(e)) for r in results)
              + "".join(cell(base[k]["ranks"].get(e)) for k in base))
    print("-" * 92)
    print(f"{'AUC':<26}" + "".join(f"{r['auc']:>12.4f}" for r in results)
          + "".join(f"{base[k]['auc']:>12.4f}" for k in base))

    print("\nLEAVE-ONE-COMPANY-OUT (honest generalisation)")
    for r in results:
        print(f"  {r['label']:<26}" + "  ".join(
            f"{display.get(h,h)[:16]}: " +
            (f"{v:,} ({100.0*v/total:.2f}%)" if v else "not found")
            for h, v in r["loco"].items()))

    print("\nFEATURE IMPORTANCE")
    for r in results:
        print(f"  {r['label']}")
        for k, v in r["importance"]:
            print(f"      {k:<16}{v*100:5.1f}%")

    with open(paths.WORK / "v3b_results.json", "w") as fh:
        json.dump(dict(
            results=[{k: v for k, v in r.items() if k != "ranks"} for r in results],
            ranks={r["label"]: {display.get(e, e): v for e, v in r["ranks"].items()}
                   for r in results},
            baselines={k: dict(auc=v["auc"],
                               ranks={display.get(e, e): x for e, x in v["ranks"].items()})
                       for k, v in base.items()},
            total=total), fh, indent=1)
    print("\nwrote v3b_results.json")


if __name__ == "__main__":
    main()
