"""Arm A for v1: where do published-story companies rank among scored companies?

No model is fitted (four usable positives). A rank test compares orderings;
AUC is biased low under PU data, but the bias is shared, so compare, don't
read absolutes.

    python3 -m pipeline.fav13.evaluate.rank_v1
"""

from .. import common
from .. import config as C
from ..score import features as F
from ..score import rank as R


def load_scored():
    ents = common.load_entities()
    evi = common.load("evidence.pkl")
    recs = F.compute(evi["evidence"], evi["dual"], evi["null_reference"],
                     ents["members"], charity_keys=F.load_charity_keys())
    F.normalise(recs)
    for r in recs:
        r["score"] = R.score(r)
    return recs, ents


def resolve_cases(ents, stories):
    """Published company -> entity id, by exact normalised name only."""
    index = {}
    for raw, e in ents["raw2ent"].items():
        index.setdefault(common.name_key(raw), e)
    for e, names in ents["members"].items():
        for n in names:
            index.setdefault(common.name_key(n), e)

    found, missed = {}, []
    for p in sorted({r["private_org"].strip() for r in stories if r["private_org"].strip()}):
        e = index.get(common.name_key(p))
        if e:
            found.setdefault(e, []).append(p)
        else:
            missed.append(p)
    return found, missed


def main():
    recs, ents = load_scored()
    stories = common.read_stories()
    found, missed = resolve_cases(ents, stories)
    n = len(recs)

    print(f"scored candidate set        : {n:,} entities")
    print(f"published companies         : "
          f"{len({r['private_org'].strip() for r in stories if r['private_org'].strip()})}")
    print(f"exact-matched to an entity  : {len(found)}")

    orderings = {
        "our score":       sorted(recs, key=lambda r: -r["score"]),
        "prominence":      R.baseline_prominence(recs, k=n),
        "raw close-count": R.baseline_raw_close(recs, k=n),
        "ratio":           R.baseline_ratio(recs, k=n),
    }
    per = {k: common.best_ranks([r["entity"] for r in v], found)
           for k, v in orderings.items()}
    present = per["our score"]
    print(f"in the scored set (obs>={C.MIN_OBS})    : {len(present)}\n")

    if missed:
        print("not matched to any entity:")
        for m in missed:
            print(f"   - {m}")
        print()
    dropped = [e for e in found if e not in present]
    if dropped:
        print(f"matched but filtered out before scoring (obs<{C.MIN_OBS} or one-sided):")
        for e in dropped:
            print(f"   - {e}  ({', '.join(found[e])})")
        print()
    if not present:
        print("No published case survives into the scored set. The rank test cannot run.")
        return

    print(f"{'company':<34}{'our score':>12}{'prominence':>12}{'raw close':>12}{'ratio':>10}")
    print("-" * 80)
    for e in sorted(present, key=lambda x: present[x]):
        print(f"{', '.join(found[e])[:32]:<34}"
              + "".join(f"{per[k][e]:>12,}" for k in ("our score", "prominence", "raw close-count"))
              + f"{per['ratio'][e]:>10,}")

    print("\n" + "-" * 80)
    print(f"{'ordering':<20}{'AUC':>8}{'median rank':>14}{'best':>8}{'top-10%':>10}{'top-25%':>10}")
    for k in orderings:
        rk = sorted(per[k].values())
        t10 = sum(1 for r in rk if r <= 0.10 * n)
        t25 = sum(1 for r in rk if r <= 0.25 * n)
        print(f"{k:<20}{common.auc(per[k], n):>8.3f}{rk[len(rk) // 2]:>14,}{rk[0]:>8,}"
              f"{t10:>7}/{len(rk):<2}{t25:>7}/{len(rk):<2}")

    print(f"\nn = {len(present)} positives. Report the comparison between "
          f"orderings, not the absolute AUC.")


if __name__ == "__main__":
    main()
