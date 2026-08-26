"""
Arm A: rank test of the scoring function against the published cases.

No model is fitted.  The published stories are a positive-only sample, so a
classifier is not merely inadvisable -- with four usable positives it is
arithmetically impossible.  What a positive-only sample *can* support is a
rank test: where do the known cases fall in the ranking, and does our score
put them higher than a baseline does?

The Mann-Whitney U statistic over those ranks is exactly AUC -- the
probability that a random known case outranks a random unlabelled entity.
Under positive-unlabelled data the absolute value is biased low, because some
"negatives" are unlabelled positives.  That bias is shared across every
scoring function measured on the same candidate set, so the *comparison*
between scoring functions stays valid even though the absolute number does
not.  Comparisons are what we report.

    python3 -m pipeline.fav13.eval_rank
"""
import csv, os, pickle, re
from . import config as C
from . import features as F
from . import rank as R

GT = os.path.join(C.GT, "favoritism_ca.csv")


def _key(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def load_scored():
    ents = pickle.load(open(os.path.join(C.WORK, "entities.pkl"), "rb"))
    evi = pickle.load(open(os.path.join(C.WORK, "evidence.pkl"), "rb"))
    recs = F.compute(evi["evidence"], evi["dual"], evi["null_reference"],
                     ents["members"], charity_keys=F.load_charity_keys())
    F.normalise(recs)
    for r in recs:
        r["score"] = R.score(r)
    return recs, ents


def resolve_cases(ents):
    """Published company name -> entity id, by exact normalised match only.

    Substring fallback was tried and rejected: it matched 'Stefano Ferrante'
    (a person) and 'Five Brandon contractors' (a description) to real
    companies.  A false positive here silently inflates the result we are
    trying to measure, so only exact matches count and the misses are
    reported.
    """
    exact = {}
    for raw, e in ents["raw2ent"].items():
        exact.setdefault(_key(raw), e)
    for e, names in ents["members"].items():
        for n in names:
            exact.setdefault(_key(n), e)

    rows = list(csv.DictReader(open(GT, newline="", encoding="utf-8")))
    orgs = sorted({r["private_org"].strip() for r in rows
                   if r["private_org"].strip()})
    found, missed = {}, []
    for p in orgs:
        e = exact.get(_key(p))
        if e:
            found.setdefault(e, []).append(p)
        else:
            missed.append(p)
    return found, missed, rows


def ranks_of(ordered, targets):
    """1-based rank of each target entity in a descending-ordered list."""
    pos = {r["entity"]: i + 1 for i, r in enumerate(ordered)}
    return {e: pos[e] for e in targets if e in pos}


def auc_from_ranks(rank_list, n_total, n_pos):
    """Mann-Whitney U -> AUC.  rank_list is 1-based ranks of the positives in
    a descending ordering, so we convert to ascending rank first."""
    if not rank_list or n_pos == 0 or n_total == n_pos:
        return float("nan")
    asc = [n_total - r + 1 for r in rank_list]
    U = sum(asc) - n_pos * (n_pos + 1) / 2.0
    return U / (n_pos * (n_total - n_pos))


def main():
    recs, ents = load_scored()
    n = len(recs)
    found, missed, rows = resolve_cases(ents)

    print(f"scored candidate set        : {n:,} entities")
    print(f"published companies         : {len({r['private_org'].strip() for r in rows if r['private_org'].strip()})}")
    print(f"exact-matched to an entity  : {len(found)}")

    orderings = {
        "our score":      sorted(recs, key=lambda r: -r["score"]),
        "prominence":     R.baseline_prominence(recs, k=n),
        "raw close-count": R.baseline_raw_close(recs, k=n),
        "ratio":          R.baseline_ratio(recs, k=n),
    }

    present = ranks_of(orderings["our score"], found)
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
        print("No published case survives into the scored set. "
              "The rank test cannot run.")
        return

    name = {r["entity"]: r for r in recs}
    print(f"{'company':<34}{'our score':>12}{'prominence':>12}"
          f"{'raw close':>12}{'ratio':>10}")
    print("-" * 80)
    per = {k: ranks_of(v, present) for k, v in orderings.items()}
    for e in sorted(present, key=lambda x: per["our score"][x]):
        label = ", ".join(found[e])[:32]
        print(f"{label:<34}"
              + "".join(f"{per[k][e]:>12,}" for k in
                        ("our score", "prominence", "raw close-count"))
              + f"{per['ratio'][e]:>10,}")

    print("\n" + "-" * 80)
    print(f"{'ordering':<20}{'AUC':>8}{'median rank':>14}{'best':>8}"
          f"{'top-10%':>10}{'top-25%':>10}")
    for k in orderings:
        rk = sorted(per[k].values())
        a = auc_from_ranks(rk, n, len(rk))
        med = rk[len(rk) // 2]
        t10 = sum(1 for r in rk if r <= 0.10 * n)
        t25 = sum(1 for r in rk if r <= 0.25 * n)
        print(f"{k:<20}{a:>8.3f}{med:>14,}{rk[0]:>8,}"
              f"{t10:>7}/{len(rk):<2}{t25:>7}/{len(rk):<2}")

    print(f"\nn = {len(present)} positives. Report the comparison between "
          f"orderings, not the absolute AUC.")


if __name__ == "__main__":
    main()
