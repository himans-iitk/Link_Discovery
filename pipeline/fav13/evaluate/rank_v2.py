"""Arm A / C for v2: where do the published cases land among scored contracts?

Rank test, no model fitted. Mann-Whitney U over the ranks is the AUC;
compare orderings, not absolute values (PU bias is shared).

    python3 -m pipeline.fav13.evaluate.rank_v2
"""

from collections import Counter

from .. import common
from ..common import NONCOMP

NO_GAP = 10 ** 9


def load_positives(raw2ent, stories):
    """-> ({entity: story company names}, unmatched names)"""
    index = common.name_index(raw2ent)
    pos, unmatched = {}, []
    for r in stories:
        nm = (r.get("private_org") or "").strip()
        if not nm:
            continue
        e = index.get(common.name_key(nm))
        if e:
            pos.setdefault(e, set()).add(nm)
        else:
            unmatched.append(nm)
    return pos, sorted(set(unmatched))


def report(name, order, positives, display, total):
    best = common.best_ranks(order, positives)
    print(f"\n--- {name} ---")
    print(f"{'company':<34}{'best rank':>11}{'percentile':>12}")
    for e in sorted(best, key=lambda x: best[x]):
        print(f"  {display.get(e, e)[:32]:<32}{best[e]:>11,}{100.0*best[e]/total:>11.3f}%")
    for e in positives:
        if e not in best:
            print(f"  {display.get(e, e)[:32]:<32}{'not scored':>11}{'':>12}")
    if best:
        rr = sorted(best.values())
        med = rr[len(rr) // 2]
        top1 = sum(1 for r in rr if r <= total * 0.01)
        top01 = sum(1 for r in rr if r <= total * 0.001)
        print(f"  found {len(best)}/{len(positives)}   median rank {med:,} "
              f"({100.0*med/total:.3f}%)   in top 1%: {top1}   top 0.1%: {top01}")
        print(f"  AUC (Mann-Whitney, rank-based)  = {common.auc(best, total):.4f}")
    return best


def main():
    ents = common.load_entities()
    display = ents.get("display", {})
    recs = common.load_scored()
    total = len(recs)
    stories = common.read_stories()

    pos, unmatched = load_positives(ents["raw2ent"], stories)
    print(f"ground truth : {len(stories)} rows, "
          f"{len({(r['private_org'], r['public_org']) for r in stories})} distinct (company, body) pairs")
    print(f"matched to an entity : {len(pos)}")
    for e, names in pos.items():
        print(f"   {display.get(e,e)[:38]:<40} <- {', '.join(sorted(names))[:52]}")
    print(f"not in the entity map ({len(unmatched)}): {', '.join(unmatched)[:150]}")
    print(f"\nscored contracts: {total:,}")

    n_pos_items = sum(1 for r in recs if r["entity"] in pos)
    ent_awards = Counter(r["entity"] for r in recs)
    gap = lambda r: r["gap"] if r["gap"] is not None else NO_GAP

    def by(key):
        return [r["entity"] for r in sorted(recs, key=key)]

    orders = {
        "v2  pi x (1-p_combined) x salience":          by(lambda r: -r["score"]),
        "v2  surprise only  (1-p_combined)":           by(lambda r: r["p_combined"]),
        "BASELINE  prominence (supplier award count)": by(lambda r: -ent_awards[r["entity"]]),
        "BASELINE  sole-source flag then recency":
            by(lambda r: (0 if r["method"] in NONCOMP else 1, gap(r))),
        "BASELINE  raw nearest-gap (no null)":         by(gap),
    }
    results = {name: report(name, order, pos, display, total)
               for name, order in orders.items()}

    print("\n\n=== SUMMARY: best rank per known company ===")
    print(f"{'company':<26}" + "".join(f"{n.split()[0][:11]:>13}" for n in orders))
    for e in sorted(pos, key=lambda e: display.get(e, e)):
        print(f"{display.get(e,e)[:24]:<26}" + "".join(
            f"{(f'{results[n][e]:,}' if results[n].get(e) else '—'):>13}" for n in orders))
    print(f"\n{'AUC':<26}" + "".join(
        f"{common.auc(results[n], total, n_pos_items):>13.4f}" for n in orders))


if __name__ == "__main__":
    main()
