#!/usr/bin/env python3
"""
Arm A / Arm C evaluation: where do the known published cases land?

No classifier is trained.  With 4 usable positives and no labelled negatives that is
arithmetically impossible.  Instead this is a RANK TEST: locate the known cases in the
scored set and compare their positions under our score against baselines.  The
Mann-Whitney U over those ranks IS the AUC -- the probability a random positive
outranks a random unlabelled instance -- computed with no model fitted.

Under positive-unlabeled data the absolute AUC is biased low (some "negatives" are
unlabelled positives).  The bias is shared across scoring functions, so COMPARISONS
between them remain valid even though the absolute number does not.
"""
import csv, os, pickle, re
from collections import Counter
from pipeline.fav13 import config as C

csv.field_size_limit(10 ** 9)
GT = os.path.join(C.GT, "favoritism_ca.csv")


def key(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def load_positives(raw2ent):
    rows = list(csv.DictReader(open(GT, encoding="utf-8", errors="replace")))
    exact = {}
    for raw, e in raw2ent.items():
        exact.setdefault(key(raw), e)
    pos = {}
    unmatched = []
    for r in rows:
        nm = (r.get("private_org") or "").strip()
        if not nm:
            continue
        e = exact.get(key(nm))
        if e:
            pos.setdefault(e, set()).add(nm)
        else:
            unmatched.append(nm)
    return pos, sorted(set(unmatched)), rows


def ranks_of(order, positives):
    """order: list of entity ids best-first (one entry per instance).
    Returns best rank (1-based) per positive entity."""
    best = {}
    for i, e in enumerate(order, 1):
        if e in positives and e not in best:
            best[e] = i
    return best


def auc_from_ranks(best, n_items, n_pos_items):
    """Mann-Whitney U -> AUC, using instance-level ranks of positive entities."""
    if not best:
        return float("nan")
    R = sum(best.values())
    n1 = len(best)
    n0 = max(n_items - n_pos_items, 1)
    U = n1 * n0 + n1 * (n1 + 1) / 2.0 - R
    return U / (n1 * n0)


def report(name, order, positives, display, total):
    best = ranks_of(order, positives)
    print(f"\n--- {name} ---")
    print(f"{'company':<34}{'best rank':>11}{'percentile':>12}")
    for e in sorted(best, key=lambda x: best[x]):
        r = best[e]
        print(f"  {display.get(e, e)[:32]:<32}{r:>11,}{100.0*r/total:>11.3f}%")
    miss = [e for e in positives if e not in best]
    for e in miss:
        print(f"  {display.get(e, e)[:32]:<32}{'not scored':>11}{'':>12}")
    if best:
        rr = sorted(best.values())
        med = rr[len(rr) // 2]
        top1 = sum(1 for r in rr if r <= total * 0.01)
        top01 = sum(1 for r in rr if r <= total * 0.001)
        print(f"  found {len(best)}/{len(positives)}   median rank {med:,} "
              f"({100.0*med/total:.3f}%)   in top 1%: {top1}   top 0.1%: {top01}")
        print(f"  AUC (Mann-Whitney, rank-based)  = {auc_from_ranks(best, total, len(best)):.4f}")
    return best


def main():
    ents = pickle.load(open(os.path.join(C.WORK, "entities.pkl"), "rb"))
    raw2ent, display = ents["raw2ent"], ents.get("display", {})
    recs = pickle.load(open(os.path.join(C.WORK, "fav13_v2_scored.pkl"), "rb"))
    total = len(recs)

    pos, unmatched, gtrows = load_positives(raw2ent)
    print(f"ground truth : {len(gtrows)} rows, "
          f"{len({(r['private_org'], r['public_org']) for r in gtrows})} distinct (company, body) pairs")
    print(f"matched to an entity : {len(pos)}")
    for e, names in pos.items():
        print(f"   {display.get(e,e)[:38]:<40} <- {', '.join(sorted(names))[:52]}")
    print(f"not in the entity map ({len(unmatched)}): {', '.join(unmatched)[:150]}")
    print(f"\nscored contracts: {total:,}")

    n_pos_items = sum(1 for r in recs if r["entity"] in pos)

    ent_awards = Counter(r["entity"] for r in recs)

    orders = {
        "v2  pi x (1-p_combined) x salience":
            [r["entity"] for r in sorted(recs, key=lambda r: -r["score"])],
        "v2  surprise only  (1-p_combined)":
            [r["entity"] for r in sorted(recs, key=lambda r: r["p_combined"])],
        "BASELINE  prominence (supplier award count)":
            [r["entity"] for r in sorted(recs, key=lambda r: -ent_awards[r["entity"]])],
        "BASELINE  sole-source flag then recency":
            [r["entity"] for r in sorted(
                recs, key=lambda r: (0 if r["method"] in C.NONCOMPETITIVE_METHODS else 1,
                                     r["gap"] if r["gap"] is not None else 10 ** 9))],
        "BASELINE  raw nearest-gap (no null)":
            [r["entity"] for r in sorted(
                recs, key=lambda r: r["gap"] if r["gap"] is not None else 10 ** 9)],
    }
    results = {}
    for name, order in orders.items():
        results[name] = report(name, order, pos, display, total)

    print("\n\n=== SUMMARY: best rank per known company ===")
    names = sorted(pos, key=lambda e: display.get(e, e))
    hdr = f"{'company':<26}" + "".join(f"{n.split()[0][:11]:>13}" for n in orders)
    print(hdr)
    for e in names:
        row = f"{display.get(e,e)[:24]:<26}"
        for n in orders:
            r = results[n].get(e)
            row += f"{(f'{r:,}' if r else '—'):>13}"
        print(row)
    print(f"\n{'AUC':<26}" + "".join(
        f"{auc_from_ranks(results[n], total, n_pos_items):>13.4f}" for n in orders))


if __name__ == "__main__":
    main()
