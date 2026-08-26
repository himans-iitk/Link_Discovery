#!/usr/bin/env python3
"""
v3b -- ablation: does removing f_prominence force the learner onto real signals?

Runs the monotone boosted-stump model twice (with and without the prominence
feature), reports leave-one-company-out for each, and puts both next to the
baselines in a single table so the comparison is honest.

Prominence is the only feature that is not a form of anomaly.  In v3 the learner
put 47.2% of its weight on it and ignored non-competitiveness, tender window,
dominance and recency entirely.  If removing it does not move the known cases,
the labels are the problem, not the feature set.
"""
import csv, json, os, pickle, random, re
from collections import Counter
from pipeline.fav13 import config as C
from pipeline.fav13 import evidence as EV
from pipeline.fav13.v3 import (MonoGBM, FEATURES, featurise, auc_ranks,
                               key, toks, GT)

random.seed(0)
NONCOMP = C.NONCOMPETITIVE_METHODS


def build_matrix():
    recs = pickle.load(open(os.path.join(C.WORK, "fav13_v2_scored.pkl"), "rb"))
    ents = pickle.load(open(os.path.join(C.WORK, "entities.pkl"), "rb"))
    raw2ent, display = ents["raw2ent"], ents.get("display", {})

    ent_awards = Counter(r["entity"] for r in recs)
    pair = Counter()
    for r in recs:
        for b in r["buyers"]:
            pair[(r["entity"], b)] += 1
    share = {k: v / ent_awards[k[0]] for k, v in pair.items()}

    X = [tuple(featurise(r, ent_awards, share)[k] for k in FEATURES) for r in recs]

    exact = {}
    for raw, e in raw2ent.items():
        exact.setdefault(key(raw), e)
    claims = []
    for g in csv.DictReader(open(GT, encoding="utf-8", errors="replace")):
        e = exact.get(key((g.get("private_org") or "").strip()))
        if not e:
            continue
        yy = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b",
                                         g.get("contract_time") or "")]
        claims.append((e, toks(g.get("public_org")),
                       (min(yy), max(yy)) if yy else (None, None)))

    y = [0] * len(recs)
    comp = [None] * len(recs)
    for i, r in enumerate(recs):
        for e, dt, (lo, hi) in claims:
            if r["entity"] != e or not dt or not (dt & toks(r["buyer_names"])):
                continue
            yr = int(EV.from_day(r["day"])[:4])
            if lo is not None and not (lo <= yr <= hi):
                continue
            y[i] = 1
            comp[i] = e
            break
    known = {e for e, _, _ in claims}
    return recs, X, y, comp, known, display, ent_awards


def subset(X, drop):
    keep = [i for i, f in enumerate(FEATURES) if f not in drop]
    names = [FEATURES[i] for i in keep]
    return [tuple(x[i] for i in keep) for x in X], names


def run_variant(label, X, names, y, comp, known, recs, display, pos_idx, neg_idx):
    total = len(X)

    def train(exclude=None):
        idx = [i for i in pos_idx if comp[i] != exclude] + neg_idx
        m = MonoGBM(rounds=45, lr=0.14)
        m.fit([X[i] for i in idx], [float(y[i]) for i in idx],
              [1.0 if y[i] else 0.25 for i in idx])
        return m

    comps = sorted({c for c in comp if c})
    loco = {}
    for held in comps:
        m = train(exclude=held)
        s = [m.score(x) for x in X]
        order = [recs[i]["entity"] for i in sorted(range(total), key=lambda i: -s[i])]
        _, best = auc_ranks(order, {held}, total)
        loco[held] = best.get(held)

    m = train()
    s = [m.score(x) for x in X]
    order = [recs[i]["entity"] for i in sorted(range(total), key=lambda i: -s[i])]
    auc, best = auc_ranks(order, known, total)

    imp = Counter()
    for t in m.trees:
        imp[names[t.j]] += abs(t.hi - t.lo)
    tot = sum(imp.values()) or 1.0
    return dict(label=label, auc=auc, ranks=best, loco=loco,
                importance=[(k, v / tot) for k, v in imp.most_common()],
                n_features=len(names))


def baselines(recs, known, ent_awards):
    total = len(recs)
    out = {}
    orders = {
        "BASELINE sole-source then recency":
            sorted(recs, key=lambda r: (0 if r["method"] in NONCOMP else 1,
                                        r["gap"] if r["gap"] is not None else 10 ** 9)),
        "BASELINE raw nearest-gap":
            sorted(recs, key=lambda r: r["gap"] if r["gap"] is not None else 10 ** 9),
        "v2 surprise only":
            sorted(recs, key=lambda r: r["p_combined"]),
        "BASELINE prominence":
            sorted(recs, key=lambda r: -ent_awards[r["entity"]]),
    }
    for k, o in orders.items():
        auc, best = auc_ranks([r["entity"] for r in o], known, total)
        out[k] = dict(label=k, auc=auc, ranks=best, loco={}, importance=[])
    return out


def main():
    print("building matrix ...", flush=True)
    recs, X, y, comp, known, display, ent_awards = build_matrix()
    total = len(recs)
    pos_idx = [i for i in range(total) if y[i]]
    neg_idx = [i for i in range(total) if not y[i]]
    random.shuffle(neg_idx)
    neg_idx = neg_idx[:25000]
    print(f"   {total:,} instances, {len(pos_idx)} weak positives, "
          f"{len(neg_idx):,} sampled unlabelled", flush=True)

    results = []
    for label, drop in [("v3  all 9 features", set()),
                        ("v3b without prominence", {"f_prominence"}),
                        ("v3c anomaly signals only", {"f_prominence", "f_channel"})]:
        Xs, names = subset(X, drop)
        print(f"\ntraining {label}  ({len(names)} features) ...", flush=True)
        r = run_variant(label, Xs, names, y, comp, known, recs, display,
                        pos_idx, neg_idx)
        results.append(r)
        print("   importance: " + ", ".join(f"{k} {v*100:.0f}%"
                                            for k, v in r["importance"]), flush=True)

    base = baselines(recs, known, ent_awards)

    print("\n\n" + "=" * 92)
    print("KNOWN-CASE RANKS  (of {:,} contracts)".format(total))
    print("=" * 92)
    names = sorted(known, key=lambda e: display.get(e, e))
    hdr = f"{'company':<26}" + "".join(f"{r['label'].split()[0][:10]:>12}"
                                       for r in results)
    hdr += "".join(f"{k.split()[-1][:10]:>12}" for k in base)
    print(hdr)
    for e in names:
        row = f"{display.get(e,e)[:24]:<26}"
        for r in results:
            v = r["ranks"].get(e)
            row += f"{(f'{v:,}' if v else '-'):>12}"
        for k in base:
            v = base[k]["ranks"].get(e)
            row += f"{(f'{v:,}' if v else '-'):>12}"
        print(row)
    print("-" * 92)
    row = f"{'AUC':<26}"
    for r in results:
        row += f"{r['auc']:>12.4f}"
    for k in base:
        row += f"{base[k]['auc']:>12.4f}"
    print(row)

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

    json.dump(dict(results=[{k: v for k, v in r.items() if k != "ranks"}
                            for r in results],
                   ranks={r["label"]: {display.get(e, e): v
                                       for e, v in r["ranks"].items()}
                          for r in results},
                   baselines={k: dict(auc=v["auc"],
                                      ranks={display.get(e, e): x
                                             for e, x in v["ranks"].items()})
                              for k, v in base.items()},
                   total=total),
              open(os.path.join(C.WORK, "v3b_results.json"), "w"), indent=1)
    print("\nwrote v3b_results.json")


if __name__ == "__main__":
    main()
