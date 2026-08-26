"""
Cache bootstrap, and the company-level baseline the award-level model replaced.

Everything downstream reads two pickles out of WORK/:

    entities.pkl   raw name -> entity id, members, display names
    evidence.pkl   per-entity lobbying and award day-numbers, plus the
                   national lobbying calendar used as the null reference

Build them once:

    python3 -m pipeline.fav13.run --build

`baseline()` scores one company as a whole, which is the v1 grain.  It is kept
because Arm C reports it: a company-level score correlates with company size
(rho vs prominence below), and that confound is the reason the motif grain
moved to one award.  `awards.py` is the current pipeline.

    python3 -m pipeline.fav13.run            company-level baseline + Arm C
"""
import argparse, os, pickle, time
from . import config as C
from . import entities as EN
from . import evidence as EV
from . import features as F
from . import rank as R


def load(rebuild=False):
    """Load the two caches, building whichever is missing."""
    ep = os.path.join(C.WORK, "entities.pkl")
    vp = os.path.join(C.WORK, "evidence.pkl")
    if rebuild:
        for p in (ep, vp):
            if os.path.exists(p):
                os.remove(p)
    if not os.path.exists(ep):
        EN.save(ep)
    ents = pickle.load(open(ep, "rb"))
    if not os.path.exists(vp):
        EV.save(ents["raw2ent"], vp)
    return ents, pickle.load(open(vp, "rb"))


def baseline(k=None, rebuild=False):
    t0 = time.time()
    ents, evi = load(rebuild)
    recs = F.compute(evi["evidence"], evi["dual"], evi["null_reference"],
                     ents["members"], charity_keys=F.load_charity_keys())
    F.normalise(recs)

    k = k or C.TOP_K
    ta = R.ThresholdAlgorithm(recs)
    top = ta.run(k)

    # Correctness: TA must return a top-k that exhaustive scoring also accepts.
    # Compare score sequences and entity sets, not entity order -- categorical
    # features tie heavily, and any order within a tie group is equally correct.
    bf = R.brute_force(recs, k)
    agree = ([round(a["score"], 12) for a in top] ==
             [round(b["score"], 12) for b in bf] and
             {a["entity"] for a in top} == {b["entity"] for b in bf})

    st = ta.stats(k)
    print(f"entities on both sides   : {len(evi['dual']):,}")
    print(f"scored (obs >= {C.MIN_OBS})      : {len(recs):,}")
    print(f"threshold algorithm      : depth {st['depth']:,} of {st['entities']:,}"
          f"  ({100*st['fraction_scanned']:.1f}% scanned,"
          f" {st['entities_touched']:,} scored)")
    print(f"matches exhaustive top-{k}: {agree}")
    print(f"elapsed                  : {time.time()-t0:.1f}s")

    # Arm C: how much of the ranking is just company size?
    rk_prom = R.ranking_of(R.baseline_prominence(recs, len(recs)))
    print("\nSpearman vs prominence (Arm C):")
    for nm, ordered in [
            ("raw close-pair count", R.baseline_raw_close(recs, len(recs))),
            ("ratio obs/expected", R.baseline_ratio(recs, len(recs))),
            ("full score", R.brute_force(recs, len(recs)))]:
        print(f"   {nm:<22} rho = {R.spearman(rk_prom, R.ranking_of(ordered)):+.3f}")
    return recs, top


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true",
                    help="build the caches and exit")
    ap.add_argument("--rebuild", action="store_true",
                    help="discard and rebuild the caches")
    ap.add_argument("--k", type=int, default=None)
    a = ap.parse_args()
    if a.build:
        ents, evi = load(a.rebuild)
        print(f"entities.pkl  {len(ents['raw2ent']):,} names -> "
              f"{len(set(ents['raw2ent'].values())):,} entities")
        print(f"evidence.pkl  {len(evi['evidence']):,} entities, "
              f"{len(evi['dual']):,} on both sides")
    else:
        baseline(k=a.k, rebuild=a.rebuild)
