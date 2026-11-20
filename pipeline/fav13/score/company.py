"""v1: score whole companies, with the threshold algorithm and Arm C.

Superseded by award-level scoring (a company score tracks company size), but
kept because Arm C reports it.

    python3 -m pipeline.fav13.score.company [--k K]
"""

import argparse
import time

from .. import config as C
from ..data.build import load
from . import features as F
from . import rank as R


def run(k=None, rebuild=False):
    t0 = time.time()
    ents, evi = load(rebuild)
    recs = F.compute(evi["evidence"], evi["dual"], evi["null_reference"],
                     ents["members"], charity_keys=F.load_charity_keys())
    F.normalise(recs)

    k = k or C.TOP_K
    ta = R.ThresholdAlgorithm(recs)
    top = ta.run(k)

    # TA must match exhaustive scoring. Compare scores and sets, not order:
    # categorical features tie heavily.
    bf = R.brute_force(recs, k)
    agree = ([round(a["score"], 12) for a in top] == [round(b["score"], 12) for b in bf]
             and {a["entity"] for a in top} == {b["entity"] for b in bf})

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


def main(argv=None):
    ap = argparse.ArgumentParser(description="v1 company-level baseline.")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--rebuild", action="store_true", help="rebuild caches first")
    a = ap.parse_args(argv)
    run(k=a.k, rebuild=a.rebuild)


if __name__ == "__main__":
    main()
