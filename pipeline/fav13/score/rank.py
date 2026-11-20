"""Top-k retrieval: Fagin's Threshold Algorithm (JCSS 2003), plus baselines.

Valid because the aggregate is a non-negative weighted sum of [0,1] features
(monotone). Sized for the FAV 12 pair space (~46 B pairs), not today's rows;
`stats()` reports how much of the candidate set was never touched.
"""

import heapq

from .. import config as C
from . import features as F


def score(rec, weights=None):
    w = weights or C.WEIGHTS
    return sum(w.get(f, 0.0) * rec.get("f_" + f, 0.0) for f in F.FEATURES)


class ThresholdAlgorithm:
    """Sorted access down each feature list; stop once the k-th best score
    reaches the aggregate of the current depth (nothing unseen can beat it)."""

    def __init__(self, recs, weights=None, features=None):
        self.recs = {r["entity"]: r for r in recs}
        self.weights = weights or C.WEIGHTS
        self.features = features or F.FEATURES
        self.lists = {f: sorted(recs, key=lambda r, f=f: -r.get("f_" + f, 0.0))
                      for f in self.features}
        self.sorted_accesses = self.random_accesses = self.depth = 0

    def run(self, k=None):
        k = k or C.TOP_K
        seen, heap = set(), []                     # min-heap of (score, entity)
        for d in range(len(self.recs)):
            self.depth = d + 1
            threshold = 0.0
            for f in self.features:
                row = self.lists[f][d]
                self.sorted_accesses += 1
                threshold += self.weights.get(f, 0.0) * row.get("f_" + f, 0.0)
                e = row["entity"]
                if e in seen:
                    continue
                seen.add(e)
                self.random_accesses += 1
                s = score(row, self.weights)
                if len(heap) < k:
                    heapq.heappush(heap, (s, e))
                elif s > heap[0][0]:
                    heapq.heapreplace(heap, (s, e))
            if len(heap) == k and heap[0][0] >= threshold:
                break
        return [dict(self.recs[e], score=s) for s, e in sorted(heap, key=lambda t: -t[0])]

    def stats(self, k):
        n = len(self.recs)
        return dict(entities=n, k=k, depth=self.depth,
                    sorted_accesses=self.sorted_accesses,
                    random_accesses=self.random_accesses,
                    entities_touched=self.random_accesses,
                    fraction_scanned=round(self.sorted_accesses /
                                           max(n * len(self.features), 1), 4))


def brute_force(recs, k=None, weights=None):
    """Exhaustive reference for checking TA."""
    scored = [dict(r, score=score(r, weights)) for r in recs]
    return sorted(scored, key=lambda r: -r["score"])[:k or C.TOP_K]


# ---------------------------------------------------------------- baselines
# Each must be beaten. Prominence is the one that can kill the thesis.

def baseline_prominence(recs, k=None):
    return sorted(recs, key=lambda r: -r["volume"])[:k or C.TOP_K]


def baseline_raw_close(recs, k=None):
    """Close pairs with no null correction."""
    return sorted(recs, key=lambda r: -r["obs"])[:k or C.TOP_K]


def baseline_ratio(recs, k=None):
    return sorted(recs, key=lambda r: (-r["ratio"], -r["obs"]))[:k or C.TOP_K]


def spearman(rank_a, rank_b):
    keys = [k for k in rank_a if k in rank_b]
    n = len(keys)
    if n < 3:
        return float("nan")
    d2 = sum((rank_a[k] - rank_b[k]) ** 2 for k in keys)
    return 1 - 6 * d2 / (n * (n * n - 1))


def ranking_of(ordered):
    return {r["entity"]: i for i, r in enumerate(ordered)}
