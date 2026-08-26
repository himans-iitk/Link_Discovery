"""
Scoring and top-k retrieval.

The aggregation is a weighted sum of features in [0,1] with non-negative
weights, so it is monotone non-decreasing in every argument.  That is the
precondition for Fagin's Threshold Algorithm (Fagin, Lotem & Naor, JCSS 66(4),
2003), which is *instance optimal*: no correct algorithm with the same access
model reads substantially fewer entries.

Why bother, on 1,022 rows?  Two reasons, and neither is today's row count.
Scoring the closed FAV 12 loop is per (lobbying episode, award) pair -- about
46 billion for `lob_target x pro_contact` -- and possible-worlds resampling
multiplies whatever we do by the number of replicates.  The access pattern has
to be right before either arrives.

TA also gives a number worth reporting: how much of the candidate set was
never touched.
"""

import heapq

from . import config as C
from . import features as F


def score(rec, weights=None):
    w = weights or C.WEIGHTS
    return sum(w.get(f, 0.0) * rec.get("f_" + f, 0.0) for f in F.FEATURES)


class ThresholdAlgorithm:
    """Fagin TA over per-feature sorted lists.

    sorted access  -- walk each feature's list in descending order
    random access  -- look up an entity's remaining feature values
    threshold      -- aggregate of the values at the current depth; once the
                      k-th best score reaches it, nothing unseen can beat it
    """

    def __init__(self, recs, weights=None, features=None):
        self.recs = {r["entity"]: r for r in recs}
        self.weights = weights or C.WEIGHTS
        self.features = features or F.FEATURES
        self.lists = {
            f: sorted(recs, key=lambda r, f=f: -r.get("f_" + f, 0.0))
            for f in self.features
        }
        self.sorted_accesses = 0
        self.random_accesses = 0
        self.depth = 0

    def run(self, k=None):
        k = k or C.TOP_K
        seen = set()
        heap = []                       # min-heap of (score, entity)
        n = len(self.recs)

        for d in range(n):
            self.depth = d + 1
            threshold = 0.0
            for f in self.features:
                row = self.lists[f][d]
                self.sorted_accesses += 1
                threshold += self.weights.get(f, 0.0) * row.get("f_" + f, 0.0)

                e = row["entity"]
                if e not in seen:
                    seen.add(e)
                    self.random_accesses += 1
                    s = score(row, self.weights)
                    if len(heap) < k:
                        heapq.heappush(heap, (s, e))
                    elif s > heap[0][0]:
                        heapq.heapreplace(heap, (s, e))

            if len(heap) == k and heap[0][0] >= threshold:
                break

        top = sorted(heap, key=lambda t: -t[0])
        return [dict(self.recs[e], score=s) for s, e in top]

    def stats(self, k):
        n = len(self.recs)
        full = n * len(self.features)
        return dict(entities=n, k=k, depth=self.depth,
                    sorted_accesses=self.sorted_accesses,
                    random_accesses=self.random_accesses,
                    entities_touched=self.random_accesses,
                    fraction_scanned=round(self.sorted_accesses / max(full, 1), 4))


def brute_force(recs, k=None, weights=None):
    """Reference implementation -- used to assert TA returns the same top-k."""
    k = k or C.TOP_K
    scored = [dict(r, score=score(r, weights)) for r in recs]
    return sorted(scored, key=lambda r: -r["score"])[:k]


# ------------------------------------------------------------------- baselines
# Arm C of the evaluation.  Each must be beaten, and the first one is the
# experiment that can kill the thesis: published stories are about prominent
# organisations, and prominence is free.

def baseline_prominence(recs, k=None):
    return sorted(recs, key=lambda r: -r["volume"])[:(k or C.TOP_K)]


def baseline_raw_close(recs, k=None):
    """Close pairs with no null correction -- the naive statistic."""
    return sorted(recs, key=lambda r: -r["obs"])[:(k or C.TOP_K)]


def baseline_ratio(recs, k=None):
    return sorted(recs, key=lambda r: (-r["ratio"], -r["obs"]))[:(k or C.TOP_K)]


def spearman(rank_a, rank_b):
    keys = [k for k in rank_a if k in rank_b]
    n = len(keys)
    if n < 3:
        return float("nan")
    d2 = sum((rank_a[k] - rank_b[k]) ** 2 for k in keys)
    return 1 - 6 * d2 / (n * (n * n - 1))


def ranking_of(ordered):
    return {r["entity"]: i for i, r in enumerate(ordered)}
