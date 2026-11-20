"""v3: learned, monotone score over explicitly linked motifs.

  - no identity gate in the score
  - each motif carries its actual linked registrations (citable)
  - score = boosted stumps, each forced to a non-negative step, so the score
    stays monotone and the threshold algorithm stays valid
  - weak labels: a contract is positive if it goes to the department, in the
    period, a published story names (~540 contracts, 2 companies).
    Labels are noisy, so evaluation is leave-one-COMPANY-out.

Result: the hand rule beats this model (see v3_ablation).

    python3 -m pipeline.fav13.score.v3     # -> work/v3_top.json, v3_scores.pkl
"""

import json
import math
import random
from collections import Counter

from pipeline import paths
from .. import common
from ..common import NONCOMP, from_day

# Every feature oriented so larger = more suspicious (keeps the aggregate monotone).
FEATURES = [
    "f_recency",       # how soon after lobbying
    "f_time_excess",   # 1 - p_time
    "f_channel",       # amount of lobbying on THIS department's channel
    "f_noncomp",       # award was non-competitive
    "f_nc_rarity",     # ...and this department rarely does that
    "f_tender_short",  # short tender window
    "f_dominance",     # supplier's outsized share of this department
    "f_dep_share",     # this department's share of the supplier's book
    "f_prominence",    # supplier size (newsworthiness, not anomaly)
]
MAX_AWARDS = 20000.0


def featurise(r, ent_awards, ent_buyer_share):
    g = r["gap"]
    nc = r["method"] in NONCOMP
    return {
        "f_recency":      (1.0 / (1.0 + g / 365.0)) if g is not None else 0.0,
        "f_time_excess":  1.0 - r["p_time"],
        "f_channel":      min(math.log1p(r["n_chan_regs"]) / math.log1p(200.0), 1.0),
        "f_noncomp":      1.0 if nc else 0.0,
        "f_nc_rarity":    (1.0 - r["p_comp"]) if nc else 0.0,
        "f_tender_short": (1.0 - r["p_proc"]) if r["tender_window"] is not None else 0.0,
        "f_dominance":    1.0 - r["p_conc"],
        "f_dep_share":    ent_buyer_share.get((r["entity"], r["buyers"][0]), 0.0)
                          if r["buyers"] else 0.0,
        "f_prominence":   min(math.log1p(ent_awards.get(r["entity"], 1)) /
                              math.log1p(MAX_AWARDS), 1.0),
    }


def feature_matrix(recs):
    """-> (X as tuples in FEATURES order, contracts per entity)"""
    ent_awards = Counter(r["entity"] for r in recs)
    pair = Counter()
    for r in recs:
        for b in r["buyers"]:
            pair[(r["entity"], b)] += 1
    share = {k: v / ent_awards[k[0]] for k, v in pair.items()}
    X = [tuple(featurise(r, ent_awards, share)[k] for k in FEATURES) for r in recs]
    return X, ent_awards


def weak_labels(recs, claims):
    """-> (y, company of each positive). Positive = story's department and period."""
    y, comp = [0] * len(recs), [None] * len(recs)
    for i, r in enumerate(recs):
        for e, dt, (lo, hi) in claims:
            if r["entity"] == e and common.dept_match(r, dt) and common.in_period(r, lo, hi):
                y[i], comp[i] = 1, e
                break
    return y, comp


# ---------------------------------------------------------------- monotone boosted stumps

class MonotoneStump:
    """Depth-1 tree whose right (higher-feature) leaf never scores below the left."""

    __slots__ = ("j", "thr", "lo", "hi")

    def __init__(self, j, thr, lo, hi):
        self.j, self.thr, self.lo, self.hi = j, thr, lo, max(hi, lo)

    def __call__(self, x):
        return self.hi if x[self.j] > self.thr else self.lo


def fit_stump(X, grad, hess, cand_thresholds, lam=1.0):
    """Best (feature, threshold) by the second-order gain; leaves clamped monotone."""
    G, H = sum(grad), sum(hess)
    best = None
    for j, thrs in enumerate(cand_thresholds):
        for t in thrs:
            gl = hl = 0.0
            for x, g_, h_ in zip(X, grad, hess):
                if x[j] <= t:
                    gl += g_
                    hl += h_
            gr, hr = G - gl, H - hl
            if hl <= 0 or hr <= 0:
                continue
            gain = gl * gl / (hl + lam) + gr * gr / (hr + lam) - G * G / (H + lam)
            if best is None or gain > best[0]:
                best = (gain, j, t, -gl / (hl + lam), -gr / (hr + lam))
    if best is None:
        return None
    _, j, t, lo, hi = best
    return MonotoneStump(j, t, lo, hi)


class MonoGBM:
    """Logistic boosting with monotone stumps."""

    def __init__(self, rounds=60, lr=0.12, lam=1.0, nbins=12):
        self.rounds, self.lr, self.lam, self.nbins = rounds, lr, lam, nbins
        self.trees, self.base = [], 0.0

    def _thresholds(self, X):
        out = []
        for j in range(len(X[0])):
            v = sorted({x[j] for x in X})
            if len(v) <= self.nbins:
                out.append(v[:-1] if len(v) > 1 else v)
            else:
                step = len(v) / (self.nbins + 1)
                out.append([v[int((i + 1) * step)] for i in range(self.nbins)])
        return out

    def fit(self, X, y, w):
        p = sum(a * b for a, b in zip(y, w)) / max(sum(w), 1e-9)
        p = min(max(p, 1e-6), 1 - 1e-6)
        self.base = math.log(p / (1 - p))
        F = [self.base] * len(X)
        thr = self._thresholds(X)
        for _ in range(self.rounds):
            grad, hess = [], []
            for f, yi, wi in zip(F, y, w):
                pi = 1.0 / (1.0 + math.exp(-max(min(f, 30), -30)))
                grad.append(wi * (pi - yi))
                hess.append(wi * pi * (1 - pi) + 1e-6)
            st = fit_stump(X, grad, hess, thr, self.lam)
            if st is None:
                break
            self.trees.append(st)
            for i, x in enumerate(X):
                F[i] += self.lr * st(x)
        return self

    def score(self, x):
        return self.base + self.lr * sum(t(x) for t in self.trees)

    def importance(self, names=FEATURES):
        c = Counter()
        for t in self.trees:
            c[names[t.j]] += abs(t.hi - t.lo)
        s = sum(c.values()) or 1.0
        return [(k, v / s) for k, v in c.most_common()]


def train(X, y, comp, pos_idx, neg_idx, exclude=None, **params):
    """Fit on all positives (minus one held-out company) + sampled unlabelled,
    with unlabelled down-weighted (PU)."""
    idx = [i for i in pos_idx if comp[i] != exclude] + neg_idx
    return MonoGBM(**params).fit([X[i] for i in idx], [float(y[i]) for i in idx],
                                 [1.0 if y[i] else 0.25 for i in idx])


def order_by(model, X, recs):
    """-> (scores, entity ids best-first)"""
    s = [model.score(x) for x in X]
    return s, [recs[i]["entity"] for i in sorted(range(len(X)), key=lambda i: -s[i])]


# ---------------------------------------------------------------- main

def main():
    random.seed(0)
    print("loading v2 instances ...", flush=True)
    recs = common.load_scored()
    ents = common.load_entities()
    display = ents.get("display", {})
    total = len(recs)

    print("building explicit registration links ...", flush=True)
    reg_day, reg_num = common.registrations(with_rnum=True)
    body_regs, _ = common.lobby_channels(ents["raw2ent"], reg_day)
    print(f"   channels: {len(body_regs):,}", flush=True)

    print("featurising ...", flush=True)
    X, _ = feature_matrix(recs)

    print("labelling ...", flush=True)
    claims = common.story_claims(common.name_index(ents["raw2ent"]))
    y, comp = weak_labels(recs, claims)
    comps = sorted({c for c in comp if c})
    print(f"   weak positives: {sum(y):,} across {len(comps)} companies", flush=True)

    neg_idx = [i for i in range(total) if not y[i]]
    random.shuffle(neg_idx)
    neg_idx = neg_idx[:40000]
    pos_idx = [i for i in range(total) if y[i]]

    print("\n=== leave-one-company-out ===", flush=True)
    for held in comps:
        _, order = order_by(train(X, y, comp, pos_idx, neg_idx, exclude=held), X, recs)
        rank = common.best_ranks(order, {held}).get(held)
        print(f"  held out {display.get(held, held)[:32]:<34} "
              f"best rank {rank:,} ({100.0*rank/total:.3f}%)" if rank else
              f"  held out {held}: not found")

    print("\n=== full model ===", flush=True)
    model = train(X, y, comp, pos_idx, neg_idx)
    s = [model.score(x) for x in X]
    order_i = sorted(range(total), key=lambda i: -s[i])
    print("  feature importance:")
    for k, v in model.importance():
        print(f"    {k:<16}{v*100:5.1f}%")

    known = {e for e, _, _ in claims}
    best = common.best_ranks([recs[i]["entity"] for i in order_i], known)
    a = common.auc(best, total)
    print(f"\n  known-company ranks under v3 (AUC {a:.4f}):")
    for e in sorted(best, key=lambda x: best[x]):
        print(f"    {display.get(e,e)[:34]:<36}{best[e]:>10,}  ({100.0*best[e]/total:.3f}%)")

    # top motifs with their explicit registration links
    out = []
    for rank, ix in enumerate(order_i[:500], 1):
        r = recs[ix]
        links = [dict(reg_rid=rid, reg_num=reg_num.get(rid, ""), reg_date=from_day(d),
                      body=b, gap=r["day"] - d)
                 for b in r["buyers"]
                 for d, rid in body_regs.get((r["entity"], b), ())[-40:]
                 if d < r["day"]]
        links.sort(key=lambda z: z["gap"])
        out.append(dict(rank=rank, score=round(s[ix], 4),
                        award_rid=r["award_rid"], entity=r["entity"],
                        name=display.get(r["entity"], r["entity"]),
                        award_date=from_day(r["day"]), method=r["method"],
                        buyers=r["buyer_names"], n_links=len(links), links=links[:5],
                        features={k: round(v, 4) for k, v in zip(FEATURES, X[ix])}))

    with open(paths.WORK / "v3_top.json", "w") as fh:
        json.dump(dict(features=FEATURES, importance=model.importance(),
                       weak_positives=sum(y), companies=len(comps),
                       auc=a, motifs=out), fh, indent=1)
    common.dump(dict(scores=s, order=order_i), "v3_scores.pkl")
    print(f"\nwrote {paths.WORK / 'v3_top.json'}")


if __name__ == "__main__":
    main()
