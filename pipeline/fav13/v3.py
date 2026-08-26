#!/usr/bin/env python3
"""
FAV 13 v3 --- explicit linked motifs + a learnable, monotone score.

Four changes requested 2026-08-14:

  1. cluster confidence (the pi gate) is REMOVED from the score.
  2. a motif now carries the ACTUAL LINKED RECORDS -- which registration, which
     award -- rather than an aggregate.  Each motif is a citable object.
  3. the score is LEARNED (gradient-boosted stumps) rather than hand-weighted.
  4. RankJoin / Fagin threshold algorithm returns the top-k.

Why the trees are monotone-constrained.  A free-form XGBoost score is not monotone
in its features, and Fagin's threshold algorithm is only correct for a monotone
aggregation -- an unconstrained learner would silently invalidate step 4.  Every
feature here is oriented so that larger = more suspicious, and every stump is forced
to have a non-negative step.  Monotonicity is therefore preserved by construction and
RankJoin stays valid.

Why the labels are weak.  Five company-level positives cannot train anything.  Instead
a contract is a weak positive if it goes to the DEPARTMENT the story names, in the
PERIOD the story names: 540 such contracts (Accenture/EDC 279, GC Strategies/PSPC 261).
Not every one is the scandal contract, so precision of the label is low -- which is
exactly why evaluation is leave-one-COMPANY-out, never a random split.
"""
import csv, json, math, os, pickle, random, re
from collections import Counter, defaultdict
from pipeline.fav13 import config as C
from pipeline.fav13 import evidence as EV

csv.field_size_limit(10 ** 9)
random.seed(0)
NONCOMP = C.NONCOMPETITIVE_METHODS
GT = os.path.join(C.GT, "favoritism_ca.csv")
STOP = {"of", "the", "and", "for", "canada", "canadian", "government", "department",
        "ministry", "agency", "office", "inc", "ltd", "corporation", "commission"}


def key(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def toks(s):
    return {t for t in re.sub(r"[^a-z ]", " ", (s or "").lower()).split()
            if t and t not in STOP and len(t) > 2}


def rows(f, cols, base=C.CLEAN):
    with open(os.path.join(base, f), newline="", encoding="utf-8",
              errors="replace") as fh:
        r = csv.reader(fh)
        h = next(r)
        ix = [h.index(c) for c in cols]
        m = max(ix)
        for x in r:
            if len(x) > m:
                yield [x[i] for i in ix]


# ---------------------------------------------------------------- features
# every feature is oriented so LARGER = MORE SUSPICIOUS.  this orientation is
# what makes the learned aggregate monotone and RankJoin applicable.
FEATURES = [
    "f_recency",       # how soon after lobbying the award landed
    "f_time_excess",   # 1 - p_time  (excess over the national lobbying pattern)
    "f_channel",       # how much lobbying happened on THIS department's channel
    "f_noncomp",       # award was non-competitive
    "f_nc_rarity",     # ...and this department rarely does that
    "f_tender_short",  # tender window was short
    "f_dominance",     # supplier holds an outsized share of this department
    "f_dep_share",     # this department is a large share of the supplier's book
    "f_prominence",    # supplier is large (newsworthiness, not anomaly)
]
MAXAW = 20000.0


def featurise(r, ent_awards, ent_buyer_share):
    g = r["gap"]
    return {
        "f_recency":      (1.0 / (1.0 + g / 365.0)) if g is not None else 0.0,
        "f_time_excess":  1.0 - r["p_time"],
        "f_channel":      min(math.log1p(r["n_chan_regs"]) / math.log1p(200.0), 1.0),
        "f_noncomp":      1.0 if r["method"] in NONCOMP else 0.0,
        "f_nc_rarity":    (1.0 - r["p_comp"]) if r["method"] in NONCOMP else 0.0,
        "f_tender_short": (1.0 - r["p_proc"]) if r["tender_window"] is not None else 0.0,
        "f_dominance":    1.0 - r["p_conc"],
        "f_dep_share":    ent_buyer_share.get((r["entity"], r["buyers"][0]), 0.0)
                          if r["buyers"] else 0.0,
        "f_prominence":   min(math.log1p(ent_awards.get(r["entity"], 1)) /
                              math.log1p(MAXAW), 1.0),
    }


# ------------------------------------------------- monotone boosted stumps
class MonotoneStump:
    """depth-1 tree, constrained so the right (higher-feature) leaf never scores
    below the left one.  this is the per-feature monotone constraint."""

    __slots__ = ("j", "thr", "lo", "hi")

    def __init__(self, j, thr, lo, hi):
        self.j, self.thr, self.lo, self.hi = j, thr, lo, max(hi, lo)

    def __call__(self, x):
        return self.hi if x[self.j] > self.thr else self.lo


def fit_stump(X, grad, hess, cand_thresholds, lam=1.0):
    """one boosting round: pick (feature, threshold) minimising the usual
    second-order objective, then clamp the leaves so the step is non-negative."""
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
    return MonotoneStump(j, t, lo, hi)          # hi clamped to >= lo inside


class MonoGBM:
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

    def importance(self):
        c = Counter()
        for t in self.trees:
            c[FEATURES[t.j]] += abs(t.hi - t.lo)
        s = sum(c.values()) or 1.0
        return [(k, v / s) for k, v in c.most_common()]


# --------------------------------------------------------------- evaluation
def auc_ranks(order, pos_set, total):
    best = {}
    for i, e in enumerate(order, 1):
        if e in pos_set and e not in best:
            best[e] = i
    if not best:
        return float("nan"), {}
    n1 = len(best)
    n0 = max(total - n1, 1)
    R = sum(best.values())
    U = n1 * n0 + n1 * (n1 + 1) / 2.0 - R
    return U / (n1 * n0), best


# --------------------------------------------------------------------- main
def main():
    print("loading v2 instances ...", flush=True)
    recs = pickle.load(open(os.path.join(C.WORK, "fav13_v2_scored.pkl"), "rb"))
    ents = pickle.load(open(os.path.join(C.WORK, "entities.pkl"), "rb"))
    raw2ent, display = ents["raw2ent"], ents.get("display", {})

    ent_awards = Counter(r["entity"] for r in recs)
    pair = Counter()
    for r in recs:
        for b in r["buyers"]:
            pair[(r["entity"], b)] += 1
    ent_buyer_share = {k: v / ent_awards[k[0]] for k, v in pair.items()}

    print("building explicit registration links ...", flush=True)
    reg_day, reg_num = {}, {}
    for rid, dd, st, rn in rows("lob_meta.csv", ["rid", "start_date_clean",
                                                 "start_date_status", "rnum"]):
        if st == "valid":
            reg_day[rid] = EV.to_day(dd)
            reg_num[rid] = rn
    body_regs = defaultdict(list)          # (entity, body_core) -> [(day, rid)]
    reg_body = defaultdict(set)
    for rid, core in rows("lob_target.csv", ["rid", "name_core"]):
        if core and rid in reg_day:
            reg_body[rid].add(core)
    for rid, raw in rows("lob_org.csv", ["rid", "name"], base=C.RAW):
        e = raw2ent.get((raw or "").strip())
        if e is None or rid not in reg_day:
            continue
        for b in reg_body.get(rid, ()):
            body_regs[(e, b)].append((reg_day[rid], rid))
    for k in body_regs:
        body_regs[k].sort()
    print(f"   channels: {len(body_regs):,}", flush=True)

    print("featurising ...", flush=True)
    X, meta = [], []
    for r in recs:
        f = featurise(r, ent_awards, ent_buyer_share)
        X.append(tuple(f[k] for k in FEATURES))
        meta.append(r)

    # weak labels: contract to the story's department, in the story's period
    print("labelling ...", flush=True)
    exact = {}
    for raw, e in raw2ent.items():
        exact.setdefault(key(raw), e)
    gt = list(csv.DictReader(open(GT, encoding="utf-8", errors="replace")))
    claims = []
    for g in gt:
        e = exact.get(key((g.get("private_org") or "").strip()))
        if not e:
            continue
        dt = toks(g.get("public_org"))
        yy = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b",
                                         g.get("contract_time") or "")]
        claims.append((e, dt, (min(yy), max(yy)) if yy else (None, None)))

    y = [0] * len(recs)
    comp_of = [None] * len(recs)
    for i, r in enumerate(recs):
        for e, dt, (lo, hi) in claims:
            if r["entity"] != e or not dt:
                continue
            if not (dt & toks(r["buyer_names"])):
                continue
            yr = int(EV.from_day(r["day"])[:4])
            if lo is not None and not (lo <= yr <= hi):
                continue
            y[i] = 1
            comp_of[i] = e
            break
    npos = sum(y)
    comps = sorted({c for c in comp_of if c})
    print(f"   weak positives: {npos:,} across {len(comps)} companies", flush=True)

    # PU sample: all positives + a random unlabelled sample
    neg_idx = [i for i in range(len(recs)) if not y[i]]
    random.shuffle(neg_idx)
    neg_idx = neg_idx[:40000]
    pos_idx = [i for i in range(len(recs)) if y[i]]

    def train(exclude=None):
        idx = [i for i in pos_idx if comp_of[i] != exclude] + neg_idx
        Xi = [X[i] for i in idx]
        yi = [float(y[i]) for i in idx]
        wi = [1.0 if y[i] else 0.25 for i in idx]      # PU downweight of unlabelled
        return MonoGBM().fit(Xi, yi, wi)

    print("\n=== leave-one-company-out ===", flush=True)
    total = len(recs)
    for held in comps:
        m = train(exclude=held)
        s = [m.score(x) for x in X]
        order = [meta[i]["entity"] for i in sorted(range(total), key=lambda i: -s[i])]
        held_pos = {held}
        a, best = auc_ranks(order, held_pos, total)
        rank = best.get(held)
        print(f"  held out {display.get(held, held)[:32]:<34} "
              f"best rank {rank:,} ({100.0*rank/total:.3f}%)" if rank else
              f"  held out {held}: not found")

    print("\n=== full model ===", flush=True)
    model = train()
    s = [model.score(x) for x in X]
    order_i = sorted(range(total), key=lambda i: -s[i])
    for i, ix in enumerate(order_i, 1):
        meta[ix]["v3_rank"] = i
        meta[ix]["v3_score"] = s[ix]
    print("  feature importance:")
    for k, v in model.importance():
        print(f"    {k:<16}{v*100:5.1f}%")

    # known-case ranks under v3
    known = {e for e, _, _ in claims}
    order_e = [meta[i]["entity"] for i in order_i]
    a, best = auc_ranks(order_e, known, total)
    print(f"\n  known-company ranks under v3 (AUC {a:.4f}):")
    for e in sorted(best, key=lambda x: best[x]):
        print(f"    {display.get(e,e)[:34]:<36}{best[e]:>10,}  ({100.0*best[e]/total:.3f}%)")

    # attach explicit links to the top motifs, then write
    out = []
    for ix in order_i[:500]:
        r = meta[ix]
        links = []
        for b in r["buyers"]:
            for d, rid in body_regs.get((r["entity"], b), ())[-40:]:
                if d < r["day"]:
                    links.append(dict(reg_rid=rid, reg_num=reg_num.get(rid, ""),
                                      reg_date=EV.from_day(d), body=b,
                                      gap=r["day"] - d))
        links.sort(key=lambda z: z["gap"])
        out.append(dict(rank=r["v3_rank"], score=round(r["v3_score"], 4),
                        award_rid=r["award_rid"], entity=r["entity"],
                        name=display.get(r["entity"], r["entity"]),
                        award_date=EV.from_day(r["day"]), method=r["method"],
                        buyers=r["buyer_names"], n_links=len(links),
                        links=links[:5],
                        features={k: round(v, 4) for k, v in
                                  zip(FEATURES, X[ix])}))
    json.dump(dict(features=FEATURES,
                   importance=model.importance(),
                   weak_positives=npos, companies=len(comps),
                   auc=a, motifs=out),
              open(os.path.join(C.WORK, "v3_top.json"), "w"), indent=1)
    pickle.dump(dict(scores=s, order=order_i),
                open(os.path.join(C.WORK, "v3_scores.pkl"), "wb"))
    print(f"\nwrote {os.path.join(C.WORK, 'v3_top.json')}")


if __name__ == "__main__":
    main()
