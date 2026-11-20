"""v2 scorer: calibrated p-values per contract, combined without weights.

  1. channel-scoped motif: one row per contract; only registrations that
     lobbied the *buying* department count
  2. four component p-values, each against its own null
       p_time  nearest channel registration vs the national lobbying calendar
               (no window: P(>=1 of n lands within the observed gap))
       p_comp  sole-source vs this buyer's own base rate
       p_proc  tender window vs the national window distribution
       p_conc  supplier's share of this buyer's book vs an even split
  3. Brown's method combines them (dependence replaces weights)
  4. score = identity gate * (1 - p) * salience;  BH q-values on p

contract_value is not in the corpus; it slots in as one more component.

    python3 -m pipeline.fav13.score.v2     # -> work/fav13_v2_scored.pkl, fav13_v2_top.csv
"""

import bisect
import csv
import math
from collections import Counter, defaultdict

from pipeline import paths
from .. import common
from ..common import NONCOMP, rows, to_day
from .features import identity_confidence

EPS = 1e-12
COMPONENTS = ("p_time", "p_comp", "p_proc", "p_conc")


# ---------------------------------------------------------------- distributions

def _gammaincc(a, x):
    """Regularised upper incomplete gamma Q(a, x)."""
    if x < 0 or a <= 0 or x == 0:
        return 1.0
    if x < a + 1.0:                                   # series for P, return 1 - P
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-14:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    tiny = 1e-300                                     # continued fraction for Q
    b, c_, d = x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a)
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c_ = b + an / c_
        if abs(c_) < tiny:
            c_ = tiny
        d = 1.0 / d
        de = d * c_
        h *= de
        if abs(de - 1.0) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def chi2_sf(x, df):
    """P(X > x), X ~ chi-square(df)."""
    return 1.0 if x <= 0 else max(_gammaincc(df / 2.0, x / 2.0), 1e-300)


def binom_sf(k, n, p):
    """P(X >= k), X ~ Binomial(n, p); k is small here."""
    if k <= 0:
        return 1.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    s = 0.0
    for i in range(k):
        s += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                      + i * math.log(p) + (n - i) * math.log1p(-p))
        if s >= 1.0:
            return 1e-300
    return max(1.0 - s, 1e-300)


# ---------------------------------------------------------------- motif instances

def build():
    ents = common.load_entities()
    raw2ent, members, display = ents["raw2ent"], ents["members"], ents.get("display", {})
    log = lambda s: print(s, flush=True)

    vocab, vlist = {}, []                 # body/buyer name <-> small int id

    def vid(s):
        i = vocab.get(s)
        if i is None:
            i = vocab[s] = len(vlist)
            vlist.append(s)
        return i

    log("[1/7] lobbying side ...")
    reg_day = common.registrations()
    reg_bodies = defaultdict(set)
    for rid, core in rows("lob_target.csv", ["rid", "name_core"]):
        if core and rid in reg_day:
            reg_bodies[rid].add(vid(core))

    channel = defaultdict(lambda: defaultdict(list))   # entity -> body -> sorted days
    ent_reg_days = defaultdict(list)
    for rid, raw in rows("lob_org.csv", ["rid", "name"], base=paths.RAW):
        e = raw2ent.get((raw or "").strip())
        if e is None or rid not in reg_day:
            continue
        d = reg_day[rid]
        ent_reg_days[e].append(d)
        for b in reg_bodies.get(rid, ()):
            channel[e][b].append(d)
    for e in channel:
        for b in channel[e]:
            channel[e][b].sort()
    for e in ent_reg_days:
        ent_reg_days[e].sort()
    log(f"      entities with registrations: {len(channel):,}")

    ref = sorted(reg_day.values())          # null: every national lobbying start
    NREF = len(ref)
    log(f"      null reference pool: {NREF:,} lobbying start dates")

    log("[2/7] procurement side ...")
    award = {}                              # rid -> (earliest day, METHOD)
    for rid, d, st, meth in rows("pro_award.csv", ["rid", "award_date_clean",
                                                   "award_date_status", "method"]):
        if st != "valid":
            continue
        day = to_day(d)
        prev = award.get(rid)
        if prev is None or day < prev[0]:
            award[rid] = (day, (meth or "").strip().upper())

    aw_buyers, aw_tender = defaultdict(set), {}
    for arid, trid, core, _kind in rows("pro_contact.csv",
                                        ["award_rid", "tender_rid", "name_core", "kind"]):
        if arid in award:
            if core:
                aw_buyers[arid].add(vid(core))
            if trid:
                aw_tender[arid] = trid

    tender_win = {}
    for rid, o, o_st, c, c_st in rows("pro_tender.csv",
                                      ["rid", "open_date_clean", "open_date_status",
                                       "close_date_clean", "close_date_status"]):
        if o_st == "valid" and c_st == "valid":
            w = to_day(c) - to_day(o)
            if 0 <= w <= 3650:
                tender_win[rid] = w
    log(f"      awards {len(award):,}  tender windows {len(tender_win):,}")

    log("[3/7] reference populations ...")
    sup_of = {}
    for arid, raw in rows("pro_supplier.csv", ["award_rid", "name"], base=paths.RAW):
        if arid in award:
            e = raw2ent.get((raw or "").strip())
            if e is not None:
                sup_of[arid] = e
    buyer_n, buyer_nc, pair_n = Counter(), Counter(), Counter()
    for arid, e in sup_of.items():
        nc = 1 if award[arid][1] in NONCOMP else 0
        for b in aw_buyers.get(arid, ()):
            buyer_n[b] += 1
            buyer_nc[b] += nc
            pair_n[(b, e)] += 1
    g_nc = sum(buyer_nc.values()) / max(sum(buyer_n.values()), 1)   # global sole-source rate
    win_sorted = sorted(tender_win.values())
    log(f"      buyers {len(buyer_n):,}  global sole-source rate {g_nc:.3f}")

    log("[4/7] motif instances + component p-values ...")
    recs = []
    for arid, e in sup_of.items():
        day, meth = award[arid]
        buyers = aw_buyers.get(arid, ())
        if not buyers:
            continue

        # p_time: nearest channel-scoped registration before the award
        best_gap, n_ch, ch_body = None, 0, None
        for b in buyers:
            days = channel.get(e, {}).get(b)
            if not days:
                continue
            n_ch += len(days)
            i = bisect.bisect_left(days, day)
            if i > 0:
                g = day - days[i - 1]
                if best_gap is None or g < best_gap:
                    best_gap, ch_body = g, b
        if best_gap is not None and n_ch > 0:
            q = (bisect.bisect_left(ref, day) - bisect.bisect_left(ref, day - best_gap)) / NREF
            p_time = 1.0 - (1.0 - q) ** n_ch
        else:
            p_time = 1.0

        # p_comp: buyer's sole-source rate, Laplace-smoothed toward global
        if meth in NONCOMP:
            p_comp = min((buyer_nc.get(b, 0) + 5.0 * g_nc) / (buyer_n.get(b, 0) + 5.0)
                         for b in buyers)
        else:
            p_comp = 1.0

        # p_proc: share of tenders with a window this short or shorter
        trid = aw_tender.get(arid)
        w = tender_win.get(trid) if trid else None
        if w is not None and win_sorted:
            p_proc = max(bisect.bisect_right(win_sorted, w) / len(win_sorted), EPS)
        else:
            p_proc = 1.0

        recs.append(dict(award_rid=arid, entity=e, day=day, method=meth,
                         buyers=tuple(buyers), n_buyers=len(buyers),
                         gap=best_gap, n_chan_regs=n_ch, chan_body=ch_body,
                         p_time=p_time, p_comp=p_comp, p_proc=p_proc,
                         tender_window=w))
    log(f"      instances: {len(recs):,}")

    # p_conc: Binomial(n_b, 1/#suppliers of b) tail at this supplier's count
    buyer_sup = defaultdict(set)
    for (b, e) in pair_n:
        buyer_sup[b].add(e)
    for r in recs:
        best = 1.0
        for b in r["buyers"]:
            n_b, k, m = buyer_n.get(b, 0), pair_n.get((b, r["entity"]), 0), len(buyer_sup.get(b, ()))
            if n_b >= 10 and m >= 2 and k >= 2:
                best = min(best, max(binom_sf(k, n_b, 1.0 / m), EPS))
        r["p_conc"] = best

    return recs, members, display, vlist, ent_reg_days


# ---------------------------------------------------------------- combination

def brown_combine(recs, comps=COMPONENTS):
    """Fisher's X = -2 sum ln p, rescaled for dependence (Brown 1975):
    X / c ~ chi2_f, with c and f from the observed variance of X."""
    k = len(comps)
    Y = [[-2.0 * math.log(max(r[c], EPS)) for c in comps] for r in recs]
    n = len(Y)
    mean = [sum(y[i] for y in Y) / n for i in range(k)]
    var = 0.0
    for i in range(k):
        for j in range(k):
            var += sum((y[i] - mean[i]) * (y[j] - mean[j]) for y in Y) / (n - 1)
    E = 2.0 * k
    if var <= 0:
        var = 4.0 * k
    c = var / (2.0 * E)
    f = 2.0 * E * E / var
    for r, y in zip(recs, Y):
        r["X"] = sum(y)
        r["p_combined"] = chi2_sf(r["X"] / c, f)
    return dict(k=k, c=c, f=f, var=var)


def bh_qvalues(recs, key="p_combined"):
    """Benjamini-Hochberg q-values into r['q']."""
    order = sorted(range(len(recs)), key=lambda i: recs[i][key])
    n = len(order)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        prev = min(prev, recs[i][key] * n / (rank + 1))
        recs[i]["q"] = min(prev, 1.0)
    return recs


def salience(r, ent_awards):
    """Would a reporter care? Kept off the p-value axis: prominence is
    important, not anomalous."""
    prom = math.log1p(ent_awards.get(r["entity"], 1)) / math.log1p(20000.0)
    named = 1.0 if r["n_buyers"] >= 1 else 0.0
    return 0.5 + 0.5 * min(prom + 0.15 * named, 1.0)


def main():
    recs, members, display, vlist, ent_reg_days = build()

    print("[5/7] Brown combination ...", flush=True)
    info = brown_combine(recs)
    print(f"      k={info['k']}  c={info['c']:.4f}  f={info['f']:.2f} "
          f"(independence would give c=1, f={2*info['k']})", flush=True)

    print("[6/7] gate x salience ...", flush=True)
    ent_awards = Counter(r["entity"] for r in recs)
    for r in recs:
        r["pi"] = identity_confidence(r["entity"], members,
                                      len(ent_reg_days.get(r["entity"], ())))
        r["salience"] = salience(r, ent_awards)
        r["score"] = r["pi"] * (1.0 - r["p_combined"]) * r["salience"]

    print("[7/7] q-values ...", flush=True)
    bh_qvalues(recs)
    recs.sort(key=lambda r: -r["score"])
    for i, r in enumerate(recs, 1):
        r["rank"] = i
        r["entity_name"] = display.get(r["entity"], r["entity"])
        r["buyer_names"] = "; ".join(vlist[b] for b in r["buyers"][:3])
        r["chan_body_name"] = vlist[r["chan_body"]] if r["chan_body"] is not None else ""

    out = common.dump(recs, "fav13_v2_scored.pkl")
    print(f"\nwrote {out}  ({len(recs):,} scored contracts)")

    cols = ["rank", "score", "p_combined", "q", "pi", "salience", "award_rid",
            "entity", "entity_name", "chan_body_name", "buyer_names", "method",
            "gap", "n_chan_regs", "tender_window", "p_time", "p_comp", "p_proc",
            "p_conc"]
    csvp = paths.WORK / "fav13_v2_top.csv"
    with open(csvp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in recs[:2000]:
            w.writerow([r.get(c, "") for c in cols])
    print(f"wrote {csvp}  (top 2000)")


if __name__ == "__main__":
    main()
