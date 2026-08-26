#!/usr/bin/env python3
"""
FAV 13 scoring, v2 -- calibrated composition instead of a weighted sum of ranks.

Pipeline (the sequence agreed 2026-08-13):

  0. resolve entities                -> identity gate  pi(e)
  1. build reference populations     -> the nulls
  2. detect motif instances          -> ONE ROW PER CONTRACT, channel-scoped
  3. aggregate lobbying evidence     -> nearest channel-scoped registration
  4. per-component p-values          -> instance vs its own null
  5. combine (Brown's method)        -> one calibrated p, no weights
  6. gate x salience                 -> pi * (1-p) * salience
  7. Benjamini-Hochberg q-values     -> "expect N in the top K to be coincidence"

Two things distinguish it from v1:

CHANNEL SCOPING.  v1 asked "did this supplier lobby ANYONE before this award".  A
company that lobbied Health Canada and later won a Defence contract scored as
suspicious.  v2 only counts registrations that lobbied *the department that bought*.

NO FREE WINDOW.  v1 needed PROXIMITY_WINDOW_DAYS=90.  v2 uses the observed minimum
gap directly: given n channel-scoped registrations drawn from the national pool of
lobbying start dates, the chance at least one lands within g days before the award is
1-(1-Q(g))^n.  The 90-day parameter disappears.

contract_value is not present in the corpus (0.63% recoverable from `desc`); the value
component is wired but disabled until the table arrives.
"""
import bisect, csv, math, os, pickle
from collections import Counter, defaultdict
from pipeline.fav13 import config as C
from pipeline.fav13 import evidence as EV

csv.field_size_limit(10 ** 9)
WORK = C.WORK
RAW, CLEAN = C.RAW, C.CLEAN

NONCOMP = C.NONCOMPETITIVE_METHODS
EPS = 1e-12


# ------------------------------------------------------------------ chi-square

def _gammaincc(a, x):
    """Regularised upper incomplete gamma Q(a,x) = Gamma(a,x)/Gamma(a)."""
    if x < 0 or a <= 0:
        return 1.0
    if x == 0:
        return 1.0
    if x < a + 1.0:                                   # series for P, then 1-P
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1.0
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-14:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    # continued fraction for Q
    tiny = 1e-300
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
    """P(X > x) for X ~ chi-square with df degrees of freedom."""
    if x <= 0:
        return 1.0
    return max(_gammaincc(df / 2.0, x / 2.0), 1e-300)


def binom_sf(k, n, p):
    """P(X >= k) for X ~ Binomial(n,p).  n is small here (awards per buyer-supplier)."""
    if k <= 0:
        return 1.0
    if p <= 0:
        return 0.0 if k > 0 else 1.0
    if p >= 1:
        return 1.0
    # sum the lower tail, subtract -- k is small
    s = 0.0
    for i in range(0, k):
        s += math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                      + i * math.log(p) + (n - i) * math.log1p(-p))
        if s >= 1.0:
            return 1e-300
    return max(1.0 - s, 1e-300)


# --------------------------------------------------------------------- loading

def rows(fname, cols, base=CLEAN):
    with open(os.path.join(base, fname), newline="", encoding="utf-8",
              errors="replace") as fh:
        r = csv.reader(fh)
        hdr = next(r)
        idx = [hdr.index(c) for c in cols]
        m = max(idx)
        for x in r:
            if len(x) > m:
                yield [x[i] for i in idx]


def load_entities():
    p = os.path.join(WORK, "entities.pkl")
    e = pickle.load(open(p, "rb"))
    return e["raw2ent"], e["members"], e.get("display", {})


def identity_gate(entity, members, n_regs):
    """pi(e) in (0,1].  Same interpretable ladder as v1, but MULTIPLICATIVE:
    a wrong identity means there is no case, not a 10% deduction."""
    names = members.get(entity, [])
    if len(names) <= 1:
        base = 1.0
    elif "." in str(entity):
        base = 0.6
    else:
        base = 0.8
    support = min(n_regs, 5) / 5.0
    return base * (0.6 + 0.4 * support)


# ------------------------------------------------------------------- main build

def build():
    raw2ent, members, display = load_entities()
    log = lambda s: print(s, flush=True)

    # ---- shared vocabulary for bodies/buyers (int ids keep the sets small)
    vocab, vlist = {}, []

    def vid(s):
        i = vocab.get(s)
        if i is None:
            i = len(vlist)
            vocab[s] = i
            vlist.append(s)
        return i

    log("[1/7] lobbying side ...")
    reg_day = {}
    for rid, d, st in rows("lob_meta.csv", ["rid", "start_date_clean",
                                            "start_date_status"]):
        if st == "valid":
            reg_day[rid] = EV.to_day(d)

    reg_bodies = defaultdict(set)
    for rid, core in rows("lob_target.csv", ["rid", "name_core"]):
        if core and rid in reg_day:
            reg_bodies[rid].add(vid(core))

    # entity -> {body_id -> sorted list of registration days}   (the CHANNELS)
    channel = defaultdict(lambda: defaultdict(list))
    ent_reg_days = defaultdict(list)
    for rid, raw in rows("lob_org.csv", ["rid", "name"], base=RAW):
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

    # null reference: every valid national lobbying start date
    ref = sorted(reg_day.values())
    NREF = len(ref)
    log(f"      null reference pool: {NREF:,} lobbying start dates")

    log("[2/7] procurement side ...")
    award = {}
    for rid, d, st, meth in rows("pro_award.csv", ["rid", "award_date_clean",
                                                   "award_date_status", "method"]):
        if st != "valid":
            continue
        day = EV.to_day(d)
        prev = award.get(rid)
        if prev is None or day < prev[0]:
            award[rid] = (day, (meth or "").strip().upper())

    aw_buyers = defaultdict(set)
    aw_tender = {}
    for arid, trid, core, kind in rows("pro_contact.csv",
                                       ["award_rid", "tender_rid", "name_core", "kind"]):
        if core and arid in award:
            aw_buyers[arid].add(vid(core))
        if trid and arid in award:
            aw_tender[arid] = trid

    tender_win = {}
    for rid, o, os_, c, cs in rows("pro_tender.csv",
                                   ["rid", "open_date_clean", "open_date_status",
                                    "close_date_clean", "close_date_status"]):
        if os_ == "valid" and cs == "valid":
            w = EV.to_day(c) - EV.to_day(o)
            if 0 <= w <= 3650:
                tender_win[rid] = w
    log(f"      awards {len(award):,}  tender windows {len(tender_win):,}")

    log("[3/7] reference populations ...")
    # buyer -> (n_awards, n_noncompetitive)
    buyer_n = Counter()
    buyer_nc = Counter()
    # (buyer, supplier) -> count      and    buyer -> supplier -> count
    sup_of = {}
    for arid, raw in rows("pro_supplier.csv", ["award_rid", "name"], base=RAW):
        if arid in award:
            e = raw2ent.get((raw or "").strip())
            if e is not None:
                sup_of[arid] = e
    for arid, e in sup_of.items():
        meth = award[arid][1]
        nc = 1 if meth in NONCOMP else 0
        for b in aw_buyers.get(arid, ()):
            buyer_n[b] += 1
            buyer_nc[b] += nc
    pair_n = Counter()
    for arid, e in sup_of.items():
        for b in aw_buyers.get(arid, ()):
            pair_n[(b, e)] += 1
    # global sole-source rate, for smoothing small buyers
    g_nc = sum(buyer_nc.values()) / max(sum(buyer_n.values()), 1)
    win_sorted = sorted(tender_win.values())
    log(f"      buyers {len(buyer_n):,}  global sole-source rate {g_nc:.3f}")

    log("[4/7] motif instances + component p-values ...")
    recs = []
    for arid, e in sup_of.items():
        day, meth = award[arid]
        buyers = aw_buyers.get(arid, ())
        if not buyers:
            continue

        # --- p_time : channel-scoped nearest registration, no window parameter
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
            lo = bisect.bisect_left(ref, day - best_gap)
            hi = bisect.bisect_left(ref, day)
            q = (hi - lo) / NREF
            p_time = 1.0 - (1.0 - q) ** n_ch
        else:
            p_time = 1.0

        # --- p_comp : sole-source against THIS buyer's own base rate
        if meth in NONCOMP:
            ps = []
            for b in buyers:
                n_b, k_b = buyer_n.get(b, 0), buyer_nc.get(b, 0)
                rate = (k_b + 5.0 * g_nc) / (n_b + 5.0)      # Laplace toward global
                ps.append(rate)
            p_comp = min(ps) if ps else 1.0
        else:
            p_comp = 1.0

        # --- p_proc : tender window against the global window distribution
        trid = aw_tender.get(arid)
        w = tender_win.get(trid) if trid else None
        if w is not None and win_sorted:
            p_proc = max(bisect.bisect_right(win_sorted, w) / len(win_sorted), EPS)
        else:
            p_proc = 1.0

        # --- p_conc : is this supplier over-represented in this buyer's book?
        p_conc = 1.0
        for b in buyers:
            n_b = buyer_n.get(b, 0)
            k = pair_n.get((b, e), 0)
            if n_b >= 10 and k >= 2:
                nsup = max(len({s for (bb, s) in pair_n if bb == b}), 1) if False else None
                # expected share under "buyer spreads evenly over its suppliers"
                # estimated as 1/number of distinct suppliers seen for that buyer
                pass
        # distinct suppliers per buyer, computed once below
        recs.append(dict(award_rid=arid, entity=e, day=day, method=meth,
                         buyers=tuple(buyers), n_buyers=len(buyers),
                         gap=best_gap, n_chan_regs=n_ch, chan_body=ch_body,
                         p_time=p_time, p_comp=p_comp, p_proc=p_proc,
                         tender_window=w))
    log(f"      instances: {len(recs):,}")

    # p_conc needs suppliers-per-buyer; compute now and fill in
    buyer_sup = defaultdict(set)
    for (b, e) in pair_n:
        buyer_sup[b].add(e)
    for r in recs:
        best = 1.0
        for b in r["buyers"]:
            n_b = buyer_n.get(b, 0)
            k = pair_n.get((b, r["entity"]), 0)
            m = len(buyer_sup.get(b, ()))
            if n_b >= 10 and m >= 2 and k >= 2:
                pv = binom_sf(k, n_b, 1.0 / m)
                best = min(best, max(pv, EPS))
        r["p_conc"] = best

    return recs, members, display, vlist, ent_reg_days, buyer_n, pair_n


# --------------------------------------------------------- Brown's combination

COMPONENTS = ("p_time", "p_comp", "p_proc", "p_conc")


def brown_combine(recs, comps=COMPONENTS):
    """Fisher's statistic rescaled for dependence (Brown 1975).

    X = -2 sum ln p_i.  Under independence X ~ chi2_2k.  The components here are
    correlated (sole-source with concentration, temporal with concentration), so
    the null is rescaled: X/c ~ chi2_f with c and f set from the observed variance.
    No weights are chosen -- the dependence structure replaces them.
    """
    k = len(comps)
    Y = [[-2.0 * math.log(max(r[c], EPS)) for c in comps] for r in recs]
    n = len(Y)
    mean = [sum(y[i] for y in Y) / n for i in range(k)]
    var = 0.0
    for i in range(k):
        for j in range(k):
            cov = sum((y[i] - mean[i]) * (y[j] - mean[j]) for y in Y) / (n - 1)
            var += cov
    E = 2.0 * k
    if var <= 0:
        var = 4.0 * k
    c = var / (2.0 * E)
    f = 2.0 * E * E / var
    for r, y in zip(recs, Y):
        X = sum(y)
        r["X"] = X
        r["p_combined"] = chi2_sf(X / c, f)
    return dict(k=k, c=c, f=f, var=var)


def bh_qvalues(recs, key="p_combined"):
    order = sorted(range(len(recs)), key=lambda i: recs[i][key])
    n = len(order)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        q = recs[i][key] * n / (rank + 1)
        prev = min(prev, q)
        recs[i]["q"] = min(prev, 1.0)
    return recs


def salience(r, ent_awards, ent_value=None):
    """Would a reporter care?  Deliberately NOT part of the surprise p-value:
    prominence is not anomalous, it is important.  Kept on its own axis so the
    null can strip volume out of the anomaly without fighting objective #2."""
    n = ent_awards.get(r["entity"], 1)
    prom = math.log1p(n) / math.log1p(20000.0)          # 0..1, log-compressed
    named = 1.0 if r["n_buyers"] >= 1 else 0.0
    return 0.5 + 0.5 * min(prom + 0.15 * named, 1.0)


def main():
    recs, members, display, vlist, ent_reg_days, buyer_n, pair_n = build()

    print("[5/7] Brown combination ...", flush=True)
    info = brown_combine(recs)
    print(f"      k={info['k']}  c={info['c']:.4f}  f={info['f']:.2f} "
          f"(independence would give c=1, f={2*info['k']})", flush=True)

    print("[6/7] gate x salience ...", flush=True)
    ent_awards = Counter(r["entity"] for r in recs)
    for r in recs:
        pi = identity_gate(r["entity"], members, len(ent_reg_days.get(r["entity"], ())))
        r["pi"] = pi
        r["salience"] = salience(r, ent_awards)
        r["score"] = pi * (1.0 - r["p_combined"]) * r["salience"]

    print("[7/7] q-values ...", flush=True)
    bh_qvalues(recs)
    recs.sort(key=lambda r: -r["score"])
    for i, r in enumerate(recs, 1):
        r["rank"] = i
        r["entity_name"] = display.get(r["entity"], r["entity"])
        r["buyer_names"] = "; ".join(vlist[b] for b in r["buyers"][:3])
        r["chan_body_name"] = vlist[r["chan_body"]] if r["chan_body"] is not None else ""

    out = os.path.join(WORK, "fav13_v2_scored.pkl")
    pickle.dump(recs, open(out, "wb"))
    print(f"\nwrote {out}  ({len(recs):,} scored contracts)")

    cols = ["rank", "score", "p_combined", "q", "pi", "salience", "award_rid",
            "entity", "entity_name", "chan_body_name", "buyer_names", "method",
            "gap", "n_chan_regs", "tender_window", "p_time", "p_comp", "p_proc",
            "p_conc"]
    csvp = os.path.join(WORK, "fav13_v2_top.csv")
    with open(csvp, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in recs[:2000]:
            w.writerow([r.get(c, "") for c in cols])
    print(f"wrote {csvp}  (top 2000)")


if __name__ == "__main__":
    main()
