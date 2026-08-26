"""
Body-conditioned temporal excess -- the closed FAV 12 + FAV 13 loop.

Why this exists.  The company-level test asks "was any award preceded by any
registration?"  For a company that lobbies constantly the answer is always
yes, so the null saturates: Accenture has 181 registrations, and 302 of its
356 award dates are *expected* to follow some registration inside 90 days
purely by construction.  Observed 296.  Ratio 0.98.  The statistic has no
headroom left on exactly the population the published stories are about.

The fix is to condition on the body.  The question becomes: was this award
made by a body that *this company lobbied*, shortly after it lobbied *that
body*?  Lobbying 138 unrelated departments now earns nothing.

Null, stated once: for each award a and each buying body z, let q(a,z) be the
share of *all national registrations targeting z* that start in [a-w, a).  If
the company holds n_z registrations against z, its chance of a coincidental
hit on that award is 1 - prod_z (1 - q(a,z))^{n_z}.  Summing over awards gives
the expected count.  Closed form, no simulation.

    python3 -m pipeline.fav13.bodycond
"""

import bisect, csv, os, pickle
from collections import defaultdict

from . import body_map as BM
from . import config as C
from . import evidence as EV

csv.field_size_limit(10 ** 9)


def canonical_bodies():
    """-> (lob_name -> body_key, pro_name -> body_key).

    The body key is the pro_contact surface form, because that is the side the
    award is recorded against.  Where a lobbying name maps to several
    procurement names, the highest-volume one wins.
    """
    m = BM.load(approved_only=True)
    pro_counts = BM._counts("pro_contact.csv", "name_core")
    lob2key, pro2key = {}, {}
    for lob, pros in m.items():
        key = max(pros, key=lambda p: pro_counts.get(p, 0))
        lob2key[lob] = key
        for p in pros:
            pro2key[p] = key
    return lob2key, pro2key


def build_events(raw2ent):
    """-> per-entity paired events, plus the per-body national reference.

    Paired means the registration keeps its own bodies rather than the union
    across registrations -- that association is what the whole test rests on,
    and `evidence.py` discards it.
    """
    lob2key, pro2key = canonical_bodies()

    reg_day = {}
    for rid, d, s in EV._rows("lob_meta.csv",
                              ["rid", "start_date_clean", "start_date_status"]):
        if s == "valid":
            reg_day[rid] = EV.to_day(d)

    reg_bodies = defaultdict(set)
    for rid, core in EV._rows("lob_target.csv", ["rid", "name_core"]):
        k = lob2key.get(core)
        if k:
            reg_bodies[rid].add(k)

    # national reference: every registration start targeting each body
    reference = defaultdict(list)
    for rid, bodies in reg_bodies.items():
        d = reg_day.get(rid)
        if d is None:
            continue
        for b in bodies:
            reference[b].append(d)
    for b in reference:
        reference[b].sort()

    award_day = {}
    for rid, d, s in EV._rows("pro_award.csv",
                              ["rid", "award_date_clean", "award_date_status"]):
        if s != "valid":
            continue
        day = EV.to_day(d)
        if rid not in award_day or day < award_day[rid]:
            award_day[rid] = day

    award_buyers = defaultdict(set)
    for arid, core in EV._rows("pro_contact.csv", ["award_rid", "name_core"]):
        k = pro2key.get(core)
        if k:
            award_buyers[arid].add(k)

    ev = defaultdict(lambda: dict(regs=[], awards=[]))
    for rid, raw in EV._rows("lob_org.csv", ["rid", "name"], base=C.RAW):
        e = raw2ent.get((raw or "").strip())
        if e is None or rid not in reg_day:
            continue
        ev[e]["regs"].append((reg_day[rid], frozenset(reg_bodies.get(rid, ()))))

    for arid, raw in EV._rows("pro_supplier.csv", ["award_rid", "name"],
                              base=C.RAW):
        e = raw2ent.get((raw or "").strip())
        if e is None or arid not in award_day:
            continue
        ev[e]["awards"].append((award_day[arid],
                                frozenset(award_buyers.get(arid, ()))))

    for r in ev.values():
        # collapse awards to distinct dates, unioning buyers -- the same
        # task-authorisation burst control as the company-level test
        byday = defaultdict(set)
        for d, bs in r["awards"]:
            byday[d] |= bs
        r["awards"] = sorted((d, frozenset(bs)) for d, bs in byday.items())
        r["regs"].sort()
    return dict(ev), dict(reference)


def observed(regs, awards, window):
    """Awards whose buying body was lobbied by this company inside the window."""
    days = [d for d, _ in regs]
    n = 0
    for a, buyers in awards:
        if not buyers:
            continue
        lo = bisect.bisect_left(days, a - window)
        hi = bisect.bisect_left(days, a)
        for i in range(lo, hi):
            if regs[i][1] & buyers:
                n += 1
                break
    return n


def expected(regs, awards, reference, window):
    """Expected count under body-conditioned coincidence."""
    n_z = defaultdict(int)
    for _, bodies in regs:
        for b in bodies:
            n_z[b] += 1
    if not n_z:
        return 0.0
    total = 0.0
    for a, buyers in awards:
        miss = 1.0
        for b in buyers:
            k = n_z.get(b, 0)
            if not k:
                continue
            ref = reference.get(b)
            if not ref:
                continue
            lo = bisect.bisect_left(ref, a - window)
            hi = bisect.bisect_left(ref, a)
            q = (hi - lo) / len(ref)
            miss *= (1.0 - q) ** k
        total += 1.0 - miss
    return total


def compute(ev, reference, window=None, min_obs=None):
    from . import features as F
    window = window or C.PROXIMITY_WINDOW_DAYS
    min_obs = C.MIN_OBS if min_obs is None else min_obs
    out = []
    for e, r in ev.items():
        if not r["regs"] or not r["awards"]:
            continue
        obs = observed(r["regs"], r["awards"], window)
        if obs < min_obs:
            continue
        exp = expected(r["regs"], r["awards"], reference, window)
        out.append(dict(entity=e, obs=obs, expected=exp,
                        ratio=(obs / exp) if exp > 0 else float("inf"),
                        nlp=F.neg_log10_tail(obs, exp),
                        n_regs=len(r["regs"]), n_awards=len(r["awards"]),
                        volume=len(r["regs"]) + len(r["awards"])))
    return out


def save(raw2ent, path=None):
    path = path or os.path.join(C.WORK, "bodycond.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ev, ref = build_events(raw2ent)
    pickle.dump(dict(events=ev, reference=ref), open(path, "wb"))
    return path


def main():
    ents = pickle.load(open(os.path.join(C.WORK, "entities.pkl"), "rb"))
    p = os.path.join(C.WORK, "bodycond.pkl")
    if os.path.exists(p):
        d = pickle.load(open(p, "rb"))
        ev, ref = d["events"], d["reference"]
    else:
        ev, ref = build_events(ents["raw2ent"])
        pickle.dump(dict(events=ev, reference=ref), open(p, "wb"))
    recs = compute(ev, ref)
    print(f"bodies in reference        : {len(ref):,}")
    print(f"entities with both sides   : "
          f"{sum(1 for r in ev.values() if r['regs'] and r['awards']):,}")
    print(f"entities passing obs>={C.MIN_OBS}   : {len(recs):,}")
    top = sorted(recs, key=lambda r: -r["nlp"])[:10]
    disp = ents["display"]
    print(f"\n{'entity':<34}{'obs':>5}{'exp':>8}{'ratio':>8}{'nlp':>8}{'regs':>6}{'awd':>6}")
    for r in top:
        print(f"{disp.get(r['entity'], r['entity'])[:32]:<34}{r['obs']:>5}"
              f"{r['expected']:>8.2f}{r['ratio']:>8.2f}{r['nlp']:>8.2f}"
              f"{r['n_regs']:>6}{r['n_awards']:>6}")


if __name__ == "__main__":
    main()
