"""Body-conditioned temporal excess (the closed FAV 12 + FAV 13 loop).

The company-level null saturates for heavy lobbyists: Accenture is *expected*
to have 302 of 356 award dates follow some registration; observed 296. So
condition on the body: was the award made by a body this company lobbied,
shortly after lobbying *that* body?

Expected count = sum over awards of 1 - prod_z (1 - q(a, z)) ** n_z, with
q(a, z) the share of national registrations targeting z in [a - w, a).

    python3 -m pipeline.fav13.score.bodycond
"""

import bisect
import os
from collections import defaultdict

from pipeline import paths
from .. import common
from .. import config as C
from ..common import rows, to_day
from ..data.body_map import canonical_bodies
from .features import neg_log10_tail

CACHE = "bodycond.pkl"


def build_events(raw2ent):
    """-> ({entity: {regs: [(day, bodies)], awards: [(day, buyers)]}}, {body: ref days}).
    Each registration keeps its own bodies (evidence.py unions them away)."""
    lob2key, pro2key = canonical_bodies()
    reg_day = common.registrations()

    reg_bodies = defaultdict(set)
    for rid, core in rows("lob_target.csv", ["rid", "name_core"]):
        k = lob2key.get(core)
        if k:
            reg_bodies[rid].add(k)

    reference = defaultdict(list)
    for rid, bodies in reg_bodies.items():
        d = reg_day.get(rid)
        if d is not None:
            for b in bodies:
                reference[b].append(d)
    for b in reference:
        reference[b].sort()

    award_day = {}
    for rid, d, s in rows("pro_award.csv", ["rid", "award_date_clean", "award_date_status"]):
        if s == "valid":
            day = to_day(d)
            if rid not in award_day or day < award_day[rid]:
                award_day[rid] = day

    award_buyers = defaultdict(set)
    for arid, core in rows("pro_contact.csv", ["award_rid", "name_core"]):
        k = pro2key.get(core)
        if k:
            award_buyers[arid].add(k)

    ev = defaultdict(lambda: dict(regs=[], awards=[]))
    for rid, raw in rows("lob_org.csv", ["rid", "name"], base=paths.RAW):
        e = raw2ent.get((raw or "").strip())
        if e is not None and rid in reg_day:
            ev[e]["regs"].append((reg_day[rid], frozenset(reg_bodies.get(rid, ()))))
    for arid, raw in rows("pro_supplier.csv", ["award_rid", "name"], base=paths.RAW):
        e = raw2ent.get((raw or "").strip())
        if e is not None and arid in award_day:
            ev[e]["awards"].append((award_day[arid], frozenset(award_buyers.get(arid, ()))))

    for r in ev.values():
        byday = defaultdict(set)           # same-day burst control, union buyers
        for d, bs in r["awards"]:
            byday[d] |= bs
        r["awards"] = sorted((d, frozenset(bs)) for d, bs in byday.items())
        r["regs"].sort()
    return dict(ev), dict(reference)


def observed(regs, awards, window):
    """Awards whose buyer this company lobbied inside the window."""
    days = [d for d, _ in regs]
    n = 0
    for a, buyers in awards:
        if not buyers:
            continue
        lo = bisect.bisect_left(days, a - window)
        hi = bisect.bisect_left(days, a)
        if any(regs[i][1] & buyers for i in range(lo, hi)):
            n += 1
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
            k, ref = n_z.get(b, 0), reference.get(b)
            if not k or not ref:
                continue
            q = (bisect.bisect_left(ref, a) - bisect.bisect_left(ref, a - window)) / len(ref)
            miss *= (1.0 - q) ** k
        total += 1.0 - miss
    return total


def compute(ev, reference, window=None, min_obs=None):
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
                        nlp=neg_log10_tail(obs, exp),
                        n_regs=len(r["regs"]), n_awards=len(r["awards"]),
                        volume=len(r["regs"]) + len(r["awards"])))
    return out


def main():
    ents = common.load_entities()
    if os.path.exists(paths.WORK / CACHE):
        d = common.load(CACHE)
    else:
        ev, ref = build_events(ents["raw2ent"])
        d = dict(events=ev, reference=ref)
        common.dump(d, CACHE)
    ev, ref = d["events"], d["reference"]
    recs = compute(ev, ref)

    print(f"bodies in reference        : {len(ref):,}")
    print(f"entities with both sides   : "
          f"{sum(1 for r in ev.values() if r['regs'] and r['awards']):,}")
    print(f"entities passing obs>={C.MIN_OBS}   : {len(recs):,}")
    disp = ents["display"]
    print(f"\n{'entity':<34}{'obs':>5}{'exp':>8}{'ratio':>8}{'nlp':>8}{'regs':>6}{'awd':>6}")
    for r in sorted(recs, key=lambda r: -r["nlp"])[:10]:
        print(f"{disp.get(r['entity'], r['entity'])[:32]:<34}{r['obs']:>5}"
              f"{r['expected']:>8.2f}{r['ratio']:>8.2f}{r['nlp']:>8.2f}"
              f"{r['n_regs']:>6}{r['n_awards']:>6}")


if __name__ == "__main__":
    main()
