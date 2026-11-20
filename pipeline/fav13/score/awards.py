"""Award-level FAV 13: one award = one motif (weighted-sum scorer).

Company scores bury the story: Accenture's scandal is one sole-source contract
among 356 award dates (non-competitive share 0.214). So the unit is the award,
and company totals are context only.

Per award a, bought by bodies Z_a, from a supplier with n_z registrations
against body z, the chance lobbying precedes it within the window by
coincidence is

    p_a = 1 - prod_z (1 - q(a, z)) ** n_z

where q(a, z) is the share of all national registrations targeting z that
start in [a - w, a). -log10(p_a) is the reportable "1 in N" surprise.

    python3 -m pipeline.fav13.score.awards
"""

import bisect
import math
from collections import defaultdict

from pipeline import paths
from .. import common
from .. import config as C
from ..common import rows, to_day
from ..data.body_map import canonical_bodies
from . import features as F

OUT = paths.WORK / "fav13_awards_topk.csv"


def build(raw2ent, members, window=None):
    """-> award rows that close the loop, unscored."""
    window = window or C.PROXIMITY_WINDOW_DAYS
    lob2key, pro2key = canonical_bodies()
    reg_day = common.registrations()

    # body -> national reference days; (entity, body) -> registration days
    rid_bodies = defaultdict(set)
    for rid, core in rows("lob_target.csv", ["rid", "name_core"]):
        k = lob2key.get(core)
        if k:
            rid_bodies[rid].add(k)
    reference = defaultdict(list)
    for rid, bodies in rid_bodies.items():
        d = reg_day.get(rid)
        if d is not None:
            for b in bodies:
                reference[b].append(d)
    for b in reference:
        reference[b].sort()

    rid_ent = {}
    for rid, raw in rows("lob_org.csv", ["rid", "name"], base=paths.RAW):
        e = raw2ent.get((raw or "").strip())
        if e is not None:
            rid_ent[rid] = e
    ent_body = defaultdict(list)
    for rid, e in rid_ent.items():
        d = reg_day.get(rid)
        if d is not None:
            for b in rid_bodies.get(rid, ()):
                ent_body[(e, b)].append(d)
    for k in ent_body:
        ent_body[k].sort()

    award = {}                          # rid -> (earliest day, method, reason, desc)
    for rid, d, s, meth, lim, desc in rows(
            "pro_award.csv", ["rid", "award_date_clean", "award_date_status",
                              "method", "limited_reason", "name"]):
        if s != "valid":
            continue
        day = to_day(d)
        if rid not in award or day < award[rid][0]:
            award[rid] = (day, meth or "", lim or "", desc or "")

    buyers = defaultdict(set)
    for arid, core in rows("pro_contact.csv", ["award_rid", "name_core"]):
        k = pro2key.get(core)
        if k:
            buyers[arid].add(k)

    sup = {}
    for arid, raw in rows("pro_supplier.csv", ["award_rid", "name"], base=paths.RAW):
        e = raw2ent.get((raw or "").strip())
        if e is not None:
            sup[arid] = e

    # company context
    ent_awards, ent_nc = defaultdict(int), defaultdict(int)
    for arid, e in sup.items():
        a = award.get(arid)
        if a:
            ent_awards[e] += 1
            ent_nc[e] += a[1] in C.NONCOMPETITIVE_METHODS
    ent_regs = defaultdict(set)
    for rid, e in rid_ent.items():
        ent_regs[e].add(rid)

    out = []
    for arid, e in sup.items():
        a = award.get(arid)
        Z = buyers.get(arid)
        if not a or not Z:
            continue
        day, meth, lim, desc = a

        hit_body, hit_day, miss = None, None, 1.0
        for b in Z:
            days = ent_body.get((e, b))
            if not days:
                continue
            lo = bisect.bisect_left(days, day - window)
            hi = bisect.bisect_left(days, day)
            if hi > lo and (hit_day is None or days[hi - 1] > hit_day):
                hit_day, hit_body = days[hi - 1], b
            ref = reference.get(b)
            if ref:
                q = (bisect.bisect_left(ref, day) - bisect.bisect_left(ref, day - window)) / len(ref)
                miss *= (1.0 - q) ** (hi - lo if hi > lo else len(days))
        if hit_body is None:
            continue

        p = max(1.0 - miss, 1e-12)
        out.append(dict(
            award_rid=arid, entity=e, body=hit_body, key=(e, hit_body, day),
            award_day=day, lobby_day=hit_day, lag_days=day - hit_day,
            p_coincidence=p, surprise=-math.log10(p),
            method=meth, limited_reason=lim, desc=desc[:120],
            noncompetitive=1.0 if meth in C.NONCOMPETITIVE_METHODS else 0.0,
            n_regs_this_body=len(ent_body.get((e, hit_body), ())),
            ctx_total_awards=ent_awards[e], ctx_total_regs=len(ent_regs[e]),
            ctx_nc_share=(ent_nc[e] / ent_awards[e]) if ent_awards[e] else 0.0,
        ))
    return collapse(out)


def collapse(recs):
    """One row per (supplier, body, award date): five task authorisations on
    one day are one story. Non-competitive if ANY award in the group is."""
    g = {}
    for r in recs:
        k = r.pop("key")
        cur = g.get(k)
        if cur is None:
            r["n_awards_same_day"] = 1
            r["award_rids"] = [r["award_rid"]]
            g[k] = r
            continue
        cur["n_awards_same_day"] += 1
        cur["award_rids"].append(r["award_rid"])
        if r["noncompetitive"] > cur["noncompetitive"]:
            cur["noncompetitive"] = r["noncompetitive"]
            cur["method"] = r["method"]
            cur["limited_reason"] = r["limited_reason"]
    return list(g.values())


# ---------------------------------------------------------------- scoring

AWARD_FEATURES = ("surprise", "proximity", "noncompetitive",
                  "commercial", "identity_confidence")

AWARD_WEIGHTS = {
    "surprise":            0.40,
    "proximity":           0.15,   # tighter lag is more suspicious
    "noncompetitive":      0.25,
    "commercial":          0.10,   # not a grant recipient
    "identity_confidence": 0.10,
}


def score_rows(recs, members, charity_keys, weights=None, window=None):
    window = window or C.PROXIMITY_WINDOW_DAYS
    for r in recs:
        gl, why = F.grant_likelihood(r["entity"], members, charity_keys)
        r["commercial"] = 1.0 - gl
        r["grant_reason"] = why
        r["identity_confidence"] = F.identity_base(r["entity"], members)
        r["proximity"] = 1.0 - (r["lag_days"] / window)
    for name in AWARD_FEATURES:
        for r, v in zip(recs, common.percentile(recs, name)):
            r["f_" + name] = v
    w = weights or AWARD_WEIGHTS
    for r in recs:
        r["score"] = sum(w[f] * r["f_" + f] for f in AWARD_FEATURES)
    return recs


def main():
    ents = common.load_entities()
    recs = build(ents["raw2ent"], ents["members"])
    n_nc = sum(r["noncompetitive"] for r in recs)
    print(f"awards closing the loop : {len(recs):,}")
    print(f"distinct suppliers      : {len({r['entity'] for r in recs}):,}")
    print(f"non-competitive among them: {n_nc:,.0f} "
          f"({100 * n_nc / max(len(recs), 1):.1f}%)")

    score_rows(recs, ents["members"], F.load_charity_keys())
    recs.sort(key=lambda r: -r["score"])
    disp = ents["display"]
    cols = ["score", "entity", "supplier", "body", "lag_days", "p_coincidence",
            "surprise", "method", "noncompetitive", "award_day", "lobby_day",
            "n_regs_this_body", "ctx_total_awards", "ctx_nc_share",
            "grant_reason", "award_rid", "desc"]
    paths.WORK.mkdir(parents=True, exist_ok=True)
    common.write_csv(OUT, [dict(r, supplier=disp.get(r["entity"], r["entity"]),
                                award_day=common.from_day(r["award_day"]),
                                lobby_day=common.from_day(r["lobby_day"]))
                           for r in recs[:2000]], cols)
    print(f"wrote {OUT}\n")

    print(f"{'supplier':<30}{'body':<26}{'lag':>5}{'1-in':>9}{'method':>10}{'score':>7}")
    for r in recs[:15]:
        print(f"{disp.get(r['entity'], r['entity'])[:28]:<30}{r['body'][:24]:<26}"
              f"{r['lag_days']:>5}{1 / r['p_coincidence']:>9,.0f}"
              f"{r['method'][:8]:>10}{r['score']:>7.3f}")


if __name__ == "__main__":
    main()
