"""
Award-level FAV 13: score contracts, not companies.

Why the unit changed.  The entity-level score dilutes the one thing the
published stories are about.  The Accenture story is a single $313M
sole-source CEBA contract; Accenture holds 356 award dates, so its
non-competitive *share* is 0.214 and the scandal disappears into the average.
ATCO: 449 awards, share 0.074.  Ontario Shipyards: 176 awards, share 0.144.
Journalists write about contracts; we were ranking companies.

The scored unit is therefore one award.  A company-level aggregate becomes
context attached to the row, never the object being ranked.

The statistic is per-award and calibrated.  For award `a` bought by bodies
`Z_a` from a supplier holding `n_z` registrations against body `z`, the
chance that lobbying would precede this award inside the window by
coincidence is

    p_a = 1 - prod_{z in Z_a} (1 - q(a,z))^{n_z}

with q(a,z) the share of *all national registrations targeting z* starting in
[a-w, a).  Because the unit is a single award, this is a probability, not a
count: surprise = -log10(p_a) is directly reportable -- "the chance this
contract would follow lobbying of this department by coincidence was 1 in
420".  That is a sentence a reporter can publish and a lawyer can check.

Contract value is not in `pro_award`; when it arrives it enters here as one
more non-negative feature and changes nothing structural.

    python3 -m pipeline.fav13.awards
"""

import bisect, csv, math, os, pickle
from collections import defaultdict

from . import body_map as BM
from . import config as C
from . import evidence as EV
from . import features as F

csv.field_size_limit(10 ** 9)

OUT = os.path.join(C.WORK, "fav13_awards_topk.csv")


def build(raw2ent, members, window=None):
    """-> list of award rows that close the loop, unscored."""
    window = window or C.PROXIMITY_WINDOW_DAYS
    lob2key, pro2key = _canonical()

    reg_day = {rid: EV.to_day(d) for rid, d, s in EV._rows(
        "lob_meta.csv", ["rid", "start_date_clean", "start_date_status"])
        if s == "valid"}

    # (entity, body) -> sorted registration days;  body -> national reference
    ent_body = defaultdict(list)
    reference = defaultdict(list)
    rid_bodies = defaultdict(set)
    for rid, core in EV._rows("lob_target.csv", ["rid", "name_core"]):
        k = lob2key.get(core)
        if k:
            rid_bodies[rid].add(k)
    for rid, bodies in rid_bodies.items():
        d = reg_day.get(rid)
        if d is None:
            continue
        for b in bodies:
            reference[b].append(d)
    for b in reference:
        reference[b].sort()

    rid_ent = {}
    for rid, raw in EV._rows("lob_org.csv", ["rid", "name"], base=C.RAW):
        e = raw2ent.get((raw or "").strip())
        if e is not None:
            rid_ent[rid] = e
    for rid, e in rid_ent.items():
        d = reg_day.get(rid)
        if d is None:
            continue
        for b in rid_bodies.get(rid, ()):
            ent_body[(e, b)].append(d)
    for k in ent_body:
        ent_body[k].sort()

    # awards
    award = {}
    for rid, d, s, meth, lim, desc in EV._rows(
            "pro_award.csv", ["rid", "award_date_clean", "award_date_status",
                              "method", "limited_reason", "name"]):
        if s != "valid":
            continue
        day = EV.to_day(d)
        if rid not in award or day < award[rid][0]:
            award[rid] = (day, meth or "", lim or "", desc or "")

    buyers = defaultdict(set)
    for arid, core in EV._rows("pro_contact.csv", ["award_rid", "name_core"]):
        k = pro2key.get(core)
        if k:
            buyers[arid].add(k)

    sup = {}
    for arid, raw in EV._rows("pro_supplier.csv", ["award_rid", "name"],
                              base=C.RAW):
        e = raw2ent.get((raw or "").strip())
        if e is not None:
            sup[arid] = e

    # company context: totals per entity, computed once
    ent_awards = defaultdict(int)
    ent_nc = defaultdict(int)
    for arid, e in sup.items():
        a = award.get(arid)
        if not a:
            continue
        ent_awards[e] += 1
        if a[1] in C.NONCOMPETITIVE_METHODS:
            ent_nc[e] += 1
    ent_regs = defaultdict(set)
    for rid, e in rid_ent.items():
        ent_regs[e].add(rid)

    rows = []
    for arid, e in sup.items():
        a = award.get(arid)
        if not a:
            continue
        day, meth, lim, desc = a
        Z = buyers.get(arid)
        if not Z:
            continue

        hit_body, hit_day, miss = None, None, 1.0
        for b in Z:
            days = ent_body.get((e, b))
            if not days:
                continue
            lo = bisect.bisect_left(days, day - window)
            hi = bisect.bisect_left(days, day)
            if hi > lo:
                d0 = days[hi - 1]
                if hit_day is None or d0 > hit_day:
                    hit_day, hit_body = d0, b
            ref = reference.get(b)
            if ref:
                rlo = bisect.bisect_left(ref, day - window)
                rhi = bisect.bisect_left(ref, day)
                q = (rhi - rlo) / len(ref)
                miss *= (1.0 - q) ** (hi - lo if hi > lo else len(days))
        if hit_body is None:
            continue

        p = max(1.0 - miss, 1e-12)
        rows.append(dict(
            award_rid=arid, entity=e, body=hit_body, key=(e, hit_body, day),
            award_day=day, lobby_day=hit_day, lag_days=day - hit_day,
            p_coincidence=p, surprise=-math.log10(p),
            method=meth, limited_reason=lim, desc=desc[:120],
            noncompetitive=1.0 if meth in C.NONCOMPETITIVE_METHODS else 0.0,
            n_regs_this_body=len(ent_body.get((e, hit_body), ())),
            ctx_total_awards=ent_awards[e], ctx_total_regs=len(ent_regs[e]),
            ctx_nc_share=(ent_nc[e] / ent_awards[e]) if ent_awards[e] else 0.0,
        ))
    return collapse(rows)


def collapse(rows):
    """One row per (supplier, body, award date).

    A department issuing five task authorisations to one supplier on one day
    is one story, not five.  Without this the top-k fills with repeats of a
    single event -- the same burst problem the entity-level test solved with
    AWARD_UNIT="date", reintroduced by moving to the award unit.  The
    non-competitive flag is kept if ANY award in the group is non-competitive,
    because that is the fact a reporter would check.
    """
    g = {}
    for r in rows:
        k = r.pop("key")
        cur = g.get(k)
        if cur is None:
            r["n_awards_same_day"] = 1
            r["award_rids"] = [r["award_rid"]]
            g[k] = r
        else:
            cur["n_awards_same_day"] += 1
            cur["award_rids"].append(r["award_rid"])
            if r["noncompetitive"] > cur["noncompetitive"]:
                cur["noncompetitive"] = r["noncompetitive"]
                cur["method"] = r["method"]
                cur["limited_reason"] = r["limited_reason"]
    return list(g.values())


def _canonical():
    m = BM.load(approved_only=True)
    pc = BM._counts("pro_contact.csv", "name_core")
    lob2key, pro2key = {}, {}
    for lob, pros in m.items():
        key = max(pros, key=lambda p: pc.get(p, 0))
        lob2key[lob] = key
        for p in pros:
            pro2key[p] = key
    return lob2key, pro2key


# ---------------------------------------------------------------- scoring

AWARD_FEATURES = ("surprise", "proximity", "noncompetitive",
                  "commercial", "identity_confidence")

AWARD_WEIGHTS = {
    "surprise":            0.40,   # calibrated per-award coincidence
    "proximity":           0.15,   # tighter lag is more suspicious
    "noncompetitive":      0.25,   # per-contract fact, no longer diluted
    "commercial":          0.10,   # not a grant recipient
    "identity_confidence": 0.10,
}


def score_rows(rows, members, charity_keys, weights=None, window=None):
    window = window or C.PROXIMITY_WINDOW_DAYS
    for r in rows:
        gl, why = F.grant_likelihood(r["entity"], members, charity_keys)
        r["commercial"] = 1.0 - gl
        r["grant_reason"] = why
        names = members.get(r["entity"], [])
        r["identity_confidence"] = (
            1.0 if len(names) <= 1 else (0.6 if "." in str(r["entity"]) else 0.8))
        r["proximity"] = 1.0 - (r["lag_days"] / window)
    F.normalise  # noqa: kept for symmetry; award features normalise below
    for name in AWARD_FEATURES:
        vals = _percentile(rows, name)
        for r, v in zip(rows, vals):
            r["f_" + name] = v
    w = weights or AWARD_WEIGHTS
    for r in rows:
        r["score"] = sum(w[f] * r["f_" + f] for f in AWARD_FEATURES)
    return rows


def _percentile(rows, key):
    order = sorted(range(len(rows)), key=lambda i: rows[i][key])
    n = len(rows)
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and rows[order[j + 1]][key] == rows[order[i]][key]:
            j += 1
        share = (i + j) / 2.0 / max(n - 1, 1)
        for t in range(i, j + 1):
            out[order[t]] = share
        i = j + 1
    return out


def main():
    ents = pickle.load(open(os.path.join(C.WORK, "entities.pkl"), "rb"))
    rows = build(ents["raw2ent"], ents["members"])
    print(f"awards closing the loop : {len(rows):,}")
    print(f"distinct suppliers      : {len({r['entity'] for r in rows}):,}")
    print(f"non-competitive among them: "
          f"{sum(r['noncompetitive'] for r in rows):,.0f} "
          f"({100 * sum(r['noncompetitive'] for r in rows) / max(len(rows), 1):.1f}%)")
    score_rows(rows, ents["members"], F.load_charity_keys())
    rows.sort(key=lambda r: -r["score"])
    disp = ents["display"]
    cols = ["score", "entity", "supplier", "body", "lag_days", "p_coincidence",
            "surprise", "method", "noncompetitive", "award_day", "lobby_day",
            "n_regs_this_body", "ctx_total_awards", "ctx_nc_share",
            "grant_reason", "award_rid", "desc"]
    os.makedirs(C.WORK, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows[:2000]:
            w.writerow(dict(r, supplier=disp.get(r["entity"], r["entity"]),
                            award_day=EV.from_day(r["award_day"]),
                            lobby_day=EV.from_day(r["lobby_day"])))
    print(f"wrote {OUT}\n")
    print(f"{'supplier':<30}{'body':<26}{'lag':>5}{'1-in':>9}{'method':>10}{'score':>7}")
    for r in rows[:15]:
        print(f"{disp.get(r['entity'], r['entity'])[:28]:<30}{r['body'][:24]:<26}"
              f"{r['lag_days']:>5}{1 / r['p_coincidence']:>9,.0f}"
              f"{r['method'][:8]:>10}{r['score']:>7.3f}")


if __name__ == "__main__":
    main()
