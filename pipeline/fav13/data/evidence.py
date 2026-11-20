"""Per-entity evidence: lobbying and award calendars, bodies and buyers.

Within-source joins (lob_org/lob_target on rid, pro_supplier/pro_contact on
award rid) are facts. The company == supplier link is the inferred edge.
Nothing here scores; assembly and scoring stay separate.
"""

from collections import Counter, defaultdict

from pipeline import paths
from .. import common
from ..common import rows, to_day


def lobbying_start_dates():
    """The null's reference population: every valid registration start."""
    return sorted(to_day(d) for d, s in
                  rows("lob_meta.csv", ["start_date_clean", "start_date_status"])
                  if s == "valid")


def build(raw2ent):
    """-> {entity: evidence dict}"""
    reg_date = {rid: to_day(d) for rid, d, s in
                rows("lob_meta.csv", ["rid", "start_date_clean", "start_date_status"])
                if s == "valid"}

    reg_bodies = defaultdict(set)
    for rid, core in rows("lob_target.csv", ["rid", "name_core"]):
        if core:
            reg_bodies[rid].add(core)

    award = {}                          # rid -> (earliest day, method, reason)
    for rid, d, s, meth, lim in rows(
            "pro_award.csv", ["rid", "award_date_clean", "award_date_status",
                              "method", "limited_reason"]):
        if s != "valid":
            continue
        day = to_day(d)
        if rid not in award or day < award[rid][0]:
            award[rid] = (day, meth or "", lim or "")

    award_buyers = defaultdict(set)
    for arid, core in rows("pro_contact.csv", ["award_rid", "name_core"]):
        if core:
            award_buyers[arid].add(core)

    ev = defaultdict(lambda: dict(
        regs=set(), reg_days=[], bodies=set(),
        awards=set(), award_days=[], buyers=set(),
        methods=Counter(), limited_reasons=Counter(),
        raw_lob=set(), raw_pro=set()))

    for rid, raw in rows("lob_org.csv", ["rid", "name"], base=paths.RAW):
        e = raw2ent.get((raw or "").strip())
        if e is None:
            continue
        r = ev[e]
        r["regs"].add(rid)
        r["raw_lob"].add(raw)
        r["bodies"] |= reg_bodies.get(rid, set())
        if rid in reg_date:
            r["reg_days"].append(reg_date[rid])

    for arid, raw in rows("pro_supplier.csv", ["award_rid", "name"], base=paths.RAW):
        e = raw2ent.get((raw or "").strip())
        if e is None:
            continue
        r = ev[e]
        r["awards"].add(arid)
        r["raw_pro"].add(raw)
        r["buyers"] |= award_buyers.get(arid, set())
        a = award.get(arid)
        if a:
            r["award_days"].append(a[0])
            r["methods"][a[1]] += 1
            if a[2]:
                r["limited_reasons"][a[2]] += 1

    for r in ev.values():
        r["reg_days"].sort()
        r["award_days"].sort()
    return dict(ev)


def dual_side(ev):
    """Entities dated on both sides: the FAV 13 candidates."""
    return [e for e, r in ev.items() if r["reg_days"] and r["award_days"]]


def save(raw2ent):
    ev = build(raw2ent)
    return common.dump(dict(evidence=ev, dual=dual_side(ev),
                            null_reference=lobbying_start_dates()), "evidence.pkl")
