"""
Evidence assembly: one record per entity, carrying everything a score or a
journalist needs.

Two kinds of edge are collected and kept distinguishable:

  CERTAIN   joins inside a source -- lob_org/lob_target share a lob_meta.rid,
            pro_supplier/pro_contact share a pro_award.rid.  These are facts.
  UNCERTAIN the identity assertion company == supplier (FAV 13), and later
            body == buyer (FAV 12).  These are inferences and carry a
            confidence.

Nothing here scores anything.  Assembly and scoring stay separate so that a
change of scoring function never silently changes what evidence exists.
"""

import csv, datetime, os, pickle
from collections import defaultdict, Counter

from . import config as C

csv.field_size_limit(10 ** 9)
_EPOCH = datetime.date(1990, 1, 1)


def to_day(iso):
    return (datetime.date(int(iso[:4]), int(iso[5:7]), int(iso[8:10])) - _EPOCH).days


def from_day(d):
    return (_EPOCH + datetime.timedelta(days=int(d))).isoformat()


def _rows(fname, cols, base=None):
    base = base or C.CLEAN
    with open(os.path.join(base, fname), newline="", encoding="utf-8",
              errors="replace") as fh:
        r = csv.reader(fh)
        hdr = next(r)
        idx = [hdr.index(c) for c in cols]
        for x in r:
            if len(x) > max(idx):
                yield [x[i] for i in idx]


def lobbying_start_dates():
    """The null's reference population: every valid registration start."""
    out = []
    for d, s in _rows("lob_meta.csv", ["start_date_clean", "start_date_status"]):
        if s == "valid":
            out.append(to_day(d))
    out.sort()
    return out


def build(raw2ent):
    """-> {entity: evidence dict}"""
    reg_date = {}
    for rid, d, s in _rows("lob_meta.csv",
                           ["rid", "start_date_clean", "start_date_status"]):
        if s == "valid":
            reg_date[rid] = to_day(d)

    # bodies lobbied per registration (certain edge; used by FAV 12 later)
    reg_bodies = defaultdict(set)
    for rid, core in _rows("lob_target.csv", ["rid", "name_core"]):
        if core:
            reg_bodies[rid].add(core)

    award = {}
    for rid, d, s, meth, lim in _rows(
            "pro_award.csv", ["rid", "award_date_clean", "award_date_status",
                              "method", "limited_reason"]):
        if s != "valid":
            continue
        day = to_day(d)
        if rid not in award or day < award[rid][0]:
            award[rid] = (day, meth or "", lim or "")

    award_buyers = defaultdict(set)
    for arid, core in _rows("pro_contact.csv", ["award_rid", "name_core"]):
        if core:
            award_buyers[arid].add(core)

    ev = defaultdict(lambda: dict(
        regs=set(), reg_days=[], bodies=set(),
        awards=set(), award_days=[], buyers=set(),
        methods=Counter(), limited_reasons=Counter(),
        raw_lob=set(), raw_pro=set()))

    for rid, raw in _rows("lob_org.csv", ["rid", "name"], base=C.RAW):
        e = raw2ent.get((raw or "").strip())
        if e is None:
            continue
        r = ev[e]
        r["regs"].add(rid)
        r["raw_lob"].add(raw)
        r["bodies"] |= reg_bodies.get(rid, set())
        if rid in reg_date:
            r["reg_days"].append(reg_date[rid])

    for arid, raw in _rows("pro_supplier.csv", ["award_rid", "name"], base=C.RAW):
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
    """Entities present on both sides with dates on both -- the FAV 13 edges."""
    return [e for e, r in ev.items() if r["reg_days"] and r["award_days"]]


def save(raw2ent, path=None):
    path = path or os.path.join(C.WORK, "evidence.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ev = build(raw2ent)
    pickle.dump(dict(evidence=ev, dual=dual_side(ev),
                     null_reference=lobbying_start_dates()), open(path, "wb"))
    return path
