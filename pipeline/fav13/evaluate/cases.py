"""Published-story cases: which entity each names, and what the corpus holds for it.
Shared by hits.py and dossier.py."""

from collections import defaultdict

from pipeline import paths
from .. import common
from ..common import rows


def stories_by_entity(raw2ent, stories):
    """-> ({entity: [story rows]}, unmatched company names)"""
    index = common.name_index(raw2ent)
    by_ent, unmatched = defaultdict(list), []
    for g in stories:
        nm = (g.get("private_org") or "").strip()
        if not nm:
            continue
        e = index.get(common.name_key(nm))
        if e:
            by_ent[e].append(g)
        else:
            unmatched.append(nm)
    return by_ent, unmatched


def holdings(raw2ent, wanted):
    """Raw records behind each wanted entity.

    -> dict(sup_names, lob_names: {e: raw names},
            regs: {e: rids sorted by date}, reg_day, reg_bodies: {rid: bodies})
    """
    sup_names, lob_names, reg_of = defaultdict(set), defaultdict(set), defaultdict(list)
    for _, raw in rows("pro_supplier.csv", ["award_rid", "name"], base=paths.RAW):
        e = raw2ent.get((raw or "").strip())
        if e in wanted:
            sup_names[e].add(raw.strip())

    reg_day = common.registrations()
    for rid, raw in rows("lob_org.csv", ["rid", "name"], base=paths.RAW):
        e = raw2ent.get((raw or "").strip())
        if e in wanted:
            lob_names[e].add(raw.strip())
            if rid in reg_day:
                reg_of[e].append(rid)

    want = {rid for v in reg_of.values() for rid in v}
    reg_bodies = defaultdict(set)
    for rid, core in rows("lob_target.csv", ["rid", "name_core"]):
        if rid in want and core:
            reg_bodies[rid].add(core)

    regs = {e: sorted(v, key=lambda r: reg_day[r]) for e, v in reg_of.items()}
    return dict(sup_names=sup_names, lob_names=lob_names, regs=regs,
                reg_day=reg_day, reg_bodies=reg_bodies)


def bodies_lobbied(h, e):
    return sorted({b for r in h["regs"].get(e, []) for b in h["reg_bodies"].get(r, ())})
