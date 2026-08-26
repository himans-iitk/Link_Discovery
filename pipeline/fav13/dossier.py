#!/usr/bin/env python3
"""
Build the evidence dossier for the ground-truth cases we hit.

For every company in favoritism_ca.csv that resolves to an entity, pull:
  - what the published story claimed
  - which records in OUR databases matched it
  - the top-ranked contracts and the lobbying that preceded them
  - which newsworthiness signals actually fired

Writes dossier.json for the LaTeX renderer.
"""
import csv, json, os, pickle, re
from collections import defaultdict
from pipeline.fav13 import config as C
from pipeline.fav13 import evidence as EV

csv.field_size_limit(10 ** 9)
GT = os.path.join(C.GT, "favoritism_ca.csv")
NONCOMP = C.NONCOMPETITIVE_METHODS


def key(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def d(day):
    return EV.from_day(day)


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


def categories(r):
    """Which newsworthiness signals actually fired for this contract."""
    out = []
    if r["method"] in NONCOMP:
        out.append(("No open competition",
                    f"awarded by method {r['method']}; this buyer sole-sources "
                    f"{100*r['p_comp']:.1f}% of the time"))
    if r["gap"] is not None and r["gap"] <= 365:
        out.append(("Lobbied, then awarded",
                    f"lobbied {r['chan_body_name'] or 'this buyer'} "
                    f"{r['gap']} days before the award"))
    if r["p_conc"] < 0.05:
        out.append(("Supplier dominance",
                    f"holds more of this buyer's contracts than an even split "
                    f"predicts (p={r['p_conc']:.2g})"))
    if r["tender_window"] is not None and r["tender_window"] <= 15:
        out.append(("Short tender window",
                    f"tender open only {r['tender_window']} days"))
    if not out:
        out.append(("Weak signal", "no component fired strongly"))
    return out


def main():
    ents = pickle.load(open(os.path.join(C.WORK, "entities.pkl"), "rb"))
    raw2ent, members, display = ents["raw2ent"], ents["members"], ents.get("display", {})
    recs = pickle.load(open(os.path.join(C.WORK, "fav13_v2_scored.pkl"), "rb"))
    total = len(recs)

    exact = {}
    for raw, e in raw2ent.items():
        exact.setdefault(key(raw), e)

    gtrows = list(csv.DictReader(open(GT, encoding="utf-8", errors="replace")))
    by_ent = defaultdict(list)
    unmatched = []
    for g in gtrows:
        nm = (g.get("private_org") or "").strip()
        if not nm:
            continue
        e = exact.get(key(nm))
        if e:
            by_ent[e].append(g)
        else:
            unmatched.append(nm)

    # index scored contracts by entity
    per_ent = defaultdict(list)
    for r in recs:
        if r["entity"] in by_ent:
            per_ent[r["entity"]].append(r)
    for e in per_ent:
        per_ent[e].sort(key=lambda r: r["rank"])

    # supplier raw names + award ids, so we can show the actual matched records
    sup_names = defaultdict(set)
    for arid, raw in rows("pro_supplier.csv", ["award_rid", "name"], base=C.RAW):
        e = raw2ent.get((raw or "").strip())
        if e in by_ent:
            sup_names[e].add(raw.strip())
    lob_names = defaultdict(set)
    reg_of = defaultdict(list)
    reg_day = {}
    for rid, dd, st in rows("lob_meta.csv", ["rid", "start_date_clean",
                                             "start_date_status"]):
        if st == "valid":
            reg_day[rid] = EV.to_day(dd)
    for rid, raw in rows("lob_org.csv", ["rid", "name"], base=C.RAW):
        e = raw2ent.get((raw or "").strip())
        if e in by_ent:
            lob_names[e].add(raw.strip())
            if rid in reg_day:
                reg_of[e].append(rid)
    reg_bodies = defaultdict(set)
    want = {rid for v in reg_of.values() for rid in v}
    for rid, core in rows("lob_target.csv", ["rid", "name_core"]):
        if rid in want and core:
            reg_bodies[rid].add(core)

    out = {"total_scored": total, "gt_rows": len(gtrows),
           "gt_pairs": len({(g["private_org"], g["public_org"]) for g in gtrows}),
           "gt_companies": len({g["private_org"].strip() for g in gtrows
                                if g["private_org"].strip()}),
           "unmatched": sorted(set(unmatched)), "cases": []}

    for e, gs in sorted(by_ent.items(),
                        key=lambda kv: per_ent[kv[0]][0]["rank"]
                        if per_ent[kv[0]] else 10 ** 9):
        cs = per_ent.get(e, [])
        stories = []
        seen = set()
        for g in gs:
            k = (g.get("title"), g.get("public_org"))
            if k in seen:
                continue
            seen.add(k)
            stories.append(dict(
                title=(g.get("title") or "")[:140],
                outlet=g.get("media_name", ""), date=g.get("publish_date", ""),
                company=g.get("private_org", ""), body=g.get("public_org", ""),
                amount=g.get("contract_amount", ""), when=g.get("contract_time", ""),
                url=g.get("url", "")))
        regs = sorted(reg_of.get(e, []), key=lambda r: reg_day[r])
        case = dict(
            entity=e, name=display.get(e, e),
            member_names=sorted(members.get(e, []))[:8],
            n_members=len(members.get(e, [])),
            supplier_variants=sorted(sup_names.get(e, []))[:6],
            lobby_variants=sorted(lob_names.get(e, []))[:6],
            n_contracts=len(cs), n_registrations=len(regs),
            reg_first=d(reg_day[regs[0]]) if regs else "",
            reg_last=d(reg_day[regs[-1]]) if regs else "",
            bodies=sorted({b for r in regs for b in reg_bodies.get(r, ())})[:10],
            stories=stories, contracts=[])
        for r in cs[:4]:
            case["contracts"].append(dict(
                rank=r["rank"], pct=100.0 * r["rank"] / total,
                award_rid=r["award_rid"][:16], date=d(r["day"]),
                method=r["method"] or "(none recorded)",
                buyers=r["buyer_names"], channel=r["chan_body_name"],
                gap=r["gap"], n_chan_regs=r["n_chan_regs"],
                tender_window=r["tender_window"],
                p_time=r["p_time"], p_comp=r["p_comp"],
                p_proc=r["p_proc"], p_conc=r["p_conc"],
                pi=r["pi"], score=r["score"],
                cats=categories(r)))
        out["cases"].append(case)

    json.dump(out, open(os.path.join(C.WORK, "dossier.json"), "w"), indent=1)
    print(f"cases: {len(out['cases'])}")
    for c in out["cases"]:
        best = c["contracts"][0]["rank"] if c["contracts"] else None
        print(f"  {c['name'][:40]:<42} contracts={c['n_contracts']:>6,} "
              f"regs={c['n_registrations']:>4}  best rank={best}")


if __name__ == "__main__":
    main()
