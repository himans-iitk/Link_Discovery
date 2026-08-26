#!/usr/bin/env python3
"""
Hit sheet: for each published story we matched, what our databases actually hold.

No commentary, no method discussion.  Per company:
  - the records that matched (name variants, counts, date ranges)
  - our procurement stats for that company
  - per story: the department and period it names, and what we hold for exactly that
"""
import csv, json, os, pickle, re
from collections import defaultdict
from pipeline.fav13 import config as C
from pipeline.fav13 import evidence as EV

csv.field_size_limit(10 ** 9)
GT = os.path.join(C.GT, "favoritism_ca.csv")
NONCOMP = C.NONCOMPETITIVE_METHODS
STOP = {"of", "the", "and", "for", "canada", "canadian", "government", "department",
        "ministry", "agency", "office", "inc", "ltd", "corporation", "commission"}


def key(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def toks(s):
    return {t for t in re.sub(r"[^a-z ]", " ", (s or "").lower()).split()
            if t and t not in STOP and len(t) > 2}


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


def years(s):
    return [int(y) for y in re.findall(r"(19|20)\d{2}", s or "")] and \
           [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", s or "")]


def main():
    ents = pickle.load(open(os.path.join(C.WORK, "entities.pkl"), "rb"))
    raw2ent, members, display = ents["raw2ent"], ents["members"], ents.get("display", {})
    recs = pickle.load(open(os.path.join(C.WORK, "fav13_v2_scored.pkl"), "rb"))

    exact = {}
    for raw, e in raw2ent.items():
        exact.setdefault(key(raw), e)

    gt = list(csv.DictReader(open(GT, encoding="utf-8", errors="replace")))
    by_ent = defaultdict(list)
    for g in gt:
        e = exact.get(key((g.get("private_org") or "").strip()))
        if e:
            by_ent[e].append(g)

    per = defaultdict(list)
    for r in recs:
        if r["entity"] in by_ent:
            per[r["entity"]].append(r)

    sup_names = defaultdict(set)
    for arid, raw in rows("pro_supplier.csv", ["award_rid", "name"], base=C.RAW):
        e = raw2ent.get((raw or "").strip())
        if e in by_ent:
            sup_names[e].add(raw.strip())

    reg_day, lob_names = {}, defaultdict(set)
    for rid, dd, st in rows("lob_meta.csv", ["rid", "start_date_clean",
                                             "start_date_status"]):
        if st == "valid":
            reg_day[rid] = EV.to_day(dd)
    reg_of = defaultdict(list)
    for rid, raw in rows("lob_org.csv", ["rid", "name"], base=C.RAW):
        e = raw2ent.get((raw or "").strip())
        if e in by_ent:
            lob_names[e].add(raw.strip())
            if rid in reg_day:
                reg_of[e].append(rid)
    reg_bodies = defaultdict(set)
    want = {r for v in reg_of.values() for r in v}
    for rid, core in rows("lob_target.csv", ["rid", "name_core"]):
        if rid in want and core:
            reg_bodies[rid].add(core)

    out = {"cases": []}
    for e, gs in sorted(by_ent.items(), key=lambda kv: -len(per[kv[0]])):
        cs = per.get(e, [])
        nc = sum(1 for r in cs if r["method"] in NONCOMP)
        yrs = sorted({EV.from_day(r["day"])[:4] for r in cs})
        regs = sorted(reg_of.get(e, []), key=lambda r: reg_day[r])
        bodies = sorted({b for r in regs for b in reg_bodies.get(r, ())})

        # per-story alignment
        stories = []
        seen = set()
        for g in gs:
            t = (g.get("title") or "").strip()
            if t in seen:
                continue
            seen.add(t)
            dept = (g.get("public_org") or "").strip()
            dt = toks(dept)
            yy = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b",
                                             g.get("contract_time") or "")]
            lo, hi = (min(yy), max(yy)) if yy else (None, None)

            match = [r for r in cs if dt and (dt & toks(r["buyer_names"]))]
            inwin = [r for r in match
                     if lo is None or lo <= int(EV.from_day(r["day"])[:4]) <= hi]
            lobbied = bool(dt and any(dt & toks(b) for b in bodies))
            stories.append(dict(
                title=t, outlet=g.get("media_name", ""), date=g.get("publish_date", ""),
                dept=dept, amount=(g.get("contract_amount") or "").strip(),
                when=(g.get("contract_time") or "").strip(),
                n_dept=len(match), n_win=len(inwin),
                nc_win=sum(1 for r in inwin if r["method"] in NONCOMP),
                lobbied_dept="yes" if lobbied else "no",
                examples=[dict(date=EV.from_day(r["day"]),
                               method=r["method"] or "-",
                               buyer=r["buyer_names"][:46],
                               gap=r["gap"])
                          for r in sorted(inwin or match,
                                          key=lambda r: r["rank"])[:3]]))

        out["cases"].append(dict(
            entity=e, name=display.get(e, e),
            variants=sorted(sup_names.get(e, set()) | lob_names.get(e, set()))[:7],
            n_variants=len(sup_names.get(e, set()) | lob_names.get(e, set())),
            n_contracts=len(cs), n_nc=nc,
            nc_share=(nc / len(cs)) if cs else 0.0,
            yr_first=yrs[0] if yrs else "", yr_last=yrs[-1] if yrs else "",
            n_buyers=len({b for r in cs for b in r["buyers"]}),
            n_regs=len(regs), n_bodies=len(bodies),
            reg_first=EV.from_day(reg_day[regs[0]])[:10] if regs else "",
            reg_last=EV.from_day(reg_day[regs[-1]])[:10] if regs else "",
            min_gap=min([r["gap"] for r in cs if r["gap"] is not None], default=None),
            n_gap_30=sum(1 for r in cs if r["gap"] is not None and r["gap"] <= 30),
            stories=stories))

    json.dump(out, open(os.path.join(C.WORK, "hits.json"), "w"), indent=1)
    for c in out["cases"]:
        print(f"{c['name'][:36]:<38} contracts {c['n_contracts']:>5,} "
              f"({c['nc_share']*100:4.1f}% no-bid)  regs {c['n_regs']:>4}  "
              f"stories {len(c['stories'])}")


if __name__ == "__main__":
    main()
