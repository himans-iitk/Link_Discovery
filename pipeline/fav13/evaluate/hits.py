"""Hit sheet: for each matched published story, what our data actually holds.
Facts only: name variants, contract and lobbying stats, and per story the
contracts in the department and period it names.

    python3 -m pipeline.fav13.evaluate.hits     # -> work/hits.json
"""

import json
from collections import defaultdict

from pipeline import paths
from .. import common
from ..common import NONCOMP, dept_match, dept_tokens, from_day, in_period, year_span
from .cases import bodies_lobbied, holdings, stories_by_entity


def story_row(g, cs, bodies):
    dept = (g.get("public_org") or "").strip()
    dt = dept_tokens(dept)
    lo, hi = year_span(g.get("contract_time"))
    match = [r for r in cs if dept_match(r, dt)]
    inwin = [r for r in match if in_period(r, lo, hi)]
    lobbied = bool(dt and any(dt & dept_tokens(b) for b in bodies))
    return dict(
        title=(g.get("title") or "").strip(), outlet=g.get("media_name", ""),
        date=g.get("publish_date", ""), dept=dept,
        amount=(g.get("contract_amount") or "").strip(),
        when=(g.get("contract_time") or "").strip(),
        n_dept=len(match), n_win=len(inwin),
        nc_win=sum(1 for r in inwin if r["method"] in NONCOMP),
        lobbied_dept="yes" if lobbied else "no",
        examples=[dict(date=from_day(r["day"]), method=r["method"] or "-",
                       buyer=r["buyer_names"][:46], gap=r["gap"])
                  for r in sorted(inwin or match, key=lambda r: r["rank"])[:3]])


def main():
    ents = common.load_entities()
    raw2ent, display = ents["raw2ent"], ents.get("display", {})
    recs = common.load_scored()

    by_ent, _ = stories_by_entity(raw2ent, common.read_stories())
    per = defaultdict(list)
    for r in recs:
        if r["entity"] in by_ent:
            per[r["entity"]].append(r)
    h = holdings(raw2ent, by_ent)
    reg_day = h["reg_day"]

    out = {"cases": []}
    for e, gs in sorted(by_ent.items(), key=lambda kv: -len(per[kv[0]])):
        cs = per.get(e, [])
        nc = sum(1 for r in cs if r["method"] in NONCOMP)
        yrs = sorted({from_day(r["day"])[:4] for r in cs})
        regs = h["regs"].get(e, [])
        bodies = bodies_lobbied(h, e)
        variants = h["sup_names"].get(e, set()) | h["lob_names"].get(e, set())

        stories, seen = [], set()
        for g in gs:
            t = (g.get("title") or "").strip()
            if t not in seen:
                seen.add(t)
                stories.append(story_row(g, cs, bodies))

        gaps = [r["gap"] for r in cs if r["gap"] is not None]
        out["cases"].append(dict(
            entity=e, name=display.get(e, e),
            variants=sorted(variants)[:7], n_variants=len(variants),
            n_contracts=len(cs), n_nc=nc, nc_share=(nc / len(cs)) if cs else 0.0,
            yr_first=yrs[0] if yrs else "", yr_last=yrs[-1] if yrs else "",
            n_buyers=len({b for r in cs for b in r["buyers"]}),
            n_regs=len(regs), n_bodies=len(bodies),
            reg_first=from_day(reg_day[regs[0]])[:10] if regs else "",
            reg_last=from_day(reg_day[regs[-1]])[:10] if regs else "",
            min_gap=min(gaps, default=None),
            n_gap_30=sum(1 for g in gaps if g <= 30),
            stories=stories))

    with open(paths.WORK / "hits.json", "w") as fh:
        json.dump(out, fh, indent=1)
    for c in out["cases"]:
        print(f"{c['name'][:36]:<38} contracts {c['n_contracts']:>5,} "
              f"({c['nc_share']*100:4.1f}% no-bid)  regs {c['n_regs']:>4}  "
              f"stories {len(c['stories'])}")


if __name__ == "__main__":
    main()
