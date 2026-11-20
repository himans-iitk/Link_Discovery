"""Evidence dossier for each published-story company we resolve: what the story
claimed, which of our records matched, the top-ranked contracts, and which
newsworthiness signals fired.

    python3 -m pipeline.fav13.evaluate.dossier     # -> work/dossier.json
"""

import json
from collections import defaultdict

from pipeline import paths
from .. import common
from ..common import NONCOMP, from_day
from .cases import bodies_lobbied, holdings, stories_by_entity


def categories(r):
    """Newsworthiness signals that fired for one contract, with reasons."""
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
        out.append(("Short tender window", f"tender open only {r['tender_window']} days"))
    return out or [("Weak signal", "no component fired strongly")]


def main():
    ents = common.load_entities()
    raw2ent, members, display = ents["raw2ent"], ents["members"], ents.get("display", {})
    recs = common.load_scored()
    total = len(recs)
    stories = common.read_stories()

    by_ent, unmatched = stories_by_entity(raw2ent, stories)
    per_ent = defaultdict(list)
    for r in recs:
        if r["entity"] in by_ent:
            per_ent[r["entity"]].append(r)
    for e in per_ent:
        per_ent[e].sort(key=lambda r: r["rank"])
    h = holdings(raw2ent, by_ent)
    reg_day = h["reg_day"]

    out = {"total_scored": total, "gt_rows": len(stories),
           "gt_pairs": len({(g["private_org"], g["public_org"]) for g in stories}),
           "gt_companies": len({g["private_org"].strip() for g in stories
                                if g["private_org"].strip()}),
           "unmatched": sorted(set(unmatched)), "cases": []}

    best_rank = lambda e: per_ent[e][0]["rank"] if per_ent[e] else 10 ** 9
    for e, gs in sorted(by_ent.items(), key=lambda kv: best_rank(kv[0])):
        cs = per_ent.get(e, [])
        story_rows, seen = [], set()
        for g in gs:
            k = (g.get("title"), g.get("public_org"))
            if k in seen:
                continue
            seen.add(k)
            story_rows.append(dict(
                title=(g.get("title") or "")[:140],
                outlet=g.get("media_name", ""), date=g.get("publish_date", ""),
                company=g.get("private_org", ""), body=g.get("public_org", ""),
                amount=g.get("contract_amount", ""), when=g.get("contract_time", ""),
                url=g.get("url", "")))
        regs = h["regs"].get(e, [])
        out["cases"].append(dict(
            entity=e, name=display.get(e, e),
            member_names=sorted(members.get(e, []))[:8],
            n_members=len(members.get(e, [])),
            supplier_variants=sorted(h["sup_names"].get(e, []))[:6],
            lobby_variants=sorted(h["lob_names"].get(e, []))[:6],
            n_contracts=len(cs), n_registrations=len(regs),
            reg_first=from_day(reg_day[regs[0]]) if regs else "",
            reg_last=from_day(reg_day[regs[-1]]) if regs else "",
            bodies=bodies_lobbied(h, e)[:10],
            stories=story_rows,
            contracts=[dict(
                rank=r["rank"], pct=100.0 * r["rank"] / total,
                award_rid=r["award_rid"][:16], date=from_day(r["day"]),
                method=r["method"] or "(none recorded)",
                buyers=r["buyer_names"], channel=r["chan_body_name"],
                gap=r["gap"], n_chan_regs=r["n_chan_regs"],
                tender_window=r["tender_window"],
                p_time=r["p_time"], p_comp=r["p_comp"],
                p_proc=r["p_proc"], p_conc=r["p_conc"],
                pi=r["pi"], score=r["score"], cats=categories(r)) for r in cs[:4]]))

    with open(paths.WORK / "dossier.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"cases: {len(out['cases'])}")
    for c in out["cases"]:
        best = c["contracts"][0]["rank"] if c["contracts"] else None
        print(f"  {c['name'][:40]:<42} contracts={c['n_contracts']:>6,} "
              f"regs={c['n_registrations']:>4}  best rank={best}")


if __name__ == "__main__":
    main()
