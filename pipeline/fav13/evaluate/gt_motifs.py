"""Complete the ground-truth motifs, then perturb them into negatives.

Stories name a company and a department, never a contract. Part 1 resolves
each story to the actual awards and registrations. Part 2 breaks ONE element
of each complete motif per technique (T1..T6), so every negative is a
counterfactual of a real case.

    python3 -m pipeline.fav13.evaluate.gt_motifs
        -> docs/gt_motifs_and_negatives.xlsx, work/gt_motifs_positive.csv
"""

import bisect
import random
from collections import Counter, defaultdict

from pipeline import paths
from .. import common
from ..common import dept_tokens as tk, first_buyer, from_day, in_period, to_day
from ..report import xlsx

STATUS_COLS = ["story", "outlet", "published", "story_company", "story_department",
               "story_amount", "story_period", "status", "n_motifs", "entity_id",
               "resolved_company", "n_contracts_company", "n_registrations_company"]
MOTIF_COLS = ["motif_id", "label", "story", "company", "entity_id", "department",
              "award_rid", "award_date", "method", "limited_reason", "lag_days",
              "n_links", "nearest_reg_rid", "nearest_reg_num", "nearest_reg_date",
              "n_regs_this_dept", "p_time", "p_comp", "p_conc", "p_proc",
              "tender_window", "company_total_awards", "company_total_regs", "source"]
NEG_COLS = ["motif_id", "label", "technique", "derived_from", "what_changed",
            "company", "entity_id", "department", "award_rid", "award_date",
            "method", "limited_reason", "lag_days", "n_links", "nearest_reg_date",
            "n_regs_this_dept", "p_time", "p_comp", "p_conc",
            "company_total_awards", "company_total_regs"]
TECHNIQUES = ["T1_SHIFT_LOBBYING", "T2_SWAP_DEPARTMENT", "T3_SWAP_COMPANY",
              "T4_MAKE_COMPETITIVE", "T5_DISTANT_TIMING", "T6_NO_LOBBYING_AT_ALL"]

# fields cleared when a negative has no lobbying link at all
NO_LINK = dict(lag_days="", n_links=0, nearest_reg_rid="", nearest_reg_num="",
               nearest_reg_date="", n_regs_this_dept=0, p_time=1.0)


def complete(ctx):
    """Part 1 -> (status row per story, positive motifs)."""
    by_ent, channel, ent_regs, ent_awards = ctx["by_ent"], ctx["channel"], ctx["ent_regs"], ctx["ent_awards"]
    display, lim, reg_num = ctx["display"], ctx["lim"], ctx["reg_num"]
    status, motifs, seen_story = [], [], set()

    for g in ctx["stories"]:
        title = (g.get("title") or "").strip()
        comp = (g.get("private_org") or "").strip()
        dept = (g.get("public_org") or "").strip()
        lo, hi = common.year_span(g.get("contract_time"))
        e = ctx["index"].get(common.name_key(comp))
        dt = tk(dept)

        row = dict(story=title[:90], outlet=g.get("media_name", ""),
                   published=g.get("publish_date", ""),
                   story_company=comp, story_department=dept,
                   story_amount=(g.get("contract_amount") or "").strip(),
                   story_period=(g.get("contract_time") or "").strip(),
                   entity_id=e or "", resolved_company=display.get(e, "") if e else "",
                   n_contracts_company=ent_awards.get(e, 0) if e else 0,
                   n_registrations_company=ent_regs.get(e, 0) if e else 0)
        status.append(row)
        if not e:
            row.update(status="company not in corpus", n_motifs=0)
            continue
        cand = [r for r in by_ent[e] if common.dept_match(r, dt)]
        use = [r for r in cand if in_period(r, lo, hi)] or cand
        if not use:
            row.update(status="department not matched to any contract", n_motifs=0)
            continue
        row.update(status="complete", n_motifs=len(use))

        if title in seen_story:
            continue
        seen_story.add(title)
        for r in sorted(use, key=lambda r: r["rank"])[:60]:
            links = sorted((r["day"] - d, rid, from_day(d), b)
                           for b in r["buyers"]
                           for d, rid in channel.get((e, b), ()) if d < r["day"])
            motifs.append(dict(
                motif_id=f"gt_{len(motifs)+1:04d}",
                story=title[:80], entity_id=e, company=display.get(e, e),
                department=first_buyer(r)[:60],
                award_rid=r["award_rid"], award_date=from_day(r["day"]),
                method=r["method"] or "", limited_reason=lim.get(r["award_rid"], ""),
                lag_days=(r["gap"] if r["gap"] is not None else ""),
                n_links=len(links),
                nearest_reg_rid=links[0][1] if links else "",
                nearest_reg_num=reg_num.get(links[0][1], "") if links else "",
                nearest_reg_date=links[0][2] if links else "",
                n_regs_this_dept=r["n_chan_regs"],
                p_time=round(r["p_time"], 6), p_comp=round(r["p_comp"], 6),
                p_conc=round(r["p_conc"], 8), p_proc=round(r["p_proc"], 6),
                tender_window=(r["tender_window"] if r["tender_window"] is not None else ""),
                company_total_awards=ent_awards[e],
                company_total_regs=ent_regs.get(e, 0),
                label="positive", source="published story"))
    return status, motifs


def perturb(ctx, motifs):
    """Part 2 -> {technique: negatives}. Draw order matters for reproducibility."""
    recs, by_ent, ent_awards, ent_regs = ctx["recs"], ctx["by_ent"], ctx["ent_awards"], ctx["ent_regs"]
    display, lim = ctx["display"], ctx["lim"]
    gt_ents = {m["entity_id"] for m in motifs}
    gt_awards = {m["award_rid"] for m in motifs}
    sheets = {}

    def mk(m, cat, **over):
        d = dict(m)
        d.update(label="negative", technique=cat, derived_from=m["motif_id"])
        d.update(over)
        d["motif_id"] = f"{cat}_{len(sheets.get(cat, []))+1:04d}"
        return d

    def from_award(r, **over):
        return dict(award_rid=r["award_rid"], award_date=from_day(r["day"]),
                    method=r["method"] or "", limited_reason=lim.get(r["award_rid"], ""),
                    **over)

    def other_company(r):
        return dict(entity_id=r["entity"], company=display.get(r["entity"], r["entity"]),
                    company_total_awards=ent_awards[r["entity"]])

    # T1: registration dates redrawn from the national calendar, count kept
    t1 = []
    reference = ctx["reference"]
    for m in motifs:
        n = max(int(m["n_regs_this_dept"] or 1), 1)
        drawn = sorted(random.choice(reference) for _ in range(min(n, 400)))
        aday = to_day(m["award_date"])
        i = bisect.bisect_left(drawn, aday)
        t1.append(mk(m, "T1_SHIFT_LOBBYING", lag_days=(aday - drawn[i - 1]) if i > 0 else "",
                     nearest_reg_date=from_day(drawn[i - 1]) if i > 0 else "",
                     nearest_reg_rid="(redrawn)", nearest_reg_num="",
                     what_changed="registration dates redrawn from the national "
                                  "lobbying calendar, count preserved"))
    sheets["T1_SHIFT_LOBBYING"] = t1

    # T2: same company, a department it never lobbied before the award
    t2 = []
    for m in motifs:
        alt = [r for r in by_ent[m["entity_id"]]
               if r["award_rid"] not in gt_awards and r["gap"] is None]
        if alt:
            r = random.choice(alt)
            t2.append(mk(m, "T2_SWAP_DEPARTMENT", department=first_buyer(r)[:60],
                         **from_award(r, **NO_LINK),
                         what_changed="same company, but a department it never "
                                      "lobbied before this award"))
    sheets["T2_SWAP_DEPARTMENT"] = t2

    # T3: same department, a company with no lobbying link to it
    dept_of = {}
    for r in recs:
        if r["buyers"]:
            dept_of.setdefault(r["buyers"][0], first_buyer(r))
    dept_pool = defaultdict(list)
    for r in recs:
        if r["gap"] is None and r["buyers"]:
            dept_pool[r["buyers"][0]].append(r)
    t3 = []
    for m in motifs:
        tgt = tk(m["department"])
        cands = next((lst for b, lst in dept_pool.items() if tgt & tk(dept_of.get(b, ""))), [])
        cands = [r for r in cands if r["entity"] not in gt_ents]
        if cands:
            r = random.choice(cands)
            t3.append(mk(m, "T3_SWAP_COMPANY", **other_company(r),
                         **from_award(r, **NO_LINK),
                         company_total_regs=ent_regs.get(r["entity"], 0),
                         what_changed="same department and period, but a company "
                                      "with no lobbying link to it"))
    sheets["T3_SWAP_COMPANY"] = t3

    # T4: same company and department, openly competed
    t4 = []
    for m in motifs:
        tgt = tk(m["department"])
        alt = [r for r in by_ent[m["entity_id"]]
               if r["method"] == "OPEN" and tgt & tk(r["buyer_names"])]
        if alt:
            r = random.choice(alt)
            t4.append(mk(m, "T4_MAKE_COMPETITIVE", award_rid=r["award_rid"],
                         award_date=from_day(r["day"]), method="OPEN",
                         lag_days=(r["gap"] if r["gap"] is not None else ""),
                         limited_reason="", p_comp=1.0,
                         what_changed="same company and department, but this award "
                                      "was won through open competition"))
    sheets["T4_MAKE_COMPETITIVE"] = t4

    # T5: same pair, nearest lobbying > 3 years earlier
    t5 = []
    for m in motifs:
        tgt = tk(m["department"])
        alt = [r for r in by_ent[m["entity_id"]]
               if r["gap"] is not None and r["gap"] > 1095 and tgt & tk(r["buyer_names"])]
        if alt:
            r = random.choice(alt)
            t5.append(mk(m, "T5_DISTANT_TIMING", **from_award(r, lag_days=r["gap"]),
                         what_changed=f"same company and department, but the nearest "
                                      f"lobbying was {r['gap']//365} years earlier"))
    sheets["T5_DISTANT_TIMING"] = t5

    # T6: a supplier that never lobbies at all
    nolob = [r for r in recs if ent_regs.get(r["entity"], 0) == 0 and r["buyers"]]
    t6 = []
    for m in motifs:
        tgt = tk(m["department"])
        cands = [r for r in nolob if tgt & tk(r["buyer_names"])][:400]
        if cands:
            r = random.choice(cands)
            t6.append(mk(m, "T6_NO_LOBBYING_AT_ALL", **other_company(r),
                         **from_award(r, **NO_LINK), company_total_regs=0,
                         what_changed="same department, but a supplier that has never "
                                      "registered to lobby anyone"))
    sheets["T6_NO_LOBBYING_AT_ALL"] = t6
    return sheets


def main():
    random.seed(0)
    print("loading ...", flush=True)
    recs = common.load_scored()
    ents = common.load_entities()
    raw2ent = ents["raw2ent"]
    reg_day, reg_num = common.registrations(with_rnum=True)
    channel, ent_regs = common.lobby_channels(raw2ent, reg_day)
    by_ent = defaultdict(list)
    for r in recs:
        by_ent[r["entity"]].append(r)
    ctx = dict(recs=recs, by_ent=by_ent, channel=channel, ent_regs=ent_regs,
               ent_awards=Counter(r["entity"] for r in recs),
               display=ents.get("display", {}), lim=common.limited_reasons(),
               reg_num=reg_num, reference=sorted(reg_day.values()),
               index=common.name_index(raw2ent), stories=common.read_stories())

    print("part 1: completing ground-truth motifs ...", flush=True)
    status, motifs = complete(ctx)
    print(f"   stories: {len(ctx['stories'])}   complete motifs: {len(motifs):,}")
    for k, v in Counter(s["status"] for s in status).most_common():
        print(f"     {k:<44}{v:>4}")

    print("part 2: perturbing ...", flush=True)
    sheets = perturb(ctx, motifs)
    for k, v in sheets.items():
        print(f"   {k:<26}{len(v):>6,}")

    def table(rs, cols):
        return [cols] + [[r.get(c, "") for c in cols] for r in rs]

    book = [("GT status", table(status, STATUS_COLS),
             [46, 16, 11, 22, 26, 13, 12, 30, 9, 12, 26, 13, 13]),
            ("GT motifs (positive)", table(motifs, MOTIF_COLS),
             [10, 8, 40, 26, 10, 30, 20, 11, 10, 30, 9, 8, 20, 16, 14, 10,
              10, 10, 12, 10, 11, 13, 13, 16])]
    book += [(k[:31], table(sheets[k], NEG_COLS),
              [18, 8, 22, 10, 52, 26, 10, 30, 20, 11, 10, 26, 9, 8, 14, 10,
               10, 10, 12, 13, 13]) for k in TECHNIQUES]
    out = xlsx.write(paths.DOCS / "gt_motifs_and_negatives.xlsx", book)
    print(f"\nwrote {out}")
    print(f"wrote {common.write_csv(paths.WORK / 'gt_motifs_positive.csv', motifs, MOTIF_COLS)}")


if __name__ == "__main__":
    main()
