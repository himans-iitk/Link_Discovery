#!/usr/bin/env python3
"""
Part 1: complete the ground-truth motifs.
Part 2: perturb them to make negatives, one technique per sheet.

The ground truth as delivered holds only story metadata: a company name, a
department name, sometimes an amount and a period.  It names no contract and no
registration.  Here we resolve each story against the corpus and fill in the
whole motif -- which award, which registrations, what the gap was, how unusual
the method was for that department, how much of that department's book the
supplier holds.

Then every complete motif is perturbed.  Each technique breaks ONE element and
holds the rest fixed, so the negative is a counterfactual of a real known case
rather than a random contract.

    python3 -m pipeline.fav13.gt_motifs
"""
import bisect, csv, os, pickle, random, re
from collections import Counter, defaultdict
from pipeline.fav13 import config as C
from pipeline.fav13 import evidence as EV
from pipeline.fav13 import xlsx

csv.field_size_limit(10 ** 9)
random.seed(0)
NONCOMP = C.NONCOMPETITIVE_METHODS
GT = os.path.join(C.GT, "favoritism_ca.csv")
STOP = {"of", "the", "and", "for", "canada", "canadian", "government", "department",
        "ministry", "agency", "office", "inc", "ltd", "corporation", "commission"}


def key(s):
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def tk(s):
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


def load():
    recs = pickle.load(open(os.path.join(C.WORK, "fav13_v2_scored.pkl"), "rb"))
    ents = pickle.load(open(os.path.join(C.WORK, "entities.pkl"), "rb"))
    return recs, ents["raw2ent"], ents.get("display", {})


def main():
    print("loading ...", flush=True)
    recs, raw2ent, display = load()

    lim = {rid: lr.strip() for rid, lr in
           rows("pro_award.csv", ["rid", "limited_reason"]) if lr.strip()}

    reg_day, reg_num = {}, {}
    for rid, d, s, rn in rows("lob_meta.csv", ["rid", "start_date_clean",
                                               "start_date_status", "rnum"]):
        if s == "valid":
            reg_day[rid] = EV.to_day(d)
            reg_num[rid] = rn
    rid_body = defaultdict(set)
    for rid, core in rows("lob_target.csv", ["rid", "name_core"]):
        if core and rid in reg_day:
            rid_body[rid].add(core)
    channel = defaultdict(list)          # (entity, dept) -> [(day, rid)]
    ent_regs = Counter()
    for rid, raw in rows("lob_org.csv", ["rid", "name"], base=C.RAW):
        e = raw2ent.get((raw or "").strip())
        if e is None or rid not in reg_day:
            continue
        ent_regs[e] += 1
        for b in rid_body.get(rid, ()):
            channel[(e, b)].append((reg_day[rid], rid))
    for k in channel:
        channel[k].sort()
    reference = sorted(reg_day.values())

    ent_awards = Counter(r["entity"] for r in recs)
    by_ent = defaultdict(list)
    for r in recs:
        by_ent[r["entity"]].append(r)
    dept_of = {}                          # buyer id -> name (first seen)
    for r in recs:
        for i, b in enumerate(r["buyers"][:1]):
            dept_of.setdefault(b, (r["buyer_names"] or "").split(";")[0].strip())

    exact = {}
    for raw, e in raw2ent.items():
        exact.setdefault(key(raw), e)

    # ============================================== PART 1: complete the motifs
    print("part 1: completing ground-truth motifs ...", flush=True)
    stories = list(csv.DictReader(open(GT, encoding="utf-8", errors="replace")))
    status, motifs = [], []
    seen_story = set()

    for g in stories:
        title = (g.get("title") or "").strip()
        comp = (g.get("private_org") or "").strip()
        dept = (g.get("public_org") or "").strip()
        yy = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b",
                                         g.get("contract_time") or "")]
        lo, hi = (min(yy), max(yy)) if yy else (None, None)
        e = exact.get(key(comp))
        dt = tk(dept)

        row = dict(story=title[:90], outlet=g.get("media_name", ""),
                   published=g.get("publish_date", ""),
                   story_company=comp, story_department=dept,
                   story_amount=(g.get("contract_amount") or "").strip(),
                   story_period=(g.get("contract_time") or "").strip(),
                   entity_id=e or "", resolved_company=display.get(e, "") if e else "",
                   n_contracts_company=ent_awards.get(e, 0) if e else 0,
                   n_registrations_company=ent_regs.get(e, 0) if e else 0)

        if not e:
            row.update(status="company not in corpus", n_motifs=0)
            status.append(row)
            continue
        cand = [r for r in by_ent[e] if dt and (dt & tk(r["buyer_names"]))]
        inwin = [r for r in cand
                 if lo is None or lo <= int(EV.from_day(r["day"])[:4]) <= hi]
        use = inwin or cand
        if not use:
            row.update(status="department not matched to any contract", n_motifs=0)
            status.append(row)
            continue

        row.update(status="complete", n_motifs=len(use))
        status.append(row)

        if title in seen_story:
            continue
        seen_story.add(title)

        for r in sorted(use, key=lambda r: r["rank"])[:60]:
            links = []
            for b in r["buyers"]:
                for d, rid in channel.get((e, b), ()):
                    if d < r["day"]:
                        links.append((r["day"] - d, rid, EV.from_day(d), b))
            links.sort()
            motifs.append(dict(
                motif_id=f"gt_{len(motifs)+1:04d}",
                story=title[:80], entity_id=e,
                company=display.get(e, e),
                department=(r["buyer_names"] or "").split(";")[0].strip()[:60],
                award_rid=r["award_rid"], award_date=EV.from_day(r["day"]),
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

    print(f"   stories: {len(stories)}   complete motifs: {len(motifs):,}")
    for k, v in Counter(s["status"] for s in status).most_common():
        print(f"     {k:<44}{v:>4}")

    # ============================================== PART 2: perturbations
    print("part 2: perturbing ...", flush=True)
    gt_ents = {m["entity_id"] for m in motifs}
    gt_awards = {m["award_rid"] for m in motifs}
    sheets = {}

    def mk(m, cat, **over):
        d = dict(m)
        d.update(label="negative", technique=cat, derived_from=m["motif_id"])
        d.update(over)
        d["motif_id"] = f"{cat}_{len(sheets.get(cat, []))+1:04d}"
        return d

    # ---- T1 shift lobbying: redraw registration dates from the national calendar
    t1 = []
    for m in motifs:
        n = max(int(m["n_regs_this_dept"] or 1), 1)
        drawn = sorted(random.choice(reference) for _ in range(min(n, 400)))
        aday = EV.to_day(m["award_date"])
        i = bisect.bisect_left(drawn, aday)
        g = (aday - drawn[i - 1]) if i > 0 else ""
        t1.append(mk(m, "T1_SHIFT_LOBBYING", lag_days=g,
                     nearest_reg_date=EV.from_day(drawn[i - 1]) if i > 0 else "",
                     nearest_reg_rid="(redrawn)", nearest_reg_num="",
                     what_changed="registration dates redrawn from the national "
                                  "lobbying calendar, count preserved"))
    sheets["T1_SHIFT_LOBBYING"] = t1

    # ---- T2 swap department: same company, a department it never lobbied
    t2 = []
    for m in motifs:
        e = m["entity_id"]
        lobbied = {b for (ee, b) in channel if ee == e}
        alt = [r for r in by_ent[e]
               if r["award_rid"] not in gt_awards and r["gap"] is None]
        if alt:
            r = random.choice(alt)
            t2.append(mk(m, "T2_SWAP_DEPARTMENT",
                         department=(r["buyer_names"] or "").split(";")[0].strip()[:60],
                         award_rid=r["award_rid"], award_date=EV.from_day(r["day"]),
                         method=r["method"] or "", lag_days="", n_links=0,
                         nearest_reg_rid="", nearest_reg_num="", nearest_reg_date="",
                         n_regs_this_dept=0, p_time=1.0,
                         limited_reason=lim.get(r["award_rid"], ""),
                         what_changed="same company, but a department it never "
                                      "lobbied before this award"))
    sheets["T2_SWAP_DEPARTMENT"] = t2

    # ---- T3 swap company: same department, no lobbying link
    dept_pool = defaultdict(list)
    for r in recs:
        if r["gap"] is None and r["buyers"]:
            dept_pool[r["buyers"][0]].append(r)
    t3 = []
    for m in motifs:
        tgt = tk(m["department"])
        cands = []
        for b, lst in dept_pool.items():
            if tgt & tk(dept_of.get(b, "")):
                cands = lst
                break
        cands = [r for r in cands if r["entity"] not in gt_ents]
        if cands:
            r = random.choice(cands)
            t3.append(mk(m, "T3_SWAP_COMPANY",
                         entity_id=r["entity"], company=display.get(r["entity"], r["entity"]),
                         award_rid=r["award_rid"], award_date=EV.from_day(r["day"]),
                         method=r["method"] or "", lag_days="", n_links=0,
                         nearest_reg_rid="", nearest_reg_num="", nearest_reg_date="",
                         n_regs_this_dept=0, p_time=1.0,
                         limited_reason=lim.get(r["award_rid"], ""),
                         company_total_awards=ent_awards[r["entity"]],
                         company_total_regs=ent_regs.get(r["entity"], 0),
                         what_changed="same department and period, but a company "
                                      "with no lobbying link to it"))
    sheets["T3_SWAP_COMPANY"] = t3

    # ---- T4 make competitive: same company and department, openly competed
    t4 = []
    for m in motifs:
        e = m["entity_id"]
        tgt = tk(m["department"])
        alt = [r for r in by_ent[e]
               if r["method"] == "OPEN" and tgt & tk(r["buyer_names"])]
        if alt:
            r = random.choice(alt)
            t4.append(mk(m, "T4_MAKE_COMPETITIVE",
                         award_rid=r["award_rid"], award_date=EV.from_day(r["day"]),
                         method="OPEN",
                         lag_days=(r["gap"] if r["gap"] is not None else ""),
                         limited_reason="", p_comp=1.0,
                         what_changed="same company and department, but this award "
                                      "was won through open competition"))
    sheets["T4_MAKE_COMPETITIVE"] = t4

    # ---- T5 distance the award: same pair, lobbying long before
    t5 = []
    for m in motifs:
        e = m["entity_id"]
        tgt = tk(m["department"])
        alt = [r for r in by_ent[e]
               if r["gap"] is not None and r["gap"] > 1095 and tgt & tk(r["buyer_names"])]
        if alt:
            r = random.choice(alt)
            t5.append(mk(m, "T5_DISTANT_TIMING",
                         award_rid=r["award_rid"], award_date=EV.from_day(r["day"]),
                         method=r["method"] or "", lag_days=r["gap"],
                         limited_reason=lim.get(r["award_rid"], ""),
                         what_changed=f"same company and department, but the nearest "
                                      f"lobbying was {r['gap']//365} years earlier"))
    sheets["T5_DISTANT_TIMING"] = t5

    # ---- T6 remove the mechanism: a company that never lobbies at all
    nolob = [r for r in recs if ent_regs.get(r["entity"], 0) == 0 and r["buyers"]]
    t6 = []
    for m in motifs:
        tgt = tk(m["department"])
        cands = [r for r in nolob if tgt & tk(r["buyer_names"])][:400]
        if cands:
            r = random.choice(cands)
            t6.append(mk(m, "T6_NO_LOBBYING_AT_ALL",
                         entity_id=r["entity"], company=display.get(r["entity"], r["entity"]),
                         award_rid=r["award_rid"], award_date=EV.from_day(r["day"]),
                         method=r["method"] or "", lag_days="", n_links=0,
                         nearest_reg_rid="", nearest_reg_num="", nearest_reg_date="",
                         n_regs_this_dept=0, p_time=1.0,
                         limited_reason=lim.get(r["award_rid"], ""),
                         company_total_awards=ent_awards[r["entity"]],
                         company_total_regs=0,
                         what_changed="same department, but a supplier that has never "
                                      "registered to lobby anyone"))
    sheets["T6_NO_LOBBYING_AT_ALL"] = t6

    for k, v in sheets.items():
        print(f"   {k:<26}{len(v):>6,}")

    # ============================================== write
    H, N, I, D = xlsx.S_HEAD, xlsx.S_NOTE, xlsx.S_INT, xlsx.S_DEC
    def tbl(recs_, cols):
        out = [cols]
        for r in recs_:
            out.append([r.get(c, "") for c in cols])
        return out

    stat_cols = ["story", "outlet", "published", "story_company", "story_department",
                 "story_amount", "story_period", "status", "n_motifs", "entity_id",
                 "resolved_company", "n_contracts_company", "n_registrations_company"]
    motif_cols = ["motif_id", "label", "story", "company", "entity_id", "department",
                  "award_rid", "award_date", "method", "limited_reason", "lag_days",
                  "n_links", "nearest_reg_rid", "nearest_reg_num", "nearest_reg_date",
                  "n_regs_this_dept", "p_time", "p_comp", "p_conc", "p_proc",
                  "tender_window", "company_total_awards", "company_total_regs", "source"]
    neg_cols = ["motif_id", "label", "technique", "derived_from", "what_changed",
                "company", "entity_id", "department", "award_rid", "award_date",
                "method", "limited_reason", "lag_days", "n_links", "nearest_reg_date",
                "n_regs_this_dept", "p_time", "p_comp", "p_conc",
                "company_total_awards", "company_total_regs"]

    book = [("GT status", tbl(status, stat_cols),
             [46, 16, 11, 22, 26, 13, 12, 30, 9, 12, 26, 13, 13]),
            ("GT motifs (positive)", tbl(motifs, motif_cols),
             [10, 8, 40, 26, 10, 30, 20, 11, 10, 30, 9, 8, 20, 16, 14, 10,
              10, 10, 12, 10, 11, 13, 13, 16])]
    for k in ["T1_SHIFT_LOBBYING", "T2_SWAP_DEPARTMENT", "T3_SWAP_COMPANY",
              "T4_MAKE_COMPETITIVE", "T5_DISTANT_TIMING", "T6_NO_LOBBYING_AT_ALL"]:
        book.append((k[:31], tbl(sheets[k], neg_cols),
                     [18, 8, 22, 10, 52, 26, 10, 30, 20, 11, 10, 26, 9, 8, 14, 10,
                      10, 10, 12, 13, 13]))

    out = os.path.join(os.path.dirname(__file__), "gt_motifs_and_negatives.xlsx")
    xlsx.write(out, book)
    print(f"\nwrote {out}")

    csvp = os.path.join(C.WORK, "gt_motifs_positive.csv")
    with open(csvp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=motif_cols)
        w.writeheader()
        for m in motifs:
            w.writerow({c: m.get(c, "") for c in motif_cols})
    print(f"wrote {csvp}")


if __name__ == "__main__":
    main()
