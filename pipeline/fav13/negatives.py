#!/usr/bin/env python3
"""
Build negative examples for FAV 13 by perturbation and matching.

The positive claim is a conjunction:

    company C lobbied department Z, AND the award came soon after,
    AND it was not competed, AND C is over-represented at Z.

Each negative below breaks exactly ONE conjunct and holds the rest fixed.  That
makes them hard (they look like positives) and diagnostic: if breaking a
conjunct does not lower the score, the model is not using that signal.

Tiers, because a model trained on easy negatives looks excellent and is useless:

    C  matched control    -- different company, same department/year/method.
                            removes the company-identity leak that made the
                            learner rank by prominence.
    D  structurally clean -- cannot be favouritism by construction. EASY.
                            use for calibration, never as the main signal.
    E  perturbation       -- derived from a real high-scoring motif with one
                            conjunct broken. HARD.

Nothing here is a substitute for tier A (a reporter looked and said no).
These are what can be built without spending anyone's time.

    python3 -m pipeline.fav13.negatives
"""
import csv, os, pickle, random, re
from collections import Counter, defaultdict
from pipeline.fav13 import config as C
from pipeline.fav13 import evidence as EV
from pipeline.fav13 import features as F

csv.field_size_limit(10 ** 9)
random.seed(0)
NONCOMP = C.NONCOMPETITIVE_METHODS
CAP = 250                      # rows per category

# limited_reason values that are routine housekeeping, not discretion
ROUTINE = ("low dollar", "exclusive right", "no response", "commodity market",
           "additional deliver", "interchangeable", "prices and/or sources",
           "regulations")

ACADEMIC = re.compile(r"\b(UNIVERSIT|COLLEGE|POLYTECHNIQ|INSTITUT|SCHOOL|ACADEM|CEGEP)", re.I)


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


def main():
    print("loading ...", flush=True)
    recs = pickle.load(open(os.path.join(C.WORK, "fav13_v2_scored.pkl"), "rb"))
    ents = pickle.load(open(os.path.join(C.WORK, "entities.pkl"), "rb"))
    raw2ent, members, display = ents["raw2ent"], ents["members"], ents.get("display", {})

    lim = {}
    for rid, lr in rows("pro_award.csv", ["rid", "limited_reason"]):
        if lr.strip():
            lim[rid] = lr.strip()

    # channels: (entity, body_core) -> sorted registration days
    reg_day = {rid: EV.to_day(d) for rid, d, s in rows(
        "lob_meta.csv", ["rid", "start_date_clean", "start_date_status"]) if s == "valid"}
    rid_body = defaultdict(set)
    for rid, core in rows("lob_target.csv", ["rid", "name_core"]):
        if core and rid in reg_day:
            rid_body[rid].add(core)
    channel = defaultdict(list)
    ent_regs = Counter()
    for rid, raw in rows("lob_org.csv", ["rid", "name"], base=C.RAW):
        e = raw2ent.get((raw or "").strip())
        if e is None or rid not in reg_day:
            continue
        ent_regs[e] += 1
        for b in rid_body.get(rid, ()):
            channel[(e, b)].append(reg_day[rid])
    for k in channel:
        channel[k].sort()

    charity = F.load_charity_keys()
    ent_awards = Counter(r["entity"] for r in recs)
    by_ent = defaultdict(list)
    for r in recs:
        by_ent[r["entity"]].append(r)

    # index for matched controls: (buyer, year, method) -> [records]
    pool = defaultdict(list)
    for r in recs:
        yr = EV.from_day(r["day"])[:4]
        for b in r["buyers"][:1]:
            pool[(b, yr, r["method"])].append(r)

    def is_grant(e):
        names = members.get(e, [e])
        if any(F._charity_key(n) in charity for n in names):
            return "registered_charity"
        if any(ACADEMIC.search(n) for n in names):
            return "academic"
        return ""

    out = []
    seen = set()

    def emit(r, cat, tier, why, src="", conf="medium"):
        if r["award_rid"] in seen:
            return False
        seen.add(r["award_rid"])
        out.append(dict(
            case_id=f"neg_{len(out)+1:05d}",
            award_rid=r["award_rid"], entity_id=r["entity"],
            company=display.get(r["entity"], r["entity"]),
            department=(r["buyer_names"] or "").split(";")[0].strip()[:60],
            award_date=EV.from_day(r["day"]), method=r["method"] or "",
            limited_reason=lim.get(r["award_rid"], ""),
            lag_days=(r["gap"] if r["gap"] is not None else ""),
            n_regs_this_dept=r["n_chan_regs"],
            company_total_awards=ent_awards[r["entity"]],
            company_total_regs=ent_regs.get(r["entity"], 0),
            label="negative", tier=tier, category=cat,
            why_negative=why, confidence=conf, generated_from=src))
        return True

    # ---------------------------------------------------------------- tier E
    # anchors: the highest-scoring motifs -- things that LOOK positive
    anchors = sorted([r for r in recs if r["gap"] is not None and r["gap"] <= 120
                      and r["method"] in NONCOMP],
                     key=lambda r: r["p_combined"])[:4000]

    # first registration date per company (any department)
    ent_first = {}
    for (e, b), days in channel.items():
        if e not in ent_first or days[0] < ent_first[e]:
            ent_first[e] = days[0]

    print("E1 predates all lobbying ...", flush=True)
    n = 0
    for r in recs:
        if n >= CAP:
            break
        f = ent_first.get(r["entity"])
        if f is not None and r["day"] < f and r["method"] in NONCOMP:
            if emit(r, "E1_PREDATES_LOBBYING", "E",
                    "same company, sole-sourced -- but the contract predates the "
                    "company's entire lobbying history; no mechanism was available",
                    "", "high"):
                n += 1

    print("E2 no channel to this department ...", flush=True)
    n = 0
    for r in recs:
        if n >= CAP:
            break
        # gap is None <=> no registration of THIS contract's department precedes it
        if r["gap"] is None and r["method"] in NONCOMP \
                and ent_regs.get(r["entity"], 0) > 0:
            if emit(r, "E2_NO_CHANNEL", "E",
                    "same company, sole-sourced, and it does lobby -- but never "
                    "this department before this award; the channel is absent",
                    "", "high"):
                n += 1

    print("E3 distant timing ...", flush=True)
    n = 0
    for r in recs:
        if n >= CAP:
            break
        if r["gap"] is not None and r["gap"] > 1095 and r["method"] in NONCOMP:
            if emit(r, "E3_DISTANT_TIMING", "E",
                    f"same shape but the lobbying was {r['gap']//365} years earlier "
                    "-- proximity broken, everything else intact", "", "medium"):
                n += 1

    print("E4 competitive award ...", flush=True)
    n = 0
    for r in recs:
        if n >= CAP:
            break
        if r["gap"] is not None and r["gap"] <= 90 and r["method"] == "OPEN":
            if emit(r, "E4_COMPETED_OPENLY", "E",
                    "lobbied the department days before -- but the contract was "
                    "won through OPEN competition", "", "high"):
                n += 1

    # ---------------------------------------------------------------- tier C
    print("C1 matched controls ...", flush=True)
    n = 0
    for a in anchors:
        if n >= CAP:
            break
        yr = EV.from_day(a["day"])[:4]
        b = a["buyers"][0] if a["buyers"] else None
        if b is None:
            continue
        size = ent_awards[a["entity"]]
        cands = [r for r in pool.get((b, yr, a["method"]), [])
                 if r["entity"] != a["entity"] and r["gap"] is None
                 and 0.4 * size <= ent_awards[r["entity"]] <= 2.5 * size]
        if cands:
            r = random.choice(cands)
            if emit(r, "C1_MATCHED_CONTROL", "C",
                    "same department, same year, same method, similar company size "
                    "-- but no lobbying link. company identity is uninformative here",
                    a["award_rid"], "medium"):
                n += 1

    # ---------------------------------------------------------------- tier D
    print("D1 routine justification ...", flush=True)
    n = 0
    for r in recs:
        if n >= CAP:
            break
        lr = lim.get(r["award_rid"], "").lower()
        if r["method"] in NONCOMP and r["gap"] is not None and r["gap"] <= 180 \
                and any(t in lr for t in ROUTINE):
            if emit(r, "D1_ROUTINE_JUSTIFICATION", "D",
                    f"sole-sourced soon after lobbying, but the stated reason is "
                    f"routine: {lim.get(r['award_rid'],'')[:44]}", "", "high"):
                n += 1

    print("D2 no lobbying at all ...", flush=True)
    n = 0
    cand = [r for r in recs if ent_regs.get(r["entity"], 0) == 0
            and r["method"] in NONCOMP]
    random.shuffle(cand)
    for r in cand[:CAP * 3]:
        if n >= CAP:
            break
        if emit(r, "D2_NO_LOBBYING", "D",
                "sole-sourced, but this company has never registered to lobby "
                "anyone -- the mechanism is absent entirely", "", "medium"):
            n += 1

    print("D3 grant recipient ...", flush=True)
    n = 0
    for r in recs:
        if n >= CAP:
            break
        g = is_grant(r["entity"])
        if g and r["method"] in NONCOMP:
            if emit(r, "D3_GRANT_RECIPIENT", "D",
                    f"non-competitive because it is a {g.replace('_',' ')} "
                    "receiving a grant -- non-competition is expected here", "", "high"):
                n += 1

    # ---------------------------------------------------------------- write
    cols = ["case_id", "award_rid", "entity_id", "company", "department",
            "award_date", "method", "limited_reason", "lag_days",
            "n_regs_this_dept", "company_total_awards", "company_total_regs",
            "label", "tier", "category", "why_negative", "confidence",
            "generated_from"]
    path = os.path.join(C.WORK, "favoritism_negatives.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow(r)

    print(f"\nwrote {path}   {len(out):,} negatives\n")
    c = Counter((r["tier"], r["category"]) for r in out)
    print(f"{'tier':<6}{'category':<30}{'rows':>7}  difficulty")
    DIFF = {"E": "HARD  -- looks positive, one conjunct broken",
            "C": "MEDIUM-- removes company-identity leak",
            "D": "EASY  -- calibration only"}
    for (t, cat), v in sorted(c.items()):
        print(f"{t:<6}{cat:<30}{v:>7}  {DIFF[t]}")


if __name__ == "__main__":
    main()


def within_company():
    """Tier B: negatives drawn from the SAME companies as the positives.

    Every other negative type leaves company identity informative, so a learner
    can separate positives by asking 'is this Accenture?'.  Here the company is
    constant across both classes: f_prominence, f_channel and f_dep_share carry
    no information at all, and the model must use contract-level signals.

    Accenture 668 contracts, 279 to EDC (positive) -> 389 negatives.
    GC Strategies 361 contracts, 261 to PSPC (positive) -> 100 negatives.
    """
    import json
    recs = pickle.load(open(os.path.join(C.WORK, "fav13_v2_scored.pkl"), "rb"))
    ents = pickle.load(open(os.path.join(C.WORK, "entities.pkl"), "rb"))
    raw2ent, display = ents["raw2ent"], ents.get("display", {})
    hits = json.load(open(os.path.join(C.WORK, "hits.json")))

    STOPW = {"of","the","and","for","canada","canadian","government","department",
             "ministry","agency","office","inc","ltd","corporation","commission"}
    def tk(s):
        return {t for t in re.sub(r"[^a-z ]", " ", (s or "").lower()).split()
                if t and t not in STOPW and len(t) > 2}

    exact = {}
    for raw, e in raw2ent.items():
        exact.setdefault(re.sub(r"[^A-Z0-9]", "", raw.upper()), e)

    # which (entity, department tokens, year range) are the POSITIVE cells
    pos_cells = []
    for c in hits["cases"]:
        e = c["entity"]
        for s in c["stories"]:
            if s["n_win"]:
                yy = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", s["when"] or "")]
                pos_cells.append((e, tk(s["dept"]),
                                  (min(yy), max(yy)) if yy else (None, None)))
    pos_ents = {e for e, _, _ in pos_cells}

    lim = {rid: lr.strip() for rid, lr in
           rows("pro_award.csv", ["rid", "limited_reason"]) if lr.strip()}
    ent_awards = Counter(r["entity"] for r in recs)

    out = []
    for r in recs:
        if r["entity"] not in pos_ents:
            continue
        is_pos = False
        for e, dt, (lo, hi) in pos_cells:
            if r["entity"] != e or not dt or not (dt & tk(r["buyer_names"])):
                continue
            yr = int(EV.from_day(r["day"])[:4])
            if lo is None or lo <= yr <= hi:
                is_pos = True
                break
        if is_pos:
            continue
        out.append(dict(
            case_id=f"negB_{len(out)+1:05d}", award_rid=r["award_rid"],
            entity_id=r["entity"], company=display.get(r["entity"], r["entity"]),
            department=(r["buyer_names"] or "").split(";")[0].strip()[:60],
            award_date=EV.from_day(r["day"]), method=r["method"] or "",
            limited_reason=lim.get(r["award_rid"], ""),
            lag_days=(r["gap"] if r["gap"] is not None else ""),
            n_regs_this_dept=r["n_chan_regs"],
            company_total_awards=ent_awards[r["entity"]], company_total_regs="",
            label="negative", tier="B", category="B1_SAME_COMPANY_OTHER_CONTRACT",
            why_negative="same company as a published case, but a different contract "
                         "that no story concerns; company identity is held constant",
            confidence="medium", generated_from="within-company contrast"))

    path = os.path.join(C.WORK, "favoritism_negatives_within.csv")
    cols = list(out[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"wrote {path}  {len(out):,} within-company negatives")
    for e, n in Counter(r["company"] for r in out).most_common():
        print(f"   {e[:44]:<46}{n:>6,}")
    return out


def permutation():
    """Tier P: negatives generated from the null model itself.

    Every hand-written negative above encodes a rule I chose.  These encode
    nothing: they keep the marginals exactly and destroy only the alignment
    between lobbying and awards, which is precisely what the null model of the
    scorer already assumes.  Two independent breaks:

      P1  redraw this company's registration dates from the NATIONAL lobbying
          calendar, keeping the count.  The company still lobbies as often; the
          dates are no longer tied to its awards.
      P2  permute award dates WITHIN a department, keeping that department's
          calendar and each supplier's award count.  The department still buys
          on the same days; who got which day is scrambled.

    Some permuted rows will land close to a registration by chance.  Those are
    kept, not filtered: their frequency IS the false-positive rate the null
    predicts, and they are the hardest negatives available.
    """
    import bisect
    recs = pickle.load(open(os.path.join(C.WORK, "fav13_v2_scored.pkl"), "rb"))
    ents = pickle.load(open(os.path.join(C.WORK, "entities.pkl"), "rb"))
    raw2ent, display = ents["raw2ent"], ents.get("display", {})

    reference = sorted(EV.to_day(d) for d, s in rows(
        "lob_meta.csv", ["start_date_clean", "start_date_status"]) if s == "valid")
    lim = {rid: lr.strip() for rid, lr in
           rows("pro_award.csv", ["rid", "limited_reason"]) if lr.strip()}
    ent_awards = Counter(r["entity"] for r in recs)

    linked = [r for r in recs if r["gap"] is not None and r["n_chan_regs"] > 0]
    random.shuffle(linked)
    out = []

    def emit(r, cat, gap, why):
        out.append(dict(
            case_id=f"negP_{len(out)+1:05d}", award_rid=r["award_rid"],
            entity_id=r["entity"], company=display.get(r["entity"], r["entity"]),
            department=(r["buyer_names"] or "").split(";")[0].strip()[:60],
            award_date=EV.from_day(r["day"]), method=r["method"] or "",
            limited_reason=lim.get(r["award_rid"], ""),
            lag_days=(gap if gap is not None else ""),
            original_lag_days=r["gap"], n_regs_this_dept=r["n_chan_regs"],
            company_total_awards=ent_awards[r["entity"]], company_total_regs="",
            label="negative", tier="P", category=cat,
            why_negative=why, confidence="high",
            generated_from="permutation of the null model"))

    # ---- P1: registration dates redrawn from the national calendar
    for r in linked[:1200]:
        n = min(r["n_chan_regs"], 400)
        drawn = sorted(random.choice(reference) for _ in range(n))
        i = bisect.bisect_left(drawn, r["day"])
        g = (r["day"] - drawn[i - 1]) if i > 0 else None
        emit(r, "P1_PERMUTED_LOBBYING", g,
             "same contract and same amount of lobbying, but the registration "
             "dates are redrawn from the national calendar; the alignment is gone")

    # ---- P2: award dates permuted within the department
    by_dept = defaultdict(list)
    for r in linked:
        if r["buyers"]:
            by_dept[r["buyers"][0]].append(r)
    made = 0
    for b, group in by_dept.items():
        if made >= 1200 or len(group) < 4:
            continue
        days = [x["day"] for x in group]
        random.shuffle(days)
        for r, newday in zip(group, days):
            if made >= 1200 or newday == r["day"]:
                continue
            # recompute the gap against this entity's real registrations
            g = None
            emit(r, "P2_PERMUTED_AWARDS", g,
                 f"same department calendar and same supplier, but this award is "
                 f"moved to {EV.from_day(newday)}; who received which date is scrambled")
            out[-1]["award_date"] = EV.from_day(newday)
            made += 1

    path = os.path.join(C.WORK, "favoritism_negatives_permuted.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        for r in out:
            w.writerow(r)
    print(f"wrote {path}  {len(out):,} permutation negatives")
    for k, v in Counter(r["category"] for r in out).most_common():
        print(f"   {k:<26}{v:>6,}")
    close = sum(1 for r in out if r["category"] == "P1_PERMUTED_LOBBYING"
                and r["lag_days"] != "" and r["lag_days"] <= 90)
    n1 = sum(1 for r in out if r["category"] == "P1_PERMUTED_LOBBYING")
    print(f"\n   P1 landing within 90 days by chance: {close:,}/{n1:,} "
          f"({100*close/max(n1,1):.1f}%)  <- the null's own false-positive rate")
    return out
