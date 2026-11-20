"""Constructed negatives for FAV 13.

The positive claim is a conjunction: C lobbied Z, AND the award came soon
after, AND it was not competed, AND C is over-represented at Z. Each negative
breaks one conjunct. Tiers, hardest first:

  E  perturbation      real high-scoring shape, one conjunct broken  (hard)
  C  matched control   other company, same dept/year/method/size
  B  within-company    same company as a story, a contract no story covers
  P  permutation       the scorer's own null: marginals kept, alignment destroyed
  D  structurally clean  cannot be favouritism; calibration only    (easy)

None replaces tier A: a reporter looked and said no.

    python3 -m pipeline.fav13.evaluate.negatives [rules|within|permuted|all]
        rules    -> work/favoritism_negatives.csv           (E, C, D; default)
        within   -> work/favoritism_negatives_within.csv    (B; needs hits.json)
        permuted -> work/favoritism_negatives_permuted.csv  (P)
"""

import bisect
import json
import random
import re
import sys
from collections import Counter, defaultdict

from pipeline import paths
from .. import common
from ..common import NONCOMP, dept_match, first_buyer, from_day, in_period
from ..data.evidence import lobbying_start_dates
from ..score.features import charity_key, load_charity_keys

CAP = 250                      # rows per category

# limited_reason values that are routine housekeeping, not discretion
ROUTINE = ("low dollar", "exclusive right", "no response", "commodity market",
           "additional deliver", "interchangeable", "prices and/or sources",
           "regulations")

# Narrower than features.ACADEMIC (no SEMINAR); kept as generated.
ACADEMIC = re.compile(r"\b(UNIVERSIT|COLLEGE|POLYTECHNIQ|INSTITUT|SCHOOL|ACADEM|CEGEP)", re.I)


def _load():
    ents = common.load_entities()
    recs = common.load_scored()
    return recs, ents, ents.get("display", {}), Counter(r["entity"] for r in recs)


def _row(r, display, ent_awards, lim, **extra):
    """Common columns for one negative contract."""
    return dict(award_rid=r["award_rid"], entity_id=r["entity"],
                company=display.get(r["entity"], r["entity"]),
                department=first_buyer(r)[:60],
                award_date=from_day(r["day"]), method=r["method"] or "",
                limited_reason=lim.get(r["award_rid"], ""),
                lag_days=(r["gap"] if r["gap"] is not None else ""),
                **extra)


def _write(name, out, cols=None):
    return common.write_csv(paths.WORK / name, out, cols or list(out[0].keys()))


# ---------------------------------------------------------------- tiers E, C, D

def rules():
    random.seed(0)
    print("loading ...", flush=True)
    recs, ents, display, ent_awards = _load()
    members = ents["members"]
    lim = common.limited_reasons()
    charity = load_charity_keys()
    channel, ent_regs = common.lobby_channels(ents["raw2ent"], common.registrations())

    # first registration per company (any department)
    ent_first = {}
    for (e, _b), regs in channel.items():
        if e not in ent_first or regs[0][0] < ent_first[e]:
            ent_first[e] = regs[0][0]

    # matched-control pool: (buyer, year, method) -> contracts
    pool = defaultdict(list)
    for r in recs:
        for b in r["buyers"][:1]:
            pool[(b, from_day(r["day"])[:4], r["method"])].append(r)

    def grant_reason(e):
        names = members.get(e, [e])
        if any(charity_key(n) in charity for n in names):
            return "registered_charity"
        if any(ACADEMIC.search(n) for n in names):
            return "academic"
        return ""

    out, seen = [], set()

    def emit(r, cat, tier, why, src="", conf="medium"):
        if r["award_rid"] in seen:
            return False
        seen.add(r["award_rid"])
        out.append(dict(case_id=f"neg_{len(out)+1:05d}",
                        **_row(r, display, ent_awards, lim,
                               n_regs_this_dept=r["n_chan_regs"],
                               company_total_awards=ent_awards[r["entity"]],
                               company_total_regs=ent_regs.get(r["entity"], 0)),
                        label="negative", tier=tier, category=cat,
                        why_negative=why, confidence=conf, generated_from=src))
        return True

    def take(candidates, cat, tier, why, conf):
        """Emit up to CAP rows from `candidates` (why may be a function of r)."""
        n = 0
        for r in candidates:
            if n >= CAP:
                break
            if emit(r, cat, tier, why(r) if callable(why) else why, "", conf):
                n += 1

    print("E1 predates all lobbying ...", flush=True)
    take((r for r in recs if r["method"] in NONCOMP
          and ent_first.get(r["entity"]) is not None and r["day"] < ent_first[r["entity"]]),
         "E1_PREDATES_LOBBYING", "E",
         "same company, sole-sourced -- but the contract predates the "
         "company's entire lobbying history; no mechanism was available", "high")

    print("E2 no channel to this department ...", flush=True)
    take((r for r in recs if r["gap"] is None and r["method"] in NONCOMP
          and ent_regs.get(r["entity"], 0) > 0),
         "E2_NO_CHANNEL", "E",
         "same company, sole-sourced, and it does lobby -- but never "
         "this department before this award; the channel is absent", "high")

    print("E3 distant timing ...", flush=True)
    take((r for r in recs if r["gap"] is not None and r["gap"] > 1095
          and r["method"] in NONCOMP),
         "E3_DISTANT_TIMING", "E",
         lambda r: f"same shape but the lobbying was {r['gap']//365} years earlier "
                   "-- proximity broken, everything else intact", "medium")

    print("E4 competitive award ...", flush=True)
    take((r for r in recs if r["gap"] is not None and r["gap"] <= 90
          and r["method"] == "OPEN"),
         "E4_COMPETED_OPENLY", "E",
         "lobbied the department days before -- but the contract was "
         "won through OPEN competition", "high")

    print("C1 matched controls ...", flush=True)
    anchors = sorted([r for r in recs if r["gap"] is not None and r["gap"] <= 120
                      and r["method"] in NONCOMP], key=lambda r: r["p_combined"])[:4000]
    n = 0
    for a in anchors:
        if n >= CAP:
            break
        if not a["buyers"]:
            continue
        size = ent_awards[a["entity"]]
        cands = [r for r in pool.get((a["buyers"][0], from_day(a["day"])[:4], a["method"]), [])
                 if r["entity"] != a["entity"] and r["gap"] is None
                 and 0.4 * size <= ent_awards[r["entity"]] <= 2.5 * size]
        if cands and emit(random.choice(cands), "C1_MATCHED_CONTROL", "C",
                          "same department, same year, same method, similar company size "
                          "-- but no lobbying link. company identity is uninformative here",
                          a["award_rid"], "medium"):
            n += 1

    print("D1 routine justification ...", flush=True)
    take((r for r in recs if r["method"] in NONCOMP and r["gap"] is not None
          and r["gap"] <= 180
          and any(t in lim.get(r["award_rid"], "").lower() for t in ROUTINE)),
         "D1_ROUTINE_JUSTIFICATION", "D",
         lambda r: f"sole-sourced soon after lobbying, but the stated reason is "
                   f"routine: {lim.get(r['award_rid'],'')[:44]}", "high")

    print("D2 no lobbying at all ...", flush=True)
    cand = [r for r in recs if ent_regs.get(r["entity"], 0) == 0 and r["method"] in NONCOMP]
    random.shuffle(cand)
    take(cand[:CAP * 3], "D2_NO_LOBBYING", "D",
         "sole-sourced, but this company has never registered to lobby "
         "anyone -- the mechanism is absent entirely", "medium")

    print("D3 grant recipient ...", flush=True)
    take((r for r in recs if r["method"] in NONCOMP and grant_reason(r["entity"])),
         "D3_GRANT_RECIPIENT", "D",
         lambda r: f"non-competitive because it is a "
                   f"{grant_reason(r['entity']).replace('_',' ')} "
                   "receiving a grant -- non-competition is expected here", "high")

    cols = ["case_id", "award_rid", "entity_id", "company", "department",
            "award_date", "method", "limited_reason", "lag_days",
            "n_regs_this_dept", "company_total_awards", "company_total_regs",
            "label", "tier", "category", "why_negative", "confidence",
            "generated_from"]
    path = _write("favoritism_negatives.csv", out, cols)
    print(f"\nwrote {path}   {len(out):,} negatives\n")
    difficulty = {"E": "HARD  -- looks positive, one conjunct broken",
                  "C": "MEDIUM-- removes company-identity leak",
                  "D": "EASY  -- calibration only"}
    print(f"{'tier':<6}{'category':<30}{'rows':>7}  difficulty")
    for (t, cat), v in sorted(Counter((r["tier"], r["category"]) for r in out).items()):
        print(f"{t:<6}{cat:<30}{v:>7}  {difficulty[t]}")
    return out


# ---------------------------------------------------------------- tier B

def within_company():
    """Same companies as the positives, other contracts. Company identity is
    constant across classes, so only contract-level signals can separate them."""
    recs, ents, display, ent_awards = _load()
    with open(paths.WORK / "hits.json") as fh:
        hits = json.load(fh)
    lim = common.limited_reasons()

    # positive cells: (entity, dept tokens, year span) of stories with matches
    cells = [(c["entity"], common.dept_tokens(s["dept"]), common.year_span(s["when"]))
             for c in hits["cases"] for s in c["stories"] if s["n_win"]]
    pos_ents = {e for e, _, _ in cells}

    out = []
    for r in recs:
        if r["entity"] not in pos_ents:
            continue
        if any(r["entity"] == e and dept_match(r, dt) and in_period(r, lo, hi)
               for e, dt, (lo, hi) in cells):
            continue
        out.append(dict(
            case_id=f"negB_{len(out)+1:05d}",
            **_row(r, display, ent_awards, lim, n_regs_this_dept=r["n_chan_regs"],
                   company_total_awards=ent_awards[r["entity"]], company_total_regs=""),
            label="negative", tier="B", category="B1_SAME_COMPANY_OTHER_CONTRACT",
            why_negative="same company as a published case, but a different contract "
                         "that no story concerns; company identity is held constant",
            confidence="medium", generated_from="within-company contrast"))

    print(f"wrote {_write('favoritism_negatives_within.csv', out)}  "
          f"{len(out):,} within-company negatives")
    for e, n in Counter(r["company"] for r in out).most_common():
        print(f"   {e[:44]:<46}{n:>6,}")
    return out


# ---------------------------------------------------------------- tier P

def permutation():
    """Negatives from the null itself (no hand-chosen rule):
      P1  redraw this company's channel registration dates from the national calendar
      P2  permute award dates within a department
    Rows that land close by chance are kept: their rate IS the null's false-positive rate."""
    random.seed(0)
    recs, _, display, ent_awards = _load()
    reference = lobbying_start_dates()
    lim = common.limited_reasons()

    linked = [r for r in recs if r["gap"] is not None and r["n_chan_regs"] > 0]
    random.shuffle(linked)
    out = []

    def emit(r, cat, gap, why):
        row = _row(r, display, ent_awards, lim)
        row["lag_days"] = gap if gap is not None else ""
        out.append(dict(case_id=f"negP_{len(out)+1:05d}", **row,
                        original_lag_days=r["gap"], n_regs_this_dept=r["n_chan_regs"],
                        company_total_awards=ent_awards[r["entity"]], company_total_regs="",
                        label="negative", tier="P", category=cat,
                        why_negative=why, confidence="high",
                        generated_from="permutation of the null model"))

    for r in linked[:1200]:
        drawn = sorted(random.choice(reference) for _ in range(min(r["n_chan_regs"], 400)))
        i = bisect.bisect_left(drawn, r["day"])
        emit(r, "P1_PERMUTED_LOBBYING", (r["day"] - drawn[i - 1]) if i > 0 else None,
             "same contract and same amount of lobbying, but the registration "
             "dates are redrawn from the national calendar; the alignment is gone")

    by_dept = defaultdict(list)
    for r in linked:
        if r["buyers"]:
            by_dept[r["buyers"][0]].append(r)
    made = 0
    for group in by_dept.values():
        if made >= 1200 or len(group) < 4:
            continue
        days = [x["day"] for x in group]
        random.shuffle(days)
        for r, newday in zip(group, days):
            if made >= 1200 or newday == r["day"]:
                continue
            # gap left blank: not recomputed against the moved date
            emit(r, "P2_PERMUTED_AWARDS", None,
                 f"same department calendar and same supplier, but this award is "
                 f"moved to {from_day(newday)}; who received which date is scrambled")
            out[-1]["award_date"] = from_day(newday)
            made += 1

    print(f"wrote {_write('favoritism_negatives_permuted.csv', out)}  "
          f"{len(out):,} permutation negatives")
    for k, v in Counter(r["category"] for r in out).most_common():
        print(f"   {k:<26}{v:>6,}")
    p1 = [r for r in out if r["category"] == "P1_PERMUTED_LOBBYING"]
    close = sum(1 for r in p1 if r["lag_days"] != "" and r["lag_days"] <= 90)
    print(f"\n   P1 landing within 90 days by chance: {close:,}/{len(p1):,} "
          f"({100*close/max(len(p1),1):.1f}%)  <- the null's own false-positive rate")
    return out


def main(argv=None):
    which = (argv if argv is not None else sys.argv[1:]) or ["rules"]
    runs = {"rules": rules, "within": within_company, "permuted": permutation}
    for name in (list(runs) if which == ["all"] else which):
        runs[name]()


if __name__ == "__main__":
    main()
