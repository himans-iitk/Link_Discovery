"""Government-body map: lob_target (bodies lobbied) <-> pro_contact (buyers).

Bodies are a closed vocabulary (684 names cover 80% of lobbying records), so
this is curation, not entity resolution. Rules, most confident first:

  exact    identical name_core
  rename   curated machinery-of-government rename (PWGSC -> PSPC, ...)
  subset   same discriminating tokens, different boilerplate
  jaccard  token overlap >= threshold -- PROPOSED only, needs review

Only approved rows (exact/rename/subset, or reviewed jaccard) are used.
The `approved` column in reference/body_map.csv is hand-curated.

    python3 -m pipeline.fav13.data.body_map      # rebuild + report
"""

import csv
import os
import re
from collections import Counter

from pipeline import paths

csv.field_size_limit(10 ** 9)

# (old or alternate name_core, current name_core). Each is a public fact.
RENAMES = [
    ("public works government services canada", "public services procurement canada"),
    ("public works government services",         "public services procurement canada"),
    ("human resources skills development canada", "employment social development canada"),
    ("foreign affairs trade development canada",  "global affairs canada"),
    ("foreign affairs international trade canada", "global affairs canada"),
    ("industry canada",                           "innovation science economic development canada"),
    ("aboriginal affairs northern development canada",
                                                  "crown indigenous relations northern affairs canada"),
    ("indigenous northern affairs canada",        "crown indigenous relations northern affairs canada"),
    ("citizenship immigration canada",            "immigration refugees citizenship canada"),
    ("finance canada",                            "department finance canada"),
    ("national revenue",                          "canada revenue agency"),
    ("treasury board canada secretariat",         "treasury board secretariat"),
    ("environment canada",                        "environment climate change canada"),
    ("infrastructure communities canada",         "infrastructure canada"),
    ("western economic diversification canada",   "prairies economic development canada"),
]

# Tokens that name no particular body.
_NOISE = {"department", "ministry", "ministere", "office", "bureau", "agency",
          "agence", "canada", "canadian", "government", "gouvernement",
          "secretariat", "service", "services", "national", "federal"}


def _toks(s):
    return [t for t in re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split() if t]


def _core(s):
    return frozenset(t for t in _toks(s) if t not in _NOISE)


def counts(fname, col):
    """Value -> row count for one column of a clean table."""
    c = Counter()
    with open(os.path.join(paths.CLEAN, fname), newline="", encoding="utf-8",
              errors="replace") as fh:
        r = csv.reader(fh)
        i = next(r).index(col)
        for x in r:
            if len(x) > i and x[i]:
                c[x[i]] += 1
    return c


def build(top_n=700, jaccard_min=0.60):
    """-> (rows, stats); one proposed (lob_target, pro_contact) pair per row."""
    lob = counts("lob_target.csv", "name_core")
    pro = counts("pro_contact.csv", "name_core")
    heads = [n for n, _ in lob.most_common(top_n)]     # the tail is noise

    pro_by_core = {}
    for p in pro:
        pro_by_core.setdefault(_core(p), []).append(p)

    ren = dict(RENAMES)
    rows, seen = [], set()

    def emit(l, p, rule):
        if (l, p) in seen:
            return
        seen.add((l, p))
        rows.append(dict(lob_target=l, pro_contact=p, rule=rule,
                         lob_records=lob.get(l, 0), pro_records=pro.get(p, 0),
                         approved="1" if rule != "jaccard" else ""))

    for l in heads:
        if l in pro:
            emit(l, l, "exact")
            continue
        if l in ren and ren[l] in pro:
            emit(l, ren[l], "rename")
            continue
        cl = _core(l)
        if not cl:
            continue
        if cl in pro_by_core:
            for p in pro_by_core[cl]:
                emit(l, p, "subset")
            continue
        best, bestj = None, 0.0
        for cp, names in pro_by_core.items():
            if not cp:
                continue
            j = len(cl & cp) / len(cl | cp)
            if j > bestj:
                best, bestj = names, j
        if best and bestj >= jaccard_min:
            for p in best:
                emit(l, p, "jaccard")

    approved = {r["lob_target"] for r in rows if r["approved"]}
    stats = Counter(r["rule"] for r in rows)
    stats["lob_bodies_considered"] = len(heads)
    stats["lob_bodies_mapped"] = len({r["lob_target"] for r in rows})
    stats["lob_bodies_mapped_approved"] = len(approved)
    stats["lob_record_coverage_pct"] = round(
        100 * sum(lob.get(l, 0) for l in approved) / sum(lob.values()), 1)
    return rows, stats


def write(rows, path=paths.BODY_MAP):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["lob_target", "pro_contact", "rule", "lob_records", "pro_records", "approved"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (-r["lob_records"], r["lob_target"])))
    return path


def load(path=paths.BODY_MAP, approved_only=True):
    """-> {lob_target name: {pro_contact names}}"""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if approved_only and not r["approved"].strip():
                continue
            out.setdefault(r["lob_target"], set()).add(r["pro_contact"])
    return out


def canonical_bodies():
    """-> (lob name -> body key, pro name -> body key).

    The key is the highest-volume pro_contact name of the mapped group, since
    awards are recorded against the procurement side.
    """
    pro_counts = counts("pro_contact.csv", "name_core")
    lob2key, pro2key = {}, {}
    for lob, pros in load().items():
        key = max(pros, key=lambda p: pro_counts.get(p, 0))
        lob2key[lob] = key
        for p in pros:
            pro2key[p] = key
    return lob2key, pro2key


def main():
    rows, stats = build()
    p = write(rows)
    print(f"wrote {p}  ({len(rows):,} proposed pairs)\n")
    for k in ("lob_bodies_considered", "lob_bodies_mapped",
              "lob_bodies_mapped_approved", "lob_record_coverage_pct"):
        print(f"  {k:<30}{stats[k]}")
    print("\n  by rule:")
    for k in ("exact", "rename", "subset", "jaccard"):
        print(f"    {k:<10}{stats.get(k, 0):>6}"
              + ("   <- needs review" if k == "jaccard" else ""))

    print("\n  highest-volume bodies still unmapped:")
    mapped = {r["lob_target"] for r in rows}
    unmapped = [(n, c) for n, c in counts("lob_target.csv", "name_core").most_common(400)
                if n not in mapped]
    for name, cnt in unmapped[:12]:
        print(f"    {cnt:>8,}  {name[:60]}")


if __name__ == "__main__":
    main()
