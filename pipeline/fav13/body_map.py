"""
Government-body map: lob_target (bodies lobbied) <-> pro_contact (buyers).

This is FAV 12 done properly.  Both sides name the same departments under
different conventions and across machinery-of-government renames, so exact
string matching resolves only 28% of the bodies that carry volume.

Government bodies are a *closed, enumerable vocabulary* -- 684 lob_target
names cover 80% of all lobbying records -- which makes this curation, not
entity resolution.  The output is a reviewable table, one row per proposed
pair, carrying the rule that produced it and the record volume behind it, so
a human can audit the head of the distribution and ignore the tail.

Rules, in decreasing confidence:

  exact       identical after the clean layer's name_core normalisation
  rename      curated machinery-of-government rename (PWGSC -> PSPC, etc.)
  subset      one token set contains the other ("national defence" inside
              "department national defence")
  jaccard     token overlap above a threshold -- PROPOSED, needs review

Only `exact`, `rename` and `subset` are used by default.  `jaccard` rows are
written out flagged so they can be promoted after review; using them
unreviewed would fabricate loops, and a fabricated loop is a fabricated
accusation.

    python3 -m pipeline.fav13.body_map          build + report
"""

import csv, os, re
from collections import Counter

from . import config as C

csv.field_size_limit(10 ** 9)

OUT = os.path.join(C.REF, "body_map.csv")

# Curated machinery-of-government renames and known convention differences.
# Each is a verifiable public fact, not a guess: the left name and the right
# name denote the same body.  Written against the clean layer's name_core
# form (lowercased, legal suffixes and stopwords stripped).
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

# Tokens that carry no discriminating information on either side.
_NOISE = {"department", "ministry", "ministere", "office", "bureau", "agency",
          "agence", "canada", "canadian", "government", "gouvernement",
          "secretariat", "service", "services", "national", "federal"}


def _toks(s):
    return [t for t in re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()).split() if t]


def _core(s):
    """Discriminating tokens only -- what actually names the body."""
    return frozenset(t for t in _toks(s) if t not in _NOISE)


def _counts(fname, col):
    c = Counter()
    with open(os.path.join(C.CLEAN, fname), newline="", encoding="utf-8",
              errors="replace") as fh:
        r = csv.reader(fh)
        i = next(r).index(col)
        for x in r:
            if len(x) > i and x[i]:
                c[x[i]] += 1
    return c


def build(top_n=None, jaccard_min=0.60):
    """-> (rows, stats).  `rows` is one proposed pair per row."""
    lob = _counts("lob_target.csv", "name_core")
    pro = _counts("pro_contact.csv", "name_core")

    # only curate the bodies that carry volume; the tail is noise
    top_n = top_n or 700
    heads = [n for n, _ in lob.most_common(top_n)]

    pro_by_core = {}
    for p in pro:
        pro_by_core.setdefault(_core(p), []).append(p)

    ren = {a: b for a, b in RENAMES}
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

        # subset: identical discriminating tokens, differing boilerplate
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

    stats = Counter(r["rule"] for r in rows)
    stats["lob_bodies_considered"] = len(heads)
    stats["lob_bodies_mapped"] = len({r["lob_target"] for r in rows})
    stats["lob_bodies_mapped_approved"] = len(
        {r["lob_target"] for r in rows if r["approved"]})
    lob_tot = sum(lob.values())
    stats["lob_record_coverage_pct"] = round(
        100 * sum(lob.get(l, 0) for l in
                  {r["lob_target"] for r in rows if r["approved"]}) / lob_tot, 1)
    return rows, stats


def write(rows, path=None):
    path = path or OUT
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cols = ["lob_target", "pro_contact", "rule", "lob_records", "pro_records",
            "approved"]
    rows = sorted(rows, key=lambda r: (-r["lob_records"], r["lob_target"]))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path


def load(path=None, approved_only=True):
    """-> {lob_target_name: {pro_contact names}} for use by the features."""
    path = path or OUT
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if approved_only and not r["approved"].strip():
                continue
            out.setdefault(r["lob_target"], set()).add(r["pro_contact"])
    return out


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
    lob = _counts("lob_target.csv", "name_core")
    mapped = {r["lob_target"] for r in rows}
    n = 0
    for name, cnt in lob.most_common(400):
        if name not in mapped:
            print(f"    {cnt:>8,}  {name[:60]}")
            n += 1
            if n >= 12:
                break


if __name__ == "__main__":
    main()
