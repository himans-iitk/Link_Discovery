#!/usr/bin/env python3
"""
Deterministic cleaning layer for the IJF link-discovery datasets.

Contract
--------
1. RAW IS NEVER MODIFIED.  Output goes to a sibling directory.
2. ADDITIVE ONLY.  Every raw column is preserved (except `fts`, a Postgres
   tsvector that is purely derived).  Cleaned values are appended as new
   columns suffixed `_clean` / `_status` / etc.
3. ROW-PRESERVING.  Output has the same number of rows, in the same order,
   with the same keys, as the input.  No dedup, no merging.
4. DETERMINISTIC ONLY.  Nothing here decides that two records are the same
   real-world entity.  Consolidation and linking are method, not cleaning.
   The single deliberate exception is `recipient_kind` on donations, which
   is an inference; it is flagged as such and carries an `unknown` state.

Name handling
-------------
No single canonical name form is imposed, because the choice is
consequential: collapsing middle initials raises apt<->don candidate
overlap by ~27%, and that increment is disproportionately false.  Several
forms are emitted side by side so the method layer can choose per case.

Date handling
-------------
Dates resolve to one of four states, never silently to NULL.  The
distinction between "no date" and "unparseable date" is what separates
"order unknown" from "order violated" downstream.
"""

import csv, json, os, re, sys, unicodedata
from collections import Counter

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "ijf-datasets")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "ijf-clean")

csv.field_size_limit(10**9)

# --------------------------------------------------------------------------
# name normalisation
# --------------------------------------------------------------------------

TITLES = {"mr", "mrs", "ms", "miss", "dr", "prof", "hon", "honourable", "rt",
          "sen", "sir", "madam", "me", "mme", "m"}
SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v", "phd", "md", "qc", "kc", "cd",
            "oc", "cm", "mp", "mla", "mpp", "mna", "esq", "ba", "bsc", "ma",
            "msc", "llb", "llm", "cpa", "ca", "pe", "peng", "rn"}

LEGAL = (r"\b(inc|incorporated|ltd|ltee|limited|limitee|llp|llc|lp|corp|"
         r"corporation|co|company|sa|sarl|plc|gmbh|nv|bv|pte|pty|sencrl|srl|"
         r"enr|cie|group|groupe|holdings|holding)\b")
ORG_STOP = r"\b(the|of|and|for|de|du|des|la|le|les|et|a|l|d)\b"


def ascii_fold(s):
    s = unicodedata.normalize("NFKD", s or "")
    return s.encode("ascii", "ignore").decode()


def _tokens(s, join_intraword):
    """Tokenise a folded name.

    Hyphens and apostrophes are intra-word connectors in ~3% of Canadian
    names ("Southern-Heathcott", "O'Hara").  Splitting on them silently
    truncates compound surnames -- "Linda Southern-Heathcott" would block
    as "linda heathcott".  Neither policy dominates: a source that writes
    the surname spaced will only match the split form.  We therefore emit
    both and let blocking union them.
    """
    s = ascii_fold(s)
    s = re.sub(r"['’ʼ-]", "" if join_intraword else " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    toks = [t for t in re.sub(r"\s+", " ", s).strip().split() if t]
    # strip leading titles and trailing suffixes so they never become the
    # "last name" -- e.g. "Seeon Smith Ii" must reduce to "seeon smith"
    while toks and toks[0] in TITLES:
        toks.pop(0)
    while toks and toks[-1] in SUFFIXES:
        toks.pop()
    return toks


def name_forms(raw):
    """-> (lower, ascii_join, ascii_split, fl_join, fl_split, ntok)"""
    lower = re.sub(r"\s+", " ", (raw or "").strip().lower())
    tj = _tokens(lower, True)
    ts = _tokens(lower, False)

    def fl(t):
        return f"{t[0]} {t[-1]}" if len(t) >= 2 else (t[0] if t else "")

    return lower, " ".join(tj), " ".join(ts), fl(tj), fl(ts), len(tj)


def org_forms(raw):
    """Return (lower, core, alt_core) for an organisation name.

    Bilingual registry names of the form "English / Francais" are split and
    BOTH halves are emitted -- picking one arbitrarily loses real matches.
    """
    lower = re.sub(r"\s+", " ", (raw or "").strip().lower())
    parts = [p.strip() for p in lower.split("/") if p.strip()]
    if not parts:
        parts = [lower]

    def core(s):
        s = ascii_fold(s)
        s = re.sub(r"\(.*?\)", " ", s)          # drop parenthetical acronyms
        s = re.sub(r"[^a-z0-9 ]", " ", s)
        s = re.sub(LEGAL, " ", s)
        s = re.sub(ORG_STOP, " ", s)
        return re.sub(r"\s+", " ", s).strip()

    primary = core(parts[0])
    alt = core(parts[1]) if len(parts) > 1 else ""
    if alt == primary:
        alt = ""
    return lower, primary, alt


# --------------------------------------------------------------------------
# date normalisation
# --------------------------------------------------------------------------

MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
MONTHS.update({m[:3]: i + 1 for m, i in
               [(k, v - 1) for k, v in MONTHS.items()]})

YEAR_MIN, YEAR_MAX = 1900, 2040
NULLISH = {"", "nan", "null", "none", "n/a", "na", "0000-00-00",
           "expired", "current", "ongoing", "indefinite", "-"}


def clean_date(raw):
    """-> (iso_or_empty, status).  status in
    {valid, missing, unparseable, out_of_range}"""
    s = (raw or "").strip()
    if s.lower() in NULLISH:
        return "", "missing"

    y = mo = d = None
    m = re.match(r"^(\d{4})[-/](\d{1,2})[-/](\d{1,2})", s)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
    if y is None:
        # "October 24, 2007" / "24 October 2007"
        t = re.sub(r"[^a-z0-9 ]", " ", ascii_fold(s.lower()))
        t = re.sub(r"\s+", " ", t).strip()
        mm = re.match(r"^([a-z]+) (\d{1,2}) (\d{4})$", t)
        if mm and mm.group(1) in MONTHS:
            mo, d, y = MONTHS[mm.group(1)], int(mm.group(2)), int(mm.group(3))
        else:
            mm = re.match(r"^(\d{1,2}) ([a-z]+) (\d{4})$", t)
            if mm and mm.group(2) in MONTHS:
                d, mo, y = int(mm.group(1)), MONTHS[mm.group(2)], int(mm.group(3))
    if y is None:
        mm = re.match(r"^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$", s)
        if mm:  # ambiguous DD/MM vs MM/DD -- day>12 disambiguates, else assume MM/DD
            a, b, y = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
            mo, d = (b, a) if a > 12 else (a, b)
    if y is None:
        return "", "unparseable"
    if not (1 <= (mo or 0) <= 12 and 1 <= (d or 0) <= 31):
        return "", "unparseable"
    if not (YEAR_MIN <= y <= YEAR_MAX):
        return "", "out_of_range"
    return f"{y:04d}-{mo:02d}-{d:02d}", "valid"


# --------------------------------------------------------------------------
# jurisdiction harmonisation
# --------------------------------------------------------------------------

JUR = {}
for code, names in {
    "fd": ["fd", "federal", "canada", "fed"],
    "ab": ["ab", "alberta"], "bc": ["bc", "british columbia", "colombie britannique"],
    "mb": ["mb", "manitoba"], "nb": ["nb", "new brunswick", "nouveau brunswick"],
    "nl": ["nl", "nf", "newfoundland and labrador", "newfoundland"],
    "ns": ["ns", "nova scotia", "nouvelle ecosse"],
    "nt": ["nt", "northwest territories"], "nu": ["nu", "nunavut"],
    "on": ["on", "ontario"], "pe": ["pe", "pei", "prince edward island"],
    "qc": ["qc", "pq", "quebec"], "sk": ["sk", "saskatchewan"],
    "yt": ["yt", "yukon"],
}.items():
    for n in names:
        JUR[n] = code


def clean_jur(raw):
    s = re.sub(r"[^a-z ]", " ", ascii_fold((raw or "").strip().lower()))
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return "", "missing"
    if s in JUR:
        return JUR[s], "valid"
    return "", "unmapped"


# --------------------------------------------------------------------------
# donation recipient typing  (the one deliberate inference in this layer)
# --------------------------------------------------------------------------

PERSON_PE = {"candidate", "candidates", "leadership contestant",
             "leadership contestants", "nomination contestant",
             "nomination contestants", "contestant", "contestants"}
ORG_PE = {"registered parties", "registered party", "party", "parties",
          "registered associations", "registered association", "association",
          "associations", "constituency", "constituencies",
          "elector organization", "elector organizations", "riding association",
          "electoral district association", "third party", "third parties",
          "leadership campaign", "nomination campaign"}


def recipient_kind(pe):
    s = re.sub(r"[^a-z ]", " ", ascii_fold((pe or "").strip().lower()))
    s = re.sub(r"\s+", " ", s).strip()
    if not s or s == "nan":
        return "unknown"
    if s in PERSON_PE:
        return "person"
    if s in ORG_PE:
        return "org"
    return "unknown"


# --------------------------------------------------------------------------
# per-table specifications
# --------------------------------------------------------------------------
# person: [(src, prefix)]   org: [(src, prefix)]
# date:   [(src, prefix)]   jur:  [(src, prefix)]

SPEC = {
    "apt.csv":            dict(person=[("name", "name")],
                               date=[("start_date", "start"), ("end_date", "end"),
                                     ("posted_date", "posted")],
                               jur=[("region", "region")]),
    "rep_target.csv":     dict(person=[("name", "name")]),
    "lob_lobbyist.csv":   dict(person=[("name", "name")]),
    "lob_rd_mv.csv":      dict(person=[("name", "name")],
                               org=[("entity", "entity")],
                               jur=[("region", "region")]),
    "lob_meta.csv":       dict(date=[("start_date", "start"), ("end_date", "end")],
                               jur=[("s", "jur")]),
    "rep_meta.csv":       dict(date=[("activity_date", "activity"),
                                     ("posted_date", "posted")],
                               jur=[("s", "jur")]),
    "lob_target.csv":     dict(org=[("name", "name")]),
    "lob_org.csv":        dict(org=[("name", "name")]),
    "lob_firm.csv":       dict(org=[("name", "name")]),
    "pro_supplier.csv":   dict(org=[("name", "name")], jur=[("prov", "prov")]),
    "pro_contact.csv":    dict(org=[("name", "name")], jur=[("prov", "prov")]),
    "pro_award.csv":      dict(date=[("award_date", "award"),
                                     ("posted_date", "posted"),
                                     ("start_date", "start"), ("end_date", "end")]),
    "pro_tender.csv":     dict(date=[("open_date", "open"), ("close_date", "close")]),
    "cha_meta.csv":       dict(org=[("name_account", "name")],
                               date=[("fiscal_period_end", "fiscal_end")]),
    "don_mv.csv":         dict(person=[("donor_full_name", "donor")],
                               date=[("donation_date", "donation")],
                               jur=[("region", "region")],
                               donation=True),
    "don_23_99_rows.csv": dict(person=[("donor_full_name", "donor")],
                               date=[("donation_date", "donation")],
                               donation=True),
}

DROP_COLS = {"fts"}


def process(fname, spec):
    src, dst = os.path.join(RAW, fname), os.path.join(OUT, fname)
    stats = Counter()
    with open(src, newline="", encoding="utf-8", errors="replace") as fi, \
         open(dst, "w", newline="", encoding="utf-8") as fo:
        r = csv.reader(fi)
        hdr = next(r)
        keep = [i for i, c in enumerate(hdr) if c not in DROP_COLS]
        out_hdr = [hdr[i] for i in keep]

        for _, pre in spec.get("person", []):
            out_hdr += [f"{pre}_lower", f"{pre}_ascii", f"{pre}_ascii_split",
                        f"{pre}_fl", f"{pre}_fl_split", f"{pre}_ntok"]
        for _, pre in spec.get("org", []):
            out_hdr += [f"{pre}_lower", f"{pre}_core", f"{pre}_alt"]
        for _, pre in spec.get("date", []):
            out_hdr += [f"{pre}_date_clean", f"{pre}_date_status"]
        for _, pre in spec.get("jur", []):
            out_hdr += [f"{pre}_code", f"{pre}_code_status"]
        if spec.get("donation"):
            out_hdr += ["recipient_kind", "recipient_lower", "recipient_ascii",
                        "recipient_fl", "recipient_fl_split"]

        w = csv.writer(fo)
        w.writerow(out_hdr)
        idx = {c: i for i, c in enumerate(hdr)}

        for row in r:
            if len(row) < len(hdr):
                row = row + [""] * (len(hdr) - len(row))
            stats["rows"] += 1
            out = [row[i] for i in keep]

            for col, pre in spec.get("person", []):
                lo, aj, asp, flj, fls, n = name_forms(row[idx[col]])
                out += [lo, aj, asp, flj, fls, str(n)]
                fl = flj
                stats[f"{pre}_blank"] += 0 if fl else 1
            for col, pre in spec.get("org", []):
                lo, core, alt = org_forms(row[idx[col]])
                out += [lo, core, alt]
                stats[f"{pre}_blank"] += 0 if core else 1
                stats[f"{pre}_has_alt"] += 1 if alt else 0
            for col, pre in spec.get("date", []):
                v, st = clean_date(row[idx[col]])
                out += [v, st]
                stats[f"{pre}_{st}"] += 1
            for col, pre in spec.get("jur", []):
                v, st = clean_jur(row[idx[col]])
                out += [v, st]
                stats[f"{pre}_{st}"] += 1
            if spec.get("donation"):
                k = recipient_kind(row[idx["political_entity"]]
                                   if "political_entity" in idx else "")
                lo, aj, _asp, flj, fls, _n = name_forms(
                    row[idx["recipient"]] if "recipient" in idx else "")
                out += [k, lo, aj, flj, fls]
                stats[f"recipient_{k}"] += 1
            w.writerow(out)
    return dict(stats)


def main():
    os.makedirs(OUT, exist_ok=True)
    targets = sys.argv[1:] or list(SPEC)
    manifest = {}
    for f in targets:
        if f not in SPEC:
            print(f"  ! no spec for {f}, skipping")
            continue
        if not os.path.exists(os.path.join(RAW, f)):
            print(f"  ! missing {f}, skipping")
            continue
        print(f"  cleaning {f} ...", flush=True)
        manifest[f] = process(f, SPEC[f])
        print(f"    {manifest[f]['rows']:,} rows", flush=True)
    mp = os.path.join(OUT, "_manifest.json")
    if os.path.exists(mp):
        manifest = {**json.load(open(mp)), **manifest}
    json.dump(manifest, open(mp, "w"), indent=2, sort_keys=True)
    print(f"\nmanifest -> {mp}")


if __name__ == "__main__":
    main()
