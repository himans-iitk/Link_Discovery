"""Helpers shared across FAV 13 stages: CSV access, day numbers, caches,
ground-truth matching, and rank-test arithmetic."""

import csv
import datetime
import os
import pickle
import re
from collections import Counter, defaultdict

from pipeline import paths
from . import config as C

csv.field_size_limit(10 ** 9)

NONCOMP = C.NONCOMPETITIVE_METHODS

# ---------------------------------------------------------------- dates

_EPOCH = datetime.date(1990, 1, 1)


def to_day(iso):
    """'YYYY-MM-DD' -> integer day number."""
    return (datetime.date(int(iso[:4]), int(iso[5:7]), int(iso[8:10])) - _EPOCH).days


def from_day(d):
    return (_EPOCH + datetime.timedelta(days=int(d))).isoformat()


def year_of(day):
    return int(from_day(day)[:4])


# ---------------------------------------------------------------- CSV + caches

def rows(fname, cols, base=paths.CLEAN):
    """Stream the named columns of a corpus table; short rows are skipped."""
    with open(os.path.join(base, fname), newline="", encoding="utf-8",
              errors="replace") as fh:
        r = csv.reader(fh)
        hdr = next(r)
        idx = [hdr.index(c) for c in cols]
        last = max(idx)
        for x in r:
            if len(x) > last:
                yield [x[i] for i in idx]


def load(name):
    with open(paths.WORK / name, "rb") as fh:
        return pickle.load(fh)


def dump(obj, name):
    os.makedirs(paths.WORK, exist_ok=True)
    with open(paths.WORK / name, "wb") as fh:
        pickle.dump(obj, fh)
    return paths.WORK / name


def load_entities():
    return load("entities.pkl")


def load_scored():
    """v2 output: one record per scored contract, rank-ordered."""
    return load("fav13_v2_scored.pkl")


def write_csv(path, recs, cols):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    return path


# ---------------------------------------------------------------- corpus lookups

def registrations(with_rnum=False):
    """rid -> registration start day (valid dates only) [, rid -> rnum]."""
    day, rnum = {}, {}
    for rid, d, st, rn in rows("lob_meta.csv", ["rid", "start_date_clean",
                                                "start_date_status", "rnum"]):
        if st == "valid":
            day[rid] = to_day(d)
            rnum[rid] = rn
    return (day, rnum) if with_rnum else day


def limited_reasons():
    """award rid -> stated reason for limited tendering (non-empty only)."""
    return {rid: lr.strip() for rid, lr in
            rows("pro_award.csv", ["rid", "limited_reason"]) if lr.strip()}


def lobby_channels(raw2ent, reg_day):
    """Lobbying channels per (entity, body name_core).

    -> ({(entity, body): sorted [(day, rid)]}, Counter entity -> n registrations)
    """
    rid_body = defaultdict(set)
    for rid, core in rows("lob_target.csv", ["rid", "name_core"]):
        if core and rid in reg_day:
            rid_body[rid].add(core)
    channel, n_regs = defaultdict(list), Counter()
    for rid, raw in rows("lob_org.csv", ["rid", "name"], base=paths.RAW):
        e = raw2ent.get((raw or "").strip())
        if e is None or rid not in reg_day:
            continue
        n_regs[e] += 1
        for b in rid_body.get(rid, ()):
            channel[(e, b)].append((reg_day[rid], rid))
    for k in channel:
        channel[k].sort()
    return channel, n_regs


def first_buyer(r):
    return (r["buyer_names"] or "").split(";")[0].strip()


# ---------------------------------------------------------------- ground truth

_DEPT_STOP = {"of", "the", "and", "for", "canada", "canadian", "government",
              "department", "ministry", "agency", "office", "inc", "ltd",
              "corporation", "commission"}


def name_key(s):
    """Exact-match key: uppercase alphanumerics only."""
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def dept_tokens(s):
    """Discriminating tokens of a department name, for loose story matching."""
    return {t for t in re.sub(r"[^a-z ]", " ", (s or "").lower()).split()
            if t and t not in _DEPT_STOP and len(t) > 2}


def year_span(s):
    """(first, last) four-digit year in free text, or (None, None)."""
    yy = [int(y) for y in re.findall(r"\b((?:19|20)\d{2})\b", s or "")]
    return (min(yy), max(yy)) if yy else (None, None)


def read_stories():
    with open(paths.STORIES, encoding="utf-8", errors="replace") as fh:
        return list(csv.DictReader(fh))


def name_index(raw2ent):
    """Exact name key -> entity. Substring matching is deliberately not used:
    it matched people and descriptions to real companies."""
    index = {}
    for raw, e in raw2ent.items():
        index.setdefault(name_key(raw), e)
    return index


def story_claims(index, stories=None):
    """-> [(entity, department tokens, (year_lo, year_hi))] for resolvable stories."""
    out = []
    for g in stories if stories is not None else read_stories():
        e = index.get(name_key((g.get("private_org") or "").strip()))
        if e:
            out.append((e, dept_tokens(g.get("public_org")),
                        year_span(g.get("contract_time"))))
    return out


def dept_match(r, dt):
    return bool(dt and (dt & dept_tokens(r["buyer_names"])))


def in_period(r, lo, hi):
    return lo is None or lo <= year_of(r["day"]) <= hi


# ---------------------------------------------------------------- rank test

def best_ranks(order, positives):
    """1-based rank of each positive's first appearance in a best-first list."""
    best = {}
    for i, e in enumerate(order, 1):
        if e in positives and e not in best:
            best[e] = i
    return best


def auc(best, n_items, n_pos=None):
    """Mann-Whitney U over the positives' ranks = AUC. Under PU data the absolute
    value is biased low; comparisons between orderings stay valid."""
    if not best:
        return float("nan")
    n1 = len(best)
    n0 = max(n_items - (n1 if n_pos is None else n_pos), 1)
    U = n1 * n0 + n1 * (n1 + 1) / 2.0 - sum(best.values())
    return U / (n1 * n0)


def percentile(recs, key):
    """Rank-normalise recs[*][key] into [0, 1]; ties share the mean rank.
    (Min-max let one heavy-tail outlier flatten everyone else.)"""
    order = sorted(range(len(recs)), key=lambda i: recs[i][key])
    n = len(recs)
    out = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and recs[order[j + 1]][key] == recs[order[i]][key]:
            j += 1
        share = (i + j) / 2.0 / max(n - 1, 1)
        for t in range(i, j + 1):
            out[order[t]] = share
        i = j + 1
    return out
