"""Entity resolution: raw organisation name -> legal entity id.

    org_preprocessed.name -> .preprocessed      (deterministic)
    cluster_node.name     -> .cluster_id        (manual)
                          -> .canonical_id      (manual, second pass)

canonical_id can span several clusters and is only ~13% populated, so both
columns are treated as equivalence relations and joined by union-find. The
resulting families are then split back into legal entities (see config).
This is the only inference in the pipeline.
"""

import csv
import os
import re
from collections import defaultdict

from pipeline import paths
from .. import common
from .. import config as C

_SQ = re.compile(r"[^A-Z0-9]")


def _sq(s):
    return _SQ.sub("", (s or "").upper())


def _trigrams(s):
    t = f"  {_sq(s)} "
    return {t[i:i + 3] for i in range(len(t) - 2)} or {t}


def _same_legal_entity(a, b, min_len=None, min_sim=0.50):
    """One name contains the other (trade name + reg. number), or the
    trigram Jaccard is high (typo variants like DELOITE ~ DELOITTE)."""
    min_len = C.SPLIT_CONTAINMENT_MIN_LEN if min_len is None else min_len
    A, B = _sq(a), _sq(b)
    if len(A) >= min_len and len(B) >= min_len and (A in B or B in A):
        return True
    TA, TB = _trigrams(a), _trigrams(b)
    return len(TA & TB) / max(len(TA | TB), 1) >= min_sim


def _components(names, related):
    """Connected components of `names` under the pairwise `related` test."""
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if related(names[i], names[j]):
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b
    out = defaultdict(list)
    for k, n in enumerate(names):
        out[find(k)].append(n)
    return list(out.values())


def _read(fname):
    with open(os.path.join(paths.RAW, fname), newline="", encoding="utf-8",
              errors="replace") as fh:
        yield from csv.DictReader(fh)


def build(granularity=None):
    """-> (raw->entity, preprocessed->entity, entity->members, lob counts, pro counts)"""
    granularity = granularity or C.ENTITY_GRANULARITY

    raw2pre, all_pre = {}, set()
    lob_n, pro_n = defaultdict(int), defaultdict(int)
    for x in _read("org_preprocessed.csv"):
        raw = (x["name"] or "").strip()
        pre = (x["preprocessed"] or "").strip()
        if not pre:
            continue
        raw2pre[raw] = pre
        all_pre.add(pre)
        lob_n[pre] += int(x["lob_count"] or 0)
        pro_n[pre] += int(x["pro_count"] or 0)

    clusters, canon = defaultdict(list), {}
    for x in _read("cluster_node.csv"):
        n = (x["name"] or "").strip()
        clusters[(x["cluster_id"] or "").strip()].append(n)
        k = (x["canonical_id"] or "").strip()
        if k:
            canon[n] = k

    # union-find over cluster_id and canonical_id -> corporate families
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for cid, names in clusters.items():
        for n in names:
            union("N:" + n, "C:" + cid)
            if n in canon:
                union("N:" + n, "K:" + canon[n])

    families = defaultdict(list)
    for pre in all_pre:
        families[find("N:" + pre)].append(pre)

    # families -> legal entities
    pre2ent, members = {}, defaultdict(list)
    for root, names in families.items():
        if granularity == "family" or len(names) == 1:
            groups = [names]
        else:
            groups = _components(names, _same_legal_entity)
        for i, g in enumerate(groups):
            eid = f"{root}.{i}" if len(groups) > 1 else root
            for n in g:
                pre2ent[n] = eid
                members[eid].append(n)

    raw2ent = {raw: pre2ent[p] for raw, p in raw2pre.items() if p in pre2ent}
    return raw2ent, pre2ent, dict(members), dict(lob_n), dict(pro_n)


def display_name(members):
    """Longest member name: the most informative surface form."""
    return {e: max(ns, key=len) for e, ns in members.items()}


def save():
    raw2ent, pre2ent, members, lob_n, pro_n = build()
    return common.dump(dict(raw2ent=raw2ent, pre2ent=pre2ent, members=members,
                            lob_n=lob_n, pro_n=pro_n,
                            display=display_name(members)), "entities.pkl")
