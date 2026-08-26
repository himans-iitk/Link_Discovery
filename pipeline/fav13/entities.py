"""
Entity map: raw organisation name -> legal entity.

Chain:
    org_preprocessed.name  ->  .preprocessed        deterministic, 1.45x
    cluster_node.name      ->  .cluster_id          manual
                           ->  .canonical_id        manual, second pass

`canonical_id` sits *above* `cluster_id` -- 189 of its values span more than
one cluster (canonical 296 covers clusters 28, 186, 441: all CORCAN).  It is
only ~13% populated and 612 clusters are mixed, so "canonical_id if present
else cluster_id" would split those clusters in half.  We therefore treat both
columns as equivalence relations and take connected components.

Then, because the chosen granularity is *legal entity* rather than corporate
family, clusters are split back apart where their members are distinct
registered names.
"""

import csv, os, pickle, re
from collections import defaultdict

from . import config as C

csv.field_size_limit(10 ** 9)

_SQ = re.compile(r"[^A-Z0-9]")


def _sq(s):
    return _SQ.sub("", (s or "").upper())


def _trigrams(s):
    t = f"  {_sq(s)} "
    return {t[i:i + 3] for i in range(len(t) - 2)} or {t}


def _same_legal_entity(a, b, min_len=None, min_sim=0.50):
    """Two cluster members are the same registered entity if one name is
    contained in the other (trade name plus registration number), or the
    strings are near-identical (data-entry variants)."""
    min_len = C.SPLIT_CONTAINMENT_MIN_LEN if min_len is None else min_len
    A, B = _sq(a), _sq(b)
    if len(A) >= min_len and len(B) >= min_len and (A in B or B in A):
        return True
    TA, TB = _trigrams(a), _trigrams(b)
    return len(TA & TB) / max(len(TA | TB), 1) >= min_sim


def _components(names, related):
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


def build(granularity=None):
    """-> (raw_name -> entity_id, preprocessed -> entity_id, entity -> members)"""
    granularity = granularity or C.ENTITY_GRANULARITY

    raw2pre, all_pre = {}, set()
    lob_n, pro_n = defaultdict(int), defaultdict(int)
    with open(os.path.join(C.RAW, "org_preprocessed.csv"), newline="",
              encoding="utf-8", errors="replace") as fh:
        for x in csv.DictReader(fh):
            raw = (x["name"] or "").strip()
            pre = (x["preprocessed"] or "").strip()
            if not pre:
                continue
            raw2pre[raw] = pre
            all_pre.add(pre)
            lob_n[pre] += int(x["lob_count"] or 0)
            pro_n[pre] += int(x["pro_count"] or 0)

    clusters = defaultdict(list)
    canon = {}
    with open(os.path.join(C.RAW, "cluster_node.csv"), newline="",
              encoding="utf-8", errors="replace") as fh:
        for x in csv.DictReader(fh):
            n = (x["name"] or "").strip()
            clusters[(x["cluster_id"] or "").strip()].append(n)
            k = (x["canonical_id"] or "").strip()
            if k:
                canon[n] = k

    # union-find over cluster_id AND canonical_id -> family-level groups
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

    pre2ent = {}
    members = defaultdict(list)
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
    """Longest member name -- the most informative surface form."""
    return {e: max(ns, key=len) for e, ns in members.items()}


def save(path=None):
    path = path or os.path.join(C.WORK, "entities.pkl")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    raw2ent, pre2ent, members, lob_n, pro_n = build()
    pickle.dump(dict(raw2ent=raw2ent, pre2ent=pre2ent, members=members,
                     lob_n=lob_n, pro_n=pro_n,
                     display=display_name(members)), open(path, "wb"))
    return path
