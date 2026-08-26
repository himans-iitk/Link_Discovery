#!/usr/bin/env python3
"""
General profile of the cleaned layer.

Answers three questions the method design depends on:
  1. How big is the problem really, after normalisation?  (entities, not rows)
  2. What does blocking cost, and where does it blow up?   (block-size tails)
  3. What is the data actually missing?                    (per-gate coverage)
"""
import csv, json, os, sys
from collections import Counter

CLEAN = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "ijf-clean")
csv.field_size_limit(10**9)


def read(f, cols):
    p = os.path.join(CLEAN, f)
    with open(p, newline="", encoding="utf-8", errors="replace") as fh:
        r = csv.reader(fh)
        hdr = next(r)
        idx = [hdr.index(c) for c in cols]
        for row in r:
            if len(row) > max(idx):
                yield [row[i] for i in idx]


def block_profile(counter, label, rows):
    """Blocking cost = sum over blocks of C(n,2) if self-joined."""
    d = len(counter)
    tot = sum(counter.values())
    singles = sum(1 for v in counter.values() if v == 1)
    sizes = sorted(counter.values(), reverse=True)
    pairs = sum(v * (v - 1) // 2 for v in counter.values())
    p99 = sizes[int(len(sizes) * 0.01)] if sizes else 0
    return dict(label=label, rows=rows, records=tot, entities=d,
                collapse=round(tot / d, 1) if d else 0,
                singleton_pct=round(100 * singles / d, 1) if d else 0,
                max_block=sizes[0] if sizes else 0,
                p99_block=p99,
                self_join_pairs=pairs)


PERSON = [("apt.csv", "name_fl", "APT appointees"),
          ("rep_target.csv", "name_fl", "REP_TARGET targets"),
          ("lob_rd_mv.csv", "name_fl", "LOB_RD offices"),
          ("lob_lobbyist.csv", "name_fl", "LOB_LOBBYIST lobbyists"),
          ("don_mv.csv", "donor_fl", "DON donors")]
ORG = [("lob_org.csv", "name_core", "LOB_ORG companies"),
       ("lob_target.csv", "name_core", "LOB_TARGET gov bodies"),
       ("pro_supplier.csv", "name_core", "PRO_SUPPLIER suppliers"),
       ("pro_contact.csv", "name_core", "PRO_CONTACT buyers"),
       ("lob_firm.csv", "name_core", "LOB_FIRM firms"),
       ("cha_meta.csv", "name_core", "CHA charities")]


def main():
    out = {"person": [], "org": [], "quality": {}, "recipient": {}}
    sets = {}

    for f, col, label in PERSON + ORG:
        if not os.path.exists(os.path.join(CLEAN, f)):
            continue
        c = Counter()
        n = 0
        for (v,) in read(f, [col]):
            n += 1
            if v:
                c[v] += 1
        sets[label] = c
        rec = block_profile(c, label, n)
        (out["person"] if (f, col, label) in PERSON else out["org"]).append(rec)

    # donation recipient typing
    if os.path.exists(os.path.join(CLEAN, "don_mv.csv")):
        k = Counter()
        rc = Counter()
        for kind, fl in read("don_mv.csv", ["recipient_kind", "recipient_fl"]):
            k[kind] += 1
            if kind == "person" and fl:
                rc[fl] += 1
        out["recipient"] = dict(counts=dict(k), person_entities=len(rc))
        sets["DON recipients (person)"] = rc

    # date + jurisdiction coverage
    DATE = [("apt.csv", "start"), ("apt.csv", "end"),
            ("lob_meta.csv", "start"), ("lob_meta.csv", "end"),
            ("rep_meta.csv", "activity"), ("don_mv.csv", "donation"),
            ("pro_award.csv", "award"), ("pro_tender.csv", "open")]
    for f, pre in DATE:
        if not os.path.exists(os.path.join(CLEAN, f)):
            continue
        c = Counter()
        for (s,) in read(f, [f"{pre}_date_status"]):
            c[s] += 1
        out["quality"][f"{f}:{pre}"] = dict(c)

    JUR = [("apt.csv", "region"), ("lob_meta.csv", "jur"),
           ("rep_meta.csv", "jur"), ("don_mv.csv", "region"),
           ("pro_contact.csv", "prov"), ("pro_supplier.csv", "prov")]
    out["jurisdiction"] = {}
    for f, pre in JUR:
        if not os.path.exists(os.path.join(CLEAN, f)):
            continue
        c = Counter()
        for (s,) in read(f, [f"{pre}_code_status"]):
            c[s] += 1
        out["jurisdiction"][f"{f}:{pre}"] = dict(c)

    # cross-table candidate volume (the real scalability number)
    out["overlap"] = {}
    keys = list(sets)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            sa, sb = set(sets[a]), set(sets[b])
            shared = sa & sb
            if not shared:
                continue
            # true candidate record-pairs, not entity-pairs
            rp = sum(sets[a][n] * sets[b][n] for n in shared)
            out["overlap"][f"{a} | {b}"] = dict(
                shared_entities=len(shared), record_pairs=rp)

    json.dump(out, open(sys.argv[1] if len(sys.argv) > 1
                        else "/tmp/profile.json", "w"), indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "overlap"}, indent=2))
    print("\n--- overlap (top 20 by record pairs) ---")
    for k, v in sorted(out["overlap"].items(),
                       key=lambda x: -x[1]["record_pairs"])[:20]:
        print(f"{v['record_pairs']:>18,}  ent={v['shared_entities']:>7,}  {k}")


if __name__ == "__main__":
    main()
