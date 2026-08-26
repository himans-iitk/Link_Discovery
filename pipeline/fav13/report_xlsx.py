#!/usr/bin/env python3
"""Build the Favouritism model-comparison workbook."""
import json, os
from pipeline.fav13 import config as C
from pipeline.fav13 import xlsx

def main():
    W = C.WORK
    res = json.load(open(os.path.join(W, "v3b_results.json")))
    hits = json.load(open(os.path.join(W, "hits.json")))
    v3 = json.load(open(os.path.join(W, "v3_top.json")))
    TOTAL = res["total"]
    H, N, I, D = xlsx.S_HEAD, xlsx.S_NOTE, xlsx.S_INT, xlsx.S_DEC

    # ---- every model, learned and baseline, in one place -----------------------
    SPEC = {
     "v3  all 9 features": ("learned", "monotone boosted stumps, 9 features",
       "540 weak positives + 25k sampled unlabelled", "yes",
       "learned company size -- 51% of weight on prominence"),
     "v3b without prominence": ("learned", "same, prominence feature removed",
       "540 weak positives + 25k sampled unlabelled", "yes",
       "channel absorbed prominence's role"),
     "v3c anomaly signals only": ("learned", "prominence and channel both removed",
       "540 weak positives + 25k sampled unlabelled", "yes",
       "best learned variant; uses dominance + recency"),
     "BASELINE sole-source then recency": ("rule", "non-competitive first, then smallest gap",
       "none", "yes", "BEST OVERALL -- two lines of code, no parameters"),
     "BASELINE raw nearest-gap": ("rule", "smallest lobbying-to-award gap, no null",
       "none", "yes", "volume artifact: big firms get tight gaps by chance"),
     "v2 surprise only": ("statistical", "4 p-values combined by Brown's method",
       "none", "yes", "combination degenerated (c=31.7, f=0.25)"),
     "BASELINE prominence": ("rule", "rank by supplier's total contract count",
       "none", "yes", "the baseline everything must beat"),
    }
    V2FULL = ("v2 full  pi x (1-p) x salience", "statistical",
              "Brown combination x identity gate x salience", "none", "yes",
              "salience reintroduced the prominence confound (-0.065 AUC)")

    comp = sorted({c for r in res["results"] for c in res["ranks"][r["label"]]} |
                  {c for b in res["baselines"].values() for c in b["ranks"]})

    # ---------------------------------------------------------------- sheet 1
    s1 = [["Model", "Type", "AUC", "Trained on", "Monotone?"] +
          [c[:24] for c in comp] + ["Note"]]
    rows = []
    for r in res["results"]:
        lab = r["label"]
        sp = SPEC.get(lab, ("", "", "", "", ""))
        rows.append((r["auc"], [lab, sp[0], (r["auc"], D), sp[2], sp[3]] +
                     [(res["ranks"][lab].get(c) or "not scored",
                       I if res["ranks"][lab].get(c) else 0) for c in comp] + [sp[4]]))
    for k, b in res["baselines"].items():
        sp = SPEC.get(k, ("", "", "", "", ""))
        rows.append((b["auc"], [k, sp[0], (b["auc"], D), sp[2], sp[3]] +
                     [(b["ranks"].get(c) or "not scored",
                       I if b["ranks"].get(c) else 0) for c in comp] + [sp[4]]))
    rows.append((0.7872, [V2FULL[0], V2FULL[1], (0.7872, D), V2FULL[3], V2FULL[4]] +
                 [(v, I) for v in [77255, 289118, 634750, "n/a", 328911, 261460][:len(comp)]] +
                 [V2FULL[5]]))
    for _, row in sorted(rows, key=lambda t: -t[0]):
        s1.append(row)
    s1 += [[], [("Ranks are the position of that company's best contract among "
                 f"{TOTAL:,} scored contracts. Lower is better.", N)],
           [("AUC 1.0 = perfect, 0.5 = random. With only 5 known companies the top "
             "three models are statistically indistinguishable.", N)],
           [("Learned models were trained on Accenture and GC Strategies, then scored "
             "on all five -- their AUC is optimistic. The baselines were not trained.", N)]]

    # ---------------------------------------------------------------- sheet 2
    s2 = [["Model", "Family", "What it does", "Training data", "Monotone", "Verdict"]]
    for k, v in SPEC.items():
        s2.append([k, v[0], v[1], v[2], v[3], v[4]])
    s2.append([V2FULL[0], V2FULL[1], V2FULL[2], V2FULL[3], V2FULL[4], V2FULL[5]])

    # ---------------------------------------------------------------- sheet 3
    s3 = [["Model", "Feature", "Share of weight", "Interpretation"]]
    MEAN = {"f_prominence": "supplier size -- NOT an anomaly signal",
            "f_channel": "how much lobbying on this department's channel",
            "f_dominance": "supplier holds outsized share of this department",
            "f_recency": "how soon after lobbying the award landed",
            "f_dep_share": "this department as a share of supplier's book",
            "f_noncomp": "award was non-competitive",
            "f_nc_rarity": "...and this department rarely does that",
            "f_tender_short": "tender window was short",
            "f_time_excess": "excess over the national lobbying pattern"}
    for r in res["results"]:
        for f, v in r["importance"]:
            s3.append([r["label"], f, (round(v, 4), D), MEAN.get(f, "")])
        for f in MEAN:
            if f not in dict(r["importance"]) and f != "f_prominence" or \
               (f == "f_prominence" and "all 9" in r["label"] and
                    f not in dict(r["importance"])):
                pass
    s3 += [[], [("Only two features are used by every variant. The learner keeps "
                 "converging on ~2 signals no matter how many it is given.", N)]]

    # ---------------------------------------------------------------- sheet 4
    s4 = [["Company", "Contracts", "No-bid", "No-bid %", "Buyers", "Registrations",
           "Depts lobbied", "Closest lobby-to-award (days)", "Contracts within 30d",
           "Years"]]
    for c in hits["cases"]:
        s4.append([c["name"], (c["n_contracts"], I), (c["n_nc"], I),
                   (round(c["nc_share"] * 100, 1), D), (c["n_buyers"], I),
                   (c["n_regs"], I), (c["n_bodies"], I),
                   (c["min_gap"] if c["min_gap"] is not None else "no lobbying",
                    I if c["min_gap"] is not None else 0),
                   (c["n_gap_30"], I),
                   f"{c['yr_first']}-{c['yr_last']}" if c["yr_first"] else ""])

    # ---------------------------------------------------------------- sheet 5
    s5 = [["Company", "Story names department", "Period", "Amount claimed",
           "Our contracts w/ that dept", "In that period", "of those no-bid",
           "Did they lobby it?"]]
    for c in hits["cases"]:
        for st in c["stories"]:
            s5.append([c["name"], st["dept"], st["when"] or "-", st["amount"] or "-",
                       (st["n_dept"], I), (st["n_win"], I), (st["nc_win"], I),
                       st["lobbied_dept"]])
    s5 += [[], [("Only two rows align: Accenture / Export Development Canada and "
                 "GC Strategies / PSPC. The rest name provincial bodies, private "
                 "utilities, or are too vague to match.", N)]]

    # ---------------------------------------------------------------- sheet 6
    s6 = [["Rank", "Score", "Company", "Award date", "Method", "Buyer(s)",
           "Linked registrations", "Nearest gap (days)", "Award ID"]]
    for m in v3["motifs"][:300]:
        s6.append([(m["rank"], I), (m["score"], D), m["name"], m["award_date"],
                   m["method"] or "-", m["buyers"][:60], (m["n_links"], I),
                   (m["links"][0]["gap"] if m["links"] else "", I if m["links"] else 0),
                   m["award_rid"][:16]])

    out = os.path.join(C.REPO, "docs", "favouritism_models.xlsx")
    xlsx.write(out, [
        ("Model comparison", s1, [30, 12, 9, 34, 11] + [14] * len(comp) + [52]),
        ("Model specs", s2, [30, 12, 44, 34, 10, 52]),
        ("Feature importance", s3, [26, 18, 15, 46]),
        ("Ground truth holdings", s4, [34, 11, 9, 10, 9, 13, 13, 26, 20, 12]),
        ("Story alignment", s5, [30, 34, 11, 15, 24, 14, 15, 16]),
        ("Top 300 motifs", s6, [7, 9, 32, 12, 10, 44, 18, 16, 18]),
    ])
    print("wrote", out)


if __name__ == "__main__":
    main()
