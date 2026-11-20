"""Model-comparison workbook -> docs/favouritism_models.xlsx.

Reads work/v3b_results.json, hits.json, v3_top.json.

    python3 -m pipeline.fav13.report.models_xlsx
"""

import json

from pipeline import paths
from . import xlsx

I, D, N = xlsx.S_INT, xlsx.S_DEC, xlsx.S_NOTE
TRAINED = "540 weak positives + 25k sampled unlabelled"

# label -> (family, what it does, training data, monotone, verdict)
SPEC = {
    "v3  all 9 features": ("learned", "monotone boosted stumps, 9 features", TRAINED, "yes",
                           "learned company size -- 51% of weight on prominence"),
    "v3b without prominence": ("learned", "same, prominence feature removed", TRAINED, "yes",
                               "channel absorbed prominence's role"),
    "v3c anomaly signals only": ("learned", "prominence and channel both removed", TRAINED, "yes",
                                 "best learned variant; uses dominance + recency"),
    "BASELINE sole-source then recency": ("rule", "non-competitive first, then smallest gap",
                                          "none", "yes",
                                          "BEST OVERALL -- two lines of code, no parameters"),
    "BASELINE raw nearest-gap": ("rule", "smallest lobbying-to-award gap, no null", "none", "yes",
                                 "volume artifact: big firms get tight gaps by chance"),
    "v2 surprise only": ("statistical", "4 p-values combined by Brown's method", "none", "yes",
                         "combination degenerated (c=31.7, f=0.25)"),
    "BASELINE prominence": ("rule", "rank by supplier's total contract count", "none", "yes",
                            "the baseline everything must beat"),
}
# v2 full score, measured separately (not in v3b_results.json).
V2_FULL = ("v2 full  pi x (1-p) x salience", "statistical",
           "Brown combination x identity gate x salience", "none", "yes",
           "salience reintroduced the prominence confound (-0.065 AUC)")
V2_FULL_AUC = 0.7872
V2_FULL_RANKS = [77255, 289118, 634750, "n/a", 328911, 261460]

FEATURE_MEANING = {
    "f_prominence": "supplier size -- NOT an anomaly signal",
    "f_channel": "how much lobbying on this department's channel",
    "f_dominance": "supplier holds outsized share of this department",
    "f_recency": "how soon after lobbying the award landed",
    "f_dep_share": "this department as a share of supplier's book",
    "f_noncomp": "award was non-competitive",
    "f_nc_rarity": "...and this department rarely does that",
    "f_tender_short": "tender window was short",
    "f_time_excess": "excess over the national lobbying pattern",
}


def _load(name):
    with open(paths.WORK / name) as fh:
        return json.load(fh)


def comparison(res, comp):
    def row(label, auc, ranks):
        sp = SPEC.get(label, ("", "", "", "", ""))
        return [label, sp[0], (auc, D), sp[2], sp[3]] + \
               [(ranks.get(c) or "not scored", I if ranks.get(c) else 0) for c in comp] + [sp[4]]

    rows = [(r["auc"], row(r["label"], r["auc"], res["ranks"][r["label"]]))
            for r in res["results"]]
    rows += [(b["auc"], row(k, b["auc"], b["ranks"])) for k, b in res["baselines"].items()]
    rows.append((V2_FULL_AUC, [V2_FULL[0], V2_FULL[1], (V2_FULL_AUC, D), V2_FULL[3], V2_FULL[4]] +
                 [(v, I) for v in V2_FULL_RANKS[:len(comp)]] + [V2_FULL[5]]))

    sheet = [["Model", "Type", "AUC", "Trained on", "Monotone?"] + [c[:24] for c in comp] + ["Note"]]
    sheet += [r for _, r in sorted(rows, key=lambda t: -t[0])]
    sheet += [[], [("Ranks are the position of that company's best contract among "
                    f"{res['total']:,} scored contracts. Lower is better.", N)],
              [("AUC 1.0 = perfect, 0.5 = random. With only 5 known companies the top "
                "three models are statistically indistinguishable.", N)],
              [("Learned models were trained on Accenture and GC Strategies, then scored "
                "on all five -- their AUC is optimistic. The baselines were not trained.", N)]]
    return sheet


def main():
    res, hits, v3 = _load("v3b_results.json"), _load("hits.json"), _load("v3_top.json")
    comp = sorted({c for r in res["results"] for c in res["ranks"][r["label"]]} |
                  {c for b in res["baselines"].values() for c in b["ranks"]})

    specs = [["Model", "Family", "What it does", "Training data", "Monotone", "Verdict"]]
    specs += [[k, *v] for k, v in SPEC.items()] + [list(V2_FULL)]

    importance = [["Model", "Feature", "Share of weight", "Interpretation"]]
    importance += [[r["label"], f, (round(v, 4), D), FEATURE_MEANING.get(f, "")]
                   for r in res["results"] for f, v in r["importance"]]
    importance += [[], [("Only two features are used by every variant. The learner keeps "
                         "converging on ~2 signals no matter how many it is given.", N)]]

    holdings = [["Company", "Contracts", "No-bid", "No-bid %", "Buyers", "Registrations",
                 "Depts lobbied", "Closest lobby-to-award (days)", "Contracts within 30d",
                 "Years"]]
    for c in hits["cases"]:
        holdings.append([c["name"], (c["n_contracts"], I), (c["n_nc"], I),
                         (round(c["nc_share"] * 100, 1), D), (c["n_buyers"], I),
                         (c["n_regs"], I), (c["n_bodies"], I),
                         (c["min_gap"] if c["min_gap"] is not None else "no lobbying",
                          I if c["min_gap"] is not None else 0),
                         (c["n_gap_30"], I),
                         f"{c['yr_first']}-{c['yr_last']}" if c["yr_first"] else ""])

    alignment = [["Company", "Story names department", "Period", "Amount claimed",
                  "Our contracts w/ that dept", "In that period", "of those no-bid",
                  "Did they lobby it?"]]
    alignment += [[c["name"], st["dept"], st["when"] or "-", st["amount"] or "-",
                   (st["n_dept"], I), (st["n_win"], I), (st["nc_win"], I), st["lobbied_dept"]]
                  for c in hits["cases"] for st in c["stories"]]
    alignment += [[], [("Only two rows align: Accenture / Export Development Canada and "
                        "GC Strategies / PSPC. The rest name provincial bodies, private "
                        "utilities, or are too vague to match.", N)]]

    top = [["Rank", "Score", "Company", "Award date", "Method", "Buyer(s)",
            "Linked registrations", "Nearest gap (days)", "Award ID"]]
    top += [[(m["rank"], I), (m["score"], D), m["name"], m["award_date"],
             m["method"] or "-", m["buyers"][:60], (m["n_links"], I),
             (m["links"][0]["gap"] if m["links"] else "", I if m["links"] else 0),
             m["award_rid"][:16]] for m in v3["motifs"][:300]]

    out = xlsx.write(paths.DOCS / "favouritism_models.xlsx", [
        ("Model comparison", comparison(res, comp), [30, 12, 9, 34, 11] + [14] * len(comp) + [52]),
        ("Model specs", specs, [30, 12, 44, 34, 10, 52]),
        ("Feature importance", importance, [26, 18, 15, 46]),
        ("Ground truth holdings", holdings, [34, 11, 9, 10, 9, 13, 13, 26, 20, 12]),
        ("Story alignment", alignment, [30, 34, 11, 15, 24, 14, 15, 16]),
        ("Top 300 motifs", top, [7, 9, 32, 12, 10, 44, 18, 16, 18]),
    ])
    print("wrote", out)


if __name__ == "__main__":
    main()
