#!/usr/bin/env python3
"""Render hits.json -> compact facts-only LaTeX."""
import json, os
from pipeline.fav13 import config as C

def main():
    D = json.load(open(os.path.join(C.WORK, "hits.json")))


    def esc(s):
        s = str(s if s is not None else "")
        for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                     ("}", r"\}"), ("~", r"\textasciitilde{}"),
                     ("^", r"\textasciicircum{}"), ("’", "'"), ("‘", "'"),
                     ("“", "``"), ("”", "''"), ("—", "--"), ("–", "-")]:
            s = s.replace(a, b)
        return s


    def tc(s, n):
        s = str(s or "")
        return s[:n] + (".." if len(s) > n else "")


    L = []
    w = L.append
    w(r"""\documentclass[10pt,a4paper]{article}
    \usepackage[margin=1.8cm,top=1.7cm,bottom=1.8cm]{geometry}
    \usepackage{mathpazo}\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
    \usepackage{booktabs,xcolor,array,titlesec}
    \definecolor{ink}{HTML}{16211E}\definecolor{pine}{HTML}{1E5B4F}
    \definecolor{slate}{HTML}{54655F}\definecolor{mulberry}{HTML}{8C3A56}
    \color{ink}\setlength{\parindent}{0pt}\setlength{\parskip}{0.32em}
    \renewcommand{\arraystretch}{1.12}
    \titleformat{\section}{\bfseries\color{pine}}{}{0em}{}
    \titlespacing*{\section}{0pt}{1.1em}{0.35em}
    \newcommand{\lbl}[1]{{\color{slate}\ttfamily\scriptsize #1}}
    \begin{document}
    \begin{center}
    {\color{slate}\ttfamily\scriptsize IJF $\cdot$ FAVOURITISM $\cdot$ GROUND-TRUTH HITS}\\[0.4em]
    {\Large\bfseries Matched Records and Aligned Statistics}\\[0.35em]
    {\color{slate}\small What our databases hold for each company named in a published
    favouritism story.}
    \end{center}
    \vspace{0.3em}""")

    w(r"""\begin{center}\small
    \begin{tabular}{@{}lrrrrrl@{}}
    \toprule
    \textbf{Company} & \textbf{Contracts} & \textbf{No-bid} & \textbf{Buyers} &
    \textbf{Regs} & \textbf{Depts} & \textbf{Contract years}\\
    \midrule""")
    for c in D["cases"]:
        w(f"{esc(tc(c['name'],31))} & {c['n_contracts']:,} & "
          f"{c['n_nc']:,} ({c['nc_share']*100:.0f}\\%) & {c['n_buyers']:,} & "
          f"{c['n_regs']:,} & {c['n_bodies']:,} & "
          f"{esc(c['yr_first'])}--{esc(c['yr_last']) if c['yr_last'] else ''}\\\\")
    w(r"\bottomrule\end{tabular}\end{center}")

    for c in D["cases"]:
        w(r"\section*{" + esc(c["name"]) + "}")
        w(r"\lbl{" + esc(c["entity"]) + r"}\quad{\small " +
          esc(f"{c['n_variants']} name variant" + ("s" if c["n_variants"] != 1 else "")) +
          ": " + esc(tc("; ".join(c["variants"]), 150)) + r"}\par")

        w(r"{\small\begin{tabular}{@{}p{3.6cm}p{12.6cm}@{}}")
        w(r"contracts held & " + f"{c['n_contracts']:,}" +
          (f", of which {c['n_nc']:,} ({c['nc_share']*100:.1f}\\%) non-competitive"
           if c["n_contracts"] else "") +
          (f"; {c['n_buyers']:,} distinct departments; {esc(c['yr_first'])}--{esc(c['yr_last'])}"
           if c["n_contracts"] else "") + r"\\")
        if c["n_regs"]:
            w(r"lobbying registrations & " + f"{c['n_regs']:,}" +
              esc(f" ({c['reg_first']} to {c['reg_last']}), {c['n_bodies']} departments lobbied") + r"\\")
            if c["min_gap"] is not None:
                w(r"closest lobby-to-award & " + f"{c['min_gap']} days" +
                  (f"; {c['n_gap_30']:,} contracts within 30 days of a registration"
                   if c["n_gap_30"] else "") + r"\\")
        else:
            w(r"lobbying registrations & \textbf{\color{mulberry}none}\\")
        w(r"\end{tabular}}\par\vspace{0.35em}")

        w(r"{\small\begin{tabular}{@{}p{5.0cm}p{1.5cm}p{1.8cm}rrrc@{}}")
        w(r"\toprule")
        w(r"\textbf{Story names department} & \textbf{Period} & \textbf{Amount} & "
          r"\textbf{Ours} & \textbf{In period} & \textbf{No-bid} & \textbf{Lobbied?}\\")
        w(r"\midrule")
        for s in c["stories"]:
            w(f"{esc(tc(s['dept'],40))} & {esc(tc(s['when'],10)) or '--'} & "
              f"{esc(tc(s['amount'],12)) or '--'} & {s['n_dept']:,} & {s['n_win']:,} & "
              f"{s['nc_win']:,} & {esc(s['lobbied_dept'])}\\\\")
        w(r"\bottomrule\end{tabular}}\par")

        ex = [x for s in c["stories"] for x in s["examples"]]
        seen, uniq = set(), []
        for x in ex:
            k = (x["date"], x["buyer"], x["method"])
            if k not in seen:
                seen.add(k)
                uniq.append(x)
        if uniq:
            w(r"\vspace{0.3em}{\small\lbl{MATCHING CONTRACTS}\par")
            w(r"\begin{tabular}{@{}llp{8.4cm}r@{}}")
            for x in uniq[:5]:
                g = f"{x['gap']} d after lobbying" if x['gap'] is not None else "--"
                w(esc(x['date']) + r" & \texttt{\footnotesize " + esc(x['method']) +
                  "} & " + esc(tc(x['buyer'], 56)) + " & " + g + r"\\")
            w(r"\end{tabular}}\par")

    w(r"\end{document}")

    out = os.path.join(C.REPO, "docs", "hits.tex")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
