"""work/hits.json -> docs/hits.tex (facts only).

    python3 -m pipeline.fav13.report.render_hits
"""

import json

from pipeline import paths
from .latex import plural
from .latex import esc as _esc
from .latex import trunc as _trunc

PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
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
\vspace{0.3em}"""


def esc(s):
    return _esc(s, em="--", en="-")


def tc(s, n):
    return _trunc(s, n, "..")


def summary_table(cases):
    L = [r"""\begin{center}\small
\begin{tabular}{@{}lrrrrrl@{}}
\toprule
\textbf{Company} & \textbf{Contracts} & \textbf{No-bid} & \textbf{Buyers} &
\textbf{Regs} & \textbf{Depts} & \textbf{Contract years}\\
\midrule"""]
    for c in cases:
        L.append(f"{esc(tc(c['name'],31))} & {c['n_contracts']:,} & "
                 f"{c['n_nc']:,} ({c['nc_share']*100:.0f}\\%) & {c['n_buyers']:,} & "
                 f"{c['n_regs']:,} & {c['n_bodies']:,} & "
                 f"{esc(c['yr_first'])}--{esc(c['yr_last']) if c['yr_last'] else ''}\\\\")
    L.append(r"\bottomrule\end{tabular}\end{center}")
    return L


def case_section(c):
    L = [r"\section*{" + esc(c["name"]) + "}",
         r"\lbl{" + esc(c["entity"]) + r"}\quad{\small " +
         esc(plural(c["n_variants"], "name variant")) +
         ": " + esc(tc("; ".join(c["variants"]), 150)) + r"}\par"]

    L.append(r"{\small\begin{tabular}{@{}p{3.6cm}p{12.6cm}@{}}")
    held = f"{c['n_contracts']:,}"
    if c["n_contracts"]:
        held += (f", of which {c['n_nc']:,} ({c['nc_share']*100:.1f}\\%) non-competitive"
                 f"; {c['n_buyers']:,} distinct departments; "
                 f"{esc(c['yr_first'])}--{esc(c['yr_last'])}")
    L.append(r"contracts held & " + held + r"\\")
    if c["n_regs"]:
        L.append(r"lobbying registrations & " + f"{c['n_regs']:,}" +
                 esc(f" ({c['reg_first']} to {c['reg_last']}), {c['n_bodies']} departments lobbied") + r"\\")
        if c["min_gap"] is not None:
            L.append(r"closest lobby-to-award & " + f"{c['min_gap']} days" +
                     (f"; {c['n_gap_30']:,} contracts within 30 days of a registration"
                      if c["n_gap_30"] else "") + r"\\")
    else:
        L.append(r"lobbying registrations & \textbf{\color{mulberry}none}\\")
    L.append(r"\end{tabular}}\par\vspace{0.35em}")

    L += [r"{\small\begin{tabular}{@{}p{5.0cm}p{1.5cm}p{1.8cm}rrrc@{}}", r"\toprule",
          r"\textbf{Story names department} & \textbf{Period} & \textbf{Amount} & "
          r"\textbf{Ours} & \textbf{In period} & \textbf{No-bid} & \textbf{Lobbied?}\\",
          r"\midrule"]
    for s in c["stories"]:
        L.append(f"{esc(tc(s['dept'],40))} & {esc(tc(s['when'],10)) or '--'} & "
                 f"{esc(tc(s['amount'],12)) or '--'} & {s['n_dept']:,} & {s['n_win']:,} & "
                 f"{s['nc_win']:,} & {esc(s['lobbied_dept'])}\\\\")
    L.append(r"\bottomrule\end{tabular}}\par")

    seen, examples = set(), []
    for x in (x for s in c["stories"] for x in s["examples"]):
        k = (x["date"], x["buyer"], x["method"])
        if k not in seen:
            seen.add(k)
            examples.append(x)
    if examples:
        L += [r"\vspace{0.3em}{\small\lbl{MATCHING CONTRACTS}\par",
              r"\begin{tabular}{@{}llp{8.4cm}r@{}}"]
        for x in examples[:5]:
            g = f"{x['gap']} d after lobbying" if x["gap"] is not None else "--"
            L.append(esc(x["date"]) + r" & \texttt{\footnotesize " + esc(x["method"]) +
                     "} & " + esc(tc(x["buyer"], 56)) + " & " + g + r"\\")
        L.append(r"\end{tabular}}\par")
    return L


def main():
    with open(paths.WORK / "hits.json") as fh:
        cases = json.load(fh)["cases"]
    L = [PREAMBLE] + summary_table(cases)
    for c in cases:
        L += case_section(c)
    L.append(r"\end{document}")

    out = paths.DOCS / "hits.tex"
    out.write_text("\n".join(L), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
