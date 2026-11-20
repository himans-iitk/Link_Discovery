"""work/dossier.json -> docs/ground_truth_cases.tex.

The closing findings are hand-written from the measured run, not generated.

    python3 -m pipeline.fav13.report.render_dossier
"""

import json
import re

from pipeline import paths
from .latex import esc, plural
from .latex import trunc as tc

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[margin=2.1cm,top=2.0cm,bottom=2.2cm]{geometry}
\usepackage{mathpazo}\usepackage[T1]{fontenc}\usepackage[utf8]{inputenc}
\usepackage{amsmath,booktabs,xcolor,titlesec,enumitem,array,longtable}
\usepackage[hidelinks]{hyperref}
\definecolor{ink}{HTML}{16211E}\definecolor{pine}{HTML}{1E5B4F}
\definecolor{mulberry}{HTML}{8C3A56}\definecolor{slate}{HTML}{54655F}
\definecolor{rulec}{HTML}{C6D2CD}\definecolor{boxbg}{HTML}{F4F7F6}
\definecolor{amber}{HTML}{8A6A1F}
\color{ink}\setlength{\parindent}{0pt}\setlength{\parskip}{0.4em}
\titleformat{\section}{\large\bfseries\color{pine}}{}{0em}{}
\titlespacing*{\section}{0pt}{1.4em}{0.5em}
\newcommand{\cat}[2]{\textbf{\color{pine}#1}\ {\color{slate}\small #2}\par}
\newcommand{\lbl}[1]{{\color{slate}\ttfamily\footnotesize #1}}
\begin{document}
\begin{center}
{\color{slate}\ttfamily\footnotesize IJF $\cdot$ FAVOURITISM $\cdot$ GROUND-TRUTH EVIDENCE}\\[0.5em]
{\LARGE\bfseries Ground-Truth Cases: What We Matched}\\[0.5em]
{\color{slate}\small The records in our databases that correspond to published
favouritism stories, and which newsworthiness signals fired.}
\end{center}
\vspace{0.4em}"""

KEY_FINDING = r"""\vspace{0.2em}
\noindent\fbox{\parbox{0.97\linewidth}{\small
\textbf{\color{mulberry}The most important finding in this document.}
Every match below is a \emph{company-level} match. In no case does the contract our
pipeline surfaced correspond to the contract the story is actually about. Accenture's
stories concern Metrolinx and Export Development Canada; we surfaced an Employment and
Social Development Canada award. ATCO's story concerns a provincial utility regulator and
is not federal procurement at all. Under the agreed definition these are
\emph{partial hits}; \textbf{we have zero exact hits.}}}"""

CATEGORIES = r"""\section*{Newsworthiness categories}
\begin{center}\small
\begin{tabular}{@{}lp{10.6cm}@{}}
\toprule
\textbf{Category} & \textbf{Fires when} \\
\midrule
No open competition   & method is \texttt{NON}, \texttt{LIMITED} or \texttt{ADVANCE}; strength depends on how rarely \emph{that} department does it \\
Lobbied, then awarded & the company lobbied \emph{this contract's own department} within a year before the award \\
Supplier dominance    & holds more of that department's contracts than an even split across its suppliers predicts \\
Short tender window   & the tender was open 15 days or fewer \\
\bottomrule\end{tabular}\end{center}"""

CONCLUSIONS = r"""\clearpage
\section*{What this tells us about the evaluation}

\textbf{1. The AUC table overstates our performance.} Every match is company-level. A
reporter handed the Accenture contract we surfaced would not be reading the Metrolinx or
CEBA story --- they would be looking at a different award entirely. The ranking finds the
right \emph{companies}; it has not yet been shown to find the right \emph{contracts}.

\textbf{2. Some ground-truth cases are not federal procurement at all.} The ATCO story
concerns a provincial utility regulator. It can never be an exact hit, because the event
it describes is not in our corpus. Cases like this should be removed from the evaluation
set rather than counted as misses.

\textbf{3. The strongest individual finding is real and was surfaced by the pipeline.}
Accenture lobbied Employment and Social Development Canada and won a sole-source contract
from that department \textbf{one day later}, with a dominance $p$-value of $10^{-12}$.
That is a lead worth checking, and it is not one of the published stories --- it is new.

\textbf{4. Duplicate award records inflate the ranking.} Accenture's ranks 77{,}255 and
77{,}256 are the same date, same department, same method, under different award
identifiers. De-duplication is needed before any top-$k$ is handed to a reporter.

\textbf{5. GC Strategies confirms the structural blind spot.} 361 contracts, zero
lobbying registrations. Any signal requiring a lobbying channel cannot reach it.

\end{document}"""

_GENERIC = {"of", "the", "and", "canada", "government"}


def body_match(story_body, our_buyers):
    words = lambda s: set(re.sub(r"[^a-z ]", " ", (s or "").lower()).split()) - _GENERIC
    return bool(words(story_body) & words(our_buyers))


def summary(D):
    L = [r"\section*{Summary}",
         f"""Of {D['gt_companies']} companies named across {D['gt_rows']} story records
({D['gt_pairs']} distinct company--department pairs), {len(D['cases'])} resolve to an
entity in our data. Below, ``rank'' is the position of that company's best contract
among {D['total_scored']:,} scored contracts.\\par""",
         r"""\begin{center}\small
\begin{tabular}{@{}lrrrr@{}}
\toprule
\textbf{Company} & \textbf{Contracts} & \textbf{Registrations} & \textbf{Best rank} & \textbf{Top 1\%?}\\
\midrule"""]
    for c in D["cases"]:
        top = c["contracts"][0] if c["contracts"] else None
        best = f"{top['rank']:,}" if top else "not scored"
        in_top = ("yes" if top["pct"] <= 1.0 else "no") if top else "---"
        L.append(f"{esc(tc(c['name'],34))} & {c['n_contracts']:,} & "
                 f"{c['n_registrations']:,} & {best} & {in_top}\\\\")
    L.append(r"""\bottomrule\end{tabular}\end{center}""")
    return L


def case_section(c):
    L = [r"\clearpage", r"\section*{" + esc(c["name"]) + "}",
         r"\lbl{ENTITY " + esc(c["entity"]) + r"}\quad\lbl{" +
         plural(c["n_members"], "name variant") + r"}\par",
         r"\vspace{0.3em}\noindent\textbf{Records matched in our databases}\par",
         r"{\small\begin{tabular}{@{}p{3.1cm}p{13.0cm}@{}}",
         r"as supplier & " + esc("; ".join(c["supplier_variants"])) + r"\\"]
    if c["lobby_variants"]:
        L.append(r"as lobbying client & " + esc("; ".join(c["lobby_variants"])) + r"\\")
    L.append(r"contracts & " + f"{c['n_contracts']:,}" + r"\\")
    if c["n_registrations"]:
        L.append(r"registrations & " + f"{c['n_registrations']:,} " +
                 esc(f"({c['reg_first']} to {c['reg_last']})") + r"\\")
        L.append(r"departments lobbied & " + esc(tc("; ".join(c["bodies"]), 240)) + r"\\")
    else:
        L.append(r"registrations & " + r"\textbf{\color{mulberry}none --- never registered as a lobbying client}\\")
    L.append(r"\end{tabular}}\par")

    L.append(r"\vspace{0.6em}\noindent\textbf{What the published stories claim}\par")
    ours = c["contracts"][0]["buyers"] if c["contracts"] else ""
    for s in c["stories"][:3]:
        L.append(r"{\small\textit{" + esc(tc(s["title"], 108)) + r"}\\")
        L.append(r"\lbl{" + esc(f"{s['outlet']} {s['date']}") + r"}\quad " +
                 esc(f"{tc(s['company'],28)} -> {tc(s['body'],34)}") +
                 (r"\quad {\color{slate}" + esc(f"amount {s['amount']}") + "}" if s["amount"] else "") +
                 (r"\quad {\color{slate}" + esc(s["when"]) + "}" if s["when"] else "") +
                 (r"\quad{\color{pine}\small [department matches ours]}" if body_match(s["body"], ours)
                  else r"\quad{\color{mulberry}\small [different department to what we surfaced]}") +
                 r"}\par\vspace{0.25em}")

    if not c["contracts"]:
        L.append(r"\vspace{0.5em}\noindent\fbox{\parbox{0.97\linewidth}{\small "
                 r"\textbf{\color{mulberry}Not scored.} " +
                 ("This company appears as a lobbying client but has no contracts in the "
                  "procurement corpus, so there is nothing to rank."
                  if c["n_registrations"] else "No usable records on either side.") + r"}}")
        return L

    L.append(r"\vspace{0.6em}\noindent\textbf{Top-ranked contracts we surfaced}\par")
    for k in c["contracts"][:3]:
        L.append(r"\vspace{0.35em}{\color{rulec}\rule{\linewidth}{0.5pt}}\par")
        L.append(r"{\small\textbf{rank " + f"{k['rank']:,}" + r"} \lbl{(top " +
                 f"{k['pct']:.2f}" + r"\%)}\quad " + esc(k["date"]) +
                 r"\quad \lbl{award " + esc(k["award_rid"]) + r"}\quad method " +
                 r"\texttt{" + esc(k["method"]) + r"}}\par")
        L.append(r"{\small\lbl{BUYER} " + esc(tc(k["buyers"], 90)) + r"}\par")
        if k["channel"]:
            L.append(r"{\small\lbl{CHANNEL} lobbied " + esc(tc(k["channel"], 60)) +
                     f", nearest registration {k['gap']} days before "
                     f"({k['n_chan_regs']} registrations on this channel)" + r"}\par")
        L.append(r"\vspace{0.2em}")
        for t, why in k["cats"]:
            L.append(r"\cat{" + esc(t) + "}{" + esc(why) + "}")
        L.append(r"{\small\lbl{P-VALUES}\ " +
                 f"time {k['p_time']:.3g} $\\cdot$ competition {k['p_comp']:.3g} $\\cdot$ "
                 f"tender {k['p_proc']:.3g} $\\cdot$ dominance {k['p_conc']:.3g}"
                 f" $\\cdot$ identity gate {k['pi']:.2f}" + r"}\par")
    return L


def main():
    with open(paths.WORK / "dossier.json") as fh:
        D = json.load(fh)
    L = [PREAMBLE] + summary(D) + [KEY_FINDING, CATEGORIES]
    for c in D["cases"]:
        L += case_section(c)
    L.append(CONCLUSIONS)

    out = paths.DOCS / "ground_truth_cases.tex"
    out.write_text("\n".join(L), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
