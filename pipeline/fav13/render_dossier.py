#!/usr/bin/env python3
"""Render dossier.json to LaTeX."""
import json, os, re
from pipeline.fav13 import config as C

def main():
    D = json.load(open(os.path.join(C.WORK, "dossier.json")))


    def esc(s):
        s = str(s or "")
        for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
                     ("$", r"\$"), ("#", r"\#"), ("_", r"\_"), ("{", r"\{"),
                     ("}", r"\}"), ("~", r"\textasciitilde{}"),
                     ("^", r"\textasciicircum{}"), ("’", "'"), ("‘", "'"),
                     ("“", "``"), ("”", "''"), ("—", "---"), ("–", "--")]:
            s = s.replace(a, b)
        return s


    def tc(s, n):
        s = str(s or "")
        return s[:n] + ("..." if len(s) > n else "")


    def norm(s):
        return re.sub(r"[^a-z ]", " ", (s or "").lower()).split()


    def body_match(story_body, our_buyers):
        a, b = set(norm(story_body)), set(norm(our_buyers))
        a -= {"of", "the", "and", "canada", "government"}
        b -= {"of", "the", "and", "canada", "government"}
        return bool(a & b)


    L = []
    w = L.append

    w(r"""\documentclass[11pt,a4paper]{article}
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
    \vspace{0.4em}""")

    w(r"\section*{Summary}")
    w(f"""Of {D['gt_companies']} companies named across {D['gt_rows']} story records
    ({D['gt_pairs']} distinct company--department pairs), {len(D['cases'])} resolve to an
    entity in our data. Below, ``rank'' is the position of that company's best contract
    among {D['total_scored']:,} scored contracts.\\par""")

    w(r"""\begin{center}\small
    \begin{tabular}{@{}lrrrr@{}}
    \toprule
    \textbf{Company} & \textbf{Contracts} & \textbf{Registrations} & \textbf{Best rank} & \textbf{Top 1\%?}\\
    \midrule""")
    for c in D["cases"]:
        br = c["contracts"][0]["rank"] if c["contracts"] else None
        pct = c["contracts"][0]["pct"] if c["contracts"] else None
        top = ("yes" if pct <= 1.0 else "no") if pct is not None else "---"
        w(f"{esc(tc(c['name'],34))} & {c['n_contracts']:,} & {c['n_registrations']:,} & "
          f"{(f'{br:,}' if br else 'not scored')} & {top}\\\\")
    w(r"""\bottomrule\end{tabular}\end{center}""")

    w(r"""\vspace{0.2em}
    \noindent\fbox{\parbox{0.97\linewidth}{\small
    \textbf{\color{mulberry}The most important finding in this document.}
    Every match below is a \emph{company-level} match. In no case does the contract our
    pipeline surfaced correspond to the contract the story is actually about. Accenture's
    stories concern Metrolinx and Export Development Canada; we surfaced an Employment and
    Social Development Canada award. ATCO's story concerns a provincial utility regulator and
    is not federal procurement at all. Under the agreed definition these are
    \emph{partial hits}; \textbf{we have zero exact hits.}}}""")

    w(r"""\section*{Newsworthiness categories}
    \begin{center}\small
    \begin{tabular}{@{}lp{10.6cm}@{}}
    \toprule
    \textbf{Category} & \textbf{Fires when} \\
    \midrule
    No open competition   & method is \texttt{NON}, \texttt{LIMITED} or \texttt{ADVANCE}; strength depends on how rarely \emph{that} department does it \\
    Lobbied, then awarded & the company lobbied \emph{this contract's own department} within a year before the award \\
    Supplier dominance    & holds more of that department's contracts than an even split across its suppliers predicts \\
    Short tender window   & the tender was open 15 days or fewer \\
    \bottomrule\end{tabular}\end{center}""")

    for c in D["cases"]:
        w(r"\clearpage")
        w(r"\section*{" + esc(c["name"]) + "}")
        w(r"\lbl{ENTITY " + esc(c["entity"]) + r"}\quad\lbl{" +
          f"{c['n_members']} name variant" + ("s" if c["n_members"] != 1 else "") + r"}\par")

        w(r"\vspace{0.3em}\noindent\textbf{Records matched in our databases}\par")
        w(r"{\small\begin{tabular}{@{}p{3.1cm}p{13.0cm}@{}}")
        w(r"as supplier & " + esc("; ".join(c["supplier_variants"])) + r"\\")
        if c["lobby_variants"]:
            w(r"as lobbying client & " + esc("; ".join(c["lobby_variants"])) + r"\\")
        w(r"contracts & " + f"{c['n_contracts']:,}" + r"\\")
        if c["n_registrations"]:
            w(r"registrations & " + f"{c['n_registrations']:,} " +
              esc(f"({c['reg_first']} to {c['reg_last']})") + r"\\")
            w(r"departments lobbied & " + esc(tc("; ".join(c["bodies"]), 240)) + r"\\")
        else:
            w(r"registrations & " + r"\textbf{\color{mulberry}none --- never registered as a lobbying client}\\")
        w(r"\end{tabular}}\par")

        w(r"\vspace{0.6em}\noindent\textbf{What the published stories claim}\par")
        for s in c["stories"][:3]:
            bm = body_match(s["body"], c["contracts"][0]["buyers"] if c["contracts"] else "")
            w(r"{\small\textit{" + esc(tc(s["title"], 108)) + r"}\\")
            w(r"\lbl{" + esc(f"{s['outlet']} {s['date']}") + r"}\quad " +
              esc(f"{tc(s['company'],28)} -> {tc(s['body'],34)}") +
              (r"\quad {\color{slate}" + esc(f"amount {s['amount']}") + "}" if s["amount"] else "") +
              (r"\quad {\color{slate}" + esc(s["when"]) + "}" if s["when"] else "") +
              (r"\quad{\color{pine}\small [department matches ours]}" if bm
               else r"\quad{\color{mulberry}\small [different department to what we surfaced]}") +
              r"}\par\vspace{0.25em}")

        if not c["contracts"]:
            w(r"\vspace{0.5em}\noindent\fbox{\parbox{0.97\linewidth}{\small "
              r"\textbf{\color{mulberry}Not scored.} " +
              ("This company appears as a lobbying client but has no contracts in the "
               "procurement corpus, so there is nothing to rank."
               if c["n_registrations"] else
               "No usable records on either side.") + r"}}")
            continue

        w(r"\vspace{0.6em}\noindent\textbf{Top-ranked contracts we surfaced}\par")
        for k in c["contracts"][:3]:
            w(r"\vspace{0.35em}{\color{rulec}\rule{\linewidth}{0.5pt}}\par")
            w(r"{\small\textbf{rank " + f"{k['rank']:,}" + r"} \lbl{(top " +
              f"{k['pct']:.2f}" + r"\%)}\quad " + esc(k["date"]) +
              r"\quad \lbl{award " + esc(k["award_rid"]) + r"}\quad method " +
              r"\texttt{" + esc(k["method"]) + r"}}\par")
            w(r"{\small\lbl{BUYER} " + esc(tc(k["buyers"], 90)) + r"}\par")
            if k["channel"]:
                w(r"{\small\lbl{CHANNEL} lobbied " + esc(tc(k["channel"], 60)) +
                  f", nearest registration {k['gap']} days before "
                  f"({k['n_chan_regs']} registrations on this channel)" + r"}\par")
            w(r"\vspace{0.2em}")
            for t, why in k["cats"]:
                w(r"\cat{" + esc(t) + "}{" + esc(why) + "}")
            w(r"{\small\lbl{P-VALUES}\ " +
              f"time {k['p_time']:.3g} $\\cdot$ competition {k['p_comp']:.3g} $\\cdot$ "
              f"tender {k['p_proc']:.3g} $\\cdot$ dominance {k['p_conc']:.3g}"
              f" $\\cdot$ identity gate {k['pi']:.2f}" + r"}\par")

    w(r"""\clearpage
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

    \end{document}""")

    out = os.path.join(C.REPO, "docs", "ground_truth_cases.tex")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
