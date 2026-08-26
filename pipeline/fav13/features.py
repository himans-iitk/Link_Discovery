"""
Features, and the null model they are measured against.

The central feature is *temporal excess*: how many contracts arrived just
after a lobbying registration, compared with how many would have arrived
there by coincidence.  Everything else is supporting evidence.

Why a null at all.  Raw closeness is a volume statistic.  A company with
17,997 contracts accumulates close pairs by arithmetic, not by influence --
in the first prototype, ranking on the raw count correlated with sheer
activity at rho = +0.74.  Subtracting the null inverts that to rho = -0.44.

The null, stated in one sentence: if this company had the same *number* of
registrations but their dates were drawn from the national pool of all
lobbying start dates, how many of its contracts would still fall within the
window?  Closed form, no simulation, exactly reproducible.
"""
import bisect, csv, math, os, re
from . import config as C

csv.field_size_limit(10 ** 9)

_STOP = r"\b(THE|OF|AND|FOR|DE|DU|DES|LA|LE|LES|ET|A|INC|LTD|LTEE|LIMITED|CORP|CORPORATION|CO|COMPANY|ASSOCIATION)\b"
_ACADEMIC = re.compile(
    r"\b(UNIVERSIT|COLLEGE|POLYTECHNIQ|INSTITUT|SCHOOL|ACADEM|CEGEP|SEMINAR)", re.I)


def _charity_key(s):
    s = re.sub(r"[^A-Z0-9 ]", " ", (s or "").upper())
    s = re.sub(_STOP, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_charity_keys():
    """Registered charities, from the CRA registry already in the clean layer."""
    keys = set()
    path = os.path.join(C.CLEAN, "cha_meta.csv")
    with open(path, newline="", encoding="utf-8", errors="replace") as fh:
        r = csv.reader(fh)
        hdr = next(r)
        i = hdr.index("name_core")
        for x in r:
            if len(x) > i and x[i]:
                keys.add(_charity_key(x[i]))
    keys.discard("")
    return keys


def grant_likelihood(entity, members, charity_keys):
    """How likely is this supplier receiving grants rather than winning tenders?

    Returns a value in [0,1] with a stateable reason attached -- a reporter can
    be told "this supplier is on the charity registry", which is checkable.
    """
    names = members.get(entity, [entity])
    if charity_keys and any(_charity_key(n) in charity_keys for n in names):
        return C.GRANT_WEIGHT_CHARITY, "registered_charity"
    if any(_ACADEMIC.search(n) for n in names):
        return C.GRANT_WEIGHT_ACADEMIC, "academic"
    return 0.0, ""


# ------------------------------------------------------------------ null model

def expected_close(award_days, n_regs, reference, window):
    """Expected number of awards preceded within `window` days by at least one
    of `n_regs` registrations drawn from `reference`.

    Each award is an independent Bernoulli: p_a = 1 - (1 - q_a)^n_regs, where
    q_a is the share of the reference population landing in [a-window, a).
    The total is Poisson-binomial; its mean is the sum of p_a.
    """
    N = len(reference)
    if N == 0 or n_regs == 0:
        return 0.0
    total = 0.0
    for a in award_days:
        lo = bisect.bisect_left(reference, a - window)
        hi = bisect.bisect_left(reference, a)
        q = (hi - lo) / N
        total += 1.0 - (1.0 - q) ** n_regs
    return total


def observed_close(award_days, reg_days, window):
    """Awards with at least one registration starting in the preceding window."""
    n = 0
    for a in award_days:
        lo = bisect.bisect_left(reg_days, a - window)
        hi = bisect.bisect_left(reg_days, a)
        if hi > lo:
            n += 1
    return n


_LN10 = math.log(10)


def neg_log10_tail(k, lam):
    """-log10 P(X >= k) for X ~ Poisson(lam), stable far into the tail.

    The naive survival sum underflows to 0 around 1e-300 and flattens the top
    of the ranking; the far tail is computed from log P(X=k) with a geometric
    correction instead.
    """
    if lam <= 0:
        return 300.0 if k > 0 else 0.0
    if k <= lam:
        s, t = 0.0, math.exp(-lam)
        for i in range(k):
            s += t
            t *= lam / (i + 1)
        return -math.log10(max(1e-300, 1.0 - s))
    lp = k * math.log(lam) - lam - math.lgamma(k + 1)
    r = lam / (k + 1)
    if r < 1:
        lp += math.log(1.0 / (1.0 - r))
    return -lp / _LN10


# -------------------------------------------------------------------- features

def _award_units(rec, unit):
    """Sector control.  IT and staffing firms receive bursts of task
    authorisations on one day; counting each separately lets a single nearby
    registration capture the whole burst.  Collapsing to distinct dates
    removes that."""
    days = rec["award_days"]
    return sorted(set(days)) if unit == "date" else days


def identity_confidence(entity, rec, members):
    """How sure are we that these records are one legal entity?

    Not a learned probability -- an interpretable ordering built from what we
    know about how the entity was formed:

      1.0  a single preprocessed name, no merge decision was ever made
      0.8  a manual cluster kept whole
      0.6  a manual cluster that we split apart on string evidence
      -    reduced when the entity rests on very few source records

    This is deliberately conservative and deliberately legible; a journalist
    can be told which of the four applies.
    """
    names = members.get(entity, [])
    if len(names) <= 1:
        base = 1.0
    elif "." in str(entity):          # produced by a legal-entity split
        base = 0.6
    else:
        base = 0.8
    support = min(len(rec["regs"]), 5) / 5.0
    return base * (0.6 + 0.4 * support)


def concentration(rec):
    """Overlap between the bodies this entity lobbied and the bodies that
    bought from it -- how tightly the loop closes.

    This is FAV 12 done by string match on the normalised body name, which is
    a LOWER BOUND, not a placeholder: it already resolves the head of the
    distribution (16 of the top 18 lobbied bodies match a procurement buyer)
    but misses renames and conventions -- `public works government services`
    against `public services procurement canada`, `national defence` against
    `department national defence`.

    Government bodies are a closed vocabulary: 684 names cover 80% of all
    lobbying records, and ~286 hand-written mappings would cover the ones that
    matter.  Until those exist, entities whose bodies are named
    inconsistently across the two registries score 0 here and are ranked
    conservatively -- the error direction is missed cases, not fabricated ones.
    """
    L, A = rec["bodies"], rec["buyers"]
    if not L or not A:
        return 0.0
    inter = len(L & A)
    return inter / min(len(L), len(A)) if inter else 0.0


def noncompetitive(rec):
    tot = sum(rec["methods"].values())
    if not tot:
        return 0.0
    nc = sum(v for k, v in rec["methods"].items()
             if k in C.NONCOMPETITIVE_METHODS)
    return nc / tot


def compute(ev, dual, reference, members, window=None, unit=None, min_obs=None,
            charity_keys=None):
    """-> list of per-entity feature records (unnormalised)."""
    window = window or C.PROXIMITY_WINDOW_DAYS
    unit = unit or C.AWARD_UNIT
    min_obs = C.MIN_OBS if min_obs is None else min_obs
    charity_keys = charity_keys if charity_keys is not None else set()

    out = []
    for e in dual:
        rec = ev[e]
        awards = _award_units(rec, unit)
        regs = rec["reg_days"]
        obs = observed_close(awards, regs, window)
        if obs < min_obs:
            continue
        exp = expected_close(awards, len(regs), reference, window)
        gl, why = grant_likelihood(e, members, charity_keys)
        out.append(dict(
            entity=e,
            obs=obs,
            expected=exp,
            ratio=(obs / exp) if exp > 0 else float("inf"),
            nlp=neg_log10_tail(obs, exp),
            n_regs=len(regs),
            n_awards=len(awards),
            n_awards_raw=len(rec["award_days"]),
            volume=len(regs) + len(awards),
            noncompetitive=noncompetitive(rec),
            grant_likelihood=gl, grant_reason=why, commercial=1.0 - gl,
            identity_confidence=identity_confidence(e, rec, members),
            concentration=concentration(rec),
            first_reg=min(regs), last_reg=max(regs),
            first_award=min(awards), last_award=max(awards),
            n_bodies=len(rec["bodies"]), n_buyers=len(rec["buyers"]),
        ))
    return out


# ------------------------------------------------------- normalisation to [0,1]

FEATURES = ("temporal_excess", "commercial", "noncompetitive",
            "identity_confidence", "concentration")


def _percentile(recs, key):
    """Rank-normalise a feature into [0,1], ties sharing the mean rank.

    Min-max was wrong here and wrong in a way that quietly inverted the
    ranking.  The tail statistic is extremely heavy-tailed -- one entity had
    -log10 p of 88 while the median was under 2 -- so dividing by the maximum
    pushed every other entity's temporal feature to nearly zero, and the
    near-constant supporting features took over the top of the list.  Rank
    normalisation is monotone (so the threshold algorithm still applies) and
    is insensitive to the outlier.
    """
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


def normalise(recs):
    """Map each scoring feature into [0,1], monotone increasing.

    Required by the aggregation contract: non-negative weights over [0,1]
    features give a monotone aggregate, which is what lets the threshold
    algorithm terminate early and provably.
    """
    if not recs:
        return recs
    big = max((r["nlp"] for r in recs if math.isfinite(r["nlp"])), default=1.0)
    for r in recs:
        if not math.isfinite(r["nlp"]):
            r["nlp"] = big
    for name, key in (("temporal_excess", "nlp"),
                      ("commercial", "commercial"),
                      ("noncompetitive", "noncompetitive"),
                      ("identity_confidence", "identity_confidence"),
                      ("concentration", "concentration")):
        vals = _percentile(recs, key)
        for r, v in zip(recs, vals):
            r["f_" + name] = v
    return recs
