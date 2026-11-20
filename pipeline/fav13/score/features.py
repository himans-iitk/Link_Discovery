"""v1 company-level features and the Poisson-binomial null.

Core feature is temporal excess: contracts that arrived just after a
registration, versus how many would by coincidence. Raw closeness is a volume
statistic (rho +0.74 with activity); subtracting the null turns it to -0.44.

Null: same number of registrations, dates drawn from the national pool of
lobbying starts. Closed form, no simulation.
"""

import bisect
import math
import re

from .. import common
from .. import config as C

_STOP = (r"\b(THE|OF|AND|FOR|DE|DU|DES|LA|LE|LES|ET|A|INC|LTD|LTEE|LIMITED|"
         r"CORP|CORPORATION|CO|COMPANY|ASSOCIATION)\b")
ACADEMIC = re.compile(
    r"\b(UNIVERSIT|COLLEGE|POLYTECHNIQ|INSTITUT|SCHOOL|ACADEM|CEGEP|SEMINAR)", re.I)


# ---------------------------------------------------------------- grant control

def charity_key(s):
    s = re.sub(r"[^A-Z0-9 ]", " ", (s or "").upper())
    s = re.sub(_STOP, " ", s)
    return re.sub(r"\s+", " ", s).strip()


def load_charity_keys():
    """Registered charity names (CRA registry, cha_meta)."""
    keys = {charity_key(n) for (n,) in common.rows("cha_meta.csv", ["name_core"]) if n}
    keys.discard("")
    return keys


def grant_likelihood(entity, members, charity_keys):
    """-> (likelihood in [0,1], checkable reason). Grants are non-competitive by design."""
    names = members.get(entity, [entity])
    if charity_keys and any(charity_key(n) in charity_keys for n in names):
        return C.GRANT_WEIGHT_CHARITY, "registered_charity"
    if any(ACADEMIC.search(n) for n in names):
        return C.GRANT_WEIGHT_ACADEMIC, "academic"
    return 0.0, ""


# ---------------------------------------------------------------- identity

def identity_base(entity, members):
    """1.0 single name (no merge) / 0.8 manual cluster kept whole /
    0.6 cluster split on string evidence (id contains '.')."""
    if len(members.get(entity, [])) <= 1:
        return 1.0
    return 0.6 if "." in str(entity) else 0.8


def identity_confidence(entity, members, n_regs):
    """identity_base, discounted when the entity rests on few registrations."""
    return identity_base(entity, members) * (0.6 + 0.4 * (min(n_regs, 5) / 5.0))


# ---------------------------------------------------------------- null model

def expected_close(award_days, n_regs, reference, window):
    """Expected awards preceded within `window` by >=1 of `n_regs` registrations
    drawn from `reference`: sum over awards of 1 - (1 - q_a)^n_regs."""
    N = len(reference)
    if N == 0 or n_regs == 0:
        return 0.0
    total = 0.0
    for a in award_days:
        lo = bisect.bisect_left(reference, a - window)
        hi = bisect.bisect_left(reference, a)
        total += 1.0 - (1.0 - (hi - lo) / N) ** n_regs
    return total


def observed_close(award_days, reg_days, window):
    """Awards with at least one registration in the preceding window."""
    n = 0
    for a in award_days:
        if bisect.bisect_left(reg_days, a) > bisect.bisect_left(reg_days, a - window):
            n += 1
    return n


_LN10 = math.log(10)


def neg_log10_tail(k, lam):
    """-log10 P(X >= k), X ~ Poisson(lam). Far tail via log P(X=k) with a
    geometric correction, so it does not underflow at 1e-300."""
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


# ---------------------------------------------------------------- features

def concentration(rec):
    """Overlap of bodies lobbied and bodies buying, by exact name. A lower
    bound: renames score 0, so errors are misses, not fabrications."""
    L, A = rec["bodies"], rec["buyers"]
    if not L or not A:
        return 0.0
    inter = len(L & A)
    return inter / min(len(L), len(A)) if inter else 0.0


def noncompetitive(rec):
    tot = sum(rec["methods"].values())
    if not tot:
        return 0.0
    return sum(v for k, v in rec["methods"].items() if k in common.NONCOMP) / tot


def compute(ev, dual, reference, members, window=None, unit=None, min_obs=None,
            charity_keys=None):
    """-> per-entity feature records (unnormalised)."""
    window = window or C.PROXIMITY_WINDOW_DAYS
    unit = unit or C.AWARD_UNIT
    min_obs = C.MIN_OBS if min_obs is None else min_obs
    charity_keys = charity_keys if charity_keys is not None else set()

    out = []
    for e in dual:
        rec = ev[e]
        days = rec["award_days"]
        awards = sorted(set(days)) if unit == "date" else days
        regs = rec["reg_days"]
        obs = observed_close(awards, regs, window)
        if obs < min_obs:
            continue
        exp = expected_close(awards, len(regs), reference, window)
        gl, why = grant_likelihood(e, members, charity_keys)
        out.append(dict(
            entity=e, obs=obs, expected=exp,
            ratio=(obs / exp) if exp > 0 else float("inf"),
            nlp=neg_log10_tail(obs, exp),
            n_regs=len(regs), n_awards=len(awards), n_awards_raw=len(days),
            volume=len(regs) + len(awards),
            noncompetitive=noncompetitive(rec),
            grant_likelihood=gl, grant_reason=why, commercial=1.0 - gl,
            identity_confidence=identity_confidence(e, members, len(rec["regs"])),
            concentration=concentration(rec),
            first_reg=min(regs), last_reg=max(regs),
            first_award=min(awards), last_award=max(awards),
            n_bodies=len(rec["bodies"]), n_buyers=len(rec["buyers"]),
        ))
    return out


FEATURES = ("temporal_excess", "commercial", "noncompetitive",
            "identity_confidence", "concentration")
_SOURCE = {"temporal_excess": "nlp"}


def normalise(recs):
    """Rank-normalise each feature into f_<name> in [0,1] (monotone, so the
    threshold algorithm still applies)."""
    if not recs:
        return recs
    big = max((r["nlp"] for r in recs if math.isfinite(r["nlp"])), default=1.0)
    for r in recs:
        if not math.isfinite(r["nlp"]):
            r["nlp"] = big
    for name in FEATURES:
        for r, v in zip(recs, common.percentile(recs, _SOURCE.get(name, name))):
            r["f_" + name] = v
    return recs
