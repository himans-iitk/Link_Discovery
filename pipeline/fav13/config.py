"""FAV 13 parameters.

Two kinds, kept apart on purpose:
  LOAD-BEARING  change what the method means; justified by mechanism, never tuned.
  FREE          no principled value; settled by sensitivity analysis.
"""

# ---------------------------------------------------------------- load-bearing

# "legal" splits corporate families; "family" keeps them merged. Merging
# manufactures loops (parent lobbies, subsidiary wins) = fabricated accusations.
# Costs ~11% of identity edges (4,857 -> 4,317).
ENTITY_GRANULARITY = "legal"

# Within a manual cluster, names are one legal entity if one contains the other
# (min length below) or they are near-identical strings.
SPLIT_CONTAINMENT_MIN_LEN = 5

# Null reference: every valid national lobbying start date. Using the entity's
# own date range would absorb the alignment we are trying to detect.
NULL_REFERENCE = "global_lobbying_starts"

# "date" counts distinct award dates once: IT/staffing firms get bursts of task
# authorisations on one day, which a single nearby registration would capture.
AWARD_UNIT = "date"

# Aggregation must be monotone with non-negative weights, or Fagin's threshold
# algorithm loses its correctness guarantee.
AGGREGATION = "weighted_sum"

# ---------------------------------------------------------------- free

# Lobby-to-award window (days). Used by v1 and the body-conditioned test;
# v2 has no window (it calibrates the observed gap instead).
PROXIMITY_WINDOW_DAYS = 90

# Minimum observed close pairs (without it, 2-and-2 entities scored 400x).
MIN_OBS = 3

TOP_K = 100

# v1 weights. Not fitted: there are no labels, and fitting to published
# stories would optimise prominence.
WEIGHTS = {
    "temporal_excess":     0.40,
    "commercial":          0.25,   # not a grant/charity recipient
    "noncompetitive":      0.20,
    "identity_confidence": 0.10,
    "concentration":       0.05,
}

# pro_award cannot tell a contract from a grant, and grants to charities and
# universities are non-competitive by design. Scored as commercial = 1 - this.
GRANT_WEIGHT_CHARITY = 1.00
GRANT_WEIGHT_ACADEMIC = 0.85

# NON = sole-source, LIMITED = limited tendering, ADVANCE = ACAN notice.
NONCOMPETITIVE_METHODS = {"NON", "LIMITED", "ADVANCE"}
