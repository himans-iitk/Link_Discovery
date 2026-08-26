"""
Configuration for the Favouritism (FAV 13) motif pipeline.

Parameters are split into two classes, and the split is deliberate:

LOAD-BEARING decisions change what the method *means*.  Each one must be
justified by a mechanism, a known property, or a prior result -- never by
"it worked better".  They are not tuned.

FREE parameters have no principled value.  They are declared with a
defensible range and settled by sensitivity analysis, not by argument.
"""

import os

# Paths resolve relative to this file, so the repo can live anywhere.
# Override the data root with IJF_DATA_ROOT.
_HERE = os.path.dirname(os.path.abspath(__file__))
REPO  = os.path.dirname(os.path.dirname(_HERE))
ROOT  = os.environ.get("IJF_DATA_ROOT", os.path.dirname(REPO))

RAW   = os.path.join(ROOT, "ijf-datasets")
CLEAN = os.path.join(ROOT, "ijf-clean")
WORK  = os.path.join(ROOT, "work")
GT    = os.path.join(REPO, "ground_truth")
REF   = os.path.join(REPO, "reference")


# ---------------------------------------------------------------- load-bearing

# Entity granularity.  "legal" splits corporate families apart; "family"
# keeps them merged.  Chosen: legal.  Merging families manufactures loops
# (parent lobbies, subsidiary wins), and a manufactured loop is a
# fabricated accusation.  Costs ~11% of identity edges (4,857 -> 4,317).
ENTITY_GRANULARITY = "legal"

# Within a manual cluster, two names are the same legal entity if one
# contains the other (trade name + registration number, e.g.
# "0434489 BC" < "0434489 BC SHARON PRUS") or they are near-identical
# strings (typo variants, "DELOITE" ~ "DELOITTE").  Containment was added
# after similarity alone split 44% of clusters and broke
# "0415527BC" from "0415527BC MEDI EXPRESS"; with it, 23%.
SPLIT_CONTAINMENT_MIN_LEN = 5

# The null's reference population: every valid lobbying start date in the
# country.  The question it answers is "given this company's *number* of
# registrations and the national temporal pattern of lobbying, how many of
# its contracts would still fall just after one?"  Using the entity's own
# date range instead would be a weaker null -- it would absorb the very
# alignment we are trying to detect.
NULL_REFERENCE = "global_lobbying_starts"

# Unit of observation on the procurement side.
#   "award" -- every award row counts
#   "date"  -- distinct award dates count once
# "date" exists because IT and staffing firms receive bursts of task
# authorisations on a single day; one nearby registration then captures
# the whole burst and inflates the statistic.  This is the sector confound
# visible in the first prototype's top-15.
AWARD_UNIT = "date"

# Aggregation must be monotone non-decreasing in every feature, with
# non-negative weights.  This is the precondition for Fagin's threshold
# algorithm (Fagin, Lotem, Naor, JCSS 2003) and is what buys instance-
# optimal early termination.  Do not add a term that can be negative.
AGGREGATION = "weighted_sum"


# ---------------------------------------------------------------- free params

# Proximity window, days.  Domain intuition says a contract weeks after
# lobbying is more suspicious than one years after; nothing fixes the
# boundary, so v1 swept 30/90/180/365 and reported sensitivity.  v2
# (score_v2.py) removed the parameter entirely: it uses the observed minimum
# gap and calibrates it against the national lobbying calendar.  Still read by
# the v1 company-level and body-conditioned paths.
PROXIMITY_WINDOW_DAYS = 90

# Minimum observed close pairs.  Without a floor, entities with two
# registrations and two awards scored 400x off a single coincidence.
# Costs ~76% of entities at 3.
MIN_OBS = 3

# Top-k returned.
TOP_K = 100

# Feature weights.  All non-negative (see AGGREGATION).  These are free and
# currently uniform-ish by design: we have no labels to fit them, and
# fitting them to the published stories would optimise prominence.
WEIGHTS = {
    "temporal_excess":     0.40,   # core signal: excess over null
    "commercial":          0.25,   # NOT a grant/contribution recipient
    "noncompetitive":      0.20,   # sole-source / limited / ACAN share
    "identity_confidence": 0.10,   # how sure are we these are one entity
    "concentration":       0.05,   # body overlap -- 0 until FAV 12 lands
}

# Grant/contribution control.
#
# `pro_award` cannot distinguish a procurement contract from a grant:
# `instrument` is 100% UNKNOWN and `type` is 97% UNKNOWN.  That matters,
# because a research grant to a university is *supposed* to be
# non-competitive, and universities lobby for research funding as a matter of
# course.  Without this control the ranking fills with Sherbrooke, Regina,
# Lethbridge, Victoria and registered charities -- the system working as
# designed, not favouritism.
#
# Two identifiable populations, both verifiable by a reader:
#   - suppliers matching the registered-charity registry (cha_meta)
#   - suppliers whose name is academic
#
# Expressed as `commercial = 1 - grant_likelihood` and given a POSITIVE
# weight, because the aggregation contract forbids negative weights: a
# negative term would break monotonicity and with it the threshold algorithm.
GRANT_WEIGHT_CHARITY = 1.00
GRANT_WEIGHT_ACADEMIC = 0.85

# Non-competitive procurement methods.  NON = non-competitive,
# LIMITED = limited tendering, ADVANCE = Advance Contract Award Notice
# (a notice-then-award mechanism that is competitive only nominally).
NONCOMPETITIVE_METHODS = {"NON", "LIMITED", "ADVANCE"}
