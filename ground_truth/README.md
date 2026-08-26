# Ground-truth data dictionary

Two files, two roles:

| File | Role | Produced by |
|---|---|---|
| `gold_cases.csv` | **Input** — documented cases transcribed from cited journalism (human-curated) | hand-authored from sources |
| `gold_links.csv` | **Output** — each named case *resolved against the real datasets* | computed by the patronage notebooks (removed; see git history) |

`gold_cases.csv` is what a human reads from an article. `gold_links.csv` is the machine's
attempt to find each named person inside `apt_rows.csv` (appointees) and `don_20_22_0.5M.csv`
(donors), so the case can become a labeled edge for evaluation.

---

## 1. `gold_cases.csv` — columns

These are **transcribed/curated**, not calculated. Each value is taken from (or judged from) the
cited `source_url`.

| Column | Meaning | How it is set |
|---|---|---|
| `case_id` | Stable id (`GC001`…) | assigned manually |
| `case_type` | `individual` (one named person) or `cohort` (an aggregate group, e.g. "83 judges") | manual |
| `person_name` | The named person (blank for cohorts) | from the article |
| `link_category` | Which of the 5 link types: Patronage / Revolving Door / Favouritism / VIP / Appointee+ | judged from the article |
| `appointment_role` | The public-side role (Senator, Justice, Judge, Director…) | from the article |
| `appointment_body` | The court/board/agency | from the article |
| `appointment_year` | Year appointed | from the article |
| `party` | Party the person donated to | from the article |
| `donation_reported` | Donation amount **as reported** (free text, e.g. "$3,535", "$14,500+ over 5 years") | from the article |
| `donation_years` | Year(s) of the donation(s) | from the article |
| `jurisdiction` | federal / Alberta / British Columbia … | from the article |
| `expected_resolvable` | **A-priori human guess** of whether both ends will exist in our loaded data (`yes`/`partial`/`no`) | judged, *before* running the resolver |
| `status` | reported / under-review / sanctioned-expression-of-concern | from the article |
| `source_outlet` | Publisher(s) | from the article |
| `source_url` | Link to the cited article | the source |
| `claim` | One-sentence verbatim-ish summary of what the source asserts | from the article |

> `expected_resolvable` is a *prediction*; the actual outcome is the `resolution` column in
> `gold_links.csv`. They can differ (and that difference is informative about data coverage).

---

## 2. `gold_links.csv` — columns (all **computed**)

Built only for `case_type == individual` rows with a non-empty `person_name`. The matcher is the
same character n-gram TF-IDF cosine used everywhere in the project.

**How the match is computed (per person):**
1. Lower-case/strip all appointee names, all donor names, and the gold person names.
2. Fit one `TfidfVectorizer(analyzer='char', ngram_range=(2,3))` on the union of those strings
   (shared vocabulary).
3. `apt_cos` = the **maximum** cosine similarity between the person's name vector and every
   appointee-name vector; the arg-max is the matched appointee.
4. `don_cos` = the same against every donor-name vector; the arg-max is the matched donor.
5. A side "counts" as found if its cosine ≥ **`TAU = 0.90`**.

| Column | Meaning | How calculated |
|---|---|---|
| `case_id`, `person`, `category` | copied from `gold_cases.csv` | copy |
| `resolution` | `resolved` / `appointee-only` / `donor-only` / `unresolved` | from `apt_cos`/`don_cos` vs `TAU` (see §3) |
| `appointee` | matched appointee name (blank if `apt_cos < TAU`) | arg-max of `apt_cos` |
| `apt_cos` | name-similarity to the best appointee (0–1; **1.0 = exact name**) | max cosine, gold×appointees |
| `apt_position`, `apt_org`, `apt_year` | attributes of the matched appointee row | looked up from `apt_rows.csv` |
| `donor` | matched donor name (blank if `don_cos < TAU`) | arg-max of `don_cos` |
| `don_cos` | name-similarity to the best donor (0–1) | max cosine, gold×donors |
| `don_party` | that donor's **most frequent** `political_party` | mode over the donor's rows |
| `don_total` | that donor's **summed** `amount_monetary` (2020–22) | sum over the donor's rows |
| `don_years` | that donor's donation year span | `min–max` of `donation_year` |
| `source_url` | copied from `gold_cases.csv` | copy |

> `apt_cos`/`don_cos` measure **name-string identity only** — not influence, not guilt. `1.0` means
> an exact name match in that table; `0.63` means the closest name is only ~63% similar (i.e. no real
> match — a different person).

---

## 3. What "fully resolvable" vs "partial" means

The flag is purely about **whether both endpoints of the link exist in the data we loaded**, at a
confident name match. It says nothing about whether the patronage allegation is true (the *article*
establishes the real-world link; the source files just may or may not contain the two rows).

A link has two endpoints: a **donor** record and an **appointee** record. We try to find each:

| `resolution` | `apt_cos ≥ 0.90`? | `don_cos ≥ 0.90`? | Meaning |
|---|:--:|:--:|---|
| **`resolved`** (fully) | ✅ | ✅ | Both endpoints found → the `(donor row → appointee row)` edge is materialized and can be used as a **gold positive** in the hypothesis tests. |
| `appointee-only` | ✅ | ❌ | We have the appointment record, but no matching donor in the data. |
| `donor-only` | ❌ | ✅ | We have the donation record, but no matching appointment. |
| `unresolved` | ❌ | ❌ | Neither endpoint found. |

**Why partials happen — it is a data-coverage gap, not a method failure:**
- `don_20_22_0.5M.csv` is **federal donations, 2020–2022, and a 0.5M sample**. So:
  - *Provincial* donors don't appear → **Daniel Robert Pahl** and **Gordon Kenneth Hatch** donated to
    the UCP via *Elections Alberta*, so their donor side is missing (`don_cos` ≈ 0.61) → `appointee-only`.
  - Donations outside 2020–22, or not in the 0.5M sample, are missing → **Diana Piccoli** is an exact
    appointee match (`apt_cos = 1.0`) but her Liberal donations aren't in this slice (`don_cos = 0.63`)
    → `appointee-only`.
- The appointments file has its own window/coverage:
  - **Daryl Fridhandler**'s 2024 Senate seat isn't in the appointments data (`apt_cos = 0.85 < 0.90`),
    though he is an exact donor match → `donor-only`.

So **"fully resolvable" = both the donation and the appointment for that named person were located in
our actual tables** (e.g. **Toni Varone**: `apt_cos = 1.0` Director @ Canada Lands Company + `don_cos
= 1.0` Liberal $1,675). Only these can serve as ready-to-use labeled positive edges; the partials
become resolvable once the missing source (provincial donations, other appointment years/snapshots)
is loaded.

**Caveats**
- A high `*_cos` is a confident *same-name* match, but common names can collide (homonyms); treat each
  resolved link as a **candidate to be human-verified** against the cited article, not a proven fact.
- `TAU = 0.90` is a knob: lower it to catch more spelling variants (more recall, more false matches),
  raise it to be stricter.
