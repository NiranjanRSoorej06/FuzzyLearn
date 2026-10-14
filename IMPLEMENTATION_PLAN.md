# FuzzyLearn — Implementation Plan (3 Phases)

Source of truth: `FuzzyLearn_Project_Improved.md`.

## Phase 1 — Fuzzy Engine & Data (Week 1–2)
- Create project skeleton: `fuzzy_engine.py`, `rules.py`, `baseline.py`, `recommendations.py`, `data/topics.csv`, `tests/`.
- Implement typed membership functions `membership_low/medium/high` with triangle/trapezoid formulas and safe endpoint handling.
- Implement `fuzzify(F, C, R)` returning all nine memberships.
- Implement central 27-rule table in `rules.py`; add `validate_rules` (each F×C×R combo exactly once).
- Implement Mamdani `infer()` (min AND, clipping, max aggregation, discrete 0–100 grid) returning aggregated curve + fired rules.
- Implement `defuzzify()` centroid with zero-total fallback (50); `priority()` returns score/label/memberships/fired_rules.
- Implement barrier→action map in `recommendations.py`; validation for unknown barrier type → "Needs review".
- Implement `baseline.py` weighted score `0.35F+0.30C+0.35R` + ranking, alternate weight sets.
- Create `data/topics.csv` (10–15 Data Structures topics, synthetic, labelled), including edge cases.
- Write unit tests: membership boundaries, rule coverage, worked example ≈79.4 for (65,82,90), output range, monotonicity grid, input validation, barrier mapping.
- Verify with scikit-fuzzy cross-check within tolerance.

## Phase 2 — Streamlit Dashboard (Week 3)
- Build `app.py`: CSV upload, validation feedback, ranked results table (rank/course/topic/score/label/action/baseline score/baseline rank).
- Per-topic explanation view: inputs, memberships, top 2–3 fired rules.
- Bar chart of fuzzy priority; fuzzy vs baseline comparison; CSV export.
- Data provenance indicator (real vs synthetic). Optional: manual entry form, filters.

## Phase 3 — Evaluation & Finalization (Week 4)
- Coverage grid (0–100 step 10), monotonicity and sensitivity plots, alternative weight combos, Spearman/top-3-overlap vs baseline, expert/TA comparison if available.
- Write `evaluation.ipynb`, README, report; polish UX; ensure acceptance criteria all met.
- Run full test suite and lint/type checks; finalize demo narrative.