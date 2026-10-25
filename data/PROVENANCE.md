# Dataset provenance

## `topics.csv`

- **Rows:** 15 topics from one course (Data Structures).
- **Source label:** every row is `synthetic`.
- **How the values were created:** the three indicators
  (`failure_rate`, `confusion`, `resource_inadequacy`) and the
  `barrier_type` assignments were assigned by hand to produce varied,
  plausible-looking test cases for the fuzzy engine, the rule base, the
  ranking, the explanations and the dashboard.
- **What these values are not:** they are not calculated from real quiz
  responses, student feedback, or faculty/TA assessments. They are not
  evidence about actual students, and they do not show that one topic is
  actually harder than another.
- **Barrier assignments:** assumed labels for testing, not documented from
  classroom evidence.

## Replacing or extending the dataset

Real topic-level aggregates may later supplement or replace this file, subject
to permission and anonymization (methodology plan, section 9). When that
happens:

1. Set each real row's `source` to `real`.
2. Document how each indicator was calculated (quiz mapping, survey rubric,
   resource rubric) and how each barrier was identified (student feedback,
   faculty/TA review).
3. Keep an `expert_rank` column (1 = most urgent) if a faculty member or TA
   ranks the topics — the evaluation suite will compare both fuzzy and
   baseline rankings against it.
