# FuzzyLearn — How to Use

FuzzyLearn ranks course topics by how urgently their teaching resources need
improvement. Give it a CSV of topics with three 0–100 indicators (quiz failure
rate, student confusion, resource inadequacy) and a barrier category; it
returns a priority score, a ranking, a suggested improvement action, and an
explanation of how it got there.

The system **supports** instructor judgment — it does not replace it, and a
high score does not prove an intervention will work.

---

## 1. Install

Requires Python 3.10+.

```bash
cd FuzzyLearn
pip install -r requirements.txt
```

---

## 2. Run the dashboard

```bash
streamlit run app.py
```

This opens a browser at `http://localhost:8501`. On first load it scores the
bundled demo dataset `data/topics.csv` (15 **synthetic** Data Structures
topics — a warning banner reminds you these values are not real student data).

### What each section does

| Section | What you get |
|---|---|
| **Filters** | Narrow by course, barrier type, or minimum fuzzy score |
| **Ranked topics** | One row per topic: fuzzy score, priority label, suggested action, baseline score/rank side by side |
| **Priority chart** | Bar chart of fuzzy scores |
| **Fuzzy vs baseline comparison** | Spearman correlation + top-3 overlap vs the weighted baseline (`0.35F + 0.30C + 0.35R`) and three alternative weight sets |
| **Worked example** | Verifies the reference case `F=65, C=82, R=90` ≈ 79.4 |
| **CSV export** | Download the ranked table including a full explanation per topic |
| **Per-topic explanation** | Pick a topic: see its input values, Low/Medium/High membership degrees, the strongest fired rules, and the suggested improvement |
| **Sensitivity viewer** | Slide one input 0–100 while holding the other two fixed and watch the score respond |
| **Membership functions** | Plots of the input and output fuzzy sets |
| **Evaluation summary** | Live run of the coverage/monotonicity checks |
| **Manual single-topic scoring** | Score one topic by hand without touching the CSV |
| **Rule base** | The full 27-rule table the engine uses |

### Using your own data

Click **Upload topics CSV** and select a file matching this schema:

```csv
course,topic,failure_rate,confusion,resource_inadequacy,barrier_type,source
Data Structures,Recursion,65,82,90,visualization,synthetic
```

| Column | Rules |
|---|---|
| `course` | non-empty text |
| `topic` | non-empty text; no duplicates within a course |
| `failure_rate` | number 0–100 (quiz failure rate for the topic) |
| `confusion` | number 0–100 (aggregated student confusion) |
| `resource_inadequacy` | number 0–100 (how poorly existing materials fit) |
| `barrier_type` | one of: `visualization`, `application`, `unclear_explanation`, `process_demo`, `self_check`, `navigation`, `relationships` |
| `source` | `synthetic` or `real` |

If anything is wrong (missing column, value out of range, unknown barrier,
duplicate topic…) the dashboard **lists every problem and refuses to score** —
fix the file and re-upload.

Each barrier type maps to a suggested improvement:

| barrier_type | Suggested improvement |
|---|---|
| `visualization` | Interactive visualization or animation |
| `application` | Worked examples and graded practice |
| `unclear_explanation` | Revised notes and step-by-step explanations |
| `process_demo` | Short instructional video |
| `self_check` | Self-assessment quiz with explanations |
| `navigation` | Restructure the resource and add summaries |
| `relationships` | Concept map or diagram |

> **On real data:** only use anonymized topic-level aggregates with the
> appropriate permission. Never put student names or IDs in the CSV. Set
> `source=real` for genuine observations; the dashboard changes its provenance
> banner accordingly. See `data/PROVENANCE.md`.

---

## 3. Run the evaluation

```bash
python evaluation.py                  # uses data/topics.csv
python evaluation.py --data mine.csv  # any valid CSV
```

This prints ten checks (membership boundaries, 27-rule coverage, output
coverage, monotonicity, sensitivity, worked example, baseline/weight-variant
comparison, limiter case, expert comparison hook, data validation) and writes:

- `reports/evaluation_report.txt` — the full text report
- `reports/sensitivity.png` — sensitivity curves
- `reports/ranked_topics.csv` — scored topics with explanations

**Optional expert comparison:** add an `expert_rank` column to your CSV
(1 = the topic a faculty member/TA considers most urgent) and the evaluation
correlates both the fuzzy and baseline rankings against it.

---

## 4. Run the tests

```bash
pytest
```

77 tests cover membership boundaries, all 27 rules, the worked example, output
range, ranking, validation, explanations, the evaluation suite, and an
independent scikit-fuzzy cross-check of the engine.

---

## 5. Use it as a library

```python
from fuzzy_engine import priority
from recommendations import suggest_action
from pipeline import score_dataframe
import pandas as pd

# Score one topic
result = priority(F=65, C=82, R=90)
print(result["score"], result["label"])     # 79.7 High
for rule in result["fired_rules"]:
    print(rule["antecedents"], "->", rule["consequent"], rule["strength"])

# Suggest an improvement
print(suggest_action("visualization"))
# 'Interactive visualization or animation'

# Score a whole CSV
df = pd.read_csv("data/topics.csv")
ranked = score_dataframe(df)                # includes scores, ranks, explanations
print(ranked[["rank", "topic", "fuzzy_score", "suggested_action"]].head())
```

`priority()` raises `ValueError` for inputs outside 0–100.
`score_dataframe()` expects a dataframe that already passed
`data_validation.validate_dataframe()`.

---

## 6. How it works (one paragraph)

Each input is fuzzified into Low/Medium/High membership degrees. All 27
combinations of those sets fire rules (min AND), whose consequent output sets
(Very Low … Very High) are clipped and merged by max. The centroid of the
merged curve on a 0–100 grid becomes the priority score; the label comes from
the output set with the highest membership at that score. The barrier type is
a **separate** lookup — urgency never influences which action is suggested.

## 7. Known behaviours

- The worked example returns **79.7** on the discrete grid (design reference
  ≈79.4 from a continuous centroid); tests allow ±0.5.
- Centroid defuzzification can wobble by ~0.2 points near membership
  transitions at low values; `python evaluation.py` reports any monotonicity
  violations honestly instead of hiding them.
- Score ties are broken alphabetically by topic name.
- The fuzzy ranking is not "better" than the baseline just because it differs.

## 8. Project files

```text
app.py                 Streamlit dashboard
fuzzy_engine.py        Membership functions + Mamdani inference
rules.py               The 27-rule base (single source of truth)
baseline.py            Weighted baseline + weight variants
recommendations.py     Barrier → action lookup
explanations.py        Explanations from actual fired rules
pipeline.py            CSV → ranked results
evaluation.py          Evaluation suite + CLI
data/topics.csv        Synthetic demo dataset
data/PROVENANCE.md     Dataset provenance and ethics notes
tests/                 77 tests
FuzzyLearn_Dataset_and_Methodology_Plan.md   Dataset + research plan
FuzzyLearn_Project_Improved.md               Full system specification
```
