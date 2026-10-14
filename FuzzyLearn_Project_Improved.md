# FuzzyLearn — Project Specification

> **Purpose:** Build an explainable fuzzy-logic decision-support system that ranks teaching-resource improvement actions for course topics.
>
> **Course:** Soft Computing  
> **Problem ID:** P40 — Ranking teaching-resource improvement actions to address documented topic-level learning barriers  
> **Primary technique:** Mamdani Fuzzy Inference System (FIS)  
> **Interface:** Streamlit web dashboard  
> **Implementation language:** Python

---

## 1. Project at a Glance

### 1.1 One-sentence description

FuzzyLearn uses three topic-level indicators—quiz failure rate, student confusion, and resource inadequacy—to calculate a fuzzy priority score from 0 to 100, rank topics by urgency, and attach a suggested teaching-resource improvement based on each topic's documented barrier.

### 1.2 The real-world problem

Instructors and teaching assistants often have limited time to improve course materials. Several topics may show learning barriers at the same time, and it may not be obvious which topic should be addressed first.

A rigid rule such as “failure rate above 50% means urgent” creates an abrupt boundary: topics with failure rates of 49% and 51% would be treated very differently despite being similar. Fuzzy logic supports gradual categories such as **Low**, **Medium**, and **High**, and combines them through readable IF–THEN rules.

### 1.3 What the system does

For every topic, FuzzyLearn:

1. Loads and validates topic-level evidence from a CSV file.
2. Converts each numeric input into fuzzy membership degrees.
3. Evaluates a complete set of 27 fuzzy rules.
4. Aggregates the rule outputs using Mamdani inference.
5. Calculates a crisp priority score using centroid defuzzification.
6. Ranks topics by priority.
7. Looks up a suggested improvement action using the recorded barrier type.
8. Explains the strongest rules that contributed to the result.
9. Compares fuzzy rankings with a weighted-score baseline and, if available, a faculty/TA ranking.
10. Displays results in a dashboard and allows CSV export.

### 1.4 What the system does not claim

- It does not grade or predict the performance of individual students.
- It does not automatically create teaching materials.
- It does not prove that a recommended intervention will improve learning.
- It does not replace instructor judgment.
- A high score means **high urgency according to the defined indicators and rules**, not proven causal effectiveness.

---

## 2. Aim, Objectives, and Scope

### 2.1 Aim

Build an interpretable fuzzy-logic system that prioritizes teaching-resource improvement actions for topics with documented learning barriers.

### 2.2 Objectives

1. Create a topic-level dataset for one course, initially containing 10–15 topics.
2. Define membership functions for all input and output variables.
3. Implement a complete, interpretable 27-rule base.
4. Implement Mamdani inference using minimum for AND, maximum for aggregation, and centroid defuzzification.
5. Map documented barrier types to plausible improvement actions.
6. Build a Streamlit dashboard with ranking, explanations, and CSV export.
7. Compare the fuzzy ranking with a weighted-sum baseline and, where possible, an expert ranking.
8. Test output coverage, monotonicity, sensitivity, and implementation correctness.

### 2.3 In scope

- One initial course, such as Data Structures.
- Approximately 10–15 topics.
- CSV-based input.
- Fuzzy scoring and ranking.
- Suggested actions based on documented barrier types.
- Per-topic explanations showing membership values and fired rules.
- Weighted-score baseline.
- Optional expert/TA comparison.
- Visualizations, evaluation, and CSV export.

### 2.4 Out of scope

- Individual student prediction or profiling.
- Automatic grading.
- Automatic generation of notes, videos, or visualizations.
- Claims that the system establishes causal learning improvement.
- Fully automated decisions without human review.

---

## 3. System Architecture and Workflow

### 3.1 High-level pipeline

```text
topics.csv
    |
    v
Load and validate data
    |
    v
Read F, C, R for each topic
    |
    v
Fuzzify each input into Low / Medium / High memberships
    |
    v
Evaluate all applicable rules from the 27-rule base
    |
    v
Clip rule output membership functions (Mamdani implication)
    |
    v
Aggregate output sets using maximum
    |
    v
Centroid defuzzification
    |
    v
Priority score (0–100) and priority label
    |
    +------------------------------+
    |                              |
    v                              v
Rank topics                 Map barrier_type to action
    |                              |
    +---------------+--------------+
                    |
                    v
Explain results, compare baseline, show dashboard
                    |
                    v
Export ranked results to CSV
```

### 3.2 Processing steps

1. **Load:** Read `data/topics.csv`.
2. **Validate:** Check required columns, missing values, numeric ranges, duplicate topics, and allowed barrier types.
3. **Fuzzify:** Calculate the Low, Medium, and High membership degree for each input.
4. **Evaluate rules:** Compute each rule's firing strength.
5. **Implication:** Clip the rule's output membership function at that firing strength.
6. **Aggregation:** Combine all clipped outputs with the maximum operator.
7. **Defuzzification:** Calculate one crisp priority score between 0 and 100.
8. **Label:** Determine the displayed priority label from the output membership functions or a documented output-label policy.
9. **Rank:** Sort topics by score, highest first.
10. **Recommend:** Map the topic's `barrier_type` to a suggested improvement.
11. **Explain:** Show input values, memberships, and the strongest fired rules.
12. **Compare and export:** Show baseline results and allow CSV download.

---

## 4. Input Data and Variable Definitions

The fuzzy engine receives three inputs. All inputs use the range **0–100**.

| Variable | Symbol | Range | Meaning | Suggested evidence source |
|---|---|---:|---|---|
| Failure rate | `F` | 0–100 | Percentage of students failing topic-related quiz questions | Anonymized quiz results |
| Student confusion | `C` | 0–100 | Aggregated confusion reported by students | Short survey rescaled to 0–100 |
| Resource inadequacy | `R` | 0–100 | How poorly current teaching materials meet the topic's needs | Faculty/TA rubric rescaled to 0–100 |
| Priority | `P` | 0–100 | Urgency of improving the topic's resources | Fuzzy engine output |

### 4.1 Why these three inputs?

- **Failure rate (`F`)** captures evidence that students are not answering topic-related questions correctly.
- **Confusion (`C`)** captures student-reported difficulty or uncertainty.
- **Resource inadequacy (`R`)** captures the extent to which existing materials may need improvement.

Do not add a second “learning difficulty” input unless it comes from an independent evidence source. If it mostly repeats quiz failure rate, it may double-count the same signal.

### 4.2 Preparing the input values

- **Failure rate:** Calculate the percentage of incorrect/failing responses on questions associated with the topic. Document the exact calculation used.
- **Confusion:** Use a short, consistently worded survey. Rescale the aggregate result to 0–100 and document the transformation.
- **Resource inadequacy:** Use a rubric that may assess clarity, examples, coverage, and ease of navigation. Aggregate and rescale the rating to 0–100.

These are topic-level aggregates, not individual student records.

---

## 5. Fuzzy Membership Functions

### 5.1 Input membership functions

The inputs `F`, `C`, and `R` all use the same three fuzzy sets: **Low**, **Medium**, and **High**.

| Set | Shape | Parameters | Interpretation |
|---|---|---|---|
| Low | Trapezoidal | `(0, 0, 25, 50)` | Fully Low through 25; gradually falls to 0 at 50 |
| Medium | Triangular | `(25, 50, 75)` | 0 at 25 and 75; fully active at 50 |
| High | Trapezoidal | `(50, 75, 100, 100)` | Rises from 0 at 50 to fully High at 75; remains fully High through 100 |

The functions overlap. This lets a value belong partly to two categories rather than forcing it into only one category.

For the specified input functions, the memberships sum to 1 across the 0–100 domain.

#### Example

For `F = 65`:

- `Low = 0.0`
- `Medium = 0.4`
- `High = 0.6`

For `C = 82` and `R = 90`:

- `Low = 0.0`
- `Medium = 0.0`
- `High = 1.0`

### 5.2 Membership-function formulas

For implementation, use standard trapezoidal and triangular membership functions.

**Triangular function** with parameters `(a, b, c)`:

- Membership is 0 outside `[a, c]`.
- It rises linearly from 0 at `a` to 1 at `b`.
- It falls linearly from 1 at `b` to 0 at `c`.
- Handle the peak and boundary cases explicitly to avoid division by zero.

**Trapezoidal function** with parameters `(a, b, c, d)`:

- Membership is 0 outside `[a, d]`.
- It rises linearly from 0 at `a` to 1 at `b`.
- It remains 1 from `b` through `c`.
- It falls linearly from 1 at `c` to 0 at `d`.
- When `a = b` or `c = d`, implement the vertical edge as a plateau boundary without dividing by zero.

Clamp or validate input values before fuzzification; valid input values are 0–100.

### 5.3 Output membership functions

The output variable `P` is **Priority**, ranging from 0 to 100. It uses five fuzzy sets.

| Output label | Shape | Parameters |
|---|---|---|
| Very Low | Left-end triangular-style set | `(0, 0, 25)` |
| Low | Triangle | `(0, 25, 50)` |
| Medium | Triangle | `(25, 50, 75)` |
| High | Triangle | `(50, 75, 100)` |
| Very High | Right-end triangular-style set | `(75, 100, 100)` |

For the end sets, treat repeated endpoints as shoulder sets: Very Low has membership 1 at 0, and Very High has membership 1 at 100. Implement these endpoint cases explicitly.

The membership-function boundaries are initial design choices. Document them and examine their effect during sensitivity analysis or faculty feedback.

---

## 6. Rule Base

### 6.1 Rule design principles

The system contains exactly **27 rules** because there are three fuzzy sets for each of three inputs:

`3 × 3 × 3 = 27 combinations`

Every combination of `F`, `C`, and `R` has one rule.

Design principles:

1. Increasing failure rate, confusion, or resource inadequacy should not reduce priority when the other inputs remain fixed.
2. Resource inadequacy acts as a limiter. When `R` is Low, the output is capped at Medium because the current resources do not appear to be the main problem.
3. The highest priority occurs when failure, confusion, and resource inadequacy are all high.
4. Rules must remain readable and explainable to instructors.

### 6.2 Rule format

Every rule follows this pattern:

```text
IF F is <Low|Medium|High>
AND C is <Low|Medium|High>
AND R is <Low|Medium|High>
THEN Priority is <Very Low|Low|Medium|High|Very High>
```

### 6.3 Complete 27-rule table

In each table, rows represent `F` and columns represent `C`. Select the table matching the fuzzy category of `R`.

#### When `R = Low`

| `F` \ `C` | Low | Medium | High |
|---|---|---|---|
| Low | Very Low | Very Low | Low |
| Medium | Very Low | Low | Low |
| High | Low | Low | Medium |

#### When `R = Medium`

| `F` \ `C` | Low | Medium | High |
|---|---|---|---|
| Low | Low | Low | Medium |
| Medium | Low | Medium | Medium |
| High | Medium | Medium | High |

#### When `R = High`

| `F` \ `C` | Low | Medium | High |
|---|---|---|---|
| Low | Medium | Medium | High |
| Medium | Medium | High | High |
| High | High | Very High | Very High |

### 6.4 Example rules

- IF `F` is High AND `C` is High AND `R` is High THEN Priority is Very High.
- IF `F` is Low AND `C` is Low AND `R` is Low THEN Priority is Very Low.
- IF `F` is High AND `C` is High AND `R` is Low THEN Priority is Medium.

The third example illustrates the resource-inadequacy limiter: students may be struggling, but the evidence does not indicate that the current resources are inadequate.

### 6.5 Rule representation in code

Store rules in one central, auditable structure. Each rule should contain:

- `failure_set`
- `confusion_set`
- `resource_set`
- `priority_set`

Generate or validate the complete 27 combinations programmatically so a rule cannot be accidentally omitted. Do not duplicate the rule table independently in the dashboard.

---

## 7. Mamdani Inference and Defuzzification

Use the following exact operators in both the implementation and the project report.

| Stage | Method | Meaning |
|---|---|---|
| Fuzzy AND | Minimum | Rule firing strength is the minimum of the three input memberships |
| Implication | Mamdani clipping | Clip the consequent membership function at the rule's firing strength |
| Aggregation | Maximum | Combine all clipped rule outputs point by point using the maximum |
| Defuzzification | Centroid | Convert the aggregated fuzzy output into one crisp score |

### 7.1 Rule firing strength

For a rule with antecedents `F is A`, `C is B`, and `R is D`:

```text
strength = min(mu_F_A(F), mu_C_B(C), mu_R_D(R))
```

A rule only contributes when its firing strength is greater than zero.

### 7.2 Mamdani implication

For output value `z` and rule consequent set `Q`:

```text
clipped_output(z) = min(rule_strength, mu_Q(z))
```

### 7.3 Aggregation

For all rules `i`:

```text
aggregated_output(z) = max(clipped_output_i(z))
```

Use a consistent discrete output grid from 0 through 100, inclusive, for the initial NumPy implementation. A step size of 1 is a straightforward default. If another resolution is used, document it and use the same resolution for cross-checking.

### 7.4 Centroid defuzzification

The continuous centroid definition is:

\[
P = \frac{\int z\,\mu(z)\,dz}{\int \mu(z)\,dz}
\]

For the discrete grid:

```text
P = sum(z * aggregated_output[z]) / sum(aggregated_output[z])
```

Here, `z` ranges from 0 to 100 on the selected grid. The output must be a number in `[0, 100]`.

If the aggregated output has zero total membership, avoid division by zero and return a documented safe fallback (for example, 50). With the complete rule base, this case should not normally occur, but it must still be handled defensively.

### 7.5 Priority label

Do not automatically assign labels using arbitrary score intervals such as 0–20, 20–40, and so on. Those intervals are not specified by the fuzzy design and may conflict with the output membership functions.

Instead, choose and document one policy:

- **Recommended:** report the crisp score and derive the displayed label from the output-set membership degrees, using the label with the highest membership; or
- Define fixed score bands explicitly, justify them, and use the same bands everywhere.

If two output labels tie, use a documented, deterministic tie-breaking policy. Keep the numeric score and the qualitative label visible separately.

---

## 8. Worked Example and Initial Unit Test

Use this example as the first engine test.

### 8.1 Input

```text
Topic: Recursion
F = 65
C = 82
R = 90
```

### 8.2 Fuzzification

| Input | Low | Medium | High |
|---|---:|---:|---:|
| `F = 65` | 0.0 | 0.4 | 0.6 |
| `C = 82` | 0.0 | 0.0 | 1.0 |
| `R = 90` | 0.0 | 0.0 | 1.0 |

### 8.3 Rules with non-zero firing strength

Only rules whose antecedent membership degrees are all non-zero can fire.

| Rule antecedents | Consequent | Firing strength |
|---|---|---:|
| `F = Medium`, `C = High`, `R = High` | High | `min(0.4, 1.0, 1.0) = 0.4` |
| `F = High`, `C = High`, `R = High` | Very High | `min(0.6, 1.0, 1.0) = 0.6` |

### 8.4 Output processing

1. Clip the High output set at 0.4.
2. Clip the Very High output set at 0.6.
3. Aggregate the clipped sets using maximum.
4. Apply centroid defuzzification on the chosen output grid.

The original project design estimates the resulting priority at approximately **79.4**. Treat this as the reference result to reproduce and verify in the implementation. If the implemented membership functions, output grid, or endpoint handling produce a different result, investigate and document the cause rather than silently changing the expected value.

Interpretation: Recursion should receive a high priority score because failure is partly High and partly Medium, while confusion and resource inadequacy are High.

---

## 9. Barrier Types and Improvement-Action Mapping

The fuzzy engine determines **how urgent** a topic is. A separate mapping determines **what kind of improvement to consider**.

Do not mix the action lookup into the fuzzy priority calculation. Keep urgency scoring and action selection as separate modules.

| `barrier_type` | Documented barrier | Suggested improvement |
|---|---|---|
| `visualization` | Students cannot visualize a process | Interactive visualization or animation |
| `application` | Students understand the concept but cannot apply it | Worked examples and graded practice |
| `unclear_explanation` | Explanations are unclear or too brief | Revised notes and step-by-step explanations |
| `process_demo` | Students need to see a process performed | Short instructional video |
| `self_check` | Students cannot check their own understanding | Self-assessment quiz with explanations |
| `navigation` | Materials are difficult to navigate | Restructure the resource and add summaries |
| `relationships` | Students struggle to understand complex relationships | Concept map or diagram |

### 9.1 Recommendation rules

- Every input topic must have a valid `barrier_type`.
- An unknown barrier type should produce a validation error or an explicit “Needs review” status; it must not silently receive an unrelated action.
- Display the action as a suggestion, not a guaranteed solution.
- Where possible, have a faculty member or TA review the mapping.

---

## 10. Dataset Specification

### 10.1 File and granularity

- **File:** `data/topics.csv`
- **One row:** one topic in one course.
- **Initial target size:** 10–15 topics from one course.
- **Required source tracking:** every row must identify whether the data is real or synthetic.

### 10.2 Required columns

| Column | Type | Validation |
|---|---|---|
| `course` | Text | Must not be empty |
| `topic` | Text | Must not be empty; duplicate topic rows should be checked |
| `failure_rate` | Numeric | Must be between 0 and 100 |
| `confusion` | Numeric | Must be between 0 and 100 |
| `resource_inadequacy` | Numeric | Must be between 0 and 100 |
| `barrier_type` | Text/category | Must match one of the seven supported barrier types |
| `source` | Text/category | `real` or `synthetic` |

### 10.3 Illustrative CSV

The following values are examples only and must not be represented as collected classroom data.

```csv
course,topic,failure_rate,confusion,resource_inadequacy,barrier_type,source
Data Structures,Recursion,65,82,90,visualization,synthetic
Data Structures,Graph algorithms,48,70,75,application,synthetic
Data Structures,Dynamic programming,60,78,80,process_demo,synthetic
Data Structures,Sorting,20,30,35,unclear_explanation,synthetic
DBMS,Normalization,40,55,65,application,synthetic
```

For the main experiment, prefer one course so that topics are comparable. The DBMS row above is illustrative and should not be mixed into a single-course evaluation unless the study design explicitly supports it.

### 10.4 Data integrity and ethics

1. If real classroom data is unavailable, mark the entire dataset as synthetic and state this in the report and dashboard.
2. Never present synthetic values as real observations.
3. Anonymize student information and use it only with permission.
4. Record the source and calculation method for every indicator.
5. Do not include names, student IDs, or other personally identifying student information.
6. Include deliberate edge cases:
   - High failure rate with low resource inadequacy.
   - All inputs Low.
   - All inputs Medium.
   - All inputs High.
   - One or more inputs at 0 and 100.
7. Keep the dataset's provenance and any assumptions in the project documentation.

---

## 11. Weighted-Score Baseline

The fuzzy system must be compared with a simple non-fuzzy baseline.

### 11.1 Initial baseline formula

```text
S = 0.35 * F + 0.30 * C + 0.35 * R
```

Where:

- `F` is failure rate.
- `C` is confusion.
- `R` is resource inadequacy.
- `S` is the baseline score from 0 to 100.

These weights are assumptions for the initial experiment, not scientifically established values.

### 11.2 Baseline requirements

1. Calculate `S` for every topic.
2. Rank topics by `S`, highest first.
3. Display fuzzy score/rank and baseline score/rank side by side.
4. Try at least two alternative weight combinations.
5. Explain where the rankings agree and differ.
6. Examine cases such as high failure with low resource inadequacy, where the fuzzy rules deliberately limit priority.
7. Do not claim the fuzzy method is better merely because it produces a different ranking.

### 11.3 Ranking evaluation

Where sufficient topics are available, report:

- **Spearman rank correlation** between the fuzzy ranking and baseline ranking.
- **Top-3 overlap** between the two rankings.
- If an expert/TA ranking is available, compare both the fuzzy ranking and the baseline ranking against it.

An expert ranking is a comparison reference, not automatically an objective ground truth.

---

## 12. Streamlit Dashboard Requirements

### 12.1 Required features

1. **CSV upload:** Allow a user to upload a topic dataset.
2. **Validation feedback:** Identify missing columns, missing values, out-of-range numeric values, duplicate topic rows, and invalid barrier types.
3. **Ranked results table:** Show rank, course, topic, fuzzy score, priority label, suggested action, baseline score, and baseline rank.
4. **Per-topic explanation:** Show original input values, Low/Medium/High membership degrees, and the top 2–3 rules that fired with their firing strengths and consequents.
5. **Priority chart:** Show a bar chart of fuzzy priority scores.
6. **Baseline comparison:** Make it easy to compare fuzzy and weighted-score rankings.
7. **CSV export:** Allow users to download the ranked results and their relevant explanations.
8. **Data provenance:** Clearly indicate whether the displayed dataset is real or synthetic.

### 12.2 Optional features

- Manual entry form for one topic.
- Membership-function plots.
- Sensitivity-curve viewer.
- Filters by course or barrier type.
- Downloadable evaluation summary.

### 12.3 Explanation example

Generate explanations from the actual rule-firing data. Do not hard-code topic-specific explanation text.

Example wording:

> **Recursion — High Priority (79.4):** Failure rate belongs partly to the Medium and High categories, while confusion and resource inadequacy are High. One of the strongest rules is `F = High AND C = High AND R = High → Very High`, with firing strength 0.60.

---

## 13. Implementation Design

### 13.1 Recommended project structure

```text
FuzzyLearn/
├── app.py                     # Streamlit dashboard
├── fuzzy_engine.py            # Membership functions and Mamdani inference
├── rules.py                   # Central 27-rule definition and validation
├── baseline.py                 # Weighted-score baseline and ranking
├── recommendations.py         # Barrier type -> suggested action lookup
├── data/
│   └── topics.csv              # Example dataset; label synthetic data clearly
├── tests/
│   ├── test_memberships.py     # Membership function tests
│   ├── test_fuzzy_engine.py    # Inference and worked-example tests
│   └── test_data_validation.py # Input validation tests
├── evaluation.ipynb            # Sensitivity and ranking comparisons
├── requirements.txt
└── README.md
```

This structure separates fuzzy scoring, rule definitions, recommendations, baseline scoring, UI, and tests. It also prevents the dashboard from becoming the only place where project logic exists.

### 13.2 Dependencies

Core dependencies:

```text
pandas
numpy
streamlit
matplotlib
scikit-fuzzy
```

Use `scikit-fuzzy` as an independent implementation cross-check if practical. The primary engine can be implemented directly in NumPy so the inference process and rule explanations remain transparent.

### 13.3 Required fuzzy-engine functions

Implement functions with clear inputs and outputs. The names below are recommendations; consistent alternatives are acceptable.

```python
membership_low(x)
membership_medium(x)
membership_high(x)

fuzzify(F, C, R)

validate_rules(rules)

infer(F, C, R)
    # Return the aggregated output curve and fired-rule details.

defuzzify(aggregated_curve)
    # Return a crisp score in [0, 100].

priority(F, C, R)
    # Return score, label, memberships, and fired-rule details.
```

### 13.4 Suggested output contract

The main `priority(F, C, R)` function should return structured data similar to:

```python
{
    "score": 79.4,
    "label": "High",
    "memberships": {
        "F": {"Low": 0.0, "Medium": 0.4, "High": 0.6},
        "C": {"Low": 0.0, "Medium": 0.0, "High": 1.0},
        "R": {"Low": 0.0, "Medium": 0.0, "High": 1.0}
    },
    "fired_rules": [
        {
            "antecedents": {
                "F": "Medium",
                "C": "High",
                "R": "High"
            },
            "consequent": "High",
            "strength": 0.4
        }
    ]
}
```

This is a schema example, not a claim that the example's `fired_rules` list is complete. The actual engine must include every rule with positive firing strength, and the UI may display the strongest 2–3.

### 13.5 Engineering requirements

- Use one authoritative definition of the rule base.
- Keep functions deterministic for identical inputs and configuration.
- Validate the 0–100 input range.
- Handle endpoints without division-by-zero errors.
- Return a valid score in `[0, 100]`.
- Retain fired-rule strengths so explanations are generated from real calculations.
- Keep ranking and recommendation lookup separate from the fuzzy inference code.
- Use the same membership functions, operators, and output grid for the engine and its cross-check.
- Write tests before connecting the engine to the dashboard.

---

## 14. Evaluation and Testing Plan

The purpose of evaluation is to determine whether the rankings are consistent, interpretable, and defensible—not merely whether the program runs.

| Test | Procedure | Expected outcome |
|---|---|---|
| Membership boundaries | Test every membership function at all corner points and at 0 and 100 | Correct values; no division-by-zero errors |
| Rule coverage | Confirm that all 27 antecedent combinations exist exactly once | No missing or duplicate combinations |
| Output coverage | Evaluate a grid of `F`, `C`, and `R` values, such as increments of 10 | Every combination returns a finite score in `[0, 100]` |
| Monotonicity | Increase one input while holding the other two fixed | Priority should not decrease, consistent with the intended rule design |
| Sensitivity | Plot priority against each input over representative fixed values of the other inputs | Curves should be interpretable and changes should be explainable |
| Worked example | Evaluate `F=65`, `C=82`, `R=90` | Investigate differences from the reference score of approximately 79.4 |
| Independent cross-check | Compare with `scikit-fuzzy` using identical membership functions and operators | Scores should agree within a documented numerical tolerance |
| Baseline comparison | Compare fuzzy and weighted-score ranks | Report Spearman correlation and top-3 overlap |
| Expert comparison | Ask a faculty member or TA to rank the topics, if possible | Compare both methods with the expert ranking and discuss disagreements |
| Data validation | Test missing columns, missing values, invalid categories, duplicates, and values outside 0–100 | Clear errors; invalid data is not silently accepted |
| Usability (optional) | Ask a potential user whether the ranking and explanations are understandable | Record short feedback and limitations |

### 14.1 Important monotonicity note

The rule base is designed to be monotonic, but this property must be tested on the implemented fuzzy system. Do not assume that a monotonic rule table alone proves that every numerical output is monotonic under every implementation detail.

If a test fails:

1. Recheck membership functions and endpoint behavior.
2. Recheck the rule table and inference operators.
3. Check output-grid resolution and defuzzification.
4. Document the failing input combination.
5. Revise the design only with a clear reason, and update the report and tests together.

### 14.2 Sensitivity analysis

Vary one input across 0–100 while holding the other two fixed at representative values. Include cases where:

- All other inputs are low.
- All other inputs are medium.
- All other inputs are high.
- Resource inadequacy is low while failure and confusion are high.

The final case is important because the rules intentionally limit priority when resources appear adequate.

---

## 15. Four-Week Implementation Plan

| Week | Tasks | Deliverable |
|---|---|---|
| 1 — Data and design | Choose one course, select 10–15 topics, define input rubrics, create `topics.csv`, document membership functions, finalize the 27-rule table | Dataset and fuzzy-system design |
| 2 — Fuzzy engine | Implement membership functions, fuzzification, rules, Mamdani inference, aggregation, centroid defuzzification, and unit tests | Tested fuzzy engine |
| 3 — Application | Implement recommendations, weighted baseline, ranking, explanations, Streamlit dashboard, and CSV export | Functional prototype |
| 4 — Evaluation and presentation | Run coverage, monotonicity, sensitivity, and baseline tests; compare with expert ranking if possible; prepare report and demo | Evaluation results, final report, presentation |

---

## 16. Expected Deliverables

The completed project should contain:

1. A working Mamdani fuzzy inference engine.
2. Documented input/output membership functions.
3. A complete, validated 27-rule base.
4. A topic dataset with clear provenance.
5. A ranked list of topics and suggested resource improvements.
6. Explanations generated from actual fired rules.
7. A Streamlit dashboard.
8. A weighted-score baseline and ranking comparison.
9. Coverage, monotonicity, sensitivity, and cross-check results.
10. An expert/TA comparison if a suitable reviewer is available.
11. Unit tests and data-validation tests.
12. A final report and demonstration.

---

## 17. Limitations, Risks, and Ethical Considerations

- **Noisy evidence:** Quiz outcomes and student survey responses may be noisy or biased.
- **Subjective design:** Membership boundaries, rule consequents, and baseline weights are design choices and must be documented.
- **No causal claim:** A high score does not prove that an intervention will work.
- **Human oversight:** Faculty should be able to inspect, question, and override the ranking.
- **Data privacy:** Student-level records must not be exposed; use anonymized topic-level aggregates and obtain permission where required.
- **Synthetic data:** Synthetic values must be clearly labelled in the dashboard, results, report, and presentation.
- **Limited generalization:** Results from one course may not transfer to another course without review.
- **Action mapping uncertainty:** Suggested actions are plausible matches for recorded barriers, not experimentally proven solutions.
- **Ranking ties:** Define a deterministic tie-breaking policy, such as secondary sorting by topic name, and document it.
- **Input quality:** The ranking is only as credible as the quality and consistency of its input evidence.

---

## 18. Possible Future Extensions

These are optional and should not distract from the initial scope:

1. Add an independent instructor-rated difficulty input.
2. Rank multiple candidate interventions per topic.
3. Support multiple courses.
4. Refine membership functions using documented faculty feedback.
5. Collect feedback after interventions and evaluate whether outcomes improve.
6. Add historical comparisons across semesters, subject to privacy and data-quality safeguards.

A future classroom study would be required to investigate whether recommended improvements actually help learning.

---

## 19. Acceptance Criteria

The project is ready for demonstration when all required criteria below are met.

### Fuzzy engine

- [ ] Input membership functions work at boundaries and across the full range.
- [ ] All 27 rules are present exactly once.
- [ ] Minimum is used for fuzzy AND.
- [ ] Mamdani clipping is used for implication.
- [ ] Maximum is used for aggregation.
- [ ] Centroid defuzzification returns a finite score in `[0, 100]`.
- [ ] The Recursion example is tested and any difference from approximately 79.4 is investigated.
- [ ] Fired-rule strengths are returned for explanations.

### Dataset and recommendations

- [ ] The CSV contains all required columns.
- [ ] Invalid values and barrier types are caught.
- [ ] Every topic has a suggested action or an explicit review status.
- [ ] Real and synthetic data are clearly distinguished.

### Dashboard and evaluation

- [ ] Topics are ranked by fuzzy score.
- [ ] Each topic shows a score, label, action, and explanation.
- [ ] The weighted-score baseline is shown beside the fuzzy ranking.
- [ ] Results can be exported as CSV.
- [ ] Coverage, monotonicity, and sensitivity checks have been run.
- [ ] Limitations and assumptions are documented.

---

## 20. Suggested Final Report Structure

1. **Abstract** — Problem, approach, implementation, and main findings.
2. **Introduction** — Teaching-resource prioritization problem and motivation.
3. **Problem Definition** — Inputs, output, scope, and assumptions.
4. **Fuzzy-System Design** — Membership functions, rule base, Mamdani inference, and centroid defuzzification.
5. **Dataset** — Collection method, transformations, provenance, and privacy.
6. **Implementation** — Architecture, modules, dashboard, and tests.
7. **Results** — Priority scores, ranked topics, suggested actions, and explanations.
8. **Evaluation** — Coverage, monotonicity, sensitivity, baseline comparison, and expert comparison if available.
9. **Discussion** — Where rankings agree or differ and why.
10. **Limitations and Ethics** — Uncertainty, subjectivity, privacy, and non-causal interpretation.
11. **Conclusion and Future Work** — What was built and what should be investigated next.

---

## 21. Summary for Developers and LLM Coding Assistants

When implementing this project, follow these rules:

1. Build **FuzzyLearn**, a Python/Streamlit decision-support application for ranking topic-level teaching-resource improvements.
2. Use exactly three fuzzy inputs initially: `failure_rate` (`F`), `confusion` (`C`), and `resource_inadequacy` (`R`), all in `[0, 100]`.
3. Use Low, Medium, and High input membership functions with the parameters defined in Section 5.
4. Use the complete 27-rule table in Section 6. Do not invent replacement rules.
5. Use Mamdani inference: minimum for AND, clipping for implication, maximum for aggregation, and centroid for defuzzification.
6. Use a discrete output grid from 0 to 100, inclusive, with a documented resolution.
7. Treat approximately 79.4 for `(F=65, C=82, R=90)` as a reference result to reproduce or investigate.
8. Keep the rule base centralized and validate its coverage.
9. Keep fuzzy urgency scoring separate from `barrier_type`-to-action mapping.
10. Keep baseline scoring separate from fuzzy scoring.
11. Generate explanations from actual rule firing, not hard-coded prose.
12. Validate input data before scoring it.
13. Never present synthetic data as real data.
14. Provide tests for membership boundaries, rule coverage, output range, monotonicity, the worked example, and invalid CSV inputs.
15. Build the dashboard only after the engine and its tests work.
16. Do not claim that the fuzzy method is superior without evidence; report comparisons honestly.
17. Keep human review, privacy, assumptions, and limitations visible.

**Core outcome:** an explainable and testable prototype that helps educators decide which teaching resources to review first, while keeping final decisions with educators.
