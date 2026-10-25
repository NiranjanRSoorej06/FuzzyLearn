# FuzzyLearn — Dataset Creation and Research Plan

## 1. Project Overview

**Project title:** Ranking Teaching-Resource Improvement Actions to Address Documented Topic-Level Learning Barriers

**Project name:** FuzzyLearn  
**Primary method:** Mamdani Fuzzy Inference System (FIS)  
**Implementation:** Python, NumPy, pandas, and Streamlit

FuzzyLearn is an explainable decision-support system that helps instructors decide which course topics should receive teaching-resource improvements first. It uses three topic-level indicators to calculate a priority score and ranks topics by urgency. A separate mapping links each topic's documented learning barrier to a suggested improvement action.

The system supports instructor judgment; it does not replace it or guarantee that a suggested intervention will improve learning.

---

## 2. The Dataset Problem

The proposed system needs three numerical indicators for each course topic:

1. **Quiz failure rate (`failure_rate`)** — percentage of incorrect or failing responses to questions associated with the topic.
2. **Student confusion (`confusion`)** — an aggregate measure derived from student feedback or a consistently worded survey, rescaled to 0–100.
3. **Teaching-resource inadequacy (`resource_inadequacy`)** — a score assessing how poorly existing materials meet the topic's needs, using a rubric such as clarity, examples, coverage, and ease of navigation.

It also needs a documented barrier category (`barrier_type`) to suggest a suitable resource improvement.

A public educational dataset such as [xAPI-Edu-Data on Kaggle](https://www.kaggle.com/datasets/aljarah/xAPI-Edu-Data) can be explored for background or initial experimentation. However, it does not directly provide all three required topic-level indicators. It should not be represented as a complete dataset for FuzzyLearn.

**Recommended approach:** create a small, clearly labelled synthetic topic-level dataset for initial implementation and testing. If permission and access are available, later supplement or replace it with anonymized real topic-level aggregates and faculty/TA assessments.

---

## 3. Is the Dataset Creatable?

Yes. The initial dataset is manageable because the project targets approximately **10–15 topics from one course**. A suitable first course is Data Structures and Algorithms.

For the prototype, each row represents one topic in one course. Synthetic values can be assigned to test the fuzzy engine, rule base, ranking, explanations, and dashboard. These values are examples for software testing—not evidence about actual students.

### Important distinction

- **Synthetic dataset:** generated values used to test whether the implementation works.
- **Real dataset:** values calculated from actual, anonymized quiz responses, student feedback, and/or faculty/TA assessments, with suitable permission.
- **Public dataset:** an existing dataset that may inform experimentation but might not match the required topic-level schema.

Never describe synthetic values as collected classroom data. Clearly label the source of every row.

---

## 4. Recommended CSV Schema

Create the file:

```text
data/topics.csv
```

Use these columns:

| Column | Type | Meaning | Validation |
|---|---|---|---|
| `course` | Text | Course containing the topic | Must not be empty |
| `topic` | Text | Topic being assessed | Must not be empty |
| `failure_rate` | Number | Topic-related quiz failure rate | 0–100 |
| `confusion` | Number | Aggregated student confusion score | 0–100 |
| `resource_inadequacy` | Number | Inadequacy of existing teaching resources | 0–100 |
| `barrier_type` | Category | Documented barrier associated with the topic | Must match a supported category |
| `source` | Category | Provenance of the values | `synthetic` or `real` |

### Example CSV

The values below are **illustrative synthetic data only**. They must not be presented as actual classroom observations.

```csv
course,topic,failure_rate,confusion,resource_inadequacy,barrier_type,source
Data Structures,Recursion,65,82,90,visualization,synthetic
Data Structures,Graph algorithms,48,70,75,application,synthetic
Data Structures,Sorting,20,30,35,unclear_explanation,synthetic
Data Structures,Dynamic programming,60,78,80,process_demo,synthetic
Data Structures,Trees,40,55,65,self_check,synthetic
Data Structures,Linked lists,32,42,40,process_demo,synthetic
Data Structures,Stacks and queues,25,35,30,application,synthetic
Data Structures,Hashing,45,60,55,unclear_explanation,synthetic
Data Structures,Binary search,28,38,25,application,synthetic
Data Structures,Heap operations,50,62,70,process_demo,synthetic
Data Structures,Graph traversal,55,68,72,visualization,synthetic
Data Structures,Greedy algorithms,42,58,48,application,synthetic
Data Structures,Divide and conquer,38,50,52,unclear_explanation,synthetic
Data Structures,Tries,35,48,60,relationships,synthetic
Data Structures,Complexity analysis,47,65,58,self_check,synthetic
```

These values are not intended to prove that one topic is actually more difficult than another. They provide varied test cases for the prototype.

---

## 5. Supported Barrier Types and Suggested Actions

The fuzzy engine determines **how urgently** a topic should be reviewed. The following separate lookup determines **what type of improvement to consider**.

| `barrier_type` | Meaning | Suggested improvement |
|---|---|---|
| `visualization` | Students struggle to visualize a process | Interactive visualization or animation |
| `application` | Students understand a concept but struggle to apply it | Worked examples and graded practice |
| `unclear_explanation` | Existing explanations are unclear or too brief | Revised notes and step-by-step explanations |
| `process_demo` | Students need to see a process performed | Short instructional video |
| `self_check` | Students cannot readily check their understanding | Self-assessment quiz with explanations |
| `navigation` | Materials are difficult to navigate | Restructure the resource and add summaries |
| `relationships` | Relationships between concepts are difficult to understand | Concept map or diagram |

Each topic should have a defensible barrier category. For synthetic rows, record that the barrier assignment is assumed for testing. For real data, document how the barrier was identified, such as student feedback or faculty/TA review.

Do not silently assign an unrelated action to an unknown barrier type. Return a validation error or a clearly labelled “Needs review” status.

---

## 6. How to Create the Dataset

### Step 1 — Choose one course

Start with one course, such as Data Structures and Algorithms, and select 10–15 topics. Keeping the initial dataset within one course makes comparisons easier to interpret.

### Step 2 — Define how each indicator is measured

- **Failure rate:** calculate the percentage of incorrect or failing responses on questions mapped to the topic. Document the calculation and minimum evidence needed.
- **Confusion:** use a consistent student survey or feedback rubric, then aggregate and rescale it to 0–100.
- **Resource inadequacy:** define a rubric covering dimensions such as clarity, examples, coverage, and navigation. Aggregate and rescale the assessment to 0–100.

For synthetic data, assign plausible values solely to exercise the software. Do not imply that they were calculated from real students.

### Step 3 — Assign the barrier category

Choose one supported `barrier_type` per topic. For the synthetic prototype, treat these as assumed test labels. For real data, record the evidence behind each label.

### Step 4 — Validate the CSV

Check that:

- All required columns exist.
- Topic and course names are non-empty.
- Numeric indicators are between 0 and 100.
- Values are not missing.
- Duplicate course-topic rows are checked.
- Every barrier category is supported.
- Every row has a valid source label (`synthetic` or `real`).

### Step 5 — Test the fuzzy system

Run the dataset through FuzzyLearn. Confirm that each row receives a finite priority score from 0 to 100, a priority label, an appropriate suggested action, and an explanation based on the actual rules fired.

### Step 6 — Record provenance

Keep a note describing how the data was created, any assumptions made, and whether each value is synthetic or real. Display this information in the dashboard and include it in the final report.

---

## 7. Planned Fuzzy Methodology

FuzzyLearn uses a **Mamdani Fuzzy Inference System** with three inputs:

- `F`: failure rate
- `C`: student confusion
- `R`: resource inadequacy

Each input ranges from 0 to 100 and has three fuzzy sets: Low, Medium, and High.

### Input membership functions

| Set | Shape | Parameters |
|---|---|---|
| Low | Trapezoidal | `(0, 0, 25, 50)` |
| Medium | Triangular | `(25, 50, 75)` |
| High | Trapezoidal | `(50, 75, 100, 100)` |

The output is a priority score from 0 to 100, represented by five fuzzy sets: Very Low, Low, Medium, High, and Very High.

### Rule base and inference

There are 27 rules because each of the three inputs has three fuzzy sets: `3 × 3 × 3 = 27`.

The planned operators are:

1. **Fuzzy AND:** minimum of input membership degrees.
2. **Implication:** Mamdani clipping.
3. **Aggregation:** maximum of clipped output sets.
4. **Defuzzification:** centroid method on a documented output grid.

The rules, membership functions, and thresholds are design choices that must be documented and evaluated. They should not be described as empirically validated until appropriate evaluation has been performed.

### Separate priority and action

- The fuzzy engine calculates urgency.
- The barrier mapping suggests an action.
- The weighted-score baseline offers a non-fuzzy comparison.
- Educators retain final decision-making authority.

---

## 8. Evaluation Plan

The initial synthetic dataset can test implementation behavior, but it cannot establish real-world educational effectiveness.

Planned checks:

1. **Membership boundaries:** test all membership functions at their endpoints and transition points.
2. **Rule coverage:** confirm that all 27 input combinations appear exactly once.
3. **Output coverage:** evaluate a grid of input combinations and verify that every output is finite and in `[0, 100]`.
4. **Monotonicity:** increase one input while holding the others fixed and check whether priority decreases unexpectedly.
5. **Sensitivity:** inspect how scores change as each input varies.
6. **Worked example:** test the specified Recursion case, `F=65`, `C=82`, `R=90`, and investigate differences from the design's approximate reference score of 79.4.
7. **Baseline comparison:** compare fuzzy scores with the weighted baseline:

   `S = 0.35 × F + 0.30 × C + 0.35 × R`

8. **Ranking comparison:** calculate Spearman rank correlation and top-3 overlap where appropriate.
9. **Expert comparison:** if faculty/TA assessments are available, compare the fuzzy and baseline rankings with those assessments.
10. **Data validation:** test missing columns, invalid categories, missing values, duplicates, and out-of-range inputs.

Do not claim that the fuzzy method is superior simply because its rankings differ from the weighted baseline.

---

## 9. Data Privacy and Ethics

- Use anonymized topic-level aggregates for any real classroom data.
- Obtain appropriate permission before using quiz results or student feedback.
- Do not include student names, IDs, or personally identifying records in the dataset.
- Label synthetic values clearly in the CSV, dashboard, report, and presentation.
- Document assumptions, transformations, and data sources.
- Treat suggested improvements as recommendations, not guaranteed solutions.
- Keep instructors involved in reviewing and overriding rankings.

---

## 10. Immediate Next Steps

1. Create `data/topics.csv` with 10–15 synthetic topics.
2. Add the `source` column and label every initial row `synthetic`.
3. Implement or verify the CSV validation module.
4. Run the fuzzy engine on the dataset and inspect scores and fired-rule explanations.
5. Compare the fuzzy ranking with the weighted-score baseline.
6. Complete coverage, monotonicity, and sensitivity tests.
7. Build or finish the Streamlit dashboard and CSV export.
8. If feasible, seek permission to collect anonymized real topic-level data or obtain faculty/TA feedback.
9. Update the report to distinguish completed implementation work from planned experiments.

---

## 11. Suggested Research-Progress Statement

> Public educational datasets will be explored for initial experimentation, but they may not provide all three required topic-level indicators. Therefore, an initial synthetic dataset of approximately 10–15 topics from one course will be developed to test the Mamdani Fuzzy Inference System. Each row will contain quiz failure rate, student confusion, resource inadequacy, a barrier category, and a source label. Synthetic data will be used only for implementation and methodological testing. Subject to data access and permission, anonymized real topic-level aggregates and faculty/TA assessments may subsequently be used for further evaluation.

## 12. Expected Outcome

The initial outcome will be a working, explainable prototype that accepts a topic-level CSV, computes fuzzy priority scores, ranks topics, maps barriers to suggested teaching-resource improvements, explains its decisions, and compares its results with a weighted-score baseline.

The prototype can demonstrate technical correctness and ranking behavior. Claims about actual educational benefit will require suitable real-world evaluation.
