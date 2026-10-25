# FuzzyLearn

FuzzyLearn contains a TOPSIS prioritisation study for teaching-resource
improvement actions, alongside an existing Mamdani fuzzy-inference demo. The
main study ranks ten proposed actions for Data Structures foundations. Its
scores support a comparison under the stated criteria; they are not learning
gain estimates, probabilities of success, or recommendations for a real
course.

## Data and criteria

The TOPSIS input is `data/actions.csv`, with one row per action (A1–A10):

| Criterion | Direction | Data source |
|---|---|---|
| Documented difficulty (`diff_*`) | Benefit | Published Delphi mean and SD, scaled from 1–10 to 0–100 |
| Documented importance (`imp_*`) | Benefit | Published Delphi mean and SD, scaled from 1–10 to 0–100 |
| Resource gap (`resource_gap`) | Benefit | Synthetic seeded draw (seed 2026) |
| Implementation effort (`effort_hours`) | Cost | Synthetic seeded draw (seed 2026) |

Difficulty and importance use triangular values `(mean − SD, mean, mean + SD)`,
clipped to the original 1–10 scale before conversion. Reading ±1 SD as a fuzzy
range is a modelling assumption. Resource gap and effort are crisp synthetic
draws. Barrier-to-action assignments are also assumptions. The resulting
dataset is **synthetic-anchored**, not a real classroom dataset; only the
difficulty and importance ratings come from published data.

The Delphi ratings are from introductory programming, discrete mathematics,
and logic-design expert panels. They are not ratings from Data Structures
students or instructors. See [`docs/project_steps_1_to_10.md`](docs/project_steps_1_to_10.md)
for the criteria definitions, data register, worked examples, assumptions, and
reported results. The independent reference implementation is
[`reference/reference_topsis.py`](reference/reference_topsis.py).

Provenance is represented differently across outputs: `data/actions.csv` and
`results/ranking.csv` contain `data_status`; the evaluation console, dashboard
warning, and figure captions identify the results as synthetic-anchored.
`results/comparisons.csv` and `results/sensitivity.csv` do not have their own
provenance column, so interpret them in the context of the input dataset.

## Methods and implementation

| File | Role |
|---|---|
| `topsis.py` | Crisp TOPSIS with benefit/cost min–max orientation, Euclidean distances, closeness, ranking, equal weights, CRITIC weights, and comparison helpers |
| `fuzzy_topsis.py` | Chen-style triangular fuzzy TOPSIS implementation, kept separate from crisp TOPSIS |
| `mamdani_adapter.py` | Adds the existing Mamdani score as a comparison by mapping difficulty, importance, and resource gap to its three inputs; effort is ignored |
| `topsis_evaluation.py` | Generates rankings, method comparisons, weight sensitivity results, and figures in `results/` |
| `app.py` | Streamlit dashboard containing both the TOPSIS prioritisation section and the Mamdani topic-level demo |

The crisp model uses equal weights by default and also computes CRITIC weights.
The sensitivity analysis compares equal weights, four scenarios that assign
0.4 to one criterion and 0.2 to each other criterion, and CRITIC weights. The
Chen fuzzy model compares triangular inputs against a degenerate fuzzy version
with the same centres. Its implementation follows the formulas recorded in
the project methodology, but the team has **not verified all formulas against
Chen's original paper**; treat those results as exploratory.

## Run

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
python topsis_evaluation.py
streamlit run app.py
python -m pytest -q
```

The evaluation script reads `data/actions.csv` and writes:

- `results/ranking.csv` — per-action scores and ranks for crisp TOPSIS, Chen
  degenerate and fuzzy TOPSIS, equal-weight sum, difficulty-only ranking, and
  the Mamdani comparison; includes `data_status`.
- `results/comparisons.csv` — Spearman correlations and top-three overlap
  between methods.
- `results/sensitivity.csv` — weights and top-three/rank-correlation results
  for each weight scenario.
- `results/ranked_closeness.png`, `results/sensitivity_heatmap.png`,
  `results/fuzzy_vs_degenerate.png`, and `results/figure_captions.txt`.

The dashboard opens with a Mamdani description and includes a separate
**TOPSIS prioritisation** section followed by the Mamdani demo. The TOPSIS
section displays the selected weight scenario, ranks and compares actions,
shows the A1 crisp calculation, and exports its ranking as a CSV. The Mamdani
section uses the synthetic topic-level example in `data/topics.csv`.

## Current results and limitations

Under equal weights, A9 and A1 lead the crisp and both Chen rankings. Adding
the triangular spread does not change Chen ranks on this dataset (Spearman
ρ = 1.000). Crisp TOPSIS and the equal-weight sum also have the same ranking
(ρ = 1.000), so this example does not show TOPSIS outperforming a weighted
sum. Crisp TOPSIS and Chen rankings differ (ρ ≈ 0.806); this comparison changes
the normalization method as well as the representation, so it does not
isolate a fuzzy-spread effect. The values are generated by
`topsis_evaluation.py` and are also recorded in `results/`.

These results have no independent validation against learning outcomes or
expert rankings. The synthetic criteria and assumed barrier mappings limit
what can be inferred. The Delphi ratings come from a different set of
introductory computing subjects. TOPSIS closeness is a relative ranking score,
not a probability; agreement between methods does not establish that a ranking
is correct.

## Research sources used

- Goldman, Gross, Heeren, Herman, Kaczmarczyk, Loui, and Zilles (2008),
  [“Identifying Important and Difficult Concepts in Introductory Computing
  Courses using a Delphi Process”](https://doi.org/10.1145/1352135.1352226),
  SIGCSE ’08. Source of the difficulty and importance Delphi means and SDs.
- Hwang and Yoon (1981),
  [*Multiple Attribute Decision Making: Methods and Applications*](https://doi.org/10.1007/978-3-642-48318-9).
  Background reference for the crisp TOPSIS method.
- Diakoulaki, Mavrotas, and Papayannakis (1995),
  [“Determining objective weights in multiple criteria problems: The CRITIC
  method”](https://doi.org/10.1016/0305-0548(94)00059-H). Reference for the
  optional CRITIC weighting implementation.
- Chen (2000),
  [“Extensions of the TOPSIS for group decision-making under fuzzy
  environment”](https://doi.org/10.1016/S0165-0114(97)00377-1), *Fuzzy Sets and
  Systems*, 114(1), 1–9. Reference for the fuzzy variant; formula verification
  is still outstanding, as noted above.

The methodology document also lists related educational studies that motivate
the problem context. They are background sources; their results are not used
as input values in this implementation.

## Project files

```text
data/actions.csv                  TOPSIS alternatives and criteria
data/topics.csv                   Synthetic Mamdani demo data
reference/reference_topsis.py     Independent data generator and reference calculation
docs/project_steps_1_to_10.md     TOPSIS methodology, sources, assumptions, and results
results/                          Generated rankings, comparisons, sensitivity, and figures
tests/                            TOPSIS and Mamdani test suite
```

The project was developed with AI assistance. Review and disclose contributions
according to the applicable course requirements.
