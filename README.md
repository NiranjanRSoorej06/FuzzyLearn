# FuzzyLearn

Explainable Mamdani fuzzy decision-support system ranking teaching-resource improvement actions.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pytest
```

## Notes

- Worked example `(F=65, C=82, R=90)` scores ~79.7 with the documented discrete 0-100 centroid; scikit-fuzzy continuous centroid is ~79.4.
- Mamdani centroid defuzzification is not guaranteed numerically monotone across membership transitions; rankings are checked in tests (marked xfail) and documented.
- Dataset rows are synthetic.
