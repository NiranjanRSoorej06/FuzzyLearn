import numpy as np

INPUT_SETS = ["Low", "Medium", "High"]
OUTPUT_SETS = ["Very Low", "Low", "Medium", "High", "Very High"]

# Input membership parameters as (a, b, c, d) trapezoids; triangles are degenerate traps (b == c).
INPUT_PARAMS = {
    "Low": (0, 0, 25, 50),
    "Medium": (25, 50, 50, 75),
    "High": (50, 75, 100, 100),
}
OUTPUT_PARAMS = {
    "Very Low": (0, 0, 0, 25),
    "Low": (0, 25, 25, 50),
    "Medium": (25, 50, 50, 75),
    "High": (50, 75, 75, 100),
    "Very High": (75, 100, 100, 100),
}

GRID = np.arange(0, 101, 1, dtype=float)


def _as_array(x):
    return np.asarray(x, dtype=float), np.ndim(x) == 0


def trapmf(x, a, b, c, d):
    """Trapezoidal membership function; triangles/shoulders are degenerate traps."""
    arr, scalar = _as_array(x)
    r = np.zeros_like(arr, dtype=float)
    if b > a:
        m = (arr >= a) & (arr <= b)
        r[m] = (arr[m] - a) / (b - a)
    m = (arr >= b) & (arr <= c)
    r[m] = 1.0
    if d > c:
        m = (arr >= c) & (arr <= d)
        r[m] = (d - arr[m]) / (d - c)
    r[arr < a] = 0.0
    r[arr > d] = 0.0
    r = np.clip(r, 0.0, 1.0)
    return float(r) if scalar else r


def _membership_value(name, value, params):
    a, b, c, d = params[name]
    return trapmf(value, a, b, c, d)


def membership_low(x):
    return _membership_value("Low", x, INPUT_PARAMS)


def membership_medium(x):
    return _membership_value("Medium", x, INPUT_PARAMS)


def membership_high(x):
    return _membership_value("High", x, INPUT_PARAMS)


def fuzzify(F, C, R):
    def _fuzzify_one(v):
        return {
            "Low": membership_low(v),
            "Medium": membership_medium(v),
            "High": membership_high(v),
        }
    return {"F": _fuzzify_one(F), "C": _fuzzify_one(C), "R": _fuzzify_one(R)}


def _out_mf(name):
    a, b, c, d = OUTPUT_PARAMS[name]
    return trapmf(GRID, a, b, c, d)


_OUT_MFS = {name: _out_mf(name) for name in OUTPUT_SETS}


def infer(F, C, R):
    from rules import RULES

    mems = fuzzify(F, C, R)
    fired = []
    clipped_list = []
    for rule in RULES:
        a = rule["failure_set"]
        b = rule["confusion_set"]
        rr = rule["resource_set"]
        strength = min(mems["F"][a], mems["C"][b], mems["R"][rr])
        if strength > 0.0:
            fired.append({
                "antecedents": {"F": a, "C": b, "R": rr},
                "consequent": rule["priority_set"],
                "strength": float(strength),
            })
            clipped = np.minimum(strength, _OUT_MFS[rule["priority_set"]])
            clipped_list.append(clipped)
    if clipped_list:
        aggregated = np.max(np.stack(clipped_list, axis=0), axis=0)
    else:
        aggregated = np.zeros_like(GRID)
    return {
        "memberships": mems,
        "fired_rules": fired,
        "aggregated": aggregated,
        "grid": GRID.copy(),
    }


def defuzzify(aggregated):
    agg = np.asarray(aggregated, dtype=float)
    denom = float(np.sum(agg))
    if denom == 0.0:
        return 50.0
    return float(np.sum(GRID * agg) / denom)


def _label_for_score(score):
    """Label by output-set membership degree at the crisp score.

    Ties are broken deterministically toward the higher-priority set
    (later in OUTPUT_SETS), which is documented in the README.
    """
    candidates = []
    for name in OUTPUT_SETS:
        a, b, c, d = OUTPUT_PARAMS[name]
        candidates.append((name, float(trapmf(score, a, b, c, d))))
    best = max(m for _, m in candidates)
    best_name = "Medium"
    for name, m in candidates:
        if m == best:
            best_name = name
    return best_name


def _check_range(value, name):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError("%s must be a number, got %r" % (name, value))
    if not (0.0 <= v <= 100.0):
        raise ValueError("%s must be in [0, 100], got %r" % (name, value))
    return v


def priority(F, C, R):
    """Return score, label, memberships and fired-rule details for one topic."""
    F = _check_range(F, "failure_rate")
    C = _check_range(C, "confusion")
    R = _check_range(R, "resource_inadequacy")
    res = infer(F, C, R)
    score = defuzzify(res["aggregated"])
    score = float(np.clip(score, 0.0, 100.0))
    label = _label_for_score(score)
    # sort fired rules by strength desc
    fired_sorted = sorted(res["fired_rules"], key=lambda r: r["strength"], reverse=True)
    return {
        "score": round(score, 1),
        "label": label,
        "memberships": res["memberships"],
        "fired_rules": fired_sorted,
        "aggregated": res["aggregated"],
    }
