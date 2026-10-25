"""Independent cross-check of the NumPy engine against scikit-fuzzy.

Uses identical membership functions, operators (min AND, Mamdani clipping,
max aggregation), the same 101-point output universe and centroid
defuzzification, so any disagreement beyond grid resolution indicates a bug.
"""

import numpy as np
import pytest

from fuzzy_engine import priority, INPUT_PARAMS, OUTPUT_PARAMS, GRID
from rules import RULES

pytest.importorskip("skfuzzy", reason="scikit-fuzzy not installed")

from skfuzzy import control as skf
from skfuzzy import trapmf as skf_trapmf

UNIVERSE = np.arange(0, 101, dtype=float)


def _make_input(name):
    a = skf.Antecedent(UNIVERSE, name)
    for label, p in INPUT_PARAMS.items():
        a[label] = skf_trapmf(UNIVERSE, p)
    return a


def _build_control_system():
    F = _make_input("F")
    C = _make_input("C")
    R = _make_input("R")
    P = skf.Consequent(UNIVERSE, "P")
    for label, p in OUTPUT_PARAMS.items():
        P[label] = skf_trapmf(UNIVERSE, p)

    cs = skf.ControlSystem()
    for rule in RULES:
        antecedent = (
            F[rule["failure_set"]] & C[rule["confusion_set"]] & R[rule["resource_set"]]
        )
        cs.addrule(skf.Rule(antecedent, P[rule["priority_set"]]))
    return cs


CS = _build_control_system()


def skfuzzy_priority(F, C, R):
    sim = skf.ControlSystemSimulation(CS)
    sim.input["F"] = F
    sim.input["C"] = C
    sim.input["R"] = R
    sim.compute()
    return float(sim.output["P"])


@pytest.mark.parametrize("point", [
    (0, 0, 0), (50, 50, 50), (100, 100, 100), (65, 82, 90),
    (20, 40, 60), (80, 20, 40), (10, 90, 50), (33, 67, 91),
])
def test_crosscheck_close(point):
    ours = priority(*point)["score"]
    theirs = skfuzzy_priority(*point)
    assert abs(ours - theirs) < 0.5, "at %s: numpy=%.3f skfuzzy=%.3f" % (point, ours, theirs)


def test_crosscheck_worked_example():
    assert abs(skfuzzy_priority(65, 82, 90) - 79.4) < 0.5


def test_crosscheck_agreement_on_grid():
    diffs = []
    for F in range(0, 101, 20):
        for C in range(0, 101, 20):
            for R in range(0, 101, 20):
                ours = priority(F, C, R)["score"]
                theirs = skfuzzy_priority(F, C, R)
                diffs.append(abs(ours - theirs))
    assert max(diffs) < 0.5
    assert np.mean(diffs) < 0.2
