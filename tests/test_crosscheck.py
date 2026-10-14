import numpy as np
import pytest
from fuzzy_engine import priority


def test_worked_example_close():
    r = priority(65, 82, 90)
    # independent skfuzzy centroid is ~79.39; allow tolerance 1.0
    assert abs(r["score"] - 79.4) < 1.0
