"""Score TOPSIS improvement actions with the existing Mamdani engine.

This is a comparison method only, not a validated use of the Mamdani
model: it maps F <- diff_c, C <- imp_c, R <- resource_gap, so the engine's
original inputs (failure_rate, confusion, resource_inadequacy) are being
reinterpreted as (difficulty, importance, resource gap). effort_hours is
ignored entirely. fuzzy_engine.py and rules.py are not modified.
"""

import numpy as np

from fuzzy_engine import priority


def score_actions(df):
    """Return a Mamdani priority score (0-100) per row of the actions df."""
    scores = np.array(
        [priority(row.diff_c, row.imp_c, row.resource_gap)["score"] for row in df.itertuples()],
        dtype=float,
    )
    return scores
