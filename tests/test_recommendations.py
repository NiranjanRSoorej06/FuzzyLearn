import math

from recommendations import (
    BARRIER_ACTIONS, NEEDS_REVIEW, suggest_action, is_valid_barrier,
)


def test_all_seven_barriers_mapped():
    assert len(BARRIER_ACTIONS) == 7
    for bt in [
        "visualization", "application", "unclear_explanation", "process_demo",
        "self_check", "navigation", "relationships",
    ]:
        action = suggest_action(bt)
        assert action is not None and action != NEEDS_REVIEW


def test_known_barrier_returns_action():
    assert suggest_action("visualization") == "Interactive visualization or animation"
    assert suggest_action(" application ") == "Worked examples and graded practice"


def test_unknown_barrier_returns_none():
    assert suggest_action("mystery") is None


def test_blank_and_missing_needs_review():
    assert suggest_action("") == NEEDS_REVIEW
    assert suggest_action(None) == NEEDS_REVIEW
    assert suggest_action(float("nan")) == NEEDS_REVIEW
    assert suggest_action(math.nan) == NEEDS_REVIEW


def test_is_valid_barrier():
    assert is_valid_barrier("process_demo")
    assert not is_valid_barrier("foo")
    assert not is_valid_barrier(None)
