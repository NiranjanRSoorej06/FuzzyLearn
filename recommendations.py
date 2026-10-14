BARRIER_ACTIONS = {
    "visualization": "Interactive visualization or animation",
    "application": "Worked examples and graded practice",
    "unclear_explanation": "Revised notes and step-by-step explanations",
    "process_demo": "Short instructional video",
    "self_check": "Self-assessment quiz with explanations",
    "navigation": "Restructure the resource and add summaries",
    "relationships": "Concept map or diagram",
}

NEEDS_REVIEW = "Needs review"


def suggest_action(barrier_type):
    if barrier_type is None or (isinstance(barrier_type, float) and barrier_type != barrier_type):
        return NEEDS_REVIEW
    bt = str(barrier_type).strip()
    if bt == "":
        return NEEDS_REVIEW
    if bt in BARRIER_ACTIONS:
        return BARRIER_ACTIONS[bt]
    return None  # unknown -> validation error signal


def is_valid_barrier(barrier_type):
    return str(barrier_type).strip() in BARRIER_ACTIONS
