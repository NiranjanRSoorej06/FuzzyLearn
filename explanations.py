"""Generate natural-language explanations from actual fired-rule data.

Every sentence is built from the real membership degrees and rule firings
returned by the fuzzy engine. Nothing here is topic-specific or hard-coded.
"""

INPUT_MEANING = {
    "F": "failure rate",
    "C": "student confusion",
    "R": "resource inadequacy",
}


def _fmt(x):
    return f"{x:.2f}"


def _describe_memberships(mems, threshold=0.05):
    active = [(name, deg) for name, deg in mems.items() if deg >= threshold]
    active.sort(key=lambda t: t[1], reverse=True)
    if not active:
        return "no non-zero membership"
    if len(active) == 1:
        name, deg = active[0]
        return f"{name} ({_fmt(deg)})"
    parts = ", ".join(f"{name} ({_fmt(deg)})" for name, deg in active)
    return parts


def _rule_phrase(fired):
    a = fired["antecedents"]
    return (
        f"F is {a['F']} AND C is {a['C']} AND R is {a['R']} "
        f"-> {fired['consequent']}"
    )


def explain(score, label, memberships, fired_rules, top_n=3):
    """Build an explanation string for one topic.

    Parameters
    ----------
    score : float
        Crisp priority score in [0, 100].
    label : str
        Priority label derived from the output membership degrees.
    memberships : dict
        {"F": {...}, "C": {...}, "R": {...}} membership degrees.
    fired_rules : list of dict
        Rules with non-zero firing strength, sorted by strength descending.
    top_n : int
        Number of strongest rules to cite.
    """
    parts = [f"Priority {score:.1f} ({label})."]

    for key in ("F", "C", "R"):
        meaning = INPUT_MEANING[key]
        parts.append(
            f"The {meaning} belongs to {_describe_memberships(memberships[key])}."
        )

    if fired_rules:
        strongest = fired_rules[:top_n]
        quoted = "; ".join(_rule_phrase(r) for r in strongest)
        parts.append(f"Strongest rules fired: {quoted}.")
        top = fired_rules[0]
        parts.append(
            f"The strongest rule has firing strength {_fmt(top['strength'])}."
        )
    else:
        parts.append("No rules fired for these inputs.")

    return " ".join(parts)


def explanation_table(ranked_df, top_n=3):
    """Return a list of {topic, explanation} dicts for a ranked dataframe."""
    out = []
    for _, row in ranked_df.iterrows():
        out.append({
            "topic": row["topic"],
            "explanation": explain(
                row["fuzzy_score"],
                row["priority_label"],
                row["memberships"],
                row["fired_rules"],
                top_n=top_n,
            ),
        })
    return out
