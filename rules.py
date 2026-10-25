INPUT_SETS = ["Low", "Medium", "High"]
OUTPUT_SETS = ["Very Low", "Low", "Medium", "High", "Very High"]

# Rows F, columns C, per R table.
_RULE_TABLE = {
    "Low": {
        "Low": {"Low": "Very Low", "Medium": "Very Low", "High": "Low"},
        "Medium": {"Low": "Very Low", "Medium": "Low", "High": "Low"},
        "High": {"Low": "Low", "Medium": "Low", "High": "Medium"},
    },
    "Medium": {
        "Low": {"Low": "Low", "Medium": "Low", "High": "Medium"},
        "Medium": {"Low": "Low", "Medium": "Medium", "High": "Medium"},
        "High": {"Low": "Medium", "Medium": "Medium", "High": "High"},
    },
    "High": {
        "Low": {"Low": "Medium", "Medium": "Medium", "High": "High"},
        "Medium": {"Low": "Medium", "Medium": "High", "High": "High"},
        "High": {"Low": "High", "Medium": "Very High", "High": "Very High"},
    },
}


def build_rules():
    rules = []
    for r_set in INPUT_SETS:
        for f_set in INPUT_SETS:
            for c_set in INPUT_SETS:
                rules.append({
                    "failure_set": f_set,
                    "confusion_set": c_set,
                    "resource_set": r_set,
                    "priority_set": _RULE_TABLE[r_set][f_set][c_set],
                })
    return rules


RULES = build_rules()


def validate_rules(rules):
    if len(rules) != 27:
        return False, "expected 27 rules, got %d" % len(rules)
    seen = set()
    for rule in rules:
        required = {"failure_set", "confusion_set", "resource_set", "priority_set"}
        if not required.issubset(rule):
            return False, "rule missing required keys"
        if rule["failure_set"] not in INPUT_SETS:
            return False, "invalid failure_set %s" % rule["failure_set"]
        if rule["confusion_set"] not in INPUT_SETS:
            return False, "invalid confusion_set %s" % rule["confusion_set"]
        if rule["resource_set"] not in INPUT_SETS:
            return False, "invalid resource_set %s" % rule["resource_set"]
        if rule["priority_set"] not in OUTPUT_SETS:
            return False, "invalid priority_set %s" % rule["priority_set"]
        key = (rule["failure_set"], rule["confusion_set"], rule["resource_set"])
        if key in seen:
            return False, "duplicate rule %s" % (key,)
        seen.add(key)
    expected = {(f, c, r) for f in INPUT_SETS for c in INPUT_SETS for r in INPUT_SETS}
    if seen != expected:
        return False, "rule set does not cover all 27 combinations"
    return True, "ok"

