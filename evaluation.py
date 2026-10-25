"""Evaluation suite implementing the ten checks of the methodology plan (section 8).

Run as a script to produce a text report and sensitivity plots:

    python evaluation.py

Everything here is deterministic and uses the same engine, membership
functions, operators and output grid as the dashboard.
"""

import argparse
import os

import numpy as np
import pandas as pd

from fuzzy_engine import (
    priority, fuzzify, defuzzify, infer, trapmf,
    INPUT_PARAMS, OUTPUT_PARAMS, OUTPUT_SETS, GRID,
)
from rules import RULES, validate_rules
from baseline import weighted_score, DEFAULT_WEIGHTS, WEIGHT_VARIANTS
from pipeline import score_dataframe
from data_validation import validate_dataframe, NUMERIC_COLUMNS

WORKED_EXAMPLE = (65, 82, 90)
WORKED_REFERENCE = 79.4
MONOTONICITY_TOLERANCE = 1e-9


# --------------------------------------------------------------------------- 1
def membership_boundary_checks():
    """Check every input membership function at its corners and at 0/100."""
    cases = [
        ("Low", INPUT_PARAMS["Low"], {0: 1.0, 25: 1.0, 50: 0.0, 100: 0.0}),
        ("Medium", INPUT_PARAMS["Medium"], {25: 0.0, 50: 1.0, 75: 0.0, 0: 0.0, 100: 0.0}),
        ("High", INPUT_PARAMS["High"], {0: 0.0, 50: 0.0, 75: 1.0, 100: 1.0}),
    ]
    results = []
    for name, params, expected in cases:
        a, b, c, d = params
        for x, want in expected.items():
            got = float(trapmf(float(x), a, b, c, d))
            results.append({
                "set": name,
                "x": x,
                "expected": want,
                "got": round(got, 6),
                "pass": abs(got - want) <= 1e-9,
            })
    for name, params in INPUT_PARAMS.items():
        a, b, c, d = params
        for x in np.linspace(0, 100, 101):
            v = float(trapmf(float(x), a, b, c, d))
            results.append({
                "set": name,
                "x": round(float(x), 3),
                "expected": "<=1",
                "got": round(v, 6),
                "pass": 0.0 <= v <= 1.0,
            })
    return results


# --------------------------------------------------------------------------- 2
def rule_coverage_check():
    ok, msg = validate_rules(RULES)
    return {"ok": bool(ok), "message": msg, "n_rules": len(RULES)}


# --------------------------------------------------------------------------- 3
def output_coverage(step=10):
    """Evaluate a grid of inputs; every score must be finite and in [0, 100]."""
    scores = []
    bad = []
    values = list(range(0, 101, step))
    if values[-1] != 100:
        values.append(100)
    for F in values:
        for C in values:
            for R in values:
                r = priority(F, C, R)
                s = r["score"]
                scores.append(s)
                if not np.isfinite(s) or not (0.0 <= s <= 100.0):
                    bad.append({"F": F, "C": C, "R": R, "score": s})
    return {
        "n_combinations": len(scores),
        "n_invalid": len(bad),
        "invalid": bad,
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "mean": float(np.mean(scores)),
    }


# --------------------------------------------------------------------------- 4
def monotonicity_report(step=5, tolerance=MONOTONICITY_TOLERANCE):
    """Increase one input while holding the others fixed; priority must not drop."""
    fixed_pairs = [(20, 20), (30, 30), (50, 50), (70, 70), (80, 80), (30, 70), (70, 30)]
    values = list(range(0, 101, step))
    if values[-1] != 100:
        values.append(100)
    violations = []
    checked = 0
    for a, b in fixed_pairs:
        for which in ("F", "C", "R"):
            prev = None
            for v in values:
                F, C, R = (v, a, b) if which == "F" else (
                    (a, v, b) if which == "C" else (a, b, v)
                )
                s = priority(F, C, R)["score"]
                checked += 1
                if prev is not None and s < prev - tolerance:
                    violations.append({
                        "input": which,
                        "fixed": (a, b),
                        "from": prev,
                        "to": s,
                    })
                prev = s
    return {"checked": checked, "n_violations": len(violations), "violations": violations[:20]}


# --------------------------------------------------------------------------- 5
def sensitivity_curves(step=5):
    """Priority as one input varies, for representative fixed values of the others."""
    scenarios = {
        "others_low": (20, 20),
        "others_medium": (50, 50),
        "others_high": (80, 80),
        "resources_low_F_C_high": (80, 80),
    }
    values = list(range(0, 101, step))
    if values[-1] != 100:
        values.append(100)
    curves = {}
    for name, (a, b) in scenarios.items():
        curves[name] = {}
        for which in ("F", "C", "R"):
            pts = []
            for v in values:
                F, C, R = (v, a, b) if which == "F" else (
                    (a, v, b) if which == "C" else (a, b, v)
                )
                pts.append({"x": v, "score": priority(F, C, R)["score"]})
            curves[name][which] = pts
    return curves


# --------------------------------------------------------------------------- 6
def worked_example_report():
    r = priority(*WORKED_EXAMPLE)
    F, C, RR = WORKED_EXAMPLE
    m = fuzzify(F, C, RR)
    return {
        "inputs": {"F": F, "C": C, "R": RR},
        "score": r["score"],
        "reference": WORKED_REFERENCE,
        "difference": round(r["score"] - WORKED_REFERENCE, 4),
        "label": r["label"],
        "memberships": m,
        "n_fired": len(r["fired_rules"]),
        "fired_rules": r["fired_rules"],
    }


# --------------------------------------------------------------------------- 7 & 8
def ranking_comparison(df, fuzzy_col="fuzzy_score", baseline_col="baseline_score"):
    """Spearman rank correlation and top-3 overlap between two rankings."""
    n = len(df)
    if n < 2:
        return {"spearman": None, "top3_overlap": None, "n_topics": n}
    spearman = float(df[fuzzy_col].rank().corr(df[baseline_col].rank()))
    k = min(3, n)
    fuzzy_top = df.sort_values([fuzzy_col, "topic"], ascending=[False, True])["topic"].head(k)
    base_top = df.sort_values([baseline_col, "topic"], ascending=[False, True])["topic"].head(k)
    overlap = sorted(set(fuzzy_top) & set(base_top))
    return {
        "spearman": spearman,
        "top3_overlap": len(overlap),
        "top3_topics": overlap,
        "top3_k": k,
        "n_topics": n,
    }


def baseline_variant_comparison(df):
    """Spearman + top-3 overlap for the default weights and each alternative set."""
    out = {"default": ranking_comparison(df)}
    for name in WEIGHT_VARIANTS:
        out[name] = ranking_comparison(
            df,
            baseline_col=f"baseline_score_{name}",
        )
    return out


def limiter_example(df=None):
    """Section 11.2.6: high failure/confusion with low resource inadequacy.

    The rule base deliberately limits priority because resources look adequate.
    """
    r = priority(85, 85, 15)
    return {
        "inputs": {"F": 85, "C": 85, "R": 15},
        "score": r["score"],
        "label": r["label"],
        "fired_rules": r["fired_rules"],
    }


# --------------------------------------------------------------------------- 9
def expert_comparison(df, expert_column="expert_rank"):
    """Compare fuzzy and baseline rankings with an expert ranking, if supplied.

    `expert_rank` is expected to use 1 for the topic the expert considers most
    urgent. It is converted to an expert *score* (higher = more urgent) before
    correlating, so a positive correlation means agreement.
    """
    if expert_column not in df.columns:
        return {"available": False, "reason": "no '%s' column in the dataset" % expert_column}
    expert = pd.to_numeric(df[expert_column], errors="coerce")
    if expert.isna().any():
        return {"available": False, "reason": "expert ranking contains missing values"}
    if len(df) < 2:
        return {"available": False, "reason": "fewer than two topics"}
    expert_score = (len(df) + 1) - expert
    return {
        "available": True,
        "fuzzy_spearman": float(df["fuzzy_score"].corr(expert_score, method="spearman")),
        "baseline_spearman": float(df["baseline_score"].corr(expert_score, method="spearman")),
        "note": "Expert ranking is a comparison reference, not objective ground truth.",
    }


# --------------------------------------------------------------------------- 10
def data_validation_checks(df):
    """Confirm the shipped dataset is valid and that invalid data is rejected."""
    ok, errors = validate_dataframe(df)

    probes = {}

    bad = df.drop(columns=["topic"], errors="ignore")
    probes["missing_column"] = not validate_dataframe(bad)[0]

    bad = df.copy()
    if len(bad) and "failure_rate" in bad:
        bad.loc[bad.index[0], "failure_rate"] = 120
        probes["out_of_range"] = not validate_dataframe(bad)[0]

    bad = df.copy()
    bad["barrier_type"] = "not_a_barrier"
    probes["invalid_barrier"] = not validate_dataframe(bad)[0]

    bad = df.copy()
    bad["source"] = "unknown"
    probes["invalid_source"] = not validate_dataframe(bad)[0]

    bad = df.copy()
    if len(bad) and "failure_rate" in bad:
        bad.loc[bad.index[0], "failure_rate"] = np.nan
        probes["missing_value"] = not validate_dataframe(bad)[0]

    bad = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    probes["duplicate_row"] = not validate_dataframe(bad)[0]

    return {
        "dataset_valid": bool(ok),
        "dataset_errors": errors,
        "invalid_input_rejected": probes,
        "all_probes_reject": all(probes.values()),
    }


# --------------------------------------------------------------------------- all
def run_all(df, step=5):
    """Run every evaluation check on a validated dataframe."""
    ranked = score_dataframe(df)
    report = {
        "n_topics": len(df),
        "sources": sorted(set(df["source"].astype(str).str.strip())),
        "membership_boundaries": membership_boundary_checks(),
        "rule_coverage": rule_coverage_check(),
        "output_coverage": output_coverage(step=10),
        "monotonicity": monotonicity_report(step=step),
        "sensitivity": sensitivity_curves(step=step),
        "worked_example": worked_example_report(),
        "ranking_default": ranking_comparison(ranked),
        "ranking_variants": baseline_variant_comparison(ranked),
        "limiter_example": limiter_example(),
        "expert_comparison": expert_comparison(ranked),
        "data_validation": data_validation_checks(df),
    }
    report["membership_boundaries_pass"] = all(c["pass"] for c in report["membership_boundaries"])
    return report, ranked


def format_report(report):
    """Render the evaluation report as readable text."""
    lines = []
    lines.append("FuzzyLearn evaluation report")
    lines.append("=" * 60)
    lines.append("Topics evaluated: %d" % report["n_topics"])
    lines.append("Data source labels: %s" % ", ".join(report["sources"]))
    lines.append("")

    mb = report["membership_boundaries"]
    lines.append("1. Membership boundaries: %s (%d checks)"
                 % ("PASS" if report["membership_boundaries_pass"] else "FAIL", len(mb)))

    rc = report["rule_coverage"]
    lines.append("2. Rule coverage: %s (%d rules) %s"
                 % ("PASS" if rc["ok"] else "FAIL", rc["n_rules"], rc["message"]))

    oc = report["output_coverage"]
    lines.append("3. Output coverage: %d combinations, invalid=%d, range=[%.1f, %.1f], mean=%.2f"
                 % (oc["n_combinations"], oc["n_invalid"], oc["min"], oc["max"], oc["mean"]))

    mo = report["monotonicity"]
    lines.append("4. Monotonicity: %d checks, %d violations"
                 % (mo["checked"], mo["n_violations"]))
    for v in mo["violations"][:5]:
        lines.append("     - %s fixed=%s: %.2f -> %.2f" % (v["input"], v["fixed"], v["from"], v["to"]))

    we = report["worked_example"]
    lines.append("5. Worked example F=%d, C=%d, R=%d: score %.2f (reference %.1f, diff %+.2f), label=%s, fired=%d"
                 % (we["inputs"]["F"], we["inputs"]["C"], we["inputs"]["R"],
                    we["score"], we["reference"], we["difference"],
                    we["label"], we["n_fired"]))

    rd = report["ranking_default"]
    lines.append("6. Ranking vs default baseline: Spearman=%.4f, top-%d overlap=%d %s"
                 % (rd["spearman"], rd["top3_k"], rd["top3_overlap"], rd["top3_topics"]))
    for name, cmp in report["ranking_variants"].items():
        if name == "default":
            continue
        lines.append("     vs %s weights: Spearman=%.4f, top-3 overlap=%d"
                     % (name, cmp["spearman"], cmp["top3_overlap"]))

    le = report["limiter_example"]
    lines.append("7. Limiter example F=85,C=85,R=15: score %.2f (%s)"
                 % (le["score"], le["label"]))

    ec = report["expert_comparison"]
    lines.append("8. Expert comparison: %s"
                 % ("available" if ec["available"] else "not available (%s)" % ec.get("reason")))

    dv = report["data_validation"]
    lines.append("9. Data validation: dataset_valid=%s, all invalid-input probes rejected=%s"
                 % (dv["dataset_valid"], dv["all_probes_reject"]))
    for k, v in dv["invalid_input_rejected"].items():
        lines.append("     - %s: %s" % (k, "rejected" if v else "ACCEPTED (bug)"))
    return "\n".join(lines)


def save_sensitivity_plots(curves, out_dir="reports"):
    """Write sensitivity curves to reports/sensitivity.png (matplotlib optional)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - plotting is optional
        return None
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for ax, (name, data) in zip(axes, curves.items()):
        for which in ("F", "C", "R"):
            xs = [p["x"] for p in data[which]]
            ys = [p["score"] for p in data[which]]
            ax.plot(xs, ys, marker=".")
        ax.set_title(name)
        ax.set_xlabel("input value")
        ax.set_ylabel("priority score")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "sensitivity.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run the FuzzyLearn evaluation suite.")
    parser.add_argument("--data", default="data/topics.csv")
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--out", default="reports")
    args = parser.parse_args(argv)

    df = pd.read_csv(args.data)
    ok, errors = validate_dataframe(df)
    if not ok:
        print("Dataset failed validation:")
        for e in errors:
            print("  - " + e)
        return 1

    report, ranked = run_all(df, step=args.step)
    text = format_report(report)
    print(text)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "evaluation_report.txt"), "w") as fh:
        fh.write(text + "\n")
    plot = save_sensitivity_plots(report["sensitivity"], out_dir=args.out)
    if plot:
        print("\nSensitivity plot written to %s" % plot)
    ranked.drop(columns=["memberships", "fired_rules"]).to_csv(
        os.path.join(args.out, "ranked_topics.csv"), index=False
    )
    print("Ranked results written to %s/ranked_topics.csv" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
