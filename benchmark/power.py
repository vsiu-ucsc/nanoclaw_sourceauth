#!/usr/bin/env python3
"""
benchmark/power.py — Minimum-detectable-effect analysis for McNemar comparisons.

Given the observed discordance counts from our McNemar tests, compute the
minimum true effect size that would have been detectable at a target power
(default 80%) and significance level (default α=0.05). This is a
*prospective* / sensitivity framing, not a post-hoc observed-power analysis:
we hold n and the discordance rate fixed (what we sampled) and vary the
hypothetical true effect, not the observed one.

Exact binomial two-sided test is used — more accurate than χ² at small n_d.

Usage
-----
  python3 benchmark/power.py                       # all models
  python3 benchmark/power.py --model claude-sonnet-4-6

Outputs a table per model and (if --csv) dumps a CSV for downstream tools.
"""
import argparse
import json
from pathlib import Path
from scipy import stats

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Core power / MDE functions
# ---------------------------------------------------------------------------

def mcnemar_reject_set(n_d: int, alpha: float = 0.05) -> set:
    """Exact binomial two-sided rejection set for H0: P(n10) = 0.5, given n_d disc pairs."""
    if n_d == 0:
        return set()
    return {k for k in range(n_d + 1)
            if stats.binomtest(k, n_d, 0.5, alternative="two-sided").pvalue <= alpha}


def mcnemar_power(n_d: int, true_p: float, alpha: float = 0.05) -> float:
    """
    Probability of rejecting H0 under H1: P(n10 | discordant) = true_p.

    true_p = 0.5 → null; true_p ≠ 0.5 → alternative. Effect size is
    |true_p - 0.5|; equivalent to true_p = 0.5 + δ/2 where δ is the
    proportion-difference effect.
    """
    if n_d == 0:
        return 0.0
    reject = mcnemar_reject_set(n_d, alpha)
    return float(sum(stats.binom.pmf(k, n_d, true_p) for k in reject))


def mde(n_d: int, target_power: float = 0.80, alpha: float = 0.05) -> float:
    """
    Smallest |true_p − 0.5| such that power ≥ target_power, given n_d pairs.
    Returned as a proportion-difference δ (i.e. 0.20 means the test has
    ≥80% power against a true 70/30 discordance skew).
    """
    if n_d == 0:
        return float("inf")
    # Discrete search over true_p ∈ {0.5, 0.51, ..., 1.0}
    for step in range(501):
        true_p = 0.5 + step * 0.001
        if mcnemar_power(n_d, true_p, alpha) >= target_power:
            return round(2 * (true_p - 0.5), 4)
    return float("inf")


# ---------------------------------------------------------------------------
# Summary ingestion
# ---------------------------------------------------------------------------

def iter_mcnemar_rows(summary: dict):
    """Yield (section, label, block) for every overall-level McNemar comparison."""
    mcn = summary.get("mcnemar", {})
    for key, by_cond in mcn.get("variant_comparison", {}).items():
        for cond, block in by_cond.items():
            overall = block.get("overall", {})
            if overall and overall.get("n10") is not None:
                yield ("variant", f"{key} @ {cond}", overall)
    for key, block in mcn.get("condition_effect", {}).items():
        overall = block.get("overall", {})
        if overall and overall.get("n10") is not None:
            yield ("condition", key, overall)
        for k, v in block.get("by_variant", {}).items():
            if v and v.get("n10") is not None:
                yield ("condition", f"{key} × variant={k}", v)


def analyse(summary_path: Path, target_power: float = 0.80, alpha: float = 0.05):
    summary = json.loads(summary_path.read_text())
    rows = []
    for section, label, b in iter_mcnemar_rows(summary):
        n_d = (b.get("n10") or 0) + (b.get("n01") or 0)
        n_tot = n_d + (b.get("n11") or 0) + (b.get("n00") or 0)
        rows.append({
            "section":  section,
            "label":    label,
            "n_total":  n_tot,
            "n_disc":   n_d,
            "p01":      b.get("p01"),
            "chi2":     b.get("chi2"),
            "p":        b.get("p"),
            "mde":      mde(n_d, target_power, alpha),
        })
    return rows


def format_table(model: str, rows: list, target_power: float) -> str:
    lines = [
        "═" * 110,
        f"  McNemar MDE analysis — {model}  (target power={target_power:.0%}, α=0.05)",
        "═" * 110,
        f"  {'section':<10} {'comparison':<60} {'n':>4} {'n_d':>4} {'p':>9} {'MDE (δ)':>10}",
        "─" * 110,
    ]
    for r in rows:
        p_str  = f"{r['p']:.4f}" if r['p'] is not None else "  —"
        mde_str = f"{r['mde']:.3f}" if r['mde'] != float("inf") else "  ∞"
        lines.append(
            f"  {r['section']:<10} {r['label']:<60} {r['n_total']:>4} {r['n_disc']:>4} {p_str:>9} {mde_str:>10}"
        )
    lines.append("─" * 110)
    lines.append(
        "  MDE (δ) = smallest proportion-difference detectable at 80% power.\n"
        "  δ = 0.20 ↔ n10/n_d = 0.6 under H1 (60/40 discordance skew)."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=None, help="Model slug (default: all)")
    p.add_argument("--target-power", type=float, default=0.80)
    p.add_argument("--alpha",        type=float, default=0.05)
    p.add_argument("--csv",          action="store_true", help="Also emit CSV per model")
    args = p.parse_args()

    model_dirs = (
        [RESULTS_DIR / args.model] if args.model
        else sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir())
    )
    for d in model_dirs:
        sp = d / "eval_summary.json"
        if not sp.exists():
            print(f"  skipping {d.name} — no eval_summary.json")
            continue
        rows = analyse(sp, args.target_power, args.alpha)
        print(format_table(d.name, rows, args.target_power))
        print()
        if args.csv:
            csv_path = d / "mcnemar_mde.csv"
            import csv as _csv
            with csv_path.open("w", newline="") as fh:
                w = _csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"  → wrote {csv_path}")


if __name__ == "__main__":
    main()
