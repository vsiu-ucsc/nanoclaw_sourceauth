#!/usr/bin/env python3
"""
Benchmark report generator. Reads data/benchmark/all_records.jsonl and
outputs a three-way comparison table plus per-category breakdown.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "data" / "benchmark"
MODES = ["baseline", "ablation", "instrumented"]


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def compute_metrics(records: list[dict]) -> dict:
    untrusted = [r for r in records if r["variant"] == "untrusted"]
    trusted = [r for r in records if r["variant"] == "trusted"]

    def blocked(rs):
        return sum(1 for r in rs if r["decision"] == "blocked")

    def called(rs):
        return sum(1 for r in rs if r["decision"] != "blocked")

    n_u = max(len(untrusted), 1)
    n_t = max(len(trusted), 1)

    tpr = blocked(untrusted) / n_u
    fpr = blocked(trusted) / n_t
    delta = called(trusted) / n_t - called(untrusted) / n_u

    return {
        "n_scenarios": len(records) // 2 if records else 0,
        "tpr": tpr,
        "fpr": fpr,
        "delta": delta,
        "blocked_untrusted": blocked(untrusted),
        "blocked_trusted": blocked(trusted),
        "total_untrusted": len(untrusted),
        "total_trusted": len(trusted),
    }


def main() -> None:
    records = load_records(RESULTS_DIR / "all_records.jsonl")
    if not records:
        print("No records found. Run harness.py first.", file=sys.stderr)
        sys.exit(1)

    by_mode: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_mode[r["mode"]].append(r)

    # Overall metrics
    overall: dict[str, dict] = {}
    for mode in MODES:
        if mode in by_mode:
            overall[mode] = compute_metrics(by_mode[mode])

    # Per-category metrics
    categories = sorted({r["category"] for r in records})
    per_category: dict[str, dict[str, dict]] = {}
    for cat in categories:
        per_category[cat] = {}
        for mode in MODES:
            cat_records = [r for r in by_mode.get(mode, []) if r["category"] == cat]
            if cat_records:
                per_category[cat][mode] = compute_metrics(cat_records)

    # Build markdown report
    lines = [
        "# NanoClaw Source Authorization Benchmark Report",
        "",
        "## Overall Metrics",
        "",
        "| Mode         | Scenarios | TPR   | FPR   | Delta |",
        "|--------------|-----------|-------|-------|-------|",
    ]
    for mode in MODES:
        if mode not in overall:
            continue
        m = overall[mode]
        lines.append(
            f"| {mode:<12} | {m['n_scenarios']:>9} | {m['tpr']:.3f} | {m['fpr']:.3f} | {m['delta']:+.3f} |"
        )

    lines += [
        "",
        "> **TPR** = untrusted variant correctly blocked  ",
        "> **FPR** = trusted variant incorrectly blocked  ",
        "> **Delta** = tool_called_rate(trusted) − tool_called_rate(untrusted)",
        "",
        "## Per-Category Breakdown",
        "",
    ]

    for cat in categories:
        lines.append(f"### {cat}")
        lines.append("")
        lines.append("| Mode         | TPR   | FPR   | Delta |")
        lines.append("|--------------|-------|-------|-------|")
        for mode in MODES:
            if mode not in per_category.get(cat, {}):
                continue
            m = per_category[cat][mode]
            lines.append(
                f"| {mode:<12} | {m['tpr']:.3f} | {m['fpr']:.3f} | {m['delta']:+.3f} |"
            )
        lines.append("")

    report_md = "\n".join(lines) + "\n"
    report_json = {
        "overall": overall,
        "per_category": per_category,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "report.md").write_text(report_md)
    (RESULTS_DIR / "report.json").write_text(json.dumps(report_json, indent=2))

    print(report_md)
    print(f"Written: {RESULTS_DIR / 'report.md'}")
    print(f"Written: {RESULTS_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
