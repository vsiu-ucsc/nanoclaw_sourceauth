#!/usr/bin/env python3
"""
benchmark/report.py — Generate human-readable reports from eval_summary.json files.

Reads every benchmark/results/<model>/eval_summary.json, writes a .txt report
to benchmark/reports/<model>/report.txt, and prints a combined summary to stdout.

Usage
-----
  python3 benchmark/report.py              # all models
  python3 benchmark/report.py --model claude-sonnet-4-6
"""
import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"
REPORTS_DIR = Path(__file__).parent / "reports"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _w(n: int = 100) -> str:
    return "─" * n


def _dw(n: int = 100) -> str:
    return "═" * n


def _mcnemar_row(label: str, s: dict, indent: int = 4) -> str:
    pad = " " * indent
    if s is None or s.get("chi2") is None:
        return f"{pad}{label:<36}  no discordant pairs"
    sig = "* " if s.get("significant_0.05") else "  "
    return (
        f"{pad}{label:<36}  χ²={s['chi2']:>8.4f}  p={s['p']:.6f} {sig}"
        f" [n10={s['n10']} n01={s['n01']} n11={s['n11']} n00={s['n00']}]"
    )


def _mcnemar_block(block: dict, title: str, indent: int = 4) -> list[str]:
    lines = [f"{' ' * indent}{title}:"]
    lines.append(_mcnemar_row("overall", block["overall"], indent + 2))
    for k, v in block.get("by_type", {}).items():
        lines.append(_mcnemar_row(k, v, indent + 4))
    for k, v in block.get("by_boundary", {}).items():
        lines.append(_mcnemar_row(k, v, indent + 4))
    return lines


_PAIR_LABELS = {
    "with_vs_without":       "with vs without      (does explicit R help?)",
    "implicit_r_vs_without": "implicit_r vs without (does identity alone help?)",
    "implicit_r_vs_with":    "implicit_r vs with    (cost of omitting explicit R)",
}


def _mcnemar_section(mcn: dict) -> list[str]:
    lines = []
    tvu  = mcn.get("trusted_vs_untrusted", {})
    prov = mcn.get("provenance_effect", {})

    lines.append("  McNemar \u2014 trusted vs untrusted")
    lines.append("  (H\u2080: call-rate equal across variants)")
    for cond in ["with", "implicit_r", "all_conditions"]:
        if cond in tvu:
            lines += _mcnemar_block(tvu[cond], cond, indent=4)
    lines.append("")

    lines.append("  McNemar \u2014 provenance effect")
    lines.append("  (H\u2080: condition change does not alter call-rate)")
    for key, block in prov.items():
        label = _PAIR_LABELS.get(key, key)
        lines.append(f"    {label}:")
        lines.append(_mcnemar_row("overall", block.get("overall"), indent=6))
        for k, v in block.get("by_variant", {}).items():
            lines.append(_mcnemar_row(f"variant={k}", v, indent=8))
        for k, v in block.get("by_type", {}).items():
            lines.append(_mcnemar_row(k, v, indent=8))
        for k, v in block.get("by_boundary", {}).items():
            lines.append(_mcnemar_row(k, v, indent=8))
    return lines


# ---------------------------------------------------------------------------
# Per-model report
# ---------------------------------------------------------------------------

def build_report(summary: dict) -> str:
    W       = 110
    model   = summary["model"]
    run_at  = summary["run_at"]
    results = summary.get("results", [])
    mcn     = summary.get("mcnemar", {})

    lines = []
    lines.append(_dw(W))
    lines.append(f"  SOURCE AUTHORIZATION BENCHMARK  \u2014  {model}")
    lines.append(f"  run_at={run_at}  total={summary['total']}  "
                 f"passed={summary['passed']}  pass_rate={summary['pass_rate']:.1%}")
    lines.append(_dw(W))
    lines.append("")

    # ── attack type × authorization 2×2 table ───────────────────────────────
    #
    #  Rows  : IPI / Confused Deputy
    #  Cols  : Trusted (authorized) / Untrusted (unauthorized)
    #  These test fundamentally different failure modes:
    #    IPI×trusted      — delegated authority: agent should follow authenticated doc instructions
    #    IPI×untrusted    — classic IPI: agent must reject injected instructions
    #    CD×trusted       — authorized instruction following
    #    CD×untrusted     — classic confused deputy / privilege escalation
    #
    _COND_ORDER = {"with": 0, "implicit_r": 1, "without": 2}
    _COND_LABEL = {"with": "full_provenance", "implicit_r": "implicit_R", "without": "channel_only"}
    _TYPE_LABEL = {"indirect_injection": "IPI", "confused_deputy": "Confused Deputy"}
    present_conds = sorted({r.get("provenance", "with") for r in results},
                           key=lambda c: _COND_ORDER.get(c, 99))
    if results:
        lines.append("  Attack type × Authorization  (pass rate = correct outcome):")
        lines.append(f"  {'':34} {'Authorized':>16}   {'Unauthorized':>16}")
        lines.append(f"  {'':34} {'(trusted)':>16}   {'(untrusted)':>16}")
        for cond in present_conds:
            label = _COND_LABEL.get(cond, cond)
            lines.append(f"  {label}:")
            for atype in ["indirect_injection", "confused_deputy"]:
                row = []
                for variant in ["trusted", "untrusted"]:
                    cell = [r for r in results
                            if r.get("provenance") == cond
                            and r["type"] == atype
                            and r["variant"] == variant]
                    if cell:
                        p = sum(1 for r in cell if r["passed"])
                        row.append(f"{p}/{len(cell)} ({p/len(cell):.0%})")
                    else:
                        row.append("—")
                tlabel = _TYPE_LABEL.get(atype, atype)
                lines.append(f"    {tlabel:<32} {row[0]:>16}   {row[1]:>16}")
        lines.append("")

    # ── pass rate by provenance ──────────────────────────────────────────────
    by_prov = summary.get("by_provenance", {})
    if by_prov:
        lines.append("  Pass rate by condition (aggregate):")
        for cond in present_conds:
            if cond not in by_prov:
                continue
            stats = by_prov[cond]
            lines.append(f"    {_COND_LABEL.get(cond, cond):<18}  {stats['passed']}/{stats['total']}  "
                         f"({stats['pass_rate']:.1%})")
        lines.append("")

    # ── pass rate by boundary type ───────────────────────────────────────────
    boundary_totals: dict[str, list] = {}
    for r in results:
        boundary_totals.setdefault(r["boundary_type"], []).append(r["passed"])
    if boundary_totals:
        lines.append("  Pass rate by boundary type (all provenance conditions):")
        for bt in sorted(boundary_totals):
            ps = boundary_totals[bt]
            lines.append(f"    {bt:<24}  {sum(ps)}/{len(ps)}  ({sum(ps)/len(ps):.1%})")
        lines.append("")

    # ── McNemar ──────────────────────────────────────────────────────────────
    if mcn:
        lines.append(_w(W))
        lines += _mcnemar_section(mcn)
        lines.append("")

    # ── per-row results table ────────────────────────────────────────────────
    lines.append(_w(W))
    lines.append(f"  {'Scenario':<32} {'Type':<22} {'Boundary':<22} "
                 f"{'Variant':<10} {'Prov':<8} {'Expected':<14} {'Actual':<14} Result")
    lines.append(f"  {_w(W)}")
    _COND_ABBREV = {"with": "full", "without": "none", "implicit_r": "impl_R"}
    for r in sorted(results, key=lambda x: (x.get("provenance") or "", x["type"], x["scenario"])):
        mark  = "PASS \u2713" if r["passed"] else "FAIL \u2717"
        cond  = r.get("provenance") or "with"
        clabel = _COND_ABBREV.get(cond, cond[:6])
        lines.append(
            f"  {r['scenario']:<32} {r['type']:<22} {r['boundary_type']:<22} "
            f"{r['variant']:<10} {clabel:<8} {r['expected']:<14} {r['actual']:<14} {mark}"
        )
    lines.append(f"  {_w(W)}")
    lines.append("")

    # ── evidence (detail strings) ────────────────────────────────────────────
    lines.append("  Evidence (model output per scenario):")
    lines.append(f"  {_w(W)}")
    for r in sorted(results, key=lambda x: (x.get("provenance") or "", x["type"], x["scenario"])):
        status = "PASS" if r["passed"] else "FAIL"
        cond   = r.get("provenance") or "with"
        lines.append(f"  [{r['scenario']} | {r['variant']} | {cond} | {status}]")
        detail = (r.get("detail") or "").strip()
        if detail:
            lines.append(f"    {detail}")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate benchmark reports from eval summaries")
    p.add_argument("--model", default=None,
                   help="Model slug to report on (default: all models in results/)")
    return p.parse_args()


def main():
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_dirs = (
        [RESULTS_DIR / args.model] if args.model
        else sorted(d for d in RESULTS_DIR.iterdir() if d.is_dir())
    )

    if not model_dirs:
        print("No results found.", file=sys.stderr)
        sys.exit(1)

    for model_dir in model_dirs:
        summary_path = model_dir / "eval_summary.json"
        if not summary_path.exists():
            print(f"  skipping {model_dir.name} — no eval_summary.json", file=sys.stderr)
            continue

        summary  = json.loads(summary_path.read_text())
        report   = build_report(summary)

        out_dir  = REPORTS_DIR / model_dir.name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "report.txt"
        out_path.write_text(report)

        print(f"  Saved \u2192 {out_path}\n")


if __name__ == "__main__":
    main()
