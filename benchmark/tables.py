#!/usr/bin/env python3
"""
benchmark/tables.py — Emit LaTeX tabulars for the paper.

Reads every benchmark/results/<model>/eval_summary.json and writes:

  tables/tab1_effects.tex    — Δ pass rate vs baseline, grouped by
                               model × condition × (attack × variant).
                               Sig asterisks inline; MDE / CIs deferred
                               to appendix.
  tables/tab2_utility.tex    — Trusted pass rate by model × condition,
                               split by attack type.

Requires booktabs + multirow in the host document.

Usage
-----
  python3 benchmark/tables.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import stats

from power import mde

RESULTS_DIR = Path(__file__).parent / "results"
TAB_DIR     = Path(__file__).parent / "tables"
TAB_DIR.mkdir(exist_ok=True)

CONDITIONS        = ["bare", "labels_unbound", "labels", "capabilities", "full"]
CONDS_VS_BASELINE = ["labels_unbound", "labels", "capabilities", "full"]
COND_LATEX = {
    "bare":           "baseline",
    "labels_unbound": "tags",
    "labels":         "tags + token",
    "capabilities":   "scopes",
    "full":           "tags + token + scopes",
}
VARIANTS = ["trusted", "untrusted", "untrusted_inline_spoof"]
VAR_SHORT = {"trusted": "trust.", "untrusted": "unauth.", "untrusted_inline_spoof": "spoof"}
TYPES = ["indirect_injection", "confused_deputy"]
TYPE_SHORT = {"indirect_injection": "IPI", "confused_deputy": "CD"}

MODEL_LABEL = {
    "claude-sonnet-4-6":              "Sonnet 4.6",
    "deepseek-deepseek-v3.2":         "DeepSeek V3.2",
    "openai-gpt-5.4":                 "GPT-5.4",
    "gemini-gemini-3.1-pro-preview":  "Gemini 3.1 Pro",
    "moonshotai-kimi-k2.6":           "Kimi K2.6",
    "minimax-minimax-m2.7":           "MiniMax M2.7",
}


# ---------------------------------------------------------------------------
# Ingestion & paired stats (shared shape with figures.py)
# ---------------------------------------------------------------------------

def load_model(results_dir: Path):
    sp = results_dir / "eval_summary.json"
    if not sp.exists():
        return None
    s = json.loads(sp.read_text())
    c, p = Counter(), Counter()
    per_scenario: dict = {}
    for r in s["results"]:
        k = (r.get("condition"), r["type"], r["variant"])
        c[k] += 1
        if r["passed"]: p[k] += 1
        scen = r["scenario"]
        per_scenario.setdefault((r["type"], r["variant"], scen), {})[r["condition"]] = r["passed"]
    cells = {k: {"n": c[k], "pass": p[k],
                 "pass_rate": p[k]/c[k] if c[k] else 0}
             for k in c}
    return {
        "slug":     results_dir.name,
        "display":  MODEL_LABEL.get(results_dir.name, results_dir.name),
        "cells":    cells,
        "per_scen": per_scenario,
    }


def discover_models():
    return [m for d in sorted(RESULTS_DIR.iterdir()) if d.is_dir()
            for m in [load_model(d)] if m is not None]


def paired_effect(model, type_, variant, cond_a, cond_b):
    """Δ pass rate (cond_b − cond_a), McNemar exact p, discordance n_d."""
    pa_list, pb_list = [], []
    for (t, v, _), passes in model["per_scen"].items():
        if t != type_ or v != variant: continue
        a, b = passes.get(cond_a), passes.get(cond_b)
        if a is None or b is None: continue
        pa_list.append(int(a)); pb_list.append(int(b))
    n = len(pa_list)
    if n == 0: return None
    pa = np.array(pa_list); pb = np.array(pb_list)
    delta = (pb.mean() - pa.mean()) * 100
    n10 = int(((pa == 0) & (pb == 1)).sum())
    n01 = int(((pa == 1) & (pb == 0)).sum())
    n_d = n10 + n01
    p_val = (stats.binomtest(min(n10, n01), n_d, 0.5, alternative="two-sided").pvalue
             if n_d else None)
    return dict(delta=delta, n10=n10, n01=n01, n_d=n_d, p=p_val, n=n)


def sig_asterisks(p):
    if p is None: return ""
    if p < 0.001: return "$^{***}$"
    if p < 0.01:  return "$^{**}$"
    if p < 0.05:  return "$^{*}$"
    return ""


def fmt_delta(eff):
    """Format a paired effect as '+12*' or '--' if unavailable."""
    if eff is None: return "--"
    return f"{eff['delta']:+.0f}{sig_asterisks(eff['p'])}"


def fmt_rate(cell):
    """Format a raw pass-rate cell as an integer percentage."""
    if cell is None or cell.get("n", 0) == 0: return "--"
    return f"{cell['pass_rate']*100:.0f}"


# ---------------------------------------------------------------------------
# Table 1 — paired Δ from baseline
# ---------------------------------------------------------------------------

def build_tab1(models) -> str:
    """
    Columns: 3 IPI variants + 3 CD variants (T / U / S)
    Rows:    per-model block of 4 conditions (all non-bare)
    Cells:   Δ pp with McNemar significance asterisks
    """
    n_var = len(VARIANTS)
    colspec = "ll" + (("r" * n_var + "|") * (len(TYPES) - 1)) + ("r" * n_var)
    # booktabs prefers no vlines; use cmidrule. We'll emit without the '|'.
    colspec = "ll" + "r" * (n_var * len(TYPES))

    lines = []
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\toprule")
    # Group header (IPI / CD)
    head_cells = []
    for t in TYPES:
        head_cells.append(r"\multicolumn{" + str(n_var) + r"}{c}{" + TYPE_SHORT[t] + "}")
    lines.append(" & ".join(["", ""] + head_cells) + r" \\")
    # cmidrules under group headers
    cmids = []
    col0 = 3
    for _ in TYPES:
        cmids.append(rf"\cmidrule(lr){{{col0}-{col0 + n_var - 1}}}")
        col0 += n_var
    lines.append(" ".join(cmids))
    # Variant subheader
    sub = ["Model", "Condition"] + [VAR_SHORT[v] for v in VARIANTS] * len(TYPES)
    lines.append(" & ".join(sub) + r" \\")
    lines.append(r"\midrule")

    # Per-model rows
    for mi, m in enumerate(models):
        for ci, cond in enumerate(CONDS_VS_BASELINE):
            cells = []
            if ci == 0:
                cells.append(r"\multirow{" + str(len(CONDS_VS_BASELINE)) + r"}{*}{" +
                             m["display"] + "}")
            else:
                cells.append("")
            cells.append(COND_LATEX[cond])
            for t in TYPES:
                for v in VARIANTS:
                    eff = paired_effect(m, t, v, "bare", cond)
                    cells.append(fmt_delta(eff))
            lines.append(" & ".join(cells) + r" \\")
        if mi != len(models) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Caption / note (commented so the author can paste into their \caption)
    notes = [
        r"% Note: cells are Δ pass rate (percentage points) vs baseline (no primitives),",
        r"% paired by scenario. T = trusted (should act), U = unauthorized (should refuse),",
        r"% S = unauthorized + inline-spoof (forged auth marker; should still refuse).",
        r"% Significance: exact McNemar two-sided p. * p<.05, ** p<.01, *** p<.001.",
        r"% Bootstrap CIs and MDE-at-80%-power reported per-cell in Appendix (Table A.x).",
    ]
    return "\n".join(notes) + "\n\n" + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Table 2 — trusted-variant pass rate (utility cost)
# ---------------------------------------------------------------------------

def build_tab2(models) -> str:
    """
    Columns: 5 conditions (incl. baseline)
    Rows:    per attack-type block of N models
    Cells:   raw trusted pass rate (%)
    """
    colspec = "ll" + "r" * len(CONDITIONS)
    lines = []
    lines.append(r"\begin{tabular}{" + colspec + "}")
    lines.append(r"\toprule")
    header = ["", "Model"] + [COND_LATEX[c] for c in CONDITIONS]
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    for ti, t in enumerate(TYPES):
        for mi, m in enumerate(models):
            cells = []
            if mi == 0:
                cells.append(r"\multirow{" + str(len(models)) + r"}{*}{" +
                             TYPE_SHORT[t] + "}")
            else:
                cells.append("")
            cells.append(m["display"])
            for c in CONDITIONS:
                cell = m["cells"].get((c, t, "trusted"))
                cells.append(fmt_rate(cell))
            lines.append(" & ".join(cells) + r" \\")
        if ti != len(TYPES) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    notes = [
        r"% Note: trusted-variant pass rate (%); agent should act in all cells.",
        r"% Drops from baseline → primitives quantify the utility cost of authorization.",
    ]
    return "\n".join(notes) + "\n\n" + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Appendix Table A — per-cell MDE and discordance for Table 1's comparisons
# ---------------------------------------------------------------------------

def build_tab_appendix_mde(models) -> str:
    """One row per (model, condition-vs-baseline, type, variant): n_d, MDE, p."""
    colspec = "lll" + "l" * 3 + "r" * 3
    lines = []
    lines.append(r"\begin{tabular}{llllllrrr}")
    lines.append(r"\toprule")
    lines.append(" & ".join([
        "Model", "Condition", "Attack", "Variant",
        r"$n_d$", r"$n_{10}$", r"$n_{01}$",
        r"$\delta$ (pp)", r"MDE (pp, 80\%)", r"$p$"
    ]) + r" \\")
    lines.append(r"\midrule")
    for m in models:
        for cond in CONDS_VS_BASELINE:
            for t in TYPES:
                for v in VARIANTS:
                    eff = paired_effect(m, t, v, "bare", cond)
                    if eff is None: continue
                    mde_pp = mde(eff["n_d"]) * 100 if eff["n_d"] else float("inf")
                    p_str = f"{eff['p']:.3g}" if eff["p"] is not None else "--"
                    mde_str = f"{mde_pp:.1f}" if mde_pp != float("inf") else "--"
                    lines.append(" & ".join([
                        m["display"], COND_LATEX[cond],
                        TYPE_SHORT[t], VAR_SHORT[v],
                        str(eff["n_d"]), str(eff["n10"]), str(eff["n01"]),
                        f"{eff['delta']:+.1f}", mde_str, p_str,
                    ]) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    models = discover_models()
    if not models:
        print("No models found in results/"); return
    print(f"Found {len(models)} models: {[m['display'] for m in models]}")

    outputs = {
        "tab1_effects.tex":     build_tab1(models),
        "tab2_utility.tex":     build_tab2(models),
        "tabA_appendix_mde.tex": build_tab_appendix_mde(models),
    }
    for name, content in outputs.items():
        path = TAB_DIR / name
        path.write_text(content)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
