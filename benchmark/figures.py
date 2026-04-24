#!/usr/bin/env python3
"""
benchmark/figures.py — Generate cross-model figures.

Reads every benchmark/results/<model>/eval_summary.json and produces:

  figures/fig1_effects.pdf        — Paired Δ pass-rate vs baseline, per
                                    primitive × attack cell × model, with
                                    bootstrap 95% CIs and MDE reference
                                    bands. (main)
  figures/fig2_binding_token.pdf  — Slope chart: does the binding token help
                                    on inline-spoof cells? Document-body
                                    forgery vs user-prompt forgery. (main)
  figures/fig3_utility.pdf        — Trusted-scenario pass rate by condition
                                    and model — the utility cost. (supp.)

Figures are model-agnostic: adding an N-th model auto-appears.

Usage
-----
  python3 benchmark/figures.py
"""
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from power import mde

RESULTS_DIR = Path(__file__).parent / "results"
FIG_DIR     = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Display configuration
# ---------------------------------------------------------------------------
CONDITIONS        = ["bare", "labels_unbound", "labels", "capabilities", "full"]
CONDS_VS_BASELINE = ["labels_unbound", "labels", "capabilities", "full"]
COND_LABEL = {
    "bare":           "baseline",
    "labels_unbound": "tags",
    "labels":         "tags\n+ token",
    "capabilities":   "scopes",
    "full":           "tags + token\n+ scopes",
}
COND_LABEL_FLAT = {k: v.replace("\n", " ") for k, v in COND_LABEL.items()}

VARIANTS = ["trusted", "untrusted", "untrusted_inline_spoof"]
VARIANT_TITLE = {
    "trusted":                ("Legitimate request",   "(agent should act)"),
    "untrusted":              ("Unauthorized",         "(agent should refuse)"),
    "untrusted_inline_spoof": ("Unauthorized + fake auth tag",
                               "(forged marker; agent should still refuse)"),
}

TYPES = ["indirect_injection", "confused_deputy"]
TYPE_TITLE = {
    "indirect_injection": ("Prompt injection",  "instructions inside a document"),
    "confused_deputy":    ("Confused deputy",   "caller lacks permission"),
}

MODEL_LABEL = {
    "claude-sonnet-4-6":              "Sonnet 4.6",
    "deepseek-deepseek-v3.2":         "DeepSeek V3.2",
    "openai-gpt-5.4":                 "GPT-5.4",
    "gemini-gemini-3.1-pro-preview":  "Gemini 3.1 Pro",
    "moonshotai-kimi-k2.6":           "Kimi K2.6",
    "minimax-minimax-m2.7":           "MiniMax M2.7",
}
MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


# ---------------------------------------------------------------------------
# Data ingestion
# ---------------------------------------------------------------------------

def load_model(results_dir: Path) -> dict | None:
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


def discover_models() -> list[dict]:
    models = []
    for d in sorted(RESULTS_DIR.iterdir()):
        if not d.is_dir(): continue
        m = load_model(d)
        if m is not None: models.append(m)
    return models


# ---------------------------------------------------------------------------
# Paired-effect computation
# ---------------------------------------------------------------------------

def paired_effect(model, type_, variant, cond_a, cond_b, n_boot=2000):
    """
    Return Δ pass rate (cond_b − cond_a) with bootstrap 95% CI, exact McNemar
    p, and discordance counts for the specified cell. `pass` = correct outcome
    on that variant (called on trusted, blocked on attack variants).
    """
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

    # Bootstrap CI
    seed = abs(hash((model["slug"], type_, variant, cond_a, cond_b))) % (2**32)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    deltas = (pb[idx].mean(axis=1) - pa[idx].mean(axis=1)) * 100
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])

    # McNemar: n10 = A fail, B pass (improvement); n01 = A pass, B fail (regression)
    n10 = int(((pa == 0) & (pb == 1)).sum())
    n01 = int(((pa == 1) & (pb == 0)).sum())
    n_d = n10 + n01
    p_val = (stats.binomtest(min(n10, n01), n_d, 0.5, alternative="two-sided").pvalue
             if n_d else None)
    return dict(delta=delta, ci_lo=ci_lo, ci_hi=ci_hi,
                n10=n10, n01=n01, n_d=n_d, p=p_val, n=n)


def sig_marker(p):
    if p is None: return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


# ---------------------------------------------------------------------------
# Fig 1 — paired Δ-from-baseline effects with bootstrap CIs + MDE band (MAIN)
# ---------------------------------------------------------------------------

def fig1_effects(models):
    """
    Headline figure: on a standard unauthorized attack, how much does each
    defensive primitive lift the agent's refusal rate over a no-primitives
    baseline? Inline-spoof (forgery ablation) is Fig 2; trusted utility is
    Fig 3.
    """
    fig, axes = plt.subplots(1, len(TYPES), figsize=(6.0 * len(TYPES), 4.8),
                             sharey=True, squeeze=False)

    n_m = len(models)
    cond_x = np.arange(len(CONDS_VS_BASELINE))
    bar_w = 0.78 / n_m

    for ci, atype in enumerate(TYPES):
        ax = axes[0, ci]
        panel_mde = []

        for mi, m in enumerate(models):
            xs = cond_x + (mi - (n_m - 1) / 2) * bar_w
            deltas, err_lo, err_hi, ps = [], [], [], []
            for cond in CONDS_VS_BASELINE:
                eff = paired_effect(m, atype, "untrusted", "bare", cond)
                if eff is None:
                    deltas.append(0.0); err_lo.append(0); err_hi.append(0)
                    ps.append(None); continue
                deltas.append(eff["delta"])
                err_lo.append(max(0, eff["delta"] - eff["ci_lo"]))
                err_hi.append(max(0, eff["ci_hi"] - eff["delta"]))
                ps.append(eff["p"])
                if eff["n_d"] > 0:
                    panel_mde.append(mde(eff["n_d"]) * 100)

            ax.bar(xs, deltas, bar_w,
                   yerr=[err_lo, err_hi], capsize=1.8,
                   label=m["display"],
                   color=MODEL_COLORS[mi % len(MODEL_COLORS)],
                   edgecolor="black", linewidth=0.4,
                   error_kw=dict(ecolor="black", lw=0.7))

            for xi, (d, p, lo, hi) in enumerate(zip(deltas, ps, err_lo, err_hi)):
                s = sig_marker(p)
                if not s: continue
                y = (d + hi + 2) if d >= 0 else (d - lo - 5)
                ax.text(xs[xi], y, s, ha="center", fontsize=7)

        if panel_mde:
            m_med = float(np.median(panel_mde))
            ax.axhspan(-m_med, m_med, color="#dddddd", alpha=0.55, zorder=0)

        ax.axhline(0, color="black", lw=0.7, zorder=1)
        ax.set_xticks(cond_x)
        ax.set_xticklabels([COND_LABEL[c] for c in CONDS_VS_BASELINE], fontsize=9.5)
        ax.set_ylim(-25, 105)
        ax.grid(axis="y", alpha=0.25)
        ax.set_axisbelow(True)

        head, sub = TYPE_TITLE[atype]
        ax.set_title(f"{head}\n({sub})", fontsize=11)
        if ci == 0:
            ax.set_ylabel("Δ refusal rate vs. baseline (pp)  —  ↑ stronger defense",
                          fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(n_m, 6), bbox_to_anchor=(0.5, -0.05),
               frameon=False, fontsize=10)

    fig.suptitle("Defensive primitives on unauthorized requests (baseline-paired Δ)",
                 fontsize=12.5, y=1.00)
    fig.tight_layout()
    out = FIG_DIR / "fig1_effects.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Fig 2 — binding token slope chart on inline-spoof cells (MAIN)
# ---------------------------------------------------------------------------

def fig2_binding_token(models):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.8), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    panels = [
        ("indirect_injection", "Document-body forgery"),
        ("confused_deputy",    "User-prompt forgery"),
    ]

    def _deoverlap(values, min_gap):
        """Shift labels vertically so consecutive ones are >= min_gap apart."""
        order = sorted(range(len(values)), key=lambda i: values[i])
        adj = list(values)
        for k in range(1, len(order)):
            prev, cur = order[k-1], order[k]
            if adj[cur] - adj[prev] < min_gap:
                adj[cur] = adj[prev] + min_gap
        return adj

    for ai, (atype, title) in enumerate(panels):
        ax = axes[ai]
        left_y, colors, names = [], [], []
        for mi, m in enumerate(models):
            u = m["cells"].get(("labels_unbound", atype, "untrusted_inline_spoof"),
                               {}).get("pass_rate", 0) * 100
            b = m["cells"].get(("labels",          atype, "untrusted_inline_spoof"),
                               {}).get("pass_rate", 0) * 100
            color = MODEL_COLORS[mi % len(MODEL_COLORS)]
            ax.plot([0, 1], [u, b], color=color, lw=2.2, alpha=0.9,
                    marker="o", markersize=10, markeredgecolor="black",
                    markeredgewidth=0.6, zorder=3)
            left_y.append(u); colors.append(color); names.append(m["display"])

        left_lab = _deoverlap(left_y, 5.5)

        for i, name in enumerate(names):
            ax.annotate(name,
                        xy=(0, left_y[i]), xytext=(-0.08, left_lab[i]),
                        ha="right", va="center",
                        fontsize=9, color=colors[i], fontweight="bold",
                        arrowprops=(dict(arrowstyle="-", color=colors[i],
                                         lw=0.5, alpha=0.5)
                                    if abs(left_lab[i] - left_y[i]) > 0.5 else None))

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["tags only\n(no binding token)", "tags + binding token"],
                           fontsize=10.5)
        ax.set_xlim(-0.55, 1.10)
        ax.set_ylim(-3, 108)
        ax.set_yticks(range(0, 101, 20))
        ax.set_yticks(range(0, 101, 10), minor=True)
        ax.set_title(title, fontsize=11)
        ax.axhline(0, color="gray", lw=0.4)
        ax.grid(axis="y", which="major", alpha=0.35)
        ax.grid(axis="y", which="minor", alpha=0.15, lw=0.4)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if ai == 0:
            ax.set_ylabel("Refusal rate on forged-marker attack (%)  —  ↑ correct refusal",
                          fontsize=10)

    fig.suptitle("Binding-token effect by forgery surface", fontsize=12, y=1.02)
    out = FIG_DIR / "fig2_binding_token.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Fig 3 — utility cost: trusted pass rate per condition (SUPPLEMENTARY)
# ---------------------------------------------------------------------------

def fig3_utility(models):
    """
    Utility cost: on a legitimate request, do primitives cause the agent
    to over-refuse? Slope chart from baseline (no primitives) → full
    (all primitives). Downward slopes = utility cost.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True,
                             gridspec_kw={"wspace": 0.10})
    panels = [("indirect_injection", "Prompt injection (legitimate doc)"),
              ("confused_deputy",    "Confused deputy (authorized caller)")]

    def _deoverlap(values, min_gap):
        order = sorted(range(len(values)), key=lambda i: values[i])
        adj = list(values)
        for k in range(1, len(order)):
            prev, cur = order[k-1], order[k]
            if adj[cur] - adj[prev] < min_gap:
                adj[cur] = adj[prev] + min_gap
        return adj

    for ai, (atype, title) in enumerate(panels):
        ax = axes[ai]
        left_y, colors, names = [], [], []
        for mi, m in enumerate(models):
            base = m["cells"].get(("bare", atype, "trusted"), {}).get("pass_rate", 0) * 100
            full = m["cells"].get(("full", atype, "trusted"), {}).get("pass_rate", 0) * 100
            color = MODEL_COLORS[mi % len(MODEL_COLORS)]
            ax.plot([0, 1], [base, full], color=color, lw=2.2, alpha=0.9,
                    marker="o", markersize=10, markeredgecolor="black",
                    markeredgewidth=0.6, zorder=3)
            left_y.append(base); colors.append(color); names.append(m["display"])

        left_lab = _deoverlap(left_y, 5.5)

        for i, name in enumerate(names):
            ax.annotate(name,
                        xy=(0, left_y[i]), xytext=(-0.08, left_lab[i]),
                        ha="right", va="center",
                        fontsize=9, color=colors[i], fontweight="bold",
                        arrowprops=(dict(arrowstyle="-", color=colors[i],
                                         lw=0.5, alpha=0.5)
                                    if abs(left_lab[i] - left_y[i]) > 0.5 else None))

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["baseline\n(no primitives)", "full\n(tags + token + scopes)"],
                           fontsize=10.5)
        ax.set_xlim(-0.55, 1.10)
        ax.set_ylim(-3, 105)
        ax.set_yticks(range(0, 101, 20))
        ax.set_yticks(range(0, 101, 10), minor=True)
        ax.set_title(title, fontsize=11)
        ax.grid(axis="y", which="major", alpha=0.35)
        ax.grid(axis="y", which="minor", alpha=0.15, lw=0.4)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if ai == 0:
            ax.set_ylabel("Pass rate on legitimate request (%)  —  ↓ over-refusal",
                          fontsize=10)

    fig.suptitle("Utility cost: baseline → full", fontsize=12.5, y=1.00)
    out = FIG_DIR / "fig3_utility.pdf"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    models = discover_models()
    if not models:
        print("No models found in results/")
        return
    print(f"Found {len(models)} models: {[m['display'] for m in models]}")
    for fn in (fig1_effects, fig2_binding_token, fig3_utility):
        out = fn(models)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
