#!/usr/bin/env python3
"""
benchmark/power_figure.py — Pre-hoc McNemar power curves.

Plots power vs effect size (p10) for fixed p01 noise floors, at n=45 pairs.
Applies to the trusted_vs_untrusted / with / by_type tests, which are the
primary discrimination hypotheses in the benchmark.

Usage
-----
  python3 benchmark/power_figure.py
  python3 benchmark/power_figure.py --n 90          # provenance_effect by_type
  python3 benchmark/power_figure.py --out fig.pdf
"""
import argparse
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.stats import norm

REPORTS_DIR = Path(__file__).parent / "reports"

# ---------------------------------------------------------------------------
# Power calculation
# ---------------------------------------------------------------------------

def mcnemar_power(n: int, p10: float, p01: float, alpha: float = 0.05) -> float:
    """
    Pre-hoc power for McNemar's test (chi-squared approximation, no Yates).

    Under H1 with n pairs and discordant probabilities p10, p01:
      ncp = sqrt(n) * |p10 - p01| / sqrt(p10 + p01)
      Power = Phi(ncp - z_{alpha/2})   [dominant term; upper tail negligible]
    """
    disc = p10 + p01
    if disc <= 0 or p10 <= p01:
        return 0.0
    z     = norm.ppf(1 - alpha / 2)
    ncp   = math.sqrt(n) * (p10 - p01) / math.sqrt(disc)
    return float(norm.cdf(ncp - z))


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def make_figure(n: int, alpha: float, out: Path) -> None:
    # p01 noise floors: fraction of pairs that go the "wrong" way
    noise_levels = [0.00, 0.05, 0.10, 0.15, 0.20]
    colors = ["#1a1a2e", "#16213e", "#0f3460", "#533483", "#e94560"]
    p10_grid = np.linspace(0, 1, 500)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for p01, color in zip(noise_levels, colors):
        # x = asymmetry d = p10 - p01; p10 = d + p01, valid while p10 <= 1
        d_grid = np.linspace(0, 1 - p01, 500)
        power  = np.array([mcnemar_power(n, d + p01, p01) for d in d_grid])
        label  = f"p₀₁ = {p01:.2f}" if p01 > 0 else "p₀₁ = 0  (no noise)"
        ax.plot(d_grid, power, color=color, linewidth=1.8, label=label)

    # Conventional 80 % reference
    ax.axhline(0.80, color="#888888", linestyle="--", linewidth=1.0, zorder=0)
    ax.text(0.01, 0.815, "80 % power", color="#666666", fontsize=8)

    # MDE dots: minimum asymmetry d needed for 80 % power
    for p01, color in zip(noise_levels, colors):
        target = 0.80
        lo, hi = 0.0, 1.0 - p01
        for _ in range(60):
            mid = (lo + hi) / 2
            if mcnemar_power(n, mid + p01, p01) < target:
                lo = mid
            else:
                hi = mid
        mde_d = (lo + hi) / 2
        if mde_d < 1.0 - p01 - 1e-6:
            ax.plot(mde_d, 0.80, "o", color=color, markersize=5, zorder=5)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Asymmetry  p₁₀ − p₀₁  (p₀₁ fixed per curve)", fontsize=10)
    ax.set_ylabel("Power  (1 − β)", fontsize=10)
    ax.set_title(
        f"Pre-hoc McNemar power  ·  n = {n} pairs,  α = {alpha}",
        fontsize=11, pad=10,
    )
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.legend(title="Noise floor", fontsize=8.5, title_fontsize=8.5,
              loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.2, linewidth=0.6)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Pre-hoc McNemar power figure")
    p.add_argument("--n",     type=int,   default=45)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--out",   type=Path,
                   default=REPORTS_DIR / "mcnemar_power.png")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_figure(args.n, args.alpha, args.out)
