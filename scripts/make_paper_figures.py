#!/usr/bin/env python3
"""Generate figure assets for the DIB paper.

F1/F2 are companion exports of the conceptual diagrams drawn inline in the
LaTeX draft. F3 is the empirical 2D scatter imported by the paper.
"""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


CLEAN_GREEN = "#2c7a3e"
BLIND_RED = "#c0392b"
NEUTRAL_GRAY = "#7f8c8d"
GRID_GRAY = "#e6e6e6"


def fig1_orthogonality(out_dir: Path, rows=None) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Behavioral acceptance  $A_S$")
    ax.set_ylabel("Identity acceptance")

    # Quadrant shading: blind spot (bottom-right) prominent, clean (top-right) soft green.
    blind_rect = Rectangle((0.50, -0.05), 0.50, 0.45, facecolor=BLIND_RED,
                           alpha=0.10, edgecolor="none", zorder=0)
    clean_rect = Rectangle((0.50, 0.55), 0.50, 0.50, facecolor=CLEAN_GREEN,
                           alpha=0.08, edgecolor="none", zorder=0)
    ax.add_patch(blind_rect)
    ax.add_patch(clean_rect)

    # Quadrant text labels
    ax.text(0.75, 0.93, "CLEAN", ha="center", va="top",
            fontsize=11, color=CLEAN_GREEN, fontweight="bold")
    ax.text(0.75, 0.86, "behavior pass $\\wedge$ identity pass",
            ha="center", va="top", fontsize=7.5, color=CLEAN_GREEN, style="italic")

    ax.text(0.75, 0.32, "BLIND SPOT", ha="center", va="top",
            fontsize=11, color=BLIND_RED, fontweight="bold")
    ax.text(0.75, 0.25, "behavior pass $\\wedge$ identity fail",
            ha="center", va="top", fontsize=7.5, color=BLIND_RED, style="italic")
    ax.text(0.75, 0.19, "what this paper names", ha="center", va="top",
            fontsize=7.5, color=BLIND_RED, style="italic")

    ax.text(0.25, 0.32, "caught by\nbehavior eval", ha="center", va="center",
            fontsize=8, color=NEUTRAL_GRAY, style="italic")

    ax.text(0.25, 0.78, "rare\n(identity intact, behavior fails)",
            ha="center", va="center", fontsize=7.5, color=NEUTRAL_GRAY, style="italic")

    # Acceptance threshold tau
    ax.axvline(0.50, color=NEUTRAL_GRAY, linestyle=":", linewidth=0.6, zorder=1)


    # Empirical points from the synthetic aggregate.
    if rows:
        for cond, r in rows.items():
            x = float(r["standard_output_pass_rate"])
            y = float(r["integrity_pass_rate"])
            if cond == "clean":
                ax.scatter([x], [y], s=110, color=CLEAN_GREEN, zorder=5, marker="o",
                           edgecolor="white", linewidth=1.2)
            elif x >= 0.80:
                ax.scatter([x], [y], s=70, color=BLIND_RED, zorder=5, marker="X",
                           edgecolor="white", linewidth=0.8)
            else:
                ax.scatter([x], [y], s=70, color=NEUTRAL_GRAY, zorder=5, marker="X",
                           edgecolor="white", linewidth=0.8)

    # Inline tiny legend, inside the upper-left quadrant
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", color="w", markerfacecolor=CLEAN_GREEN,
               markersize=8, markeredgecolor="white", label="clean"),
        Line2D([], [], marker="X", color="w", markerfacecolor=BLIND_RED,
               markersize=8, markeredgecolor="white", label="behavior-preserving tamper"),
        Line2D([], [], marker="X", color="w", markerfacecolor=NEUTRAL_GRAY,
               markersize=8, markeredgecolor="white", label="behavior-visible tamper"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.02, 0.62),
              frameon=False, fontsize=7.5)

    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticks([0, 0.5, 1.0])
    ax.grid(True, color=GRID_GRAY, linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "fig1_orthogonality.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig1_orthogonality.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig1_orthogonality.{pdf,png}")


def fig2_stack(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    components = [
        ("Postprocessor",   "rewrite / filter / redact",            "#e57373"),
        ("Runtime",         "packages, container, hardware",        "#ffb74d"),
        ("Retrieval",       "policy + corpus + retrieved docs",     "#fff176"),
        ("Tooling",         "tool specs + router",                  "#aed581"),
        ("System prompt",   "policy text + developer prompt",       "#4dd0e1"),
        ("Model + tokenizer", "weights, vocab, chat template",       "#7986cb"),
    ]

    n = len(components)
    bar_h = 0.78
    y_positions = list(range(n))

    # Horizontal header
    ax.text(0.30, -1.0, "Deployment stack", ha="center", va="center",
            fontsize=10, fontweight="bold", color="#263238")
    ax.text(0.95, -1.0, "identity channel", ha="center", va="center",
            fontsize=9, color="#37474f", style="italic")

    for y, (name, detail, color) in zip(y_positions, components):
        box = FancyBboxPatch(
            (0.05, y - bar_h / 2), 0.50, bar_h,
            boxstyle="round,pad=0.005,rounding_size=0.02",
            facecolor=color, edgecolor="#37474f", linewidth=0.6, alpha=0.85,
        )
        ax.add_patch(box)
        ax.text(0.080, y, name, va="center", ha="left", fontsize=9, fontweight="bold")
        ax.text(0.57, y, detail, va="center", ha="left", fontsize=8, color="#37474f")
        # Per-component hash token on the right
        ax.text(0.95, y, "$h_c$", ha="center", va="center", fontsize=9,
                color="#37474f", family="serif")
        # Dotted leader from row to its hash
        ax.plot([0.90, 0.93], [y, y], color="#90a4ae", linewidth=0.5, linestyle=":")

    # Vertical guide on the right joining the hash column
    ax.plot([0.95, 0.95], [-0.5, n - 0.5], color="#cfd8dc", linewidth=0.8, zorder=0)

    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(-1.4, n - 0.4)
    ax.invert_yaxis()
    ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "fig2_stack.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig2_stack.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2_stack.{pdf,png}")


def _read_condition_rows(csv_path: Path):
    rows = {}
    with csv_path.open() as f:
        for r in csv.DictReader(f):
            rows[r["condition"]] = r
    return rows


def fig3_result_scatter(out_dir: Path, repo_root: Path) -> None:
    """Evaluator-profile plot on the synthetic suite.

    Each condition is traced as a profile across four output-only evaluators
    (Standard, Strict, Forensic, Canary) and then the identity channel. The
    point the figure makes that the table cannot: every tamper rides high
    alongside clean through all four behavioral evaluators, then drops to zero
    only at the identity channel. Adds the Forensic/Canary evaluators omitted
    from the consolidated table and shows the divergence as a single visual
    cliff rather than a grid of numbers.
    """
    synth = _read_condition_rows(repo_root / "paper_runs/aggregate/aggregate_condition.csv")

    evaluators = [
        ("standard_output_pass_rate", "Standard"),
        ("strict_output_pass_rate", "Strict"),
        ("output_forensic_pass_rate", "Forensic"),
        ("canary_probe_pass_rate", "Canary"),
    ]
    x_behav = [0.0, 1.0, 2.0, 3.0]
    x_id = 4.6
    xs_all = x_behav + [x_id]

    def profile(cond_key):
        r = synth.get(cond_key, {})
        ys = []
        for col, _ in evaluators:
            try:
                ys.append(float(r.get(col, 0)) * 100.0)
            except (TypeError, ValueError):
                ys.append(0.0)
        try:
            ys.append(float(r.get("integrity_pass_rate", 0)) * 100.0)
        except (TypeError, ValueError):
            ys.append(0.0)
        return ys

    behavior_preserving = ["model_substitution", "prompt_tamper", "tool_tamper",
                           "retrieval_tamper", "dependency_tamper"]

    CLEAN = "#2c7a3e"
    TAMP = "#c0392b"
    POST = "#d98014"

    fig, ax = plt.subplots(figsize=(8.4, 3.9))

    # Behavior-preserving tampers drawn as one visual class.
    for ck in behavior_preserving:
        ax.plot(xs_all, profile(ck), color=TAMP, alpha=0.42, lw=1.6, marker="o",
                markersize=4, markerfacecolor=TAMP, markeredgecolor="white",
                markeredgewidth=0.5, zorder=3)

    # Postprocess: the one behavior-visible control.
    ax.plot(xs_all, profile("postprocess_tamper"), color=POST, alpha=0.95, lw=1.9,
            ls="--", marker="s", markersize=4.5, markerfacecolor=POST,
            markeredgecolor="white", markeredgewidth=0.5, zorder=4)

    # Clean reference: stays high everywhere, including identity.
    ax.plot(xs_all, profile("clean"), color=CLEAN, lw=2.5, marker="o",
            markersize=5.5, markerfacecolor=CLEAN, markeredgecolor="white",
            markeredgewidth=0.8, zorder=5)

    # Acceptance threshold across the behavioral region.
    ax.plot([-0.3, 3.3], [80, 80], color="black", lw=0.7, ls=":", alpha=0.5, zorder=1)
    ax.text(-0.28, 82, r"$\tau=80\%$", fontsize=8, color="black", alpha=0.65,
            style="italic", va="bottom", ha="left")

    # Separator between the behavioral evaluators and the identity channel.
    ax.axvline(3.8, color="#9aa0a6", lw=0.9, ls="--", alpha=0.8, zorder=1)

    # Region headers.
    ax.text(1.5, 114, "behavioral evidence (output-only)", ha="center", va="bottom",
            fontsize=9.5, color="#333333", fontweight="bold")
    ax.text(4.6, 114, "identity evidence", ha="center", va="bottom",
            fontsize=9.5, color="#333333", fontweight="bold")

    # The cliff annotation.
    ax.annotate("every tamper\ndrops to 0", xy=(x_id, 5), xytext=(3.95, 46),
                fontsize=8.5, color=TAMP, fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=TAMP, lw=1.0))

    ax.set_xticks(xs_all)
    ax.set_xticklabels([lbl for _, lbl in evaluators] + ["Identity"])
    ax.set_ylabel("pass rate (%)")
    ax.set_ylim(-4, 122)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.45, 5.25)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=CLEAN, lw=2.5, marker="o", markersize=6,
               markeredgecolor="white", label="clean"),
        Line2D([], [], color=TAMP, lw=1.8, alpha=0.6, marker="o", markersize=5,
               markeredgecolor="white", label=r"behavior-preserving tampers ($\times$5)"),
        Line2D([], [], color=POST, lw=1.9, ls="--", marker="s", markersize=5,
               markeredgecolor="white", label="postprocess (behavior-visible)"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, 0.01),
              frameon=True, framealpha=0.92, fontsize=8.5, edgecolor="#cccccc")

    fig.tight_layout()
    fig.savefig(out_dir / "fig3_results.pdf", bbox_inches="tight")
    fig.savefig(out_dir / "fig3_results.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig3_results.{pdf,png}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="paper/figures")
    parser.add_argument("--repo_root", default=".")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(args.repo_root)

    synth = _read_condition_rows(repo_root / "paper_runs/aggregate/aggregate_condition.csv")
    fig1_orthogonality(out_dir, synth)
    fig2_stack(out_dir)
    fig3_result_scatter(out_dir, repo_root)


if __name__ == "__main__":
    main()
