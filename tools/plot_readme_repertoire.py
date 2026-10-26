"""Render the concise main README figure from retained repertoire measurements.

Example (inside the verified plotting runtime):
    python tools/plot_readme_repertoire.py \
        --data docs/data/abodybuilder3_repertoire_scale_261026.json \
        --output docs/assets/abodybuilder3_main_throughput_memory_261026

The detailed study and this generator stay on dev; main ships the PNG/SVG.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    counts = [1_000, 10_000, 100_000]
    models = [("base", "Base", "#087F8C"), ("plddt", "pLDDT", "#D77819")]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 12,
        "svg.fonttype": "none", "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.4))
    fig.subplots_adjust(left=.075, right=.98, bottom=.25, top=.70, wspace=.24)
    dark = "#172B42"
    fig.suptitle("~7,700 antibody structures/minute on one DGX Spark",
                 x=.075, y=.975, ha="left", fontsize=20, weight="bold", color=dark)
    fig.text(.075, .902,
             "100,000 distinct pairs in ~13 minutes per model · strict FP32 · every PDB written",
             fontsize=12, color="#47586C")

    positions = np.arange(len(counts))
    width = .34
    for model_index, (model, label, color) in enumerate(models):
        rows = [next(row for row in data["summaries"]
                     if row["cohort"] == f"oas_{count}" and row["model"] == model)
                for count in counts]
        x = positions + (model_index - .5) * .38
        for panel, ax in enumerate(axes):
            if panel == 0:
                values = np.array([row["structures_per_minute"] for row in rows])
                bounds = np.array([row["throughput_range"] for row in rows])
                texts = [f"{value:,.0f}" for value in values]
                offset = 160
            else:
                values = np.array([row["peak_additional_ram_gib"] for row in rows])
                bounds = np.array([row["ram_sensitivity_gib"] for row in rows])
                texts = [f"~{value:.2f}" for value in values]
                offset = .17
            bars = ax.bar(x, values, width=width, color=color,
                          label=label, zorder=3)
            ax.errorbar(x, values, yerr=[values - bounds[:, 0], bounds[:, 1] - values],
                        fmt="none", color=dark, capsize=4, lw=1.4, zorder=4)
            for bar, text in zip(bars, texts):
                ax.text(bar.get_x() + width / 2, bar.get_height() + offset,
                        text, ha="center", va="bottom", fontsize=11.5,
                        color=dark, weight="bold")

    for ax in axes:
        ax.set_xticks(positions, [f"{count:,}" for count in counts])
        ax.set_xlabel("Distinct antibody pairs per job", labelpad=10)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E5EBF0", lw=.9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#CBD3DC")
        ax.tick_params(axis="both", length=0, pad=7)
    axes[0].set_title("End-to-end throughput", loc="left", pad=18,
                      fontsize=15, weight="bold", color=dark)
    axes[0].set_ylim(0, 9200)
    axes[0].set_yticks([0, 2000, 4000, 6000, 8000])
    axes[0].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[0].set_ylabel("Structures/minute · higher is better", labelpad=10)
    axes[1].set_title("Low unified RAM usage", loc="left", pad=18,
                      fontsize=15, weight="bold", color=dark)
    axes[1].set_ylim(0, 6)
    axes[1].set_ylabel("Peak additional system RAM (GiB)", labelpad=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2,
               loc="upper left", bbox_to_anchor=(.067, .866))
    fig.text(.075, .123,
             "Two fresh runs per model and panel. Throughput: median complete-job time, including startup and output.",
             fontsize=10.5, color="#47586C")
    fig.text(.075, .077,
             "RAM: higher peak drop in system MemAvailable; CPU/GPU memory counted together. Idle drift <0.16 GiB.",
             fontsize=10.5, color="#47586C")
    fig.text(.075, .031,
             "Whiskers: timing repeat range or RAM repeat/baseline sensitivity; not confidence intervals.",
             fontsize=10.5, color="#47586C")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        destination = args.output.with_suffix("." + extension)
        fig.savefig(destination, dpi=180)
        if extension == "svg":
            destination.write_text("\n".join(
                line.rstrip() for line in destination.read_text().splitlines()
            ) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
