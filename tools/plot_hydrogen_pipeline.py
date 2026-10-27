"""Scientific plots for the GPU hydrogen / bounded-streaming comparison."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    d = json.loads(args.data.read_text()); out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"})
    cpu, gpu, reduce, dark = "#008477", "#3268BC", "#BD6423", "#172B42"

    def style(ax, both=False):
        ax.spines[["top", "right"]].set_visible(False); ax.set_axisbelow(True)
        ax.grid(axis="both" if both else "y", color="#E4E9EF", lw=.8)

    def save(fig, name):
        for ext in ("png", "svg"):
            path = out/(name+"_271026."+ext)
            fig.savefig(path, dpi=180, facecolor="white")
            if ext == "svg": path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines())+"\n")
        plt.close(fig)

    grid = [r for r in d["summaries"] if r["study"] in ("pipeline64", "pipeline64_followup")]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.6))
    fig.subplots_adjust(left=.075, right=.975, bottom=.23, top=.73, wspace=.22)
    fig.suptitle("GPU hydrogen preparation scales with concurrent workers", x=.075, y=.97, ha="left", fontsize=20, weight="bold", color=dark)
    fig.text(.075, .90, "64 distinct antibodies · full job including startup, neural prediction, chemistry and writing", color="#48596D")
    for engine, color in [("cpu1", cpu), ("gpu_mixed", gpu)]:
        for schedule, marker in [("stream", "o"), ("phased", "D")]:
            rows = sorted([r for r in grid if r["hydrogen"] == engine and r["schedule"] == schedule], key=lambda r: r["workers"])
            xs = [r["workers"] for r in rows]
            for i, ax in enumerate(axes):
                ys = [r["structures_per_minute"] if i == 0 else r["peak_additional_ram_gib"] for r in rows]
                ax.plot(xs, ys, color=color, marker=marker, ms=8, lw=2 if schedule == "stream" else 0,
                        markerfacecolor=color if schedule == "stream" else "white", markeredgewidth=1.8)
                if i == 0:
                    for r, y in zip(rows, ys):
                        reps = [60*64/t for t in r["times_seconds"]]
                        ax.vlines(r["workers"], min(reps), max(reps), color=color, lw=1.5)
    for ax in axes:
        ax.set_xticks([1, 4, 8, 12, 16, 20]); ax.set_xlabel("Concurrent preparation workers", labelpad=10); style(ax)
    axes[0].set_ylabel("Structures / minute"); axes[0].set_ylim(0, 145)
    axes[1].set_ylabel("Peak additional unified RAM (GiB)"); axes[1].set_ylim(0, 11)
    handles = [Line2D([], [], color=cpu, lw=2, label="CPU hydrogens"), Line2D([], [], color=gpu, lw=2, label="GPU hydrogens, mixed precision"),
               Line2D([], [], marker="o", color=dark, ls="", label="Bounded streaming"), Line2D([], [], marker="D", color=dark, mfc="white", ls="", label="Phased")]
    fig.legend(handles=handles, ncol=2, frameon=False, loc="upper left", bbox_to_anchor=(.065, .865), fontsize=11)
    fig.text(.075, .105, "Points: median of two fresh runs; vertical bars: repeat range. RAM: higher measured peak across repeats.", fontsize=10.5, color="#48596D")
    fig.text(.075, .06, "Worker grid was extended in stages; 12–20 GPU workers reached a plateau. No configuration is enabled on main or dev.", fontsize=10.5, color="#48596D")
    save(fig, "hydrogen_pipeline_workers")

    conditions = ["cpu20_phase", "gpu12_stream", "gpu20_stream", "gpu20_phase"]
    names = ["CPU · 20 workers\nphased", "GPU · 12 workers\nstreaming", "GPU · 20 workers\nstreaming", "GPU · 20 workers\nphased"]
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 6.5))
    fig.subplots_adjust(left=.2, right=.96, bottom=.23, top=.73, wspace=.28)
    fig.suptitle("128-antibody confirmation: ~2.2× and ~1.7× faster", x=.045, y=.97, ha="left", fontsize=21, weight="bold", color=dark)
    fig.text(.045, .90, "Compared with the tuned 20-worker CPU candidate · same neural predictions and preparation settings", color="#48596D")
    for route, color, offset, marker in [("openmm", gpu, -.14, "o"), ("reduce", reduce, .14, "D")]:
        for i, condition in enumerate(conditions):
            row = next(r for r in d["summaries"] if r["study"] == "confirmation128" and r["route"] == route and r["condition"] == condition)
            for panel, ax in enumerate(axes):
                if panel == 0:
                    v = row["median_seconds"]; low, high = min(row["times_seconds"]), max(row["times_seconds"])
                    label = f"{v:.1f} s"
                else:
                    v = row["peak_additional_ram_gib"]; low, high = row["ram_sensitivity_gib"]; label = f"{v:.2f}"
                ax.errorbar(v, i+offset, xerr=[[v-low], [high-v]], fmt=marker, color=color, ms=8, capsize=4, lw=1.7)
                ax.annotate(label, (v, i+offset), xytext=(9, 0), textcoords="offset points", va="center", fontsize=11,
                            weight="bold" if condition == "gpu20_phase" else "normal", color=dark)
    for i, ax in enumerate(axes):
        ax.set_ylim(3.6, -.6); ax.set_yticks(range(4), names if i == 0 else [""]*4)
        ax.spines[["left", "top", "right"]].set_visible(False); ax.tick_params(length=0, pad=8); ax.set_axisbelow(True); ax.grid(axis="x", color="#E4E9EF")
    axes[0].set_xlim(0, 137); axes[0].set_xlabel("Complete-job seconds · lower is faster", labelpad=12)
    axes[1].set_xlim(0, 13.3); axes[1].set_xlabel("Peak additional unified RAM (GiB)", labelpad=12)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=gpu, label="OpenMM hydrogens"),
                        Line2D([], [], marker="D", ls="", color=reduce, label="Reduce flips + OpenMM hydrogens")],
               loc="upper left", bbox_to_anchor=(.035, .86), ncol=2, frameon=False, fontsize=11)
    fig.text(.045, .115, "Time: median and range of two fresh runs. RAM: higher peak drop in MemAvailable; bars include baseline sensitivity.", fontsize=10.5, color="#48596D")
    fig.text(.045, .07, "GPU hydrogens alter prepared coordinates and introduce some new geometry defects. These remain research candidates.", fontsize=10.5, color="#48596D")
    save(fig, "hydrogen_pipeline_confirmation")

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 10.4))
    fig.subplots_adjust(left=.085, right=.965, bottom=.16, top=.80, hspace=.38, wspace=.28)
    fig.suptitle("Chemical geometry: mixed changes, not a clean correctness pass", x=.065, y=.975, ha="left", fontsize=19, weight="bold", color=dark)
    fig.text(.065, .923, "CPU 20 phased versus GPU 20 phased · 128 antibodies per route · fixed PDB coordinates", color="#48596D")
    titles = ["H bond errors > 0.10 Å", "H-involving angle errors > 30°", "α-H on the wrong side of N–CA–C", "Severe H-involving overlaps"]
    keys = ["H_bond_errors_over_0_10_A", "H_angle_errors_over_30_deg", "alpha_H_wrong_side", "severe_H_clashes"]
    for ax, key, title in zip(axes.flat, keys, titles):
        maximum = 1
        for route, color, marker in [("openmm", gpu, "o"), ("reduce", reduce, "D")]:
            row = next(r for r in d["geometry"] if r["study"] == "confirmation128" and r["route"] == route and r["condition"] == "gpu20_phase")
            x = [p["baseline"][key] for p in row["paired"]]; y = [p["candidate"][key] for p in row["paired"]]
            maximum = max(maximum, *x, *y)
            ax.scatter(x, y, color=color, marker=marker, s=35, alpha=.55, edgecolors="white", linewidths=.3)
        ax.plot([0, maximum*1.2], [0, maximum*1.2], color="#8793A4", lw=1.3, ls="--", zorder=0)
        if maximum > 8:
            ax.set_xscale("symlog", linthresh=1); ax.set_yscale("symlog", linthresh=1)
        ticks = [v for v in [0, 1, 2, 5, 20, 100, 500, 2000] if v <= maximum*1.2]
        if maximum <= 3: ticks = list(range(maximum+1))
        ax.set_xticks(ticks, [str(t) for t in ticks]); ax.set_yticks(ticks, [str(t) for t in ticks])
        ax.set_xlim(-.05, maximum*1.2+.05); ax.set_ylim(-.05, maximum*1.2+.05)
        ax.set_xlabel("CPU flags per antibody"); ax.set_ylabel("GPU flags per antibody")
        ax.set_title(title, fontsize=12, loc="left", weight="bold", pad=12); style(ax, both=True)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=gpu, label="OpenMM route"),
                        Line2D([], [], marker="D", ls="", color=reduce, label="Reduce route")],
               loc="upper left", bbox_to_anchor=(.055, .89), ncol=2, frameon=False)
    fig.text(.065, .085, "Above the diagonal: more flags on GPU. Points can overlap; a flag count is not experimental structural accuracy.", fontsize=10.5, color="#48596D")
    fig.text(.065, .05, "Both CPU and GPU retain severe outliers. Histograms of counts can hide newly introduced defects; paired identities were checked.", fontsize=10.5, color="#48596D")
    fig.text(.065, .017, "Count axes: linear from 0 to 1, logarithmic above 1; the α-H panel is linear throughout.", fontsize=10, color="#48596D")
    save(fig, "hydrogen_pipeline_geometry")

    fig, ax = plt.subplots(figsize=(13.8, 6.5))
    fig.subplots_adjust(left=.20, right=.96, bottom=.23, top=.73)
    fig.suptitle("Streaming overlaps chemistry, but does not shorten these jobs", x=.045, y=.97, ha="left", fontsize=20, weight="bold", color=dark)
    fig.text(.045, .90, "128 antibodies · OpenMM route, first repetition · clock starts at the first completed raw PDB", color="#48596D")
    for i, condition in enumerate(["cpu20_phase", "gpu20_phase", "gpu20_stream"]):
        p = next(p for p in d["phases"] if p["study"] == "confirmation128" and p["route"] == "openmm" and p["condition"] == condition and p["replicate"] == 1)
        ev = p["events_from_first_raw_seconds"]
        start = p.get("first_preparation_relative_seconds", ev.get("phased_preparation_start", 0))
        ax.barh(i-.13, p["raw_write_span_seconds"], left=0, height=.18, color="#8793A4")
        ax.barh(i+.13, ev["all_preparation_finished"]-start, left=start, height=.18, color=cpu if i == 0 else gpu, alpha=.85)
        ax.annotate(f"{ev['all_preparation_finished']:.1f} s", (ev["all_preparation_finished"], i+.13), xytext=(7, 0), textcoords="offset points", va="center", fontsize=11)
    ax.set_yticks(range(3), ["CPU 20 · phased", "GPU 20 · phased", "GPU 20 · streamed"])
    ax.set_ylim(2.55, -.55); ax.set_xlim(0, 122); ax.set_xlabel("Seconds since the first raw PDB was written", labelpad=12)
    ax.spines[["top", "left", "right"]].set_visible(False); ax.grid(axis="x", color="#E4E9EF"); ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([], [], color="#8793A4", lw=7, label="First-to-last raw PDB window"),
                        Line2D([], [], color=gpu, lw=7, label="Preparation window (CPU in green)")],
               loc="upper left", bbox_to_anchor=(.035, .86), ncol=2, frameon=False, fontsize=11)
    fig.text(.045, .115, "With bounded streaming, raw-output completion stretches from ~1 s to ~32 s; the preparation queue reaches its credit limit.", fontsize=10.5, color="#48596D")
    fig.text(.045, .07, "These are host pipeline windows, not GPU kernel time. Full-container timings, including startup, are shown in the confirmation figure.", fontsize=10.5, color="#48596D")
    save(fig, "hydrogen_pipeline_streaming")


if __name__ == "__main__":
    main()
