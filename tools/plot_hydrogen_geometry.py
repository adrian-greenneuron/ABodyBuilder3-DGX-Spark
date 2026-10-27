"""Static scientific figures for the fixed-heavy hydrogen correction."""
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); d = json.loads(args.data.read_text()); out = args.output
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "svg.fonttype": "none"})
    blue, orange, green, dark = "#3268BC", "#B85C23", "#008477", "#172B42"

    def save(fig, name):
        for ext in ("png", "svg"):
            p = out/(name+"_271026."+ext)
            fig.savefig(p, dpi=180, facecolor="white")
            if ext == "svg": p.write_text("\n".join(s.rstrip() for s in p.read_text().splitlines())+"\n")
        plt.close(fig)

    def style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True); ax.grid(color="#E4E9EF", lw=.8)

    names = ["Earlier CPU · 20 workers", "Earlier GPU · 20 workers", "Corrected CPU · 20 workers", "Corrected GPU · 12 workers"]
    conditions = ["cpu_original20", "gpu_original20", "cpu_adaptive20", "gpu_adaptive12"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.6))
    fig.subplots_adjust(left=.24, right=.96, top=.76, bottom=.23, wspace=.27)
    fig.suptitle("Corrected GPU hydrogen preparation retains its speed advantage", x=.04, y=.975, ha="left", fontsize=20, weight="bold", color=dark)
    fig.text(.04, .90, "127 native antibody pairs · two fresh full jobs · prediction, repair, hydrogens, writing and startup included", color="#48596D")
    for route, color, marker, offset in [("openmm", blue, "o", -.13), ("reduce", orange, "D", .13)]:
        for i, condition in enumerate(conditions):
            r = next(x for x in d["summaries"] if (x["cohort"], x["route"], x["condition"]) == ("127", route, condition))
            for panel, ax in enumerate(axes):
                v = r["median_seconds"] if panel == 0 else r["peak_additional_ram_gib"]
                lo, hi = (min(r["seconds"]), max(r["seconds"])) if panel == 0 else r["ram_sensitivity_gib"]
                ax.errorbar(v, i+offset, xerr=[[max(0, v-lo)], [max(0, hi-v)]], fmt=marker, ms=9, color=color, lw=1.8, capsize=4)
                ax.annotate(f"{v:.1f} s" if panel == 0 else f"{v:.2f}", (v, i+offset), xytext=(9, 0), textcoords="offset points", va="center", weight="bold" if i == 3 else "normal", fontsize=11, color=dark)
    for i, ax in enumerate(axes):
        ax.set_ylim(3.6, -.6); ax.set_yticks(range(4), names if i == 0 else [""]*4)
        ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0, pad=12)
        ax.set_axisbelow(True); ax.grid(axis="x", color="#E4E9EF")
    axes[0].set_xlim(0, 255); axes[0].set_xlabel("Complete-job time (seconds)", labelpad=12)
    axes[1].set_xlim(0, 12.2); axes[1].set_xlabel("Additional unified RAM (GiB)", labelpad=12)
    fig.legend(handles=[Line2D([], [], ls="", marker="o", color=blue, label="OpenMM hydrogens"), Line2D([], [], ls="", marker="D", color=orange, label="Reduce flips + OpenMM hydrogens")], loc="upper left", bbox_to_anchor=(.03, .86), frameon=False, ncol=2, fontsize=11)
    fig.text(.04, .12, "Time: median and repeat range. RAM: higher measured MemAvailable drop; bars show pre/post-baseline sensitivity.", fontsize=10.5, color="#48596D")
    fig.text(.04, .07, "Earlier = experimental CPU/GPU candidates, not main or dev. Corrected hydrogens use a different, explicit chemistry protocol.", fontsize=10.5, color="#48596D")
    save(fig, "hydrogen_geometry_performance")

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 6.8))
    fig.subplots_adjust(left=.07, right=.97, top=.75, bottom=.28, wspace=.30)
    fig.suptitle("The correction removes the tested hydrogen-geometry defects", x=.045, y=.975, ha="left", fontsize=20, weight="bold", color=dark)
    fig.text(.045, .90, "OpenMM route · 127 matched structures · coordinates read from saved PDBs, without further optimisation", color="#48596D")
    specs = [("max_H_bond_error_A", "Largest X–H bond error (Å)", .1), ("max_H_angle_error_deg", "Largest H angle error (degrees)", 30), ("max_planar_H_deviation_deg", "Largest sp²-H plane deviation (degrees)", 1)]
    for condition, label, color in [("cpu_original20", "Earlier CPU", orange), ("gpu_original20", "Earlier GPU", blue), ("gpu_adaptive12", "Corrected GPU", green)]:
        rows = [r for r in d["geometry"] if r["condition"] == condition and r["route"] == "openmm" and r["stage"] in ("planar127", "adaptive127")]
        assert len(rows) == 127
        for ax, (key, title, cutoff) in zip(axes, specs):
            values = np.sort([r[key] for r in rows]); ax.step(values, np.arange(1, len(values)+1)/len(values), where="post", lw=2.2, color=color, label=label)
    for i, (ax, (key, title, cutoff)) in enumerate(zip(axes, specs)):
        ax.axvline(cutoff, color="#69798D", ls="--", lw=1.2)
        ax.set_xlabel(title, labelpad=10); ax.set_ylim(0, 1.025); style(ax)
        if i == 0: ax.set_xscale("log"); ax.set_xlim(.0005, 12); ax.set_ylabel("Cumulative fraction of antibodies")
        else: ax.set_xlim(0, 125 if i == 1 else 95)
        ax.annotate(f"Screen: {cutoff:g}"+(" Å" if i == 0 else "°"), (cutoff, .48), xytext=(7, 0), textcoords="offset points", rotation=90, va="center", fontsize=10, color="#48596D")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.035, .855), ncol=3, frameon=False)
    fig.text(.045, .105, "Both corrected engines pass bond, angle, tetrahedral/α-H, plane and outward-trigonal checks on all 509 compatible inputs.", fontsize=10.5, color="#48596D")
    fig.text(.045, .055, "Three of the original 512 require heavy-atom repair and are explicitly rejected. Geometric checks do not establish experimental accuracy.", fontsize=10.5, color="#48596D")
    save(fig, "hydrogen_geometry_checks")

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.8))
    fig.subplots_adjust(left=.09, right=.97, top=.78, bottom=.24, wspace=.24)
    fig.suptitle("Fewer hydrogen clashes, with residual contacts still visible", x=.06, y=.975, ha="left", fontsize=20, weight="bold", color=dark)
    fig.text(.06, .90, "127 paired structures per route · independent, unmodified legacy Probe · no hydrogen regeneration", color="#48596D")
    for ax, route, color, title in [(axes[0], "openmm", blue, "OpenMM route"), (axes[1], "reduce", orange, "Reduce + OpenMM route")]:
        old = {r["target"]: r for r in d["probe"] if r["stage"] == "planar127" and r["route"] == route and r["condition"] == "gpu_original20"}
        new = {r["target"]: r for r in d["probe"] if r["stage"] == "adaptive127" and r["route"] == route and r["condition"] == "gpu_adaptive12"}
        assert old.keys() == new.keys()
        x = np.array([old[k]["H_clash_pairs"] for k in sorted(old)]); y = np.array([new[k]["H_clash_pairs"] for k in sorted(old)])
        maximum = max(x.max(), y.max())*1.09
        ax.scatter(x, y, s=45, color=color, alpha=.65, edgecolor="white", lw=.4)
        ax.plot([0, maximum], [0, maximum], color="#8793A4", lw=1.5, ls="--")
        ax.set_xlim(0, maximum); ax.set_ylim(0, maximum); ax.set_aspect("equal", adjustable="box"); style(ax)
        reduction = 100*(1-y.sum()/x.sum())
        ax.set_title(title+f" · {reduction:.1f}% fewer pairs", fontsize=13, weight="bold", loc="left", pad=13)
        ax.set_xlabel("Earlier GPU: H-involving bad-overlap pairs", labelpad=9)
        ax.set_ylabel("Corrected GPU: H-involving pairs", labelpad=9)
        ax.text(.04, .94, f"{sum(y<x)} improve · {sum(y==x)} tie · {sum(y>x)} worsen", transform=ax.transAxes, va="top", fontsize=10.5, color=dark)
    fig.text(.06, .14, "Below the diagonal = fewer flagged pairs. Nuclear radii and standard residue bonding; unique unordered pairs.", fontsize=10.5, color="#48596D")
    fig.text(.06, .09, "These are contact diagnostics, not experimental structural accuracy. Fixed-heavy clashes and some H contacts remain.", fontsize=10.5, color="#48596D")
    fig.text(.06, .04, "The candidate is retained for review only. Main, dev and their default images remain unchanged.", fontsize=10.5, color="#48596D")
    save(fig, "hydrogen_geometry_contacts")


if __name__ == "__main__":
    main()
