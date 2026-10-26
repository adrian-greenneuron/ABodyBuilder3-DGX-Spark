"""Create compact evidence and plots for the ordered A–E/G screens."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics as st
import zipfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    data = repo / "docs/data"; assets = repo / "docs/assets"
    data.mkdir(parents=True, exist_ok=True); assets.mkdir(parents=True, exist_ok=True)
    def read(path):
        return json.loads((args.runs / path).read_text())
    records = {"profile_controls": read("aot_pipeline_profile_261026/measurements_261026.json"),
               "deferred_1000": read("aot_deferred_pipeline_261026/measurements_n1000_261026.json"),
               "deferred_10000": read("aot_deferred_pipeline_261026/measurements_n10000_261026.json"),
               "shapes": read("aot_shape_schedule_261026/measurements_n1000_261026.json"),
               "resident": read("aot_resident_jobs_261026/measurements_261026.json"),
               "geometry_update": read("aot_geometry_update_261026/measurements_261026.json")}
    profile = read("aot_pipeline_profile_261026/nsight/stage_profile_261026.json")
    instrumented = read("aot_pipeline_profile_261026/instrumented/stage_profile_261026.json")
    host = collections.defaultdict(float); gpu = collections.defaultdict(float)
    for event in instrumented["events"]:
        host[event["stage"]] += event["end"] - event["start"]
    for event in instrumented["cuda_events"]:
        gpu[event["stage"]] += event["stream_elapsed_ms"] / 1000
    records["profile_totals"] = {"host_seconds_overlapping": dict(host), "gpu_stream_seconds": dict(gpu),
                                 "note": "Host intervals overlap; stream elapsed includes device work and possible stalls. Neither is an additive whole-job decomposition."}
    connection = sqlite3.connect(args.runs / "aot_pipeline_profile_261026/nsight/pipeline.sqlite")
    nvtx_start = connection.execute("SELECT MIN(start) FROM NVTX_EVENTS WHERE text='ABB3.checkpoint_loading'").fetchone()[0] / 1e9
    offset = nvtx_start - next(e["start"] for e in profile["events"] if e["stage"] == "checkpoint_loading")
    kernels = [(a / 1e9 - offset, (b-a) / 1e9) for a,b in connection.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL")]
    top = [{"kernel": name, "gpu_seconds": duration / 1e9, "calls": count}
           for name, duration, count in connection.execute(
               "SELECT s.value,SUM(k.end-k.start),COUNT(*) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.shortName=s.id GROUP BY k.shortName ORDER BY SUM(k.end-k.start) DESC LIMIT 20")]
    records["top_kernels"] = top
    records["trace_notes"] = "Raw Nsight/pstats/coordinate payloads remain local. Only timing aggregates, bounded queue events and source identities are packaged."
    plt.rcParams.update({"font.size": 12, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    def save(fig, name):
        for suffix in ("png", "svg"):
            path = assets / f"{name}_261026.{suffix}"
            fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
            if suffix == "svg":
                path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines()) + "\n")
        plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    lanes = [("checkpoint_loading", "Weights", "#456478"), ("package_loading", "AOT packages", "#456478"),
             ("feature_construction", "CPU features", "#48a099"), ("aot_forward", "Host enqueue", "#b2925f"),
             ("atom_conversion_and_download", "Host conversion / wait", "#d7a55f"),
             ("output_batch", "Output workers", "#8d69a9")]
    for i, (stage, label, colour) in enumerate(lanes):
        spans = [(e["start"], e["end"] - e["start"]) for e in profile["events"] if e["stage"] == stage]
        axes[0].broken_barh(spans, (i-.32,.64), facecolors=colour)
    axes[0].broken_barh(kernels, (len(lanes)-.32,.64), facecolors="#1372a0")
    axes[0].set_yticks(range(len(lanes)+1), [x[1] for x in lanes] + ["GPU kernels"])
    axes[0].invert_yaxis(); axes[0].set_title("A · The model runs while host conversion waits; writing overlaps it", loc="left", fontweight="bold")
    queue = profile["queue_events"]
    for key, label in (("ready", "Feature batches ready"), ("waiting_to_write", "Batches waiting to write")):
        axes[1].step([e["seconds"] for e in queue], [e[key] for e in queue], where="post", label=label, linewidth=2)
    axes[1].set(xlabel="Seconds from instrumented script start", ylabel="Batches")
    axes[1].legend(loc="upper right"); axes[1].set_ylim(bottom=-.1)
    for ax in axes: ax.grid(axis="x", alpha=.2)
    fig.suptitle("1,000 pLDDT predictions · strict FP32 AOT · actual device timeline", fontweight="bold")
    fig.text(.5,-.01,"Host bars indicate occupied intervals, including waits, not CPU utilisation. The Nsight job includes profiling overhead.",ha="center",fontsize=10)
    fig.tight_layout(); save(fig,"aot_pipeline_timeline")
    fig, axes = plt.subplots(1,3,figsize=(15,4.9))
    colours = {"base":"#1372a0", "plddt":"#8a4fa2"}
    deferred = records["deferred_10000"]
    for i, model in enumerate(("base","plddt")):
        for rep in (1,2):
            rows=[r for r in deferred if r["model"]==model and r["replicate"]==rep]
            values=[next(r["container_seconds"] for r in rows if r["mode"]==m) for m in ("reference","deferred")]
            axes[0].plot([i-.15,i+.15],values,"o-",color=colours[model],markersize=7,linewidth=1.8)
    axes[0].set(xticks=[0,1],xticklabels=["base","pLDDT"],ylabel="Complete 10,000-input job (s)",title="B · Deferred copies: 0.6–1.2%")
    axes[0].text(.5,.02,"Left: current   Right: deferred",transform=axes[0].transAxes,ha="center",fontsize=10)
    labels={"accepted32":"current 32","dynamic16":"16","dynamic32":"32","dynamic64":"64","dynamic128":"128","fixed32":"fixed 32"}
    for model in ("base","plddt"):
        for condition in labels:
            rows=[r for r in records["shapes"] if r["model"]==model and r["condition"]==condition]
            if not rows: continue
            x=max(r["metrics"]["peak_cuda_allocated_bytes"] for r in rows)/2**30
            y=st.median(r["container_seconds"] for r in rows)
            axes[1].scatter(x,y,s=70,color=colours[model],marker="o" if model=="base" else "D")
            if model=="base": axes[1].annotate(labels[condition],(x,y),xytext=(5,5),textcoords="offset points",fontsize=9)
            axes[1].vlines(x,min(r["container_seconds"] for r in rows),max(r["container_seconds"] for r in rows),color=colours[model],lw=2)
    axes[1].set(xlabel="Peak CUDA allocation (GiB)",ylabel="Complete 1,000-input job (s)",title="C · More memory, little benefit")
    for model, label, marker in (("base", "base", "o"), ("plddt", "pLDDT", "D")):
        axes[1].scatter([], [], color=colours[model], marker=marker, label=label)
    axes[1].legend(fontsize=9, loc="upper right")
    groups=[(5,"base"),(5,"plddt"),(100,"base"),(100,"plddt")]
    for i,(size,model) in enumerate(groups):
        for delta,mode,colour in ((-.18,"fresh","#74818d"),(.18,"resident","#1372a0")):
            values=[r["total_container_seconds"] for r in records["resident"] if r["inputs_per_job"]==size and r["model"]==model and r["mode"]==mode]
            axes[2].bar(i+delta,st.median(values),.34,color=colour,label=mode if i==0 else None)
            axes[2].scatter([i+delta]*len(values),values,s=16,color="black",zorder=3)
    axes[2].set(xticks=range(4),xticklabels=["10×5\nbase","10×5\npLDDT","5×100\nbase","5×100\npLDDT"],ylabel="All requests completed (s)",title="E · Reuse startup for small jobs")
    axes[2].legend()
    for ax in axes: ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.suptitle("Ordered pipeline screens · two repetitions per condition",fontweight="bold")
    fig.text(.5,-.02,"Resident totals include one initial startup versus one startup per fresh request. These are implementation screens, not new accuracy results.",ha="center",fontsize=10)
    fig.tight_layout();save(fig,"aot_pipeline_screens")
    evidence = data/"aot_pipeline_measurements_261026.json"
    evidence.write_text(json.dumps(records,indent=2)+"\n")
    members={"measurements_261026.json":evidence.read_bytes(),"profile_events_261026.json":json.dumps(profile,indent=2).encode()+b"\n"}
    for study in ("aot_pipeline_profile_261026","aot_deferred_pipeline_261026","aot_shape_schedule_261026","aot_resident_jobs_261026","aot_geometry_update_261026"):
        for path in (args.runs/study).glob("*protocol*.json"):
            members[study+"/"+path.name]=path.read_bytes()
    with zipfile.ZipFile(data/"aot_pipeline_evidence_261026.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,value in members.items():archive.writestr(name,value)
        archive.writestr("SHA256SUMS","".join(f"{hashlib.sha256(v).hexdigest()}  {k}\n" for k,v in sorted(members.items())))


if __name__ == "__main__": main()
