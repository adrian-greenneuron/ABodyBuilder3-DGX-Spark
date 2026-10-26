"""Plot optional-stage results, including the failure to qualify exact outputs."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import statistics as st
import zipfile
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    repo=Path(__file__).resolve().parents[1]
    def read(path): return json.loads((args.runs/path).read_text())
    profile=read("optional_preparation_profile_261026/profile_measurements_261026.json")
    initial=read("optional_preparation_optimisation_261026/workers/measurements_261026.json")
    relaxed=read("optional_preparation_optimisation_261026/relax_workers/measurements_261026.json")
    threads=read("optional_preparation_optimisation_261026/thread_screen/measurements_261026.json")
    differences=read("optional_preparation_optimisation_261026/paired_preparation_differences_261026.json")
    cli=read("optional_preparation_cli_261026/measurements_261026.json")
    cap=read("optional_preparation_cli_261026/worker_cap_measurements_261026.json")
    fast=read("optional_preparation_cli_261026/matched_fast_checks_261026.json")
    repeat=read("optional_preparation_cli_261026/repeatability_trace/repeatability_261026.json")
    repeat_root=args.runs/"optional_preparation_cli_261026/repeatability_trace"
    reference=np.load(repeat_root/repeat[0]["outputs"][0]["directory"]/"stages.npz")
    summary=[]
    for group in repeat:
        changes={}
        for stage in ("repair_positions_nm","hydrogenated_positions_nm"):
            changes[stage+"_max_difference_angstrom"]=max(float(np.abs(
                np.load(repeat_root/v["directory"]/"stages.npz")[stage]-reference[stage]).max())*10
                for v in group["outputs"])
            changes[stage+"_unique_hashes"]=len({v["stage_sha256"][stage] for v in group["outputs"]})
        summary.append({"workers":group["workers"],"calls":len(group["outputs"]),
                        "unique_prepared_pdbs":len({v["prepared_sha256"] for v in group["outputs"]}),**changes})
    evidence={"profile":profile,"preparation_workers":initial,"relaxation_workers":relaxed,
              "cpu_thread_screen":threads,"paired_differences":differences,"installed_cli":cli,
              "worker_cap":cap,"plain_prediction_regression":fast,"repeatability":summary,
              "repair_platform":read("optional_preparation_cli_261026/repair_platform_261026.json")["contexts"],
              "qualification_parity":read("optional_preparation_cli_261026/qualification_parity_261026.json"),
              "decision":"Parallel preparation and altered engine threading/relaxation remain isolated for review. No chemistry default change on dev or main.",
              "fast_condition_key":"The reused regression tool labels old/current dev as accepted and the preparation candidate as language. Image identities, not these historical labels, identify the comparison."}
    plt.rcParams.update({"font.size":12,"svg.fonttype":"none","axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    labels=["OpenMM H","Reduce + H","H + CUDA relax"]
    stages=[("_repair","Repair + checks","#89a0af"),("_reduce","Reduce wrapper","#9c6da6"),
            ("OpenMM.addHydrogens","Hydrogen addition","#217ba6"),("_minimise","Restrained stages","#dd9f3f"),("other","Other checks / output","#91b9aa")]
    for i,mode in enumerate(("openmm","reduce","restrained")):
        rows=[r for r in profile["rows"] if r["mode"]==mode and r["profiled"]]
        totals=[]
        for row in rows:
            total=collections.defaultdict(float)
            for e in row["stages"]:total[e["stage"]]+=e["seconds"]
            total["other"]=max(0,row["seconds"]-sum(total[name] for name,_,_ in stages[:-1]))
            totals.append(total)
        bottom=0
        for name,label,colour in stages:
            value=st.mean(row[name] for row in totals)
            axes[0].bar(i,value,bottom=bottom,color=colour,label=label if i==0 else None)
            bottom+=value
    axes[0].set(xticks=range(3),xticklabels=["OpenMM\nH","Reduce\n+ H","H + CUDA\nrelax"],ylabel="Seconds per antibody",title="Where the optional time goes")
    axes[0].legend(fontsize=9,loc="upper left",bbox_to_anchor=(0,1.02))
    axes[0].set_ylim(0,18)
    colours={"openmm":"#1372a0","reduce":"#8a4fa2"}
    for mode in ("openmm","reduce"):
        x=[];y=[]
        for workers in (1,4,8):
            rows=([r for r in cli["rows"] if r["mode"]==mode and r["condition"]=="serial" and not r["eager"]]
                  if workers==1 else [r for r in cap["rows"] if r["mode"]==mode and r["workers"]==workers])
            x.append(max(r["metrics"]["container_lifetime_memory_peak_bytes"] for r in rows)/2**30)
            y.append(st.median(r["container_seconds"] for r in rows))
            axes[1].annotate(str(workers), (x[-1],y[-1]), xytext=(5,5),textcoords="offset points",fontsize=10)
        axes[1].plot(x,y,"o-",markersize=8,linewidth=2,label="OpenMM H" if mode=="openmm" else "Reduce + H",color=colours[mode])
    axes[1].set(xlabel="Higher measured container peak (GiB)",ylabel="Complete eight-antibody CLI job (s)",title="Faster with independent workers")
    axes[1].legend(fontsize=10);axes[1].set_ylim(0,95)
    for mode in ("openmm","reduce"):
        values=[]
        for workers in (1,4,8):
            rows=([r for r in cli["rows"] if r["mode"]==mode and r["condition"]=="serial" and not r["eager"]]
                  if workers==1 else [r for r in cap["rows"] if r["mode"]==mode and r["workers"]==workers])
            values.append(st.median(r["container_seconds"] for r in rows))
        axes[2].plot([1,4,8],np.array(values[0])/np.array(values),"o-",markersize=8,linewidth=2,color=colours[mode])
    axes[2].set(xticks=[1,4,8],xlabel="Preparation processes (CPU thread cap 1)",ylabel="Speedup versus serial CLI",title="Promising; not enabled on dev",ylim=(0,5.5))
    axes[2].text(.5,.05,"A later stress check changed\nH coordinates at PDB rounding\nprecision. Raw predictions unchanged.",transform=axes[2].transAxes,ha="center",fontsize=10,bbox=dict(facecolor="#fff2db",edgecolor="#caa775",pad=7))
    for ax in axes:ax.grid(axis="y",alpha=.2);ax.set_axisbelow(True)
    fig.suptitle("Optional chemistry · same settings · concurrency still needs review",fontweight="bold")
    fig.text(.5,-.02,"Profiles: two retained public predictions per route. Complete CLI: eight targets, two runs per condition.\nContainer memory includes shared-memory accounting; it must not be added to process RSS or PyTorch CUDA allocation.",ha="center",fontsize=10)
    fig.tight_layout()
    assets=repo/"docs/assets";data=repo/"docs/data"
    for suffix in ("png","svg"):
        path=assets/f"preparation_optimisation_261026.{suffix}";fig.savefig(path,dpi=160,bbox_inches="tight",facecolor="white")
        if suffix=="svg":path.write_text("\n".join(s.rstrip() for s in path.read_text().splitlines())+"\n")
    plt.close(fig)
    target=data/"preparation_optimisation_measurements_261026.json"
    target.write_text(json.dumps(evidence,indent=2)+"\n")
    members={target.name:target.read_bytes()}
    for study in ("optional_preparation_profile_261026","optional_preparation_optimisation_261026","optional_preparation_cli_261026"):
        for p in (args.runs/study).glob("*protocol*.json"):members[study+"/"+p.name]=p.read_bytes()
    for name in ("failure_fixture_clarification_261026.json","final_source_manifest_261026.json","measurement_tools_manifest_261026.json"):
        members[name]=(args.runs/"optional_preparation_cli_261026"/name).read_bytes()
    members["failure_validation_261026.json"]=(args.runs/"optional_preparation_cli_261026/failure_checks_v2/validation_261026.json").read_bytes()
    with zipfile.ZipFile(data/"preparation_optimisation_evidence_261026.zip","w",compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,value in members.items():z.writestr(name,value)
        z.writestr("SHA256SUMS","".join(f"{hashlib.sha256(v).hexdigest()}  {k}\n" for k,v in sorted(members.items())))


if __name__=="__main__":main()
