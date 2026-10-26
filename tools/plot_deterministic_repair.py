"""Plot complete-job speed/RAM and the repeatability/output trade-off."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--hydrogens",type=Path,required=True)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    data=json.loads(args.data.read_text());hydrogens=json.loads(args.hydrogens.read_text())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":12,"svg.fonttype":"none","savefig.facecolor":"white"})
    colors={"openmm":"#087F8C","reduce":"#C97015"};dark="#172B42"
    conditions=["reference_serial","cpu_serial","cpu_parallel","double_serial","double_parallel"]
    names=["Current CUDA single\nserial","CPU deterministic\nserial","CPU deterministic\n8 workers","CUDA double deterministic\nserial","CUDA double deterministic\n8 workers"]

    def style(ax):
        ax.spines[["top","right","left"]].set_visible(False)
        ax.spines["bottom"].set_color("#CBD3DC")
        ax.set_axisbelow(True);ax.grid(axis="x",color="#E5EBF0");ax.tick_params(length=0,pad=7)

    def save(fig,name):
        for ext in ["png","svg"]:
            p=args.output_dir/(name+"."+ext);fig.savefig(p,dpi=180)
            if ext=="svg":p.write_text("\n".join(line.rstrip() for line in p.read_text().splitlines())+"\n")
        plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(14,7.6))
    fig.subplots_adjust(left=.235,right=.965,bottom=.28,top=.75,wspace=.25)
    fig.suptitle("Deterministic CPU repair + parallel preparation: ~5× faster",x=.04,y=.975,ha="left",weight="bold",fontsize=20,color=dark)
    fig.text(.04,.91,"Eight distinct antibodies per job · two fresh runs per condition · model and hydrogen engine unchanged",fontsize=12,color="#47586C")
    fig.legend(handles=[Line2D([],[],marker="o",color=colors["openmm"],ls="",label="OpenMM hydrogens"),Line2D([],[],marker="D",color=colors["reduce"],ls="",label="Reduce flips + OpenMM hydrogens")],loc="upper left",bbox_to_anchor=(.035,.87),ncol=2,frameon=False)
    for panel,ax in enumerate(axes):
        for route,offset,marker in [("openmm",-.14,"o"),("reduce",.14,"D")]:
            for i,condition in enumerate(conditions):
                row=next(x for x in data["summaries"] if x["route"]==route and x["condition"]==condition)
                y=i+offset
                if panel==0:
                    value=row["median_seconds"];lo,hi=min(row["times_seconds"]),max(row["times_seconds"])
                    label=f"{value:.1f} s";reps=row["times_seconds"]
                else:
                    value=row["peak_additional_ram_gib"];lo,hi=row["ram_sensitivity_gib"]
                    label=f"~{value:.2f}"+("*" if row["baseline_drift_flags"] else "");reps=row["ram_repeats_gib"]
                ax.errorbar(value,y,xerr=[[value-lo],[hi-value]],fmt=marker,ms=7,color=colors[route],capsize=4,lw=1.5)
                ax.scatter(reps,[y]*len(reps),s=16,color=colors[route],zorder=4)
                ax.annotate(label,(value,y),xytext=(8,0),textcoords="offset points",ha="left",va="center",fontsize=10.5,color=dark,weight="bold" if condition=="cpu_parallel" else "normal")
        ax.set_ylim(4.6,-.6);ax.set_yticks(range(5),names if panel==0 else [""]*5);style(ax)
    axes[0].set_xlim(0,108);axes[0].set_xlabel("Complete-job seconds · lower is better",labelpad=12);axes[0].set_title("Time includes prediction, preparation and writing",loc="left",fontsize=12,weight="bold",pad=15)
    axes[1].set_xlim(0,5.1);axes[1].set_xlabel("Peak additional unified RAM (GiB)",labelpad=12);axes[1].set_title("CPU and GPU pressure counted together",loc="left",fontsize=12,weight="bold",pad=15)
    fig.text(.04,.13,"Time: median and repeat range. RAM: higher peak drop in system MemAvailable; whiskers show repeat/baseline sensitivity.",fontsize=10.5,color="#47586C")
    fig.text(.04,.085,"* One Reduce-route repetition exceeded the 0.25 GiB idle-drift flag in each marked condition; RAM values are approximate.",fontsize=10.5,color="#47586C")
    fig.text(.04,.04,"Candidates remain isolated: prepared hydrogen coordinates differ from the current repair. Main and dev defaults are unchanged.",fontsize=10.5,color="#47586C")
    save(fig,"deterministic_repair_performance_261026")

    fig,axes=plt.subplots(1,2,figsize=(14,6.5),gridspec_kw={"width_ratios":[1,1.1]})
    fig.subplots_adjust(left=.075,right=.97,bottom=.25,top=.74,wspace=.3)
    fig.suptitle("Deterministic repair: repeatability and hydrogen-coordinate differences",x=.075,y=.97,ha="left",weight="bold",fontsize=19,color=dark)
    fig.text(.075,.90,"Original heavy atoms and protonation choices match; changes remain after equivalent-hydrogen matching.",fontsize=12,color="#47586C")
    policies=["reference","cuda_deterministic","cpu1","cuda_double"]
    variants=[next(x for x in data["screen_summaries"] if x["policy"]==p)["pdb_variants"] for p in policies]
    ax=axes[0];ax.bar(range(4),variants,color=["#8793A4","#8793A4","#087F8C","#5266A5"],width=.65)
    ax.set_xticks(range(4),["Current\nCUDA single","CUDA single\n+ det. forces","CPU 1 thread\n+ det. forces","CUDA double\n+ det. forces"],fontsize=10)
    ax.set_ylim(0,2.5);ax.set_yticks([0,1,2]);ax.set_ylabel("Distinct prepared PDB variants");ax.set_title("A   Stress input: 40 preparations per policy",loc="left",weight="bold",fontsize=12,pad=16)
    for i,value in enumerate(variants):ax.text(i,value+.09,str(value),ha="center",weight="bold",fontsize=14)
    ax.spines[["top","right"]].set_visible(False);ax.set_axisbelow(True);ax.grid(axis="y",color="#E5EBF0")
    ax=axes[1];labels=[]
    for i,(route,condition) in enumerate([(r,c) for r in ["openmm","reduce"] for c in ["cpu_serial","double_serial"]]):
        rows=[x for x in hydrogens["rows"] if x["route"]==route and x["condition"]==condition]
        for row in rows:
            ax.plot([i-.1,i+.1],[row["hydrogen_max_by_name_angstrom"],row["hydrogen_max_after_matching_angstrom"]],color=colors[route],alpha=.5,lw=1)
        ax.scatter([i-.1]*8,[x["hydrogen_max_by_name_angstrom"] for x in rows],s=34,facecolors="none",edgecolors=colors[route])
        ax.scatter([i+.1]*8,[x["hydrogen_max_after_matching_angstrom"] for x in rows],s=28,color=colors[route])
        labels.append(("OpenMM" if route=="openmm" else "Reduce + H")+"\n"+("CPU repair" if condition=="cpu_serial" else "Double repair"))
    ax.set_xticks(range(4),labels,fontsize=10);ax.set_ylim(-.07,2.4);ax.set_ylabel("Largest H displacement vs current repair (Å)");ax.set_title("B   Eight targets per route and repair policy",loc="left",weight="bold",fontsize=12,pad=16)
    ax.spines[["top","right"]].set_visible(False);ax.set_axisbelow(True);ax.grid(axis="y",color="#E5EBF0")
    fig.legend(handles=[Line2D([],[],marker="o",ls="",markerfacecolor="none",markeredgecolor="#47586C",label="By atom name"),Line2D([],[],marker="o",ls="",color="#47586C",label="After same-parent H matching")],loc="upper left",bbox_to_anchor=(.56,.86),ncol=2,frameon=False,fontsize=10)
    fig.text(.075,.13,"Stress: serial/concurrent preparations in two rounds, one previously variable input; full-float repair/H arrays also repeat exactly for CPU/double.",fontsize=10,color="#47586C")
    fig.text(.075,.08,"Right: one point per antibody, using PDB-rounded coordinates; connected markers show before/after H-label matching. No alignment.",fontsize=10,color="#47586C")
    fig.text(.075,.03,"These are numerical/chemical output checks, not experimental-accuracy measurements. No result-changing candidate is enabled.",fontsize=10,color="#47586C")
    save(fig,"deterministic_repair_outputs_261026")


if __name__=="__main__":main()
