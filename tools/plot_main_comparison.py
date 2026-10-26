"""Render the README comparison from the committed matched measurements."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--data',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True,help='Output basename, without extension.')
args=parser.parse_args();data=json.loads(args.data.read_text())['summary']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':15,'axes.labelsize':12,'svg.fonttype':'none','savefig.facecolor':'white'})
old_color='#758294';new_color='#087F8C';dark='#172B42'
fig,axes=plt.subplots(1,2,figsize=(13.8,6.0),gridspec_kw={'width_ratios':[1.12,1]})
fig.subplots_adjust(left=.14,right=.97,bottom=.20,top=.70,wspace=.24)
for col,(metric,title,xlabel) in enumerate([('median_seconds','Complete job time','Seconds · lower is better'),('peak_summed_rss_gib','Peak process memory','Summed RSS (GiB) · lower is better')]):
 ax=axes[col];max_value=max(row['conditions']['previous'][metric] for row in data)
 ax.set_xlim(0,max_value*1.53);ax.set_ylim(-.65,2.65);ax.invert_yaxis()
 for i,row in enumerate(data):
  old=row['conditions']['previous'];new=row['conditions']['new']
  ax.plot([new[metric],old[metric]],[i,i],color='#D4DAE2',lw=4,zorder=1)
  for value,condition,color,offset in [(old,'previous',old_color,12),(new,'new',new_color,-19)]:
   v=value[metric]
   if col==0:
    err=np.array([[v-value['min_seconds']],[value['max_seconds']-v]])
    ax.errorbar(v,i,xerr=err,fmt='none',color=color,capsize=5,lw=2,zorder=2)
   ax.scatter(v,i,s=110,c=color,edgecolors='white',linewidths=1.2,zorder=3,marker='o' if condition=='previous' else 'D')
   label=f'{v:.1f} s' if col==0 else f'{v:.2f}'
   ax.annotate(label,(v,i),xytext=(0,offset),textcoords='offset points',ha='center',va='center',fontsize=11,color=dark,weight='bold' if condition=='new' else 'normal')
  benefit=f"{row['speedup']:.1f}× faster" if col==0 else f"{row['summed_rss_reduction_percent']:.0f}% lower"
  ax.text(max_value*1.51,i,benefit,ha='right',va='center',fontsize=12,weight='bold',color=new_color)
 ax.set_title(title,loc='left',weight='bold',color=dark,pad=18)
 ax.set_xlabel(xlabel,labelpad=10)
 ax.grid(axis='x',color='#E9EDF1',lw=1)
 ax.set_axisbelow(True)
 ax.set_yticks(range(3))
 ax.tick_params(axis='both',length=0,labelsize=11)
 if col==0:
  ax.set_yticklabels([f"{'pLDDT' if x['model']=='plddt' else x['model'].capitalize()}\n{x['count']:,} inputs" for x in data],fontsize=12)
 else:ax.set_yticklabels([])
 for spine in ['top','left','right']:ax.spines[spine].set_visible(False)
 ax.spines['bottom'].set_color('#CBD3DC')
fig.suptitle('ABodyBuilder3: previous main vs new main',x=.14,y=.98,ha='left',fontsize=20,fontweight='bold',color=dark)
fig.text(.14,.90,'DGX Spark · identical inputs within each model · two fresh-container runs per version',fontsize=12,color='#47586C')
handles=[Line2D([],[],marker='o',markersize=9,color=old_color,lw=0,label='Previous main'),Line2D([],[],marker='D',markersize=8,color=new_color,lw=0,label='New main · strict FP32')]
fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.135,.88),frameon=False,ncol=2,fontsize=12)
fig.text(.14,.09,'Time: median and range. Memory: higher sampled peak. Startup and PDB writing included.',fontsize=11,color='#47586C')
fig.text(.14,.045,'RSS counts shared pages per process; it is not total physical RAM. CUDA allocation is discussed below.',fontsize=10.5,color='#47586C')
args.output.parent.mkdir(parents=True,exist_ok=True)
for extension in ['png','svg']:fig.savefig(args.output.with_suffix('.'+extension),dpi=180)
plt.close(fig)
