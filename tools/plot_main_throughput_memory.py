"""Plot matched throughput and independently measured shared-RAM pressure."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=json.loads(a.data.read_text())['summary']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'svg.fonttype':'none','savefig.facecolor':'white'})
old_color='#758294';new_color='#087F8C';dark='#172B42'
fig,axes=plt.subplots(1,2,figsize=(14.3,6.2),gridspec_kw={'width_ratios':[1.12,1]})
fig.subplots_adjust(left=.14,right=.98,bottom=.28,top=.70,wspace=.22)
fig.suptitle('~4,500 antibody structures/minute on one DGX Spark',x=.14,y=.98,ha='left',fontsize=21,fontweight='bold',color=dark)
fig.text(.14,.90,'Measured base/pLDDT throughput includes startup, model loading and every PDB write.',fontsize=12,color='#47586C')
fig.legend(handles=[Line2D([],[],marker='o',markersize=9,color=old_color,lw=0,label='Previous main'),Line2D([],[],marker='D',markersize=8,color=new_color,lw=0,label='New main · strict FP32')],loc='upper left',bbox_to_anchor=(.135,.87),frameon=False,ncol=2,fontsize=12)
for col,ax in enumerate(axes):
 ax.set_ylim(-.65,2.65);ax.invert_yaxis();ax.set_yticks(range(3));ax.set_axisbelow(True);ax.grid(axis='x',color='#E9EDF1',lw=1);ax.tick_params(length=0,labelsize=11)
 if col==0:
  ax.set_xscale('log');ax.set_xlim(.8,20000);ax.set_xticks([1,10,100,1000,5000]);ax.set_xticklabels(['1','10','100','1,000','5,000']);ax.minorticks_off();ax.set_xlabel('Structures/minute · log scale · higher is better',labelpad=11);ax.set_title('End-to-end prediction throughput',loc='left',weight='bold',fontsize=15,pad=18,color=dark)
  ax.set_yticklabels([f"{'pLDDT' if r['model']=='plddt' else r['model'].capitalize()}\n{r['count']:,} inputs" for r in rows],fontsize=12)
 else:
  ax.set_xlim(0,23.5);ax.set_xticks([0,5,10,15,20]);ax.set_yticklabels([]);ax.set_xlabel('Additional unified RAM (GiB) · lower is better',labelpad=11);ax.set_title('Estimated RAM pressure at peak',loc='left',weight='bold',fontsize=15,pad=18,color=dark)
 for spine in ['top','left','right']:ax.spines[spine].set_visible(False)
 ax.spines['bottom'].set_color('#CBD3DC')
 for i,row in enumerate(rows):
  values={c:(row['timing'][c]['structures_per_minute'] if col==0 else row['conditions'][c]['higher_peak_gib']) for c in ['previous','new']}
  ax.plot([values['previous'],values['new']],[i,i],color='#D4DAE2',lw=4,zorder=1)
  for condition,color,dy in [('previous',old_color,14),('new',new_color,-20)]:
   value=values[condition]
   if col==0:
    t=row['timing'][condition];lo=60*row['count']/t['max_seconds'];hi=60*row['count']/t['min_seconds'];label=f'{value:,.0f}' if value>=100 else f'{value:.1f}'
   else:
    lo,hi=row['conditions'][condition]['pre_or_post_baseline_envelope_gib'];label=f'~{value:.1f}'
   ax.errorbar(value,i,xerr=np.array([[value-lo],[hi-value]]),fmt='none',color=color,capsize=4,lw=1.6,zorder=2)
   ax.scatter(value,i,s=110,c=color,marker='o' if condition=='previous' else 'D',edgecolor='white',lw=1.2,zorder=3)
   ax.annotate(label,(value,i),xytext=(0,dy),textcoords='offset points',ha='center',va='center',fontsize=11,color=dark,weight='bold' if condition=='new' else 'normal')
  label=f"{row['speedup']:.1f}×" if col==0 else f"~{row['reduction_percent']:.0f}% less"
  ax.text(19500 if col==0 else 23.1,i,label,ha='right',va='center',color=new_color,weight='bold',fontsize=12)
fig.text(.14,.135,'Two fresh runs per version/model in each study. Throughput: median and range; RAM: higher observed peak.',fontsize=11,color='#47586C')
fig.text(.14,.09,'RAM uses the baseline-adjusted drop in system MemAvailable, capturing shared CPU/GPU pressure together.',fontsize=10.5,color='#47586C')
fig.text(.14,.045,'RAM whiskers show repeat/baseline sensitivity, not confidence intervals. Idle drift reached 0.94 GiB; values are estimates.',fontsize=10.5,color='#47586C')
a.output.parent.mkdir(parents=True,exist_ok=True)
for ext in ['png','svg']:fig.savefig(a.output.with_suffix('.'+ext),dpi=180)
plt.close(fig)
