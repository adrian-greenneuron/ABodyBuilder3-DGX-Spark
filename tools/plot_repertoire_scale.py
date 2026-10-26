"""Plot repertoire throughput, unified-memory scaling, diversity and sustained jobs."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();x=json.loads(a.data.read_text());a.output_dir.mkdir(exist_ok=True,parents=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':13,'axes.labelsize':11,'svg.fonttype':'none','savefig.facecolor':'white'})
colors={'base':'#087F8C','plddt':'#D77819'};labels={'base':'Base','plddt':'pLDDT'}
def style(ax):
 ax.spines[['top','right']].set_visible(False);ax.grid(color='#E9EDF1',lw=.8);ax.set_axisbelow(True)
def save(fig,name):
 for ext in ['png','svg']:fig.savefig(a.output_dir/(name+'.'+ext),dpi=170)
 plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(13.8,10.2));fig.subplots_adjust(left=.08,right=.97,bottom=.09,top=.87,hspace=.46,wspace=.36)
fig.suptitle('Repertoire-scale prediction: 100,000 distinct antibodies',x=.08,y=.98,ha='left',weight='bold',fontsize=20,color='#172B42')
fig.text(.08,.936,'Current main · complete PDB output · two fresh runs per model and panel · no repeated pairs within a job',fontsize=11,color='#47586C')
for model in ['base','plddt']:
 rows=[v for v in x['summaries'] if v['model']==model and v['cohort'].startswith('oas_')];rows.sort(key=lambda v:v['count']);n=np.array([v['count'] for v in rows]);offset=.97 if model=='base' else 1.03
 ax=axes[0,0];values=np.array([v['structures_per_minute'] for v in rows]);ax.plot(n*offset,values,'o-',color=colors[model],lw=2.2,ms=7,label=labels[model])
 for v in rows:ax.scatter([v['count']*offset]*2,v['throughput_range'],color=colors[model],s=24,zorder=4)
 ax=axes[0,1];values=np.array([v['peak_additional_ram_gib'] for v in rows]);ax.plot(n*offset,values,'o-',color=colors[model],lw=2.2,ms=7,label=labels[model]);error=np.array([[v['peak_additional_ram_gib']-v['ram_sensitivity_gib'][0] for v in rows],[v['ram_sensitivity_gib'][1]-v['peak_additional_ram_gib'] for v in rows]]);ax.errorbar(n*offset,values,yerr=error,fmt='none',color=colors[model],capsize=4)
for ax in axes[0]:
 ax.set_xscale('log');ax.set_xlim(800,135000);ax.set_xticks([1000,10000,100000]);ax.set_xticklabels(['1,000','10,000','100,000']);ax.set_xlabel('Distinct native heavy/light pairs');style(ax)
axes[0,0].axhline(4500,color='#778294',ls='--',lw=1.5,label='README ~4,500/min');axes[0,0].set_ylim(0,9000);axes[0,0].set_ylabel('Structures/minute · higher is better');axes[0,0].set_title('A   End-to-end throughput',loc='left',weight='bold');axes[0,0].legend(frameon=False,fontsize=10,loc='lower right')
axes[0,1].set_ylim(0,6);axes[0,1].set_ylabel('Additional unified RAM at peak (GiB)');axes[0,1].set_title('B   System-level memory estimate',loc='left',weight='bold');axes[0,1].legend(frameon=False,fontsize=10,loc='lower right')
for key,label,color in [('earlier','Earlier 100-pair panel','#778294'),('oas_100000.csv','100k distinct repertoire pairs',colors['base']),('structural_diverse.csv','3,756 distinct structural pairs','#7850A8')]:
 hist=x['earlier_reference_input']['length_histogram'] if key=='earlier' else next(z for z in x['selection']['inputs'] if z['file']==key)['length_histogram'];pairs=sorted((int(k),v) for k,v in hist.items());n=np.array([v[0] for v in pairs]);c=np.cumsum([v[1] for v in pairs]);axes[1,0].step(n,c/c[-1],where='post',color=color,lw=2,label=label)
ax=axes[1,0];ax.axvspan(272,285,color='#ECE7F3',alpha=.8);ax.axvline(272,color='#9C8AAC',ls=':',lw=1);ax.set_xlim(200,285);ax.set_ylim(0,1.01);ax.set_xlabel('Combined heavy + light length (residues)');ax.set_ylabel('Cumulative fraction');ax.set_title('C   Broader natural sequence lengths',loc='left',weight='bold');ax.legend(frameon=False,fontsize=9,loc='upper left');style(ax)
study=next(z for z in x['selection']['inputs'] if z['file']=='oas_100000.csv')['study_counts'];pairs=sorted(study.items(),key=lambda z:z[1]);ax=axes[1,1];ax.barh([z[0].replace('_',' ') for z in pairs],[z[1] for z in pairs],color='#2D8D99',height=.68);ax.set_xlabel('Distinct pairs in the 100k panel');ax.set_title('D   13 public studies; original pairing retained',loc='left',weight='bold');ax.tick_params(axis='y',labelsize=8.5);style(ax)
fig.text(.08,.025,'Timing: medians; dots show repeats. RAM whiskers: repeat/pre–post baseline sensitivity, not confidence intervals. Study-balanced panels are engineering samples, not population-frequency estimates.',fontsize=9,color='#47586C')
save(fig,'abodybuilder3_repertoire_scaling_261026')
fig,axes=plt.subplots(2,2,figsize=(13.8,9.8));fig.subplots_adjust(left=.08,right=.97,bottom=.16,top=.87,hspace=.42,wspace=.29)
fig.suptitle('Sustained 100,000-antibody jobs',x=.08,y=.98,ha='left',weight='bold',fontsize=20,color='#172B42');fig.text(.08,.935,'Four complete jobs · ~13 minutes each · no CUDA OOM retries · all 100k-pair batches used the saved executable',fontsize=11,color='#47586C')
for item in x['sustained']:
 color=colors[item['model']];alpha=1 if item['replicate']==1 else .55;ls='-' if item['replicate']==1 else '--';label=labels[item['model']] if item['replicate']==1 else None
 n=[v['completed'] for v in item['intervals']];rate=[v['structures_per_minute'] for v in item['intervals']];axes[0,0].plot(n,rate,color=color,alpha=alpha,ls=ls,lw=1.8,label=label)
 row=next(z for z in x['rows'] if z['label']==item['label']);curve=item['ram_curve'];time=[(v['monotonic']-row['launch_monotonic'])/60 for v in curve];count=[v['completed'] for v in curve];mem=[v['additional_ram_gib'] for v in curve]
 axes[0,1].plot(time,np.array(count)/1000,color=color,alpha=alpha,ls=ls,lw=1.8,label=label);axes[1,0].plot(np.array(count)/1000,mem,color=color,alpha=alpha,ls=ls,lw=1.3,label=label)
ax=axes[0,0];ax.set_ylim(6500,8500);ax.xaxis.set_major_formatter(FuncFormatter(lambda v,p:f'{int(v/1000)}k'));ax.set_xlabel('Completed structures');ax.set_ylabel('Structures/minute · zoomed axis');ax.set_title('A   Throughput in 5,000-output blocks',loc='left',weight='bold');ax.legend(frameon=False)
ax=axes[0,1];ax.set_xlabel('Elapsed job time (minutes)');ax.set_ylabel('Completed structures (thousands)');ax.set_title('B   Output writing progresses steadily',loc='left',weight='bold');ax.set_ylim(0,105);ax.legend(frameon=False)
ax=axes[1,0];ax.set_xlabel('Completed structures (thousands)');ax.set_ylabel('Additional unified RAM (GiB)');ax.set_ylim(0,6);ax.set_title('C   RAM pressure stays bounded through each job',loc='left',weight='bold');ax.legend(frameon=False)
ax=axes[1,1];cohorts=['oas_1000','oas_10000','oas_100000','structural_diverse'];names=['OAS 1k','OAS 10k','OAS 100k','Structural 3.8k'];bottom=np.zeros(4);bins=[(1,1,'1'),(2,8,'2–8'),(9,16,'9–16'),(17,24,'17–24'),(25,31,'25–31'),(32,32,'32')];palette=['#D9E2E8','#B9CFD5','#8BB8C1','#5EA2B1','#2C8899','#00667B']
for (lo,hi,label),color in zip(bins,palette):
 values=[]
 for cohort in cohorts:
  rows=[v for v in x['rows'] if v['label'].startswith(cohort+'_')];h=rows[0]['batch_size_histogram'];assert all(z['batch_size_histogram']==h for z in rows);values.append(100*sum(int(size)*freq for size,freq in h.items() if lo<=int(size)<=hi)/rows[0]['count'])
 ax.bar(names,values,bottom=bottom,label=label,color=color);bottom+=values
ax.set_ylim(0,100);ax.set_ylabel('Inputs processed in each batch-size group (%)');ax.set_title('D   Batch occupancy',loc='left',weight='bold');ax.legend(title='Batch size',frameon=False,ncol=6,fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.16));ax.tick_params(axis='x',labelsize=9)
for ax in axes.flat:style(ax)
fig.text(.08,.04,'Solid/dashed: fresh repetitions. Throughput blocks interpolate 1-second output counts; startup block omitted.',fontsize=9,color='#47586C')
fig.text(.08,.018,'RAM displays 1-second maxima from 20 ms system MemAvailable samples. CUDA allocation and RSS are not added.',fontsize=9,color='#47586C')
save(fig,'abodybuilder3_repertoire_sustained_261026')
