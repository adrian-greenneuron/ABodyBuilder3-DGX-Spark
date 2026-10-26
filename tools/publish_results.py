"""Plot and package compact language-lifecycle evidence without model payloads."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True)
    args=parser.parse_args();root=args.study;repo=Path(__file__).resolve().parents[1]
    records=json.loads((root/'measurements_261026.json').read_text())
    expanded=json.loads((root/'expanded_cli_measurements_261026.json').read_text())
    fast=json.loads((root/'matched_fast_checks_261026.json').read_text())
    assets=repo/'docs/assets';data=repo/'docs/data';assets.mkdir(exist_ok=True);data.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colours={'reference':'#687888','direct':'#16729e'}
    fig,axes=plt.subplots(1,3,figsize=(13,4.9))
    for index,mode in enumerate(('reference','direct')):
        rows=[row for row in records if row['mode']==mode]
        elapsed=[row['container_seconds'] for row in rows]
        embeddings=[value['seconds'] for row in rows for value in row['metrics']['stages'] if value['stage']=='embedding']
        for axis,values,title in ((axes[0],elapsed,'Five-antibody complete job (s)'),
                                  (axes[1],embeddings,'Embedding time per antibody (s)')):
            value=statistics.median(values)
            if axis is axes[0]:
                axis.bar(index,value,color=colours[mode],width=.6)
            else:
                axis.plot([index-.2,index+.2],[value,value],color=colours[mode],linewidth=4)
            axis.scatter(np.full(len(values),index),values,color='black',s=13,zorder=3)
            axis.text(index,(value+max(values)*.06 if axis is axes[0] else value*1.35),f'{value:.2f}',ha='center',fontweight='bold')
            axis.set_title(title);axis.set_xticks([0,1],['Trainer','Resident encoder'])
            if axis is axes[0]:
                axis.set_ylim(0,axis.get_ylim()[1]*1.2)
            else:
                axis.set_yscale('log');axis.set_ylim(.06,100)
                axis.set_yticks([.1,1,10,100],['0.1','1','10','100'])
                axis.set_title(title+' · log scale')
            axis.grid(axis='y',alpha=.2)
        for j,metric in enumerate(('peak_cuda_allocated_bytes','sampled_process_tree_rss_peak_bytes')):
            value=max(row['metrics'][metric] for row in rows)/2**30
            x=j+(-.18 if mode=='reference' else .18)
            axes[2].bar(x,value,.34,color=colours[mode],label=mode if j==0 else None)
            axes[2].text(x,value+.12,f'{value:.2f}',ha='center',fontsize=10)
    axes[2].set(title='Higher measured peak (GiB)',xticks=[0,1],xticklabels=['CUDA allocated','Process RSS'])
    axes[2].set_ylim(0,10);axes[2].grid(axis='y',alpha=.2)
    fig.suptitle('Same batch-1 encoder, same embeddings: remove repeated teardown',fontweight='bold',y=1.03)
    fig.text(.5,-.02,'Strict FP32 · five published inputs · two fresh runs per condition · black points are observations.\n'
             'CPU thread cap 1 for this stage comparison. CUDA allocation and RSS overlap; do not add them.',ha='center',fontsize=10)
    fig.tight_layout()
    for suffix in ('png','svg'):
        path=assets/f'prott5_lifecycle_261026.{suffix}';fig.savefig(path,dpi=160,bbox_inches='tight',facecolor='white')
        if suffix=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    plt.close(fig)
    evidence={'initial':records,'installed_cli':expanded,'fast_path':fast,
              'exactness':json.loads((root/'expanded_cli_validation_261026.json').read_text()),
              'provenance':json.loads((repo/'provenance_261026.json').read_text())}
    (data/'prott5_lifecycle_measurements_261026.json').write_text(json.dumps(evidence,indent=2)+'\n')
    members={}
    for name in ('protocol_261026.json','measurements_261026.json','expanded_protocol_261026.json',
                 'expanded_selection_261026.json','expanded_cli_measurements_261026.json',
                 'expanded_cli_validation_261026.json','matched_fast_checks_261026.json',
                 'fast_timing_clarification_261026.json'):
        members[name]=(root/name).read_bytes()
    archive=data/'prott5_lifecycle_evidence_261026.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name,value in members.items():z.writestr(name,value)
        z.writestr('SHA256SUMS',''.join(f'{hashlib.sha256(value).hexdigest()}  {name}\n' for name,value in sorted(members.items())))
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise ValueError('Archive CRC failed.')
    (data/'prott5_lifecycle_checksums_261026.txt').write_text(f'{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n')
    print(json.dumps({'completed':True,'archive_bytes':archive.stat().st_size}),flush=True)


if __name__=='__main__':main()
