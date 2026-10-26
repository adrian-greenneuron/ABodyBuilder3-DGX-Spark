from pathlib import Path
import subprocess,json,time,csv,os,argparse
parser=argparse.ArgumentParser(description='Replay the frozen main-version comparison in a prepared study directory.')
parser.add_argument('--study',type=Path,required=True)
parser.add_argument('--prott5',type=Path,required=True)
args=parser.parse_args()
root=args.study.resolve()
images={'previous':'abodybuilder3-official-spark:preparation-251026','new':'abodybuilder3-official-spark:accepted-runtime-261026'}
protocol=json.loads((root/'protocol_261026.json').read_text())
for condition,image in images.items():
    protocol['versions'][condition]['image']=image
    protocol['versions'][condition]['image_id']=subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip()
(root/'protocol_261026.json').write_text(json.dumps(protocol,indent=2)+'\n')
records=[]
for model,count in [('base',1000),('plddt',1000),('language',5)]:
    for replicate in (1,2):
        for condition in (['previous','new'] if replicate==1 else ['new','previous']):
            label=f'{model}_{condition}_r{replicate}';dest=root/label;dest.mkdir();(dest/'pdbs').mkdir();(dest/'lightning_logs').mkdir()
            command=['docker','run','--rm','--network','none','--gpus','all','--ipc=host','--user',f'{os.getuid()}:{os.getgid()}','-e','HOME=/tmp','-e','PYTHONHASHSEED=0','-e','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True','-e','PYTHONPATH=/measurement/instrumentation:/measurement','-e','ABB3_RESOURCE_DIR=/results/resources','-v',f'{root}/tools:/measurement:ro','-v',f'{root}/inputs:/inputs:ro','-v',f'{dest}:/results']
            if model=='language':command+=['-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-v',f'{args.prott5.resolve()}:/opt/abodybuilder3/Rostlab/prot_t5_xl_uniref50:ro','-v',f'{dest}/lightning_logs:/opt/abodybuilder3/lightning_logs']
            command+=[images[condition],'python','/measurement/measure_installed_cli.py','predict','--csv',f'/inputs/antibodies_{count}.csv','--output','/results/pdbs/','--model',model,'--seed','42','--quiet']
            (dest/'command_261026.json').write_text(json.dumps(command,indent=2)+'\n')
            print('START',label,flush=True);start=time.perf_counter()
            with (dest/'inference.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
            wall=time.perf_counter()-start
            if result.returncode:raise RuntimeError(f'{label} failed: {result.returncode}')
            expected=list(csv.DictReader((root/f'inputs/antibodies_{count}.csv').open()))
            assert sorted(p.name for p in (dest/'pdbs').glob('*.pdb'))==sorted(row['name']+'.pdb' for row in expected)
            import math
            for path in (dest/'pdbs').glob('*.pdb'):
                text=path.read_text();assert text.endswith('END\n')
                assert all(math.isfinite(float(line[k:k+8])) for line in text.splitlines() if line.startswith('ATOM  ') for k in (30,38,46))
            metrics=json.loads((dest/'metrics_261026.json').read_text());assert metrics['completed']
            current={};peak=0;events=[];exact=[]
            for path in (dest/'resources').glob('*.csv'):
                for x in csv.DictReader(path.open()):events.append((float(x['monotonic_seconds']),path.stem,int(x['cuda_allocated_bytes'])))
            for stamp,pid,value in sorted(events):current[pid]=value;peak=max(peak,sum(current.values()))
            for path in (dest/'resources').glob('*.json'):exact.append(json.loads(path.read_text()))
            assert peak>0
            record={'model':model,'count':count,'condition':condition,'replicate':replicate,'complete_container_seconds':wall,**metrics,'sampled_concurrent_cuda_allocated_peak_bytes':peak,'per_process_cuda_peaks':exact,'pdb_count':len(expected)}
            records.append(record);(root/'measurements_261026.json').write_text(json.dumps({'completed':len(records)==12,'protocol':protocol,'rows':records},indent=2)+'\n')
            print('DONE',label,round(wall,2),'seconds',flush=True)
print('ALL_MATCHED_BENCHMARKS_COMPLETE',flush=True)
