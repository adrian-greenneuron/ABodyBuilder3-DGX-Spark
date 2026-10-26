"""Compare system-level UMA pressure around unchanged installed CLI runs."""
import argparse,csv,json,os,statistics,subprocess,time,threading,math
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True);parser.add_argument('--prott5',type=Path,required=True);args=parser.parse_args();root=args.study.resolve()
protocol=json.loads((root/'protocol_261026.json').read_text());records=[]
fields=['MemTotal','MemFree','MemAvailable','Buffers','Cached','SwapFree','SReclaimable','SUnreclaim','AnonPages','Shmem','PageTables','KernelStack','Dirty']
images={x:protocol['versions'][x]['image'] for x in ['previous','new']}
for model,count in [('base',1000),('plddt',1000),('language',5)]:
 for rep in (1,2):
  for condition in (['previous','new'] if rep==1 else ['new','previous']):
   label=f'{model}_{condition}_r{rep}';out=root/label;out.mkdir();(out/'pdbs').mkdir();(out/'lightning_logs').mkdir()
   samples=[];stop=threading.Event();phase='pre'
   def observe():
    with (out/'system_memory_trace_261026.csv').open('w') as stream:
     writer=csv.writer(stream);writer.writerow(['monotonic','phase',*fields])
     while not stop.is_set():
      now=time.monotonic();mem={k:int(v.split()[0])*1024 for line in Path('/proc/meminfo').read_text().splitlines() for k,v in [line.split(':',1)]};row={'monotonic':now,'phase':phase,**{k:mem[k] for k in fields}};samples.append(row);writer.writerow([now,phase,*[mem[k] for k in fields]]);stop.wait(.02)
   thread=threading.Thread(target=observe,daemon=True);thread.start();time.sleep(5)
   command=['docker','run','--rm','--network','none','--gpus','all','--ipc=host','--user',f'{os.getuid()}:{os.getgid()}','-e','HOME=/tmp','-e','PYTHONHASHSEED=0','-e','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True','-v',f'{root}/tools:/measurement:ro','-v',f'{root}/inputs:/inputs:ro','-v',f'{out}:/results']
   if model=='language':command+=['-e','HF_HUB_OFFLINE=1','-e','TRANSFORMERS_OFFLINE=1','-v',f'{args.prott5.resolve()}:/opt/abodybuilder3/Rostlab/prot_t5_xl_uniref50:ro','-v',f'{out}/lightning_logs:/opt/abodybuilder3/lightning_logs']
   command+=[images[condition],'python','/measurement/measure_pss.py','predict','--csv',f'/inputs/antibodies_{count}.csv','--output','/results/pdbs/','--model',model,'--seed','42','--quiet']
   (out/'command_261026.json').write_text(json.dumps(command,indent=2)+'\n');print('START',label,flush=True);phase='job';start=time.perf_counter()
   with (out/'inference.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
   elapsed=time.perf_counter()-start;phase='post';time.sleep(8);stop.set();thread.join()
   if result.returncode:raise RuntimeError(f'{label} failed: {result.returncode}')
   before=[x for x in samples if x['phase']=='pre'];during=[x for x in samples if x['phase']=='job'];after=[x for x in samples if x['phase']=='post']
   baseline=statistics.median(x['MemAvailable'] for x in before);post=statistics.median(x['MemAvailable'] for x in after[-100:]);lowest=min(during,key=lambda x:x['MemAvailable']);delta=baseline-lowest['MemAvailable'];drift=post-baseline
   expected=list(csv.DictReader((root/f'inputs/antibodies_{count}.csv').open()));assert sorted(p.name for p in (out/'pdbs').glob('*.pdb'))==sorted(x['name']+'.pdb' for x in expected)
   for p in (out/'pdbs').glob('*.pdb'):
    text=p.read_text();assert text.endswith('END\n');assert all(math.isfinite(float(line[k:k+8])) for line in text.splitlines() if line.startswith('ATOM  ') for k in [30,38,46])
   pss=json.loads((out/'process_pss_261026.json').read_text());assert pss['completed']
   row={'model':model,'count':count,'condition':condition,'replicate':rep,'complete_container_seconds':elapsed,'baseline_available_bytes':baseline,'min_job_available_bytes':lowest['MemAvailable'],'peak_additional_unavailable_bytes':delta,'pre_idle_range_bytes':max(x['MemAvailable'] for x in before)-min(x['MemAvailable'] for x in before),'post_idle_drift_bytes':drift,'idle_drift_flag':abs(drift)>0.25*2**30,'swap_change_bytes':before[0]['SwapFree']-min(x['SwapFree'] for x in during),'peak_snapshot':lowest,'peak_pss_bytes':pss['peak_pss_bytes'],'peak_private_bytes':pss['peak_private_bytes'],'peak_summed_rss_bytes':pss['peak_summed_rss_bytes'],'pdb_count':len(expected)}
   records.append(row);(root/'measurements_261026.json').write_text(json.dumps({'completed':len(records)==12,'protocol':protocol,'rows':records},indent=2)+'\n');print('DONE',label,'peak additional GiB',round(delta/2**30,3),'idle drift GiB',round(drift/2**30,3),flush=True)
print('ALL_UNIFIED_MEMORY_MEASUREMENTS_COMPLETE',flush=True)
