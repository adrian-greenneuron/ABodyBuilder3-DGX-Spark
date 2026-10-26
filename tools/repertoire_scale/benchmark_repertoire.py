"""Measure installed repertoire CLI throughput, output progression and shared RAM."""
import argparse,collections,csv,ctypes,hashlib,json,math,os,select,shutil,statistics,struct,subprocess,threading,time
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True);args=parser.parse_args();root=args.study.resolve();protocol=json.loads((root/'protocol_261026.json').read_text());records=[]
fields=['MemTotal','MemFree','MemAvailable','Cached','SReclaimable','SUnreclaim','AnonPages','Shmem','SwapFree','Dirty','PageTables']
libc=ctypes.CDLL(None,use_errno=True);libc.inotify_init1.argtypes=[ctypes.c_int];libc.inotify_add_watch.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_uint32]
for case in protocol['cases']:
 label=case['label'];out=root/label;out.mkdir();pdbs=out/'pdbs';pdbs.mkdir()
 assert shutil.disk_usage(root).free>100*2**30,'Less than 100 GiB disk reserve'
 inputs=root/'inputs'/case['input'];assert hashlib.sha256(inputs.read_bytes()).hexdigest()==case['input_sha256']
 expected=list(csv.DictReader(inputs.open()));expected_by_name={x['name']:x for x in expected};assert len(expected_by_name)==case['count']
 fd=libc.inotify_init1(os.O_NONBLOCK|os.O_CLOEXEC);assert fd>=0;assert libc.inotify_add_watch(fd,os.fsencode(pdbs),0x8)>=0
 stop=threading.Event();phase='pre';samples=[];completed=[];overflows=[];first_event=threading.Event();launch_time=0.
 def observe():
  count=0;last_progress=0.
  with (out/'system_memory_trace_261026.csv').open('w') as memstream,(out/'output_progress_261026.csv').open('w') as progress:
   mw=csv.writer(memstream);mw.writerow(['monotonic','phase',*fields]);pw=csv.writer(progress);pw.writerow(['monotonic','completed_pdbs'])
   while not stop.is_set():
    now=time.monotonic();mem={k:int(v.split()[0])*1024 for line in Path('/proc/meminfo').read_text().splitlines() for k,v in [line.split(':',1)]};row={'monotonic':now,'phase':phase,**{k:mem[k] for k in fields}};samples.append(row);mw.writerow([now,phase,*[mem[k] for k in fields]])
    try:
     while True:
      block=os.read(fd,65536)
      offset=0
      while offset<len(block):
       wd,mask,cookie,length=struct.unpack_from('iIII',block,offset);name=block[offset+16:offset+16+length].split(b'\0',1)[0];offset+=16+length
       if mask&0x4000:overflows.append(now)
       if mask&0x8 and name.endswith(b'.pdb'):
        count+=1;completed.append((now,count))
        if count==1:print('FIRST_PDB',label,flush=True);first_event.set()
    except BlockingIOError:pass
    if phase!='pre' and (now-last_progress>=1 or stop.is_set()):pw.writerow([now,count]);last_progress=now
    stop.wait(.02)
   pw.writerow([time.monotonic(),count])
 thread=threading.Thread(target=observe,daemon=True);thread.start();time.sleep(5)
 command=['docker','run','--rm','--network','none','--gpus','all','--ipc=host','--user',f'{os.getuid()}:{os.getgid()}','-e','HOME=/tmp','-e','PYTHONHASHSEED=0','-e','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True','-v',f'{root}/inputs:/inputs:ro','-v',f'{out}:/results',protocol['image_id'],'python','scripts/predict.py','predict','--csv',f'/inputs/{case["input"]}','--output','/results/pdbs/','--model',case['model'],'--seed','42','--quiet','--batch-report','/results/batching.json']
 (out/'command_261026.json').write_text(json.dumps(command,indent=2)+'\n');print('START',label,flush=True);phase='job';launch_time=time.monotonic();started=time.perf_counter()
 with (out/'inference.log').open('w') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
 elapsed=time.perf_counter()-started;finish=time.monotonic();phase='post';time.sleep(8);stop.set();thread.join();os.close(fd)
 if result.returncode:raise RuntimeError(f'{label} failed with exit {result.returncode}')
 # Verify all names and complete output endings; sample finite coordinates outside timing.
 actual={p.stem:p for p in pdbs.iterdir() if p.suffix=='.pdb'};assert set(actual)==set(expected_by_name)
 for path in actual.values():
  with path.open('rb') as stream:stream.seek(-4,2);assert stream.read()==b'END\n'
 sample_names=set(sorted(expected_by_name,key=lambda x:hashlib.sha256(x.encode()).digest())[:128]);sample_names.update(x['name'] for x in expected if not 200<=len(x['heavy'])+len(x['light'])<=272)
 for name in sample_names:
  lines=actual[name].read_text().splitlines();atoms=[x for x in lines if x.startswith('ATOM  ')];assert atoms
  assert all(math.isfinite(float(x[k:k+8])) for x in atoms for k in [30,38,46]);assert sum(x[12:16].strip()=='CA' for x in atoms)==len(expected_by_name[name]['heavy'])+len(expected_by_name[name]['light'])
 batching=json.loads((out/'batching.json').read_text());assert batching['completed'] and batching['saved']==case['count'] and batching['skipped']==0
 assert len(completed)==case['count'] and not overflows,('inotify count',len(completed),len(overflows))
 before=[x for x in samples if x['phase']=='pre'];during=[x for x in samples if x['phase']=='job'];after=[x for x in samples if x['phase']=='post'];baseline=statistics.median(x['MemAvailable'] for x in before);post=statistics.median(x['MemAvailable'] for x in after[-100:]);minimum=min(x['MemAvailable'] for x in during)
 row={**case,'completed':True,'exit_code':result.returncode,'complete_container_seconds':elapsed,'structures_per_second':case['count']/elapsed,'structures_per_minute':60*case['count']/elapsed,'first_pdb_seconds':completed[0][0]-launch_time,'last_pdb_seconds':completed[-1][0]-launch_time,'first_to_last_pdb_seconds':completed[-1][0]-completed[0][0],'baseline_available_bytes':baseline,'min_job_available_bytes':minimum,'peak_additional_unavailable_bytes':baseline-minimum,'post_idle_drift_bytes':post-baseline,'idle_drift_flag':abs(post-baseline)>.25*2**30,'pre_idle_range_bytes':max(x['MemAvailable'] for x in before)-min(x['MemAvailable'] for x in before),'swap_change_bytes':before[0]['SwapFree']-min(x['SwapFree'] for x in during),'pdb_count':len(actual),'all_pdb_endings_checked':True,'finite_coordinate_and_CA_count_samples':len(sample_names),'batch_count':len(batching['batches']),'aot_batches':batching['deployment']['aot_batches'],'eager_fallback_batches':batching['deployment']['eager_shape_fallback_batches'],'fallback_reasons':batching['deployment']['fallback_reasons'],'oom_retries':len(batching['oom_retries']),'batch_size_histogram':dict(collections.Counter(x['size'] for x in batching['batches'])),'launch_monotonic':launch_time,'finish_monotonic':finish,'output_bytes':sum(p.stat().st_size for p in actual.values())}
 records.append(row);(root/'measurements_261026.json').write_text(json.dumps({'completed':len(records)==len(protocol['cases']),'protocol':protocol,'rows':records},indent=2)+'\n');print('DONE',label,'per_min',round(row['structures_per_minute'],1),'peak_ram_gib',round(row['peak_additional_unavailable_bytes']/2**30,3),flush=True)
print('ALL_REPERTOIRE_BENCHMARKS_COMPLETE',flush=True)
