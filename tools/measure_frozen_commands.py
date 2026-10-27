"""Measure a finite frozen command list with host unified-memory accounting.

Commands and output directories are explicit protocol fields. This measures
whole commands, not summed process RSS or CUDA allocations. It never clears
caches or retries a scientific job. Failed commands retain logs and traces.
"""
import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
import threading
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--resume',action='store_true');args=p.parse_args()
    protocol=json.loads(args.protocol.read_text());root=args.protocol.parent
    results=root/protocol['measurements'];rows=json.loads(results.read_text())['rows'] if args.resume else []
    done={r['label'] for r in rows}
    for case in protocol['cases']:
        if case['label'] in done:continue
        folder=root/case['directory'];folder.mkdir(parents=True)
        samples=[];stop=threading.Event();phase='pre';errors=[]
        def sample():
            try:
                with (folder/'unified_memory_trace_271026.csv').open('w') as f:
                    writer=csv.writer(f);writer.writerow(['monotonic','phase','MemAvailable','SwapFree'])
                    while not stop.is_set():
                        mem={k:int(v.split()[0])*1024 for line in Path('/proc/meminfo').read_text().splitlines() for k,v in [line.split(':',1)]}
                        row=(time.monotonic(),phase,mem['MemAvailable'],mem['SwapFree']);samples.append(row);writer.writerow(row)
                        stop.wait(.02)
            except Exception as exc:errors.append(repr(exc))
        thread=threading.Thread(target=sample,daemon=True);thread.start();time.sleep(5)
        print('START',case['label'],flush=True);phase='job';started=time.perf_counter()
        try:
            with (folder/'command.log').open('w') as log:
                result=subprocess.run(case['command'],stdout=log,stderr=subprocess.STDOUT)
            elapsed=time.perf_counter()-started;phase='post';time.sleep(8)
        finally:stop.set();thread.join()
        assert not errors,errors
        pre=[r for r in samples if r[1]=='pre'];during=[r for r in samples if r[1]=='job'];post=[r for r in samples if r[1]=='post']
        baseline=statistics.median(r[2] for r in pre);after=statistics.median(r[2] for r in post[-100:]);low=min(r[2] for r in during)
        row={**{k:v for k,v in case.items() if k!='command'},'exit_code':result.returncode,
             'command_seconds':elapsed,'baseline_available_bytes':baseline,'post_available_bytes':after,
             'peak_additional_unavailable_bytes':baseline-low,'ram_sensitivity_bytes':sorted([baseline-low,after-low]),
             'pre_idle_range_bytes':max(r[2] for r in pre)-min(r[2] for r in pre),
             'post_idle_drift_bytes':after-baseline,'baseline_drift_flag':abs(after-baseline)>.25*2**30,
             'swap_change_bytes':pre[0][3]-min(r[3] for r in during)}
        rows.append(row);results.write_text(json.dumps({'completed':len(rows)==len(protocol['cases']),'rows':rows},indent=2)+'\n')
        print('DONE',case['label'],round(elapsed,3),'s',round((baseline-low)/2**30,3),'GiB',result.returncode,flush=True)
        if result.returncode not in case.get('allowed_exit_codes',[0]):
            raise RuntimeError('Command failed; inspect retained log. No automatic retry.')
        if case.get('verification_command'):
            subprocess.run(case['verification_command'],check=True)


if __name__=='__main__':main()
