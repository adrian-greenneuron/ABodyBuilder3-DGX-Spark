"""Check Linux UMA accounting using touched, known-size CPU/CUDA allocations."""
import gc,json,time
from pathlib import Path
import torch

def phase(name):
    rows=[]
    for _ in range(30):
        data={k:int(v.split()[0])*1024 for line in Path('/proc/meminfo').read_text().splitlines() for k,v in [line.split(':',1)]}
        rows.append({'monotonic':time.monotonic(),'MemFree':data['MemFree'],'MemAvailable':data['MemAvailable'],'Cached':data['Cached'],'SReclaimable':data['SReclaimable'],'SwapFree':data['SwapFree']})
        time.sleep(.1)
    record={'phase':name,'samples':rows}
    records.append(record);print(name,flush=True)

records=[]
torch.cuda.init();torch.cuda.synchronize();phase('cuda_context_baseline')
size=4*1024**3
cpu=torch.empty(size,dtype=torch.uint8,device='cpu');cpu.fill_(17);phase('cpu_4gib_touched')
gpu=torch.empty(size,dtype=torch.uint8,device='cuda');gpu.fill_(23);torch.cuda.synchronize();phase('cpu4_cuda4_touched')
del cpu;gc.collect();phase('cuda_4gib_only')
del gpu;gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize();phase('allocations_released')
Path('/results/calibration_261026.json').write_text(json.dumps({'allocation_bytes':size,'phases':records},indent=2)+'\n')
