"""Bounded geometry/pair-update kernel screen; no learned model changes.

Compare the existing strided distance-channel update with the existing full
concatenation alternative. Micro timings are not full-pipeline speedups.
"""
import argparse
import json
from pathlib import Path
import statistics

import torch
from abodybuilder3.aot_runtime import configure_precision


def strided_update(trans, mask, static, buffer):
    distance = torch.norm(trans[:, :, None] - trans[:, None, :], dim=-1)
    distance = distance * (mask[:, :, None] * mask[:, None, :])
    buffer[..., -1].copy_(distance)
    return buffer


def concatenated_update(trans, mask, static, buffer):
    distance = torch.norm(trans[:, :, None] - trans[:, None, :], dim=-1)
    distance = distance * (mask[:, :, None] * mask[:, None, :])
    return torch.cat((static, distance.unsqueeze(-1)), dim=-1)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();configure_precision('strict');torch.set_num_threads(1);torch.manual_seed(42)
    functions={name:torch.compile(fn,fullgraph=True,dynamic=True)
               for name,fn in [('strided',strided_update),('concatenated',concatenated_update)]}
    rows=[]
    with torch.no_grad():
        for batch,length in ((32,232),(32,248),(64,232)):
            trans=torch.randn(batch,length,3,device='cuda')
            mask=torch.ones(batch,length,device='cuda');mask[:,-7:]=0
            static=torch.randn(batch,length,length,127,device='cuda')
            buffer=torch.empty(batch,length,length,128,device='cuda');buffer[...,:-1].copy_(static)
            for value in (trans,static,buffer):torch._dynamo.mark_static(value,-1)
            reference=functions['strided'](trans,mask,static,buffer).clone()
            actual=functions['concatenated'](trans,mask,static,buffer)
            exact=torch.equal(reference,actual);max_difference=float((reference-actual).abs().max())
            timings={}
            for name,fn in functions.items():
                for _ in range(3):fn(trans,mask,static,buffer)
                values=[]
                for _ in range(3):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(50):output=fn(trans,mask,static,buffer)
                    end.record();end.synchronize();values.append(start.elapsed_time(end)/50)
                timings[name]=values
            rows.append({'batch':batch,'length':length,'outputs_exact':exact,'max_difference':max_difference,
                         'milliseconds_per_update':timings,
                         'concatenation_speedup':statistics.median(timings['strided'])/statistics.median(timings['concatenated'])})
            print(json.dumps(rows[-1]),flush=True)
            del reference,actual,output,static,buffer,trans,mask
    args.output.write_text(json.dumps({'completed':True,'scope':'Standalone compiled geometry/pair update; no whole-model or accuracy claim.','rows':rows},indent=2)+'\n')


if __name__=='__main__':main()
