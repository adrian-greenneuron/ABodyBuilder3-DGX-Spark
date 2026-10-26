"""Stage and full-job comparison of the original/direct encoder lifecycle."""
import time
START = time.perf_counter()
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from benchmark_runtime import Memory
from prediction_fingerprint import fingerprint


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['reference','direct'],required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    sys.path.insert(0,'/opt/abodybuilder3/scripts')
    if args.mode=='direct':from candidate.model import ProtT5
    else:from reference.model import ProtT5
    spec=importlib.util.spec_from_file_location('language_component_cli','/opt/abodybuilder3/scripts/predict.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    from abodybuilder3.aot_runtime import configure_precision
    configure_precision('strict');torch.set_num_threads(1);torch.manual_seed(42)
    rows=list(csv.DictReader(args.input.open()));args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'pdbs').mkdir();(args.output/'raw').mkdir();(args.output/'embeddings').mkdir()
    stage=[];fingerprints=[]
    with Memory() as memory:
        started=time.perf_counter();model=cli.load_model('language',torch.device('cuda'));torch.cuda.synchronize()
        stage.append({'stage':'structure_model_load','seconds':time.perf_counter()-started})
        started=time.perf_counter();encoder=ProtT5(device_map='cuda');torch.cuda.synchronize()
        stage.append({'stage':'encoder_and_tokeniser_load','seconds':time.perf_counter()-started})
        dtype_counts={}
        for parameter in encoder.embedding_module.model.parameters():
            key=str(parameter.dtype);dtype_counts[key]=dtype_counts.get(key,0)+parameter.numel()
        original_embeddings=encoder.get_embeddings
        collected=[]
        def embeddings(heavy,light):
            torch.cuda.synchronize();started=time.perf_counter()
            result=original_embeddings(heavy,light);torch.cuda.synchronize()
            stage.append({'stage':'embedding','seconds':time.perf_counter()-started,
                          'encoder_device_after':str(next(encoder.embedding_module.model.parameters()).device)})
            collected.append(result[0].detach().cpu().numpy().copy())
            return result
        encoder.get_embeddings=embeddings
        for row in rows:
            started=time.perf_counter()
            inp=cli.prepare_input_with_embeddings(row['heavy'],row['light'],encoder,torch.device('cuda'))
            torch.cuda.synchronize();mark=time.perf_counter()
            output=cli.run_inference(model,inp,torch.device('cuda'));torch.cuda.synchronize()
            stage.append({'stage':'structure_inference','seconds':time.perf_counter()-mark})
            pdb=cli.generate_pdb(output,inp);(args.output/'pdbs'/(row['name']+'.pdb')).write_text(pdb)
            fingerprints.append({'name':row['name'],**fingerprint(output,inp)})
            np.save(args.output/'embeddings'/(row['name']+'.npy'),collected[-1])
            np.savez(args.output/'raw'/(row['name']+'.npz'),positions=output['atom37'].detach().cpu().numpy(),
                     masks=output['atom37_atom_exists'].detach().cpu().numpy(),
                     heavy=inp['is_heavy'].detach().cpu().numpy().astype(bool),aatype=inp['aatype'].cpu().numpy())
            stage.append({'stage':'complete_antibody','name':row['name'],'seconds':time.perf_counter()-started})
            print('ANTIBODY_COMPLETE',row['name'],stage[-1]['seconds'],flush=True)
        finished=time.perf_counter()
    result={'completed':True,'mode':args.mode,'count':len(rows),'encoder_parameter_dtypes':dtype_counts,
            'script_seconds':finished-START,'stages':stage,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),**memory.result()}
    (args.output/'metrics_261026.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'fingerprints_261026.json').write_text(json.dumps(fingerprints,indent=2)+'\n')


if __name__=='__main__':main()
