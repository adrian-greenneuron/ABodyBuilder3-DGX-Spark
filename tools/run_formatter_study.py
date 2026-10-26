"""Matched installed-CLI comparisons of original and native PDB formatting."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

IMAGE='abodybuilder3-component-spark:native-formatter-261026'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--dev',type=Path,required=True)
    p.add_argument('--truth',type=Path,required=True)
    args=p.parse_args();root=args.study.resolve();root.mkdir(parents=True,exist_ok=False)
    component=Path(__file__).resolve().parents[1]
    frozen=root/'frozen';frozen.mkdir()
    for name in ['predict.py','batch_pipeline.py','structure_preparation.py']:
        shutil.copy2(args.dev/'scripts'/name,frozen/('predict_main.py' if name=='predict.py' else name))
    targets=json.loads((args.truth/'base/compiled/metadata_251026.json').read_text())['targets']
    for count in [5,100,1000,10000]:
        with (frozen/f'antibodies_{count}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['name','heavy','light'],lineterminator='\n');w.writeheader()
            for i in range(count):
                name=list(targets)[i%100]
                w.writerow({'name':f'{i:05d}_{name}','heavy':targets[name]['heavy'],'light':targets[name]['light']})
    runtime=root/'measurement';shutil.copytree(component/'tools/runtime',runtime,ignore=shutil.ignore_patterns('__pycache__'))
    protocol={'purpose':'Change only PDB text formatting; compare complete jobs, raw fingerprints and PDB bytes.',
              'parent_image':'sha256:18456b5147ea1ec35d8de4f664962baeea6af1bb473ad9107bb1cbfa37fd844d',
              'dev_commit':'bd8a12ea9f58fa3577215532653a05c47e90e256','models':['base','plddt'],
              'precisions':['normal','strict'],'counts':[100,1000,10000], 'repetitions':2,
              'input_scope':'Repeat the same 100 published experimental antibodies; no prediction caching.',
              'source_sha256':{str(p.relative_to(component)):hashlib.sha256(p.read_bytes()).hexdigest() for p in component.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix!='.so'}}
    (root/'protocol_261026.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with (root/'image_build_261026.log').open('w') as log:
        subprocess.run(['docker','build','--network','none','--pull=false','-t',IMAGE,str(component)],check=True,stdout=log,stderr=subprocess.STDOUT)
    image=subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip()
    records=[];checks=[]
    for count in [5,100,1000,10000]:
        for model in ['base','plddt']:
            for precision in ['normal','strict']:
                for rep in ([1] if count==5 else [1,2]):
                    paths={}
                    for native in ([False,True] if rep==1 else [True,False]):
                        mode='native' if native else 'original'
                        dest=root/f'n{count}'/model/precision/mode/f'run{rep}';dest.mkdir(parents=True)
                        paths[mode]=dest
                        command=['docker','run','--rm','--network','none','--gpus','all','--ipc=host',
                            '-e','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True','-e','PYTHONHASHSEED=0',
                            '-e',f'ABB3_NATIVE_FORMATTER={int(native)}','-e',f'ABB3_STRICT_FP32={int(precision=="strict")}',
                            '-e','ABB3_FORMATTER_USAGE=/output/formatter_usage.json',
                            '-e','PYTHONPATH=/measurement/instrumentation:/measurement',
                            '-e','ABB3_RESOURCE_DIR=/output/resources','-e','ABB3_FINGERPRINT_PATH=/output/prediction_fingerprints.json',
                            '-v',f'{runtime}:/measurement:ro','-v',f'{frozen}:/frozen:ro','-v',f'{dest}:/output',
                            '--entrypoint','python',image,'/measurement/measure_cli.py','predict',
                            '--csv',f'/frozen/antibodies_{count}.csv','--output','/output/pdbs/',
                            '--model',model,'--workers','1','--seed','42','--quiet',
                            '--batch-size','auto','--cpu-workers','2','--batch-report','/output/batching.json']
                        (dest/'command_261026.json').write_text(json.dumps(command,indent=2)+'\n')
                        print('START',count,model,precision,mode,rep,flush=True);start=time.perf_counter()
                        with (dest/'inference.log').open('w') as log:
                            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
                        if result.returncode:
                            print((dest/'inference.log').read_text()[-10000:],flush=True);raise SystemExit(result.returncode)
                        seconds=time.perf_counter()-start
                        usage=json.loads((dest/'formatter_usage.json').read_text())
                        assert usage['native_calls']==(count if native else 0)
                        peaks=[json.loads(p.read_text()) for p in (dest/'resources').glob('*.json')]
                        row={'count':count,'model':model,'precision':precision,'mode':mode,'replicate':rep,
                             'container_seconds':seconds,'image':image,'relative_output':str(dest.relative_to(root)),
                             'usage':usage,'cuda_peak_allocated_bytes':max(p['peak_cuda_allocated_bytes'] for p in peaks),
                             'cuda_peak_reserved_bytes':max(p['peak_cuda_reserved_bytes'] for p in peaks),
                             **json.loads((dest/'cli_metrics.json').read_text())}
                        assert row['completed'];records.append(row)
                        (root/'measurements_261026.json').write_text(json.dumps(records,indent=2)+'\n')
                        print('DONE',count,model,precision,mode,rep,round(seconds,3),flush=True)
                    a=json.loads((paths['original']/'prediction_fingerprints.json').read_text())
                    b=json.loads((paths['native']/'prediction_fingerprints.json').read_text())
                    assert a==b,'Raw prediction differences are not accepted for a formatter rewrite'
                    files=sorted((paths['original']/'pdbs').glob('*.pdb'))
                    assert len(files)==count
                    for path in files:assert path.read_bytes()==(paths['native']/'pdbs'/path.name).read_bytes(),path.name
                    checks.append({'count':count,'model':model,'precision':precision,'replicate':rep,
                                   'all_raw_fingerprints_exact':True,'all_pdb_bytes_exact':True})
                    (root/'validation_261026.json').write_text(json.dumps(checks,indent=2)+'\n')
    (root/'completed_261026.json').write_text(json.dumps({'completed':True,'jobs':len(records),'matched_predictions':sum(c['count'] for c in checks)})+'\n')


if __name__=='__main__':main()
