"""Finite fresh-container complete-pipeline comparison; no installed changes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

IMAGE = 'sha256:18456b5147ea1ec35d8de4f664962baeea6af1bb473ad9107bb1cbfa37fd844d'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(args):
    root = args.study.resolve()
    root.mkdir(parents=True, exist_ok=False)
    frozen = root/'frozen_dev'; frozen.mkdir()
    for name in ['predict.py', 'batch_pipeline.py', 'structure_preparation.py']:
        shutil.copy2(args.dev/'scripts'/name, frozen/('predict_main.py' if name=='predict.py' else name))
    component = root/'frozen_component'
    shutil.copytree(Path(__file__).resolve().parents[1], component,
                    ignore=shutil.ignore_patterns('.git', '__pycache__', 'docs'))
    measurement = root/'frozen_measurement'; measurement.mkdir()
    for name in ['benchmark_runtime.py', 'prediction_fingerprint.py']:
        shutil.copy2(args.dev/'tools'/name, measurement/name)
    metadata = json.loads((args.previous/'base/compiled/metadata_251026.json').read_text())
    targets = metadata['targets']
    assert len(targets) == 100
    identities = {}
    for count in args.counts:
        identities[str(count)] = {}
        with (frozen/f'antibodies_{count}.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['name','heavy','light'], lineterminator='\n')
            writer.writeheader()
            for i in range(count):
                target = list(targets)[i % 100]
                name = f'{i:04d}_{target}'
                identities[str(count)][name] = target
                writer.writerow({'name':name, 'heavy':targets[target]['heavy'], 'light':targets[target]['light']})
    (frozen/'input_identities_251026.json').write_text(json.dumps(identities, indent=2)+'\n')
    protocol = {'study_id':root.name, 'image':IMAGE, 'models':['base','plddt'],
        'conditions':['eager/normal','compiled/normal','combined/normal','eager/strict','compiled/strict','combined/strict'],
        'counts':args.counts, 'fresh_process_repetitions':2,
        'input_source':'Same 100 experimental antibodies; 1000 workload cycles them ten times, without prediction caching.',
        'compiler':'torch.compile default mode, dynamic=True to accommodate the unchanged production scheduler; fresh empty disk cache for every job.',
        'comparison':'Full frozen dev CLI including imports, checkpoint load, feature preparation, transfers, atom conversion and writing every PDB; common fingerprint/capture instrumentation. Raw evidence flush also measured separately.',
        'batching':'Unchanged auto caps (16 for <=128, otherwise32), original bounded length bins/pair budget, two preparation and two writer workers.',
        'no_enabling':True, 'no_chemistry':True,
        'main_commit':subprocess.check_output(['git','-C',str(args.main),'rev-parse','HEAD'],text=True).strip(),
        'dev_commit':subprocess.check_output(['git','-C',str(args.dev),'rev-parse','HEAD'],text=True).strip(),
        'component_parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'experimental_tensor_sha256':{name:entry['source_sha256'] for name,entry in targets.items()},
        'frozen_source_sha256':{str(p.relative_to(root)):sha(p) for parent in [frozen,component,measurement] for p in parent.rglob('*') if p.is_file()}}
    (root/'protocol_251026.json').write_text(json.dumps(protocol, indent=2)+'\n')
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--dev', type=Path, required=True)
    p.add_argument('--main', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--counts', type=int, nargs='+', default=[100,1000])
    args = p.parse_args()
    for count in args.counts:
        if count < 100 or count % 100:
            p.error('Workloads must contain whole repeats of the same 100 targets.')
    root = prepare(args)
    records = []
    conditions = [('eager','normal'),('compiled','normal'),('combined','normal'),
                  ('eager','strict'),('compiled','strict'),('combined','strict')]
    image = subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip()
    assert image == IMAGE
    for count in args.counts:
        for model in ['base','plddt']:
            for rep in [1,2]:
                order = conditions if (rep==1) == (model=='base') else conditions[::-1]
                for mode,precision in order:
                    label = f'{mode}_{precision}'
                    dest = root/f'n{count}'/model/label/f'run{rep}'
                    dest.mkdir(parents=True, exist_ok=False)
                    name = f'abb3-full-{root.name}-{count}-{model}-{label}-r{rep}'
                    command = ['docker','run','--rm','--name',name,'--network','none','--gpus','all','--ipc=host',
                        '--user','1000:1000', '-e','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
                        '-e','PYTHONHASHSEED=0','-e','PYTHONPATH=/component/tools:/measurement:/frozen',
                        '-e','TORCHINDUCTOR_CACHE_DIR=/output/compiler_cache','-e','TRITON_CACHE_DIR=/output/triton_cache',
                        '-e','TORCHINDUCTOR_COMPILE_THREADS=4',
                        '-v',f'{root}/frozen_component:/component:ro','-v',f'{root}/frozen_dev:/frozen:ro',
                        '-v',f'{root}/frozen_measurement:/measurement:ro','-v',f'{dest}:/output',
                        '--entrypoint','python',image,'-u','/component/tools/measure_pipeline.py',
                        '--mode',mode,'--precision',precision,'--model',model,'--count',str(count)]
                    (dest/'command_251026.json').write_text(json.dumps(command, indent=2)+'\n')
                    print('START',count,model,mode,precision,rep,flush=True)
                    start = time.perf_counter()
                    with (dest/'inference_251026.log').open('w') as log:
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                    elapsed = time.perf_counter()-start
                    if result.returncode:
                        print((dest/'inference_251026.log').read_text()[-12000:],flush=True)
                        raise SystemExit(result.returncode)
                    metrics = json.loads((dest/'metrics_251026.json').read_text())
                    batches = json.loads((dest/'batching.json').read_text())
                    assert metrics['completed'] and batches['completed'] and batches['saved']==count
                    assert not batches['oom_retries'], 'OOM batch splitting would confound the matched comparison'
                    text = (dest/'inference_251026.log').read_text()
                    if 'hit config.recompile_limit' in text or 'hit config.cache_size_limit' in text:
                        raise RuntimeError(f'Compiler fell back after exhausting its cache: {dest}')
                    records.append({'count':count,'model':model,'mode':mode,'precision':precision,'replicate':rep,
                        'container_command_seconds':elapsed,'evidence_writing_seconds':metrics['raw_evidence_writing_seconds'],
                        'relative_output':str(dest.relative_to(root)), 'metrics':metrics,'batching':batches})
                    (root/'measurements_251026.json').write_text(json.dumps(records, indent=2)+'\n')
                    print('DONE',count,model,mode,precision,rep,round(elapsed,3),'seconds',flush=True)
    (root/'completed_251026.json').write_text(json.dumps({'completed':True,'jobs':len(records)})+'\n')
    print('COMPLETE',len(records),'fresh-container jobs',flush=True)


if __name__ == '__main__':
    main()
