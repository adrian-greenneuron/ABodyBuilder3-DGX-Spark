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
    protocol = {'study_id':root.name, 'image':args.image, 'models':['base','plddt'],
        'conditions':[f'{mode}/{precision}' for precision in ['normal','strict'] for mode in args.modes],
        'counts':args.counts, 'fresh_process_repetitions':2,
        'input_source':'Same 100 experimental antibodies; 1000 workload cycles them ten times, without prediction caching.',
        'compiler':'Compiled modes use torch.compile default/dynamic=True. Graph mode replays unchanged eager operations with private CUDA graph pools; eager mode is the current dev control.',
        'comparison':'Full frozen dev CLI including imports, checkpoint load, feature preparation, transfers, atom conversion and writing every PDB; common fingerprint/capture instrumentation. Raw evidence flush also measured separately.',
        'batching':'Unchanged auto caps (16 for <=128, otherwise32), original bounded length bins/pair budget, two preparation and two writer workers.',
        'no_enabling':True, 'no_chemistry':True,
        'native_formatter':args.native_formatter,
        'requested_policy':{'batch_size':args.batch_size,'cpu_workers':args.cpu_workers,'batch_memory_gib':args.batch_memory_gib},
        'main_commit':subprocess.check_output(['git','-C',str(args.main),'rev-parse','HEAD'],text=True).strip(),
        'dev_commit':subprocess.check_output(['git','-C',str(args.dev),'rev-parse','HEAD'],text=True).strip(),
        'component_parent_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'experimental_tensor_sha256':{name:entry['source_sha256'] for name,entry in targets.items()},
        'frozen_source_sha256':{str(p.relative_to(root)):sha(p) for parent in [frozen,component,measurement] for p in parent.rglob('*') if p.is_file()}}
    protocol['cache_source'] = str(args.cache_source.resolve()) if args.cache_source else None
    protocol['cache_policy'] = ('Seed prior caches, prepare both workloads once on the GPU, then bake the caches into Docker layers; measured jobs use no host cache mounts.' if args.cache_source else protocol['compiler'])
    (root/f'protocol_{args.date_code}.json').write_text(json.dumps(protocol, indent=2)+'\n')
    return root


def cached_images(root, args, reference):
    """Prepare caches outside timed jobs, then package the exact prepared files."""
    base = root/'package_context';base.mkdir()
    for source,name in [(root/'frozen_component','component'),(root/'frozen_dev','frozen'),(root/'frozen_measurement','measurement')]:
        shutil.copytree(source,base/name)
    (base/'Dockerfile').write_text('ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\nCOPY component/ /component/\nCOPY frozen/ /frozen/\nCOPY measurement/ /measurement/\nENV PYTHONPATH=/component/tools:/measurement:/frozen\n')
    code_tag=f'abodybuilder3-component-spark:cache-code-{args.date_code}'
    parent_tag='abodybuilder3-dev-spark:optimised-251026'
    assert subprocess.check_output(['docker','image','inspect',parent_tag,'--format','{{.Id}}'],text=True).strip()==reference
    build=['docker','build','--network','none','--pull=false','--build-arg',f'BASE_IMAGE={parent_tag}','-t',code_tag,str(base)]
    with (root/f'code_image_build_{args.date_code}.log').open('w') as log:
        subprocess.run(build,check=True,stdout=log,stderr=subprocess.STDOUT)
    images={};preparation=[]
    for model in ['base','plddt']:
        for mode in ['compiled','combined']:
            for precision in ['normal','strict']:
                key=f'{model}/{mode}_{precision}'
                bank=root/'cache_banks'/model/f'{mode}_{precision}';bank.mkdir(parents=True)
                for count in args.counts:
                    old=args.cache_source/f'n{count}'/model/f'{mode}_{precision}'/'run1'
                    for cache in ['compiler_cache','triton_cache']:
                        shutil.copytree(old/cache,bank/cache,dirs_exist_ok=True,
                                        ignore=shutil.ignore_patterns('locks','*.lock','__pycache__'))
                # A prep run also checks the adapter with its stable packaged
                # paths. It is excluded from all measured job medians.
                for count in args.counts:
                    dest=bank/f'prepare_n{count}';dest.mkdir()
                    cmd=['docker','run','--rm','--network','none','--gpus','all','--ipc=host','--user','1000:1000',
                        '-e','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True','-e','PYTHONHASHSEED=0',
                        '-e','TORCHINDUCTOR_CACHE_DIR=/output/compiler_cache','-e','TRITON_CACHE_DIR=/output/triton_cache',
                        '-e','TORCHINDUCTOR_COMPILE_THREADS=4',
                        '-v',f'{bank}/compiler_cache:/output/compiler_cache','-v',f'{bank}/triton_cache:/output/triton_cache',
                        '-v',f'{dest}:/results','--entrypoint','python',code_tag,'-u','/component/tools/measure_pipeline.py',
                        '--mode',mode,'--precision',precision,'--model',model,'--count',str(count),
                        '--output','/results','--date-code',args.date_code]
                    print('PREPARE_CACHE',key,count,flush=True)
                    (dest/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
                    start=time.perf_counter()
                    with (dest/'inference.log').open('w') as log:
                        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
                    if result.returncode:
                        print((dest/'inference.log').read_text()[-12000:],flush=True)
                        raise SystemExit(result.returncode)
                    preparation.append({'condition':key,'count':count,'wall_seconds':time.perf_counter()-start,
                                        'metrics':json.loads((dest/f'metrics_{args.date_code}.json').read_text())})
                    (root/f'cache_preparation_{args.date_code}.json').write_text(json.dumps(preparation,indent=2)+'\n')
                manifest={str(p.relative_to(bank)):sha(p) for cache in ['compiler_cache','triton_cache']
                    for p in (bank/cache).rglob('*') if p.is_file() and 'locks' not in p.parts and p.suffix!='.lock'}
                (bank/'cache_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
                # Docker context omits all prediction outputs and private data.
                (bank/'.dockerignore').write_text('prepare_*/\n**/locks/\n**/*.lock\n')
                (bank/'Dockerfile').write_text('ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\nCOPY --chown=1000:1000 compiler_cache/ /output/compiler_cache/\nCOPY --chown=1000:1000 triton_cache/ /output/triton_cache/\nCOPY cache_manifest.json /opt/abodybuilder3_compilation_cache_manifest.json\nENV TORCHINDUCTOR_CACHE_DIR=/output/compiler_cache TRITON_CACHE_DIR=/output/triton_cache TORCHINDUCTOR_COMPILE_THREADS=4\n')
                tag=f'abodybuilder3-component-spark:{model}-{mode}-{precision}-cached-{args.date_code}'
                cmd=['docker','build','--network','none','--pull=false','--build-arg',f'BASE_IMAGE={code_tag}','-t',tag,str(bank)]
                with (bank/'build.log').open('w') as log:subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
                image=subprocess.check_output(['docker','image','inspect',tag,'--format','{{.Id}}'],text=True).strip()
                images[key]={'image':image,'tag':tag,'cache_files':len(manifest),
                             'cache_bytes':sum((bank/n).stat().st_size for n in manifest),
                             'manifest_sha256':sha(bank/'cache_manifest.json')}
                (root/f'cached_images_{args.date_code}.json').write_text(json.dumps(images,indent=2)+'\n')
                print('PACKAGED_CACHE',key,image,flush=True)
    return images


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--dev', type=Path, required=True)
    p.add_argument('--main', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--counts', type=int, nargs='+', default=[100,1000])
    p.add_argument('--cache-source', type=Path, help='Prepare and package caches from a completed cold-cache comparison.')
    p.add_argument('--date-code', default='251026')
    p.add_argument('--resume', action='store_true', help='Continue a failed controller using its frozen source and already prepared images.')
    p.add_argument('--modes',nargs='+',choices=['eager','compiled','combined','graph'],default=['eager','compiled','combined'])
    p.add_argument('--image',default=IMAGE,help='Pinned runtime or qualified formatter-only extension image.')
    p.add_argument('--native-formatter',action='store_true')
    p.add_argument('--batch-size',default='auto')
    p.add_argument('--cpu-workers',type=int,default=2)
    p.add_argument('--batch-memory-gib',type=float,default=6.)
    args = p.parse_args()
    for count in args.counts:
        if count < 100 or count % 100:
            p.error('Workloads must contain whole repeats of the same 100 targets.')
    root = args.study.resolve() if args.resume else prepare(args)
    measurements_path=root/f'measurements_{args.date_code}.json'
    records = json.loads(measurements_path.read_text()) if measurements_path.exists() else []
    conditions = [(mode,precision) for precision in ['normal','strict'] for mode in args.modes]
    image = subprocess.check_output(['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()
    if args.image==IMAGE:assert image == IMAGE
    images=(json.loads((root/f'cached_images_{args.date_code}.json').read_text()) if args.resume else cached_images(root,args,image)) if args.cache_source else {}
    for count in args.counts:
        for model in ['base','plddt']:
            for rep in [1,2]:
                order = conditions if (rep==1) == (model=='base') else conditions[::-1]
                for mode,precision in order:
                    label = f'{mode}_{precision}'
                    dest = root/f'n{count}'/model/label/f'run{rep}'
                    if any((r['count'],r['model'],r['mode'],r['precision'],r['replicate'])==(count,model,mode,precision,rep) for r in records):
                        continue
                    dest.mkdir(parents=True, exist_ok=False)
                    name = f'abb3-full-{root.name}-{count}-{model}-{label}-r{rep}'
                    active_image=images[f'{model}/{label}']['image'] if mode!='eager' and images else image
                    command = ['docker','run','--rm','--name',name,'--network','none','--gpus','all','--ipc=host',
                        '--user','1000:1000', '-e','PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True',
                        '-e','PYTHONHASHSEED=0','-e','PYTHONPATH=/component/tools:/measurement:/frozen',
                        '-e',f'TORCHINDUCTOR_CACHE_DIR={"/results" if mode=="eager" else "/output"}/compiler_cache',
                        '-e',f'TRITON_CACHE_DIR={"/results" if mode=="eager" else "/output"}/triton_cache',
                        '-e','TORCHINDUCTOR_COMPILE_THREADS=4']
                    if args.native_formatter:
                        command+=['-e','ABB3_NATIVE_FORMATTER=1','-e','ABB3_FORMATTER_USAGE=/results/formatter_usage.json']
                    if mode=='eager' or not images:
                        command+=['-v',f'{root}/frozen_component:/component:ro','-v',f'{root}/frozen_dev:/frozen:ro',
                                  '-v',f'{root}/frozen_measurement:/measurement:ro']
                    if not images:
                        command+=['-v',f'{dest}:/output']
                    command+=['-v',f'{dest}:/results','--entrypoint','python',active_image,'-u','/component/tools/measure_pipeline.py',
                        '--mode',mode,'--precision',precision,'--model',model,'--count',str(count),
                        '--output','/results','--date-code',args.date_code]
                    if (args.batch_size,args.cpu_workers,args.batch_memory_gib)!=('auto',2,6.):
                        command+=['--batch-size',args.batch_size,'--cpu-workers',str(args.cpu_workers),
                                  '--batch-memory-gib',str(args.batch_memory_gib)]
                    (dest/'command_251026.json').write_text(json.dumps(command, indent=2)+'\n')
                    print('START',count,model,mode,precision,rep,flush=True)
                    start = time.perf_counter()
                    with (dest/'inference_251026.log').open('w') as log:
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                    elapsed = time.perf_counter()-start
                    if result.returncode:
                        print((dest/'inference_251026.log').read_text()[-12000:],flush=True)
                        raise SystemExit(result.returncode)
                    metrics = json.loads((dest/f'metrics_{args.date_code}.json').read_text())
                    batches = json.loads((dest/'batching.json').read_text())
                    assert metrics['completed'] and batches['completed'] and batches['saved']==count
                    assert not batches['oom_retries'], 'OOM batch splitting would confound the matched comparison'
                    text = (dest/'inference_251026.log').read_text()
                    if 'hit config.recompile_limit' in text or 'hit config.cache_size_limit' in text:
                        raise RuntimeError(f'Compiler fell back after exhausting its cache: {dest}')
                    records.append({'count':count,'model':model,'mode':mode,'precision':precision,'replicate':rep,
                        'image':active_image,'packaged_cache':bool(images and mode!='eager'),
                        'container_command_seconds':elapsed,'evidence_writing_seconds':metrics['raw_evidence_writing_seconds'],
                        'relative_output':str(dest.relative_to(root)), 'metrics':metrics,'batching':batches})
                    (root/f'measurements_{args.date_code}.json').write_text(json.dumps(records, indent=2)+'\n')
                    print('DONE',count,model,mode,precision,rep,round(elapsed,3),'seconds',flush=True)
    (root/f'completed_{args.date_code}.json').write_text(json.dumps({'completed':True,'jobs':len(records)})+'\n')
    print('COMPLETE',len(records),'fresh-container jobs',flush=True)


if __name__ == '__main__':
    main()
