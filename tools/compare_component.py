"""Compare isolated inference changes with frozen reference code and weights.

This is a numerical-regression/implementation benchmark, not a structural
accuracy experiment. Nonidentical candidates are recorded, never auto-accepted.
"""
import argparse
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixtures(stress):
    rows = json.loads((ROOT/'tests/benchmark_antibodies.json').read_text())
    if stress:
        first = rows[0]
        rows += [dict(name='padding_short', heavy=first['heavy'][:-20], light=first['light'][:-10]),
                 dict(name='padding_long', heavy=first['heavy']+'G'*20, light=first['light']+'S'*15)]
        for i in range(8):
            # Synthetic single-residue substitutions exercise numerical diversity;
            # they are not experimental antibodies or an accuracy assessment.
            heavy = list(first['heavy'])
            pos = 25+i*9
            heavy[pos] = 'S' if heavy[pos] != 'S' else 'A'
            rows.append(dict(name=f'substitution_{i}', heavy=''.join(heavy), light=first['light']))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=['base','plddt'], default=['base','plddt'])
    parser.add_argument('--implementation',choices=['candidate','accepted'],default='candidate')
    parser.add_argument('--variants', nargs='+', choices=['compatibility','pair_lookup','point_matmul','final_only','lookup_final','pair_buffer','exact_all','combined','compiled'],
                        default=['pair_lookup','point_matmul','final_only'])
    parser.add_argument('--count', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--public-only', action='store_true', help='Bound an initial compiler feasibility screen.')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.implementation=='accepted' and any(v in ('point_matmul','combined') for v in args.variants):
        parser.error('Nonidentical point-matmul experiments are absent from the accepted implementation.')
    if min(args.count, args.batch_size, args.repeats) < 1:
        parser.error('Counts must be positive.')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    sys.path.insert(0, '/opt/abodybuilder3/scripts')
    from batch_pipeline import BatchPolicy, plan_batches
    from abodybuilder3.batched_inference import prepare_cpu_batch, batch_add_atom37_to_output
    cli = import_file('component_cli', '/opt/abodybuilder3/scripts/predict.py')
    cli.VERBOSE = False
    ref_module = import_file('frozen_structure_module', ROOT/'reference/structure_module.py')
    module = import_file('candidate_structure_module', ROOT/args.implementation/'structure_module.py')
    compact = import_file('compact_features', ROOT/args.implementation/'features.py')
    # Reuse the existing resource sampler rather than another measurement policy.
    from benchmark_runtime import Memory
    cases = fixtures(not args.public_only)
    regression_groups = list(plan_batches([(x['name'],x['heavy'],x['light']) for x in cases],
                                          BatchPolicy(args.batch_size)))
    public = fixtures(False)
    throughput_groups = list(plan_batches([(str(i),public[i%len(public)]['heavy'],public[i%len(public)]['light'])
                                           for i in range(args.count)], BatchPolicy(args.batch_size)))
    records = []
    for model_name in args.models:
        original = cli.load_model(model_name, torch.device('cuda'))
        original.__class__ = ref_module.StructureModule
        for layer in original.ipa_layers:
            layer.__class__ = ref_module.InvariantPointAttention
        model = copy.deepcopy(original)
        model.__class__ = module.StructureModule
        for layer in model.ipa_layers:
            layer.__class__ = module.InvariantPointAttention
        assert list(original.state_dict())==list(model.state_dict())
        assert all(torch.equal(a,b) for a,b in zip(original.state_dict().values(),model.state_dict().values()))
        lookup = compact.PairProjection(model.linear_in_edge)
        compiled = None

        def predict(group, variant, format_pdb=False):
            compact_input = variant in ('pair_lookup','lookup_final','exact_all','combined')
            batch, inputs = (compact.prepare_compact_batch if compact_input else prepare_cpu_batch)(
                [(h,l) for _,h,l in group])
            batch = {k:v.to('cuda') for k,v in batch.items()}
            active = original if variant == 'reference' else model
            kwargs = {}
            if compact_input:
                batch['pair'] = lookup(batch.pop('pair_code'))
                kwargs['_inference_pair_projected'] = True
            if variant in ('point_matmul','combined'):
                kwargs['_inference_point_matmul'] = True
            if variant in ('final_only','lookup_final','exact_all','combined','compiled'):
                kwargs['_inference_final_only'] = True
            if variant in ('pair_buffer','exact_all'):
                kwargs['_inference_pair_buffer'] = True
            output = (compiled if variant == 'compiled' else active)(
                {'single':batch['single'],'pair':batch['pair']}, batch['aatype'],batch['seq_mask'],**kwargs)
            cpu = batch_add_atom37_to_output(output,batch['aatype'],[len(x['aatype']) for x in inputs])
            state = output['single'].detach().cpu().numpy() if not format_pdb else None
            pdbs = [cli.generate_pdb(out,inp) for out,inp in zip(cpu,inputs)]
            return cpu, state, pdbs

        # Reference repeatability must pass before testing an implementation.
        reference = []
        with torch.no_grad():
            for group in regression_groups:
                reference.append(predict(group,'reference'))
        for variant in dict.fromkeys(['compatibility', *args.variants]):
            if variant == 'compiled':
                compiled = torch.compile(model, mode='default')
            comparisons = []
            with torch.no_grad():
                for group, (ref, states, pdbs) in zip(regression_groups,reference):
                    actual,actual_states,actual_pdbs = predict(group,variant)
                    for i,((name,_,_),a,b) in enumerate(zip(group,ref,actual)):
                        assert np.array_equal(a['atom37_atom_exists'],b['atom37_atom_exists'])
                        mask = a['atom37_atom_exists'].astype(bool)
                        assert np.isfinite(b['atom37'][mask]).all()
                        comparisons.append({'name':name,
                            'coordinates_exact':bool(np.array_equal(a['atom37'],b['atom37'])),
                            'confidence_exact':bool(np.array_equal(a.get('plddt',[]),b.get('plddt',[]))),
                            'final_states_exact':bool(np.array_equal(states[i],actual_states[i])),
                            'pdb_exact':pdbs[i] == actual_pdbs[i],
                            'max_atom_difference':float(np.abs(a['atom37'][mask]-b['atom37'][mask]).max()),
                            'max_ca_difference':float(np.abs(a['atom37'][:,1]-b['atom37'][:,1]).max())})
                    del actual,actual_states
            exact = all(all(x[k] for k in ('coordinates_exact','confidence_exact','final_states_exact','pdb_exact'))
                        for x in comparisons)
            if variant == 'compatibility' and not exact:
                raise AssertionError('Candidate default/reference copy changed numerical behaviour.')
            rows = []
            # Alternate control/candidate order in the two measured repetitions.
            for repeat in range(args.repeats+1):
                for condition in (['reference',variant] if repeat%2 == 0 else [variant,'reference']):
                    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    with Memory() as memory, torch.no_grad():
                        started = time.perf_counter()
                        for group in throughput_groups:
                            result = predict(group,condition,format_pdb=True)
                            del result
                        torch.cuda.synchronize()
                        elapsed = time.perf_counter()-started
                    rows.append({'condition':condition,'repeat':repeat,'warmup':repeat==0,'seconds':elapsed,
                                 'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
                                 'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),**memory.result()})
            medians = {condition:statistics.median(x['seconds'] for x in rows
                       if x['condition']==condition and not x['warmup']) for condition in ['reference',variant]}
            record={'model':model_name,'variant':variant,'exact':exact,'comparisons':comparisons,
                    'count':args.count,'batch_size':args.batch_size,'measurements':rows,
                    'medians_seconds':medians,'reference_over_candidate':medians['reference']/medians[variant],
                    'acceptance':'eligible_for_full_job_check' if exact and medians[variant]<medians['reference'] else
                                 ('requires_user_discussion' if not exact else 'no_measured_speed_improvement')}
            records.append(record)
            (args.output/'component_comparison_251026.json').write_text(json.dumps(records,indent=2)+'\n')
            print(json.dumps({k:v for k,v in record.items() if k not in ('comparisons','measurements')}),flush=True)
        del reference,model,original,lookup,compiled
        gc.collect();torch.cuda.empty_cache()
    manifest={'torch':torch.__version__,'cuda':torch.version.cuda,
              'float32_matmul_precision':torch.get_float32_matmul_precision(),
              'allow_tf32':torch.backends.cuda.matmul.allow_tf32,
              'implementation':args.implementation,
              'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                               for folder in ('reference','candidate','accepted','tools') for p in (ROOT/folder).glob('*.py')},
              'scope':'FP32 matched model weights; engineering fixtures, not biological accuracy'}
    (args.output/'source_manifest_251026.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()
