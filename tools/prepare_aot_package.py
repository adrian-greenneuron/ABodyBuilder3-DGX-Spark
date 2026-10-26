"""Export the pinned inference model and prepare a loadable AOTI package.

Run preparation separately from fresh-process deployment benchmarks. Failures
remain in their own output directory. No fallback changes the model or precision.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch

from aot_model import ModelExportWrapper, symbolic_shape_identities

ROOT = Path(__file__).resolve().parents[1]


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', choices=['base','plddt'], required=True)
    parser.add_argument('--csv', type=Path, default=Path('/frozen/antibodies_1000.csv'))
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--dynamic', action='store_true')
    parser.add_argument('--min-length', type=int, default=200)
    parser.add_argument('--max-length', type=int, default=272)
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, '/opt/abodybuilder3/scripts')
    from batch_pipeline import BatchPolicy, plan_batches
    from abodybuilder3.inference_features import prepare_compact_batch, PairProjection
    cli = import_file('aot_preparation_cli', '/frozen/predict_main.py')
    cli.VERBOSE = False
    candidate = import_file('aot_structure_candidate', ROOT/'candidate/structure_module.py')
    started = time.perf_counter()
    model = cli.load_model(args.model, torch.device('cuda'))
    model.__class__ = candidate.StructureModule
    for layer in model.ipa_layers:
        layer.__class__ = candidate.InvariantPointAttention
    model.eval()
    rows = list(csv.DictReader(args.csv.open()))
    batches = plan_batches([(r['name'],r['heavy'],r['light']) for r in rows],
                           BatchPolicy(args.batch_size))
    group = next(iter(batches))
    cpu, _ = prepare_compact_batch([(h,l) for _,h,l in group])
    batch = {k:v.to('cuda') for k,v in cpu.items()}
    wrapper = ModelExportWrapper(model).eval()
    with torch.no_grad():
        batch['pair'] = PairProjection(model.linear_in_edge)(batch.pop('pair_code'))
        values = (batch['single'],batch['pair'],batch['aatype'],batch['seq_mask'])
        expected = wrapper(*values)  # Initialise the existing residue buffers.
        torch.cuda.synchronize()
        torch.save(tuple(v.cpu() for v in values), args.output/'example_inputs.pt')
        torch.save({k:v.cpu() for k,v in expected.items()}, args.output/'eager_output.pt')
        example_ready = time.perf_counter()
        dynamic_shapes = None
        if args.dynamic:
            length = torch.export.Dim('length', min=args.min_length, max=args.max_length)
            dims = {1:length}
            if len(group)>1:
                size = torch.export.Dim('batch', min=2, max=args.batch_size)
                dims = {0:size, **dims}
            dynamic_shapes = (dict(dims), {**dims,2:length}, dict(dims), dict(dims))
        print(json.dumps({'event':'export_start','batch_size':len(group),
                          'length':values[0].shape[1], 'dynamic':args.dynamic}),flush=True)
        with symbolic_shape_identities():
            uncached = wrapper(*values)
            assert expected.keys() == uncached.keys()
            assert all(torch.equal(expected[k],uncached[k]) for k in expected), 'Identity-cache bypass changed eager outputs.'
            ep = torch.export.export(wrapper, values, dynamic_shapes=dynamic_shapes, strict=False)
        exported = time.perf_counter()
        torch.export.save(ep, args.output/'exported_model.pt2')
        record = {'model':args.model,'precision':'strict_fp32','attention':'original_broadcast',
                  'torch':torch.__version__,'cuda':torch.version.cuda,
                  'batch_size':len(group),'length':values[0].shape[1],
                  'dynamic':args.dynamic,'min_length':args.min_length,'max_length':args.max_length,
                  'example_setup_seconds':example_ready-started,'export_seconds':exported-example_ready,
                  'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in [Path(__file__),ROOT/'tools/aot_model.py',ROOT/'candidate/structure_module.py']},
                  'export_graph_nodes':len(list(ep.graph.nodes)),
                  'identity_cache_bypass_eager_exact':True}
        (args.output/'preparation_261026.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'event':'export_complete',**record}),flush=True)
        if args.export_only:
            return
        compiled_started = time.perf_counter()
        path = torch._inductor.aoti_compile_and_package(ep, package_path=str(args.output/'model.pt2'))
        record['package_compile_seconds'] = time.perf_counter()-compiled_started
        record['package_bytes'] = Path(path).stat().st_size
        record['package_sha256'] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        (args.output/'preparation_261026.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({'event':'package_complete',**record}),flush=True)


if __name__ == '__main__':
    main()
