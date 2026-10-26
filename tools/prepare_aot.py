"""Build both released checkpoints' AOT executables in the pinned GPU runtime.

This is an explicit build step, outside timed prediction. It needs no retained
experimental image, compiler cache, biological dataset or network connection.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import torch

from abodybuilder3.aot_runtime import (
    ModelExportWrapper, SCHEMA_VERSION, configure_precision, runtime_identity,
    sha256, source_identity, symbolic_shape_identities,
)
from abodybuilder3.inference_features import PairProjection, prepare_compact_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--min-length', type=int, default=200)
    parser.add_argument('--max-length', type=int, default=272)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configure_precision('strict')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'scripts'))
    spec = importlib.util.spec_from_file_location('package_build_cli', root / 'scripts/predict.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cli.VERBOSE = False
    fixture = json.loads((root / 'tests/benchmark_antibodies.json').read_text())[0]
    manifest = {'schema_version': SCHEMA_VERSION, 'runtime': runtime_identity(),
                'sources': source_identity(), 'models': {}}
    for name in ('base', 'plddt'):
        model = cli.load_model(name, torch.device('cuda'))
        entry = {'checkpoint_sha256': model._deployment_checkpoint_sha256,
                 'precision': 'strict_fp32', 'attention': 'original_broadcast', 'domains': {}}
        wrapper = ModelExportWrapper(model).eval()
        for domain, count in (('single', 1), ('multi', 32)):
            dest = args.output / name / domain
            dest.mkdir(parents=True)
            cpu, _ = prepare_compact_batch([(fixture['heavy'], fixture['light'])] * count)
            batch = {key: value.to('cuda') for key, value in cpu.items()}
            with torch.no_grad():
                pair = PairProjection(model.linear_in_edge)(batch['pair_code'])
                values = (batch['single'], pair, batch['aatype'], batch['seq_mask'])
                expected = wrapper(*values)
                torch.cuda.synchronize()
                dims = {1: torch.export.Dim('length', min=args.min_length, max=args.max_length)}
                if count > 1:
                    dims[0] = torch.export.Dim('batch', min=2, max=count)
                dynamic = (dict(dims), {**dims, 2: dims[1]}, dict(dims), dict(dims))
                print(json.dumps({'event': 'export_start', 'model': name, 'domain': domain}), flush=True)
                started = time.perf_counter()
                with symbolic_shape_identities():
                    actual = wrapper(*values)
                    if not all(torch.equal(expected[key], actual[key]) for key in expected):
                        raise ValueError('Export-only identity-cache bypass changed eager output.')
                    exported = torch.export.export(wrapper, values, dynamic_shapes=dynamic, strict=False)
                package = dest / 'model.pt2'
                torch._inductor.aoti_compile_and_package(exported, package_path=str(package))
                # Check the saved package through the loader, not the export object.
                from torch._inductor import codecache  # Registers the pinned package loader.
                execute = torch._inductor.aoti_load_package(str(package))
                actual = execute(*values)
                torch.cuda.synchronize()
                comparisons = {}
                for key in expected:
                    delta = (actual[key] - expected[key]).abs()
                    comparisons[key] = {'max_abs': float(delta.max()),
                                        'mean_abs': float(delta.mean())}
                    if not torch.isfinite(actual[key]).all():
                        raise ValueError(f'Nonfinite AOT output: {key}')
                # Detailed structure qualification is separate; this catches
                # gross export/package corruption before assembling the image.
                if comparisons['positions']['max_abs'] > .01:
                    raise ValueError(f'AOT position smoke comparison failed: {comparisons}')
                entry['domains'][domain] = {
                    'path': str(package.relative_to(args.output)), 'sha256': sha256(package),
                    'min_batch': 1 if count == 1 else 2, 'max_batch': count,
                    'min_length': args.min_length, 'max_length': args.max_length,
                    'build_seconds': time.perf_counter() - started,
                    'build_example_comparison': comparisons}
                print(json.dumps({'event': 'package_complete', 'model': name, 'domain': domain,
                                  **entry['domains'][domain]}), flush=True)
                del expected, actual, exported, execute, pair, batch, cpu, values
        manifest['models'][name] = entry
        del wrapper, model
        torch.cuda.empty_cache()
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'event': 'all_packages_complete', 'models': list(manifest['models'])}), flush=True)


if __name__ == '__main__':
    main()
