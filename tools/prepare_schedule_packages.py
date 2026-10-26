"""Isolated AOT shape-bank preparation; installed model/source remain unchanged."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import torch
import torch.nn.functional as F
from abodybuilder3.aot_runtime import ModelExportWrapper, configure_precision, sha256, symbolic_shape_identities
from abodybuilder3.inference_features import PairProjection, prepare_compact_batch


def pad_batch(batch, length):
    n = batch['aatype'].shape[1]
    if length < n:
        raise ValueError('Cannot truncate an export example.')
    result = {}
    for key, value in batch.items():
        if key == 'pair_code':
            result[key] = F.pad(value, (0, length-n, 0, length-n), value=387)
        elif key == 'single':
            result[key] = F.pad(value, (0, 0, 0, length-n))
        else:
            result[key] = F.pad(value, (0, length-n))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-batch', type=int, default=128)
    p.add_argument('--fixed-length', type=int)
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    configure_precision('strict'); torch.set_num_threads(1); torch.manual_seed(42)
    sys.path.insert(0, '/opt/abodybuilder3/scripts')
    spec = importlib.util.spec_from_file_location('schedule_build_cli', '/opt/abodybuilder3/scripts/predict.py')
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    old = Path('/opt/abodybuilder3/aot_packages')
    manifest = copy.deepcopy(json.loads((old / 'manifest.json').read_text()))
    fixture = json.loads(Path('/opt/abodybuilder3/tests/benchmark_antibodies.json').read_text())[0]
    for model_name in ('base', 'plddt'):
        model = cli.load_model(model_name, torch.device('cuda'))
        wrapper = ModelExportWrapper(model).eval()
        single = manifest['models'][model_name]['domains']['single']
        target = args.output / single['path']; target.parent.mkdir(parents=True)
        shutil.copy2(old / single['path'], target)
        batch, _ = prepare_compact_batch([(fixture['heavy'], fixture['light'])] * 32)
        if args.fixed_length:
            batch = pad_batch(batch, args.fixed_length)
        batch = {key: value.to('cuda') for key, value in batch.items()}
        with torch.no_grad():
            pair = PairProjection(model.linear_in_edge)(batch['pair_code'])
            values = (batch['single'], pair, batch['aatype'], batch['seq_mask'])
            expected = wrapper(*values)
            dims = {0: torch.export.Dim('batch', min=2, max=args.max_batch)}
            if args.fixed_length is None:
                dims[1] = torch.export.Dim('length', min=200, max=272)
            dynamic = (dict(dims), {**dims, **({2: dims[1]} if 1 in dims else {})}, dict(dims), dict(dims))
            print('EXPORT', model_name, args.fixed_length, args.max_batch, flush=True)
            start = time.perf_counter()
            with symbolic_shape_identities():
                exported = torch.export.export(wrapper, values, dynamic_shapes=dynamic, strict=False)
            target = args.output / model_name / 'multi/model.pt2'; target.parent.mkdir(parents=True)
            torch._inductor.aoti_compile_and_package(exported, package_path=str(target))
            from torch._inductor import codecache
            execute = torch._inductor.aoti_load_package(str(target)); actual = execute(*values)
            difference = float((actual['positions'] - expected['positions']).abs().max())
            if not torch.isfinite(actual['positions']).all() or difference > .01:
                raise ValueError(f'Package smoke failed: {difference}')
            manifest['models'][model_name]['domains']['multi'] = {
                'path': str(target.relative_to(args.output)), 'sha256': sha256(target),
                'min_batch': 2, 'max_batch': args.max_batch,
                'min_length': args.fixed_length or 200, 'max_length': args.fixed_length or 272,
                'build_seconds': time.perf_counter() - start, 'example_max_position_difference': difference}
            print('PREPARED', model_name, difference, time.perf_counter() - start, flush=True)
            del actual, expected, execute, exported, pair, values, batch
        del model, wrapper
        torch.cuda.empty_cache()
    manifest['experimental_shape_profile'] = {'max_batch': args.max_batch, 'fixed_length': args.fixed_length}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
