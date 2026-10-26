"""GPU qualification of packaged domains, partial batches and mixed padding.

Length edits are engineering fixtures, not experimental accuracy observations.
"""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
from abodybuilder3.aot_runtime import AOTForward, configure_precision
from abodybuilder3.batched_inference import batch_add_atom37_to_output
from abodybuilder3.inference_features import PairProjection, prepare_compact_batch


def main():
    out = Path(sys.argv[1])
    configure_precision('strict')
    torch.set_num_threads(1)
    torch.manual_seed(42)
    spec = importlib.util.spec_from_file_location('shape_cli', '/opt/abodybuilder3/scripts/predict.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    fixture = json.loads(Path('/opt/abodybuilder3/tests/benchmark_antibodies.json').read_text())[0]
    rows = []
    for model_name in ('base', 'plddt'):
        model = cli.load_model(model_name, torch.device('cuda'))
        adapter = AOTForward('/opt/abodybuilder3/aot_packages', model_name,
                             model._deployment_checkpoint_sha256, model.forward)
        projector = PairProjection(model.linear_in_edge)
        cases = [(size, length, False) for size in (1, 2, 3, 16, 31, 32)
                 for length in (199, 200, 201, 225, 271, 272, 273)]
        cases += [(33, 225, False), (32, 242, True), (3, 272, True)]
        for size, length, mixed in cases:
            pairs = []
            for i in range(size):
                n = length - (i % 9) if mixed else length
                heavy = (fixture['heavy'] + 'G' * 100)[:n - len(fixture['light'])]
                pairs.append((heavy, fixture['light']))
            cpu, inputs = prepare_compact_batch(pairs)
            batch = {key: value.to('cuda') for key, value in cpu.items()}
            options = dict(_inference_pair_projected=True, _inference_final_only=True, _inference_pair_buffer=True)
            with torch.no_grad():
                features = {'single': batch['single'], 'pair': projector(batch['pair_code'])}
                reference = model(features, batch['aatype'], batch['seq_mask'], **options)
                actual = adapter(features, batch['aatype'], batch['seq_mask'], **options)
                a = batch_add_atom37_to_output(reference, batch['aatype'], [len(item['aatype']) for item in inputs])
                b = batch_add_atom37_to_output(actual, batch['aatype'], [len(item['aatype']) for item in inputs])
            ca_max = atom_max = 0.
            for first, second in zip(a, b):
                if not np.array_equal(first['atom37_atom_exists'], second['atom37_atom_exists']):
                    raise ValueError('Atom masks changed.')
                valid = first['atom37_atom_exists'].astype(bool)
                ca = valid[:, 1]
                delta = np.linalg.norm(first['atom37'] - second['atom37'], axis=-1)
                atom_max = max(atom_max, float(delta[valid].max()))
                ca_max = max(ca_max, float(delta[ca, 1].max()))
            record = {'model': model_name, 'size': size, 'padded_length': length,
                      'mixed_lengths': mixed, 'max_valid_atom_displacement_A': atom_max,
                      'max_CA_displacement_A': ca_max,
                      'expected_fallback': length < 200 or length > 272 or size > 32}
            rows.append(record)
            if atom_max > .01 or ca_max > .001:
                out.write_text(json.dumps({'completed': False, 'rows': rows}, indent=2) + '\n')
                raise ValueError(f'Shape numerical gate failed: {record}')
            print(json.dumps(record), flush=True)
            del a, b, actual, reference, features, batch, cpu
        expected_fallbacks = sum(row['expected_fallback'] for row in rows if row['model'] == model_name)
        if adapter.stats['eager_shape_fallback_batches'] != expected_fallbacks:
            raise ValueError('Shape fallback accounting differs.')
        del model, adapter, projector
        torch.cuda.empty_cache()
    out.write_text(json.dumps({'completed': True, 'cases': len(rows), 'rows': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
