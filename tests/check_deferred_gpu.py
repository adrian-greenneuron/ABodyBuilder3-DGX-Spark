"""Paired raw/PDB equality and two in-flight output batches, including fallback."""
import importlib.util
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
sys.path.insert(0, '/opt/abodybuilder3/scripts')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'candidate'))
import deferred_inference as candidate
from abodybuilder3 import batched_inference as original
from abodybuilder3.aot_runtime import AOTForward, configure_precision


def main():
    dest = Path(sys.argv[1])
    configure_precision('strict'); torch.set_num_threads(1); torch.manual_seed(42)
    spec = importlib.util.spec_from_file_location('deferred_cli', '/opt/abodybuilder3/scripts/predict.py')
    cli = importlib.util.module_from_spec(spec); spec.loader.exec_module(cli)
    fixture = json.loads(Path('/opt/abodybuilder3/tests/benchmark_antibodies.json').read_text())[0]
    rows = []
    for name in ('base', 'plddt'):
        model = cli.load_model(name, torch.device('cuda'))
        model.forward = AOTForward('/opt/abodybuilder3/aot_packages', name,
                                  model._deployment_checkpoint_sha256, model.forward)
        for size, length, mixed in ((1, 199, False), (1, 200, False), (3, 200, False),
                                    (2, 272, False), (3, 273, False), (32, 230, False), (3, 242, True)):
            pairs = [((fixture['heavy'] + 'G' * 100)[:length - len(fixture['light']) - (i % 4 if mixed else 0)],
                      fixture['light']) for i in range(size)]
            prepared = original.prepare_cpu_batch(pairs, compact=True)
            reference, inputs = original.infer_prepared_batch(model, prepared, 'cuda')
            # Enqueue two batches without waiting for CPU arrays. The first
            # buffers must remain correct while the second model pass is queued.
            first, _ = candidate.infer_prepared_batch(model, candidate.pin_prepared(prepared), 'cuda')
            second, _ = candidate.infer_prepared_batch(model, candidate.pin_prepared(prepared), 'cuda')
            def check(predictions):
                for expected, actual, inp in zip(reference, predictions, inputs):
                    if set(actual) != set(expected) or not all(np.array_equal(actual[key], expected[key]) for key in expected):
                        raise ValueError('Deferred raw prediction differs.')
                    if cli.generate_pdb(actual, inp) != cli.generate_pdb(expected, inp):
                        raise ValueError('Deferred PDB bytes differ.')
                return True
            with ThreadPoolExecutor(max_workers=2) as pool:
                if not all(pool.map(check, (first, second))):
                    raise ValueError('Concurrent output check failed.')
            rows.append({'model': name, 'size': size, 'length': length, 'mixed': mixed,
                         'raw_arrays_exact': True, 'PDB_bytes_exact': True, 'in_flight_batches': 2})
            print(json.dumps(rows[-1]), flush=True)
        del model
        torch.cuda.empty_cache()
    dest.write_text(json.dumps({'completed': True, 'cases': rows}, indent=2) + '\n')


if __name__ == '__main__':
    main()
