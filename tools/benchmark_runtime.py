"""Matched FP32 inference measurements; run inside one deployment container."""
import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import resource
import threading
import time

import numpy as np
import psutil
import torch
import ml_collections

torch.serialization.add_safe_globals([ml_collections.ConfigDict])


def load_cli():
    spec = importlib.util.spec_from_file_location('abb3_frozen_cli', '/frozen/predict_main.py')
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    cli.VERBOSE = False
    return cli


def cuda_sync():
    torch.cuda.synchronize()


class Memory:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.process = psutil.Process()

    def sample(self):
        processes = [self.process, *self.process.children(recursive=True)]
        rss = 0
        for p in processes:
            try:
                rss += p.memory_info().rss
            except psutil.Error:
                pass
        self.samples.append((time.perf_counter(), rss))

    def __enter__(self):
        self.sample()
        def loop():
            while not self.stop.wait(.05):
                self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.sample()

    def result(self):
        cgroup = Path('/sys/fs/cgroup/memory.peak')
        return {
            'sampled_process_tree_rss_peak_bytes': max(x[1] for x in self.samples),
            'main_process_lifetime_rss_peak_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            'container_lifetime_memory_peak_bytes': int(cgroup.read_text()) if cgroup.exists() else None,
        }


def save_prediction(folder, name, output, inp, pdb):
    folder.mkdir(parents=True, exist_ok=True)
    positions = output['atom37']
    masks = output['atom37_atom_exists']
    if torch.is_tensor(positions):
        positions = positions.detach().cpu().numpy()
    if torch.is_tensor(masks):
        masks = masks.detach().cpu().numpy()
    confidence = output.get('plddt')
    if confidence is None:
        confidence = np.array([])
    elif torch.is_tensor(confidence):
        confidence = confidence.detach().cpu().numpy()
    if not np.isfinite(positions[masks.astype(bool)]).all():
        raise ValueError(f'Nonfinite coordinates for {name}')
    np.savez(folder / f'{name}.npz', positions=positions, masks=masks,
             aatype=inp['aatype'].detach().cpu().numpy(),
             is_heavy=inp['is_heavy'].detach().cpu().numpy(), confidence=confidence)
    (folder / f'{name}.pdb').write_text(pdb)


def compact_prediction(name, output, inp, pdb):
    # Do not retain the full GPU model output or pair features between antibodies.
    def cpu(value):
        return value.detach().cpu() if torch.is_tensor(value) else value
    out = {k: cpu(output[k]) for k in ['atom37', 'atom37_atom_exists', 'plddt'] if k in output}
    inputs = {k: cpu(inp[k]) for k in ['aatype', 'is_heavy']}
    return name, out, inputs, pdb


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['base', 'plddt', 'language'], required=True)
    p.add_argument('--mode', choices=['sequential', 'dev_batch', 'dev_language_adapter'], default='sequential')
    p.add_argument('--count', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    fixtures = json.loads(Path('/frozen/antibodies.json').read_text())
    data = [fixtures[i % len(fixtures)] for i in range(args.count)]
    if args.model == 'language' and args.mode == 'dev_batch':
        raise ValueError('Existing dev helper does not implement language embeddings.')
    if args.mode != 'sequential':
        from abodybuilder3.batched_inference import (
            strings_to_batch_input, batch_add_atom37_to_output, batch_output_to_pdbs,
        )
    cli = load_cli()
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device('cuda:0')
    cuda_sync()
    torch.cuda.reset_peak_memory_stats()
    with Memory() as load_mem:
        start = time.perf_counter()
        model = cli.load_model(args.model, device)
        cuda_sync()
        load_seconds = time.perf_counter() - start
        embedder = None
        embedding_load_seconds = 0.
        if args.model == 'language':
            from abodybuilder3.language.model import ProtT5
            start = time.perf_counter()
            embedder = ProtT5(weights_dir=Path('/prott5'), device_map='cuda:0')
            cuda_sync()
            embedding_load_seconds = time.perf_counter() - start
    loading = {'model_load_seconds': load_seconds, 'prott5_load_seconds': embedding_load_seconds,
               'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(), **load_mem.result()}
    records = []
    for rep in range(3):
        torch.manual_seed(42)
        gc.collect()
        torch.cuda.empty_cache()
        cuda_sync()
        torch.cuda.reset_peak_memory_stats()
        components = {'prepare_seconds': 0., 'inference_seconds': 0., 'pdb_seconds': 0.}
        results = []
        with Memory() as mem:
            start = time.perf_counter()
            for first in range(0, len(data), 1 if args.mode == 'sequential' else args.batch_size):
                group = data[first:first + (1 if args.mode == 'sequential' else args.batch_size)]
                mark = time.perf_counter()
                if args.mode == 'sequential':
                    item = group[0]
                    if embedder is None:
                        inp = cli.prepare_input(item['heavy'], item['light'])
                    else:
                        inp = cli.prepare_input_with_embeddings(item['heavy'], item['light'], embedder, device)
                    cuda_sync()
                    components['prepare_seconds'] += time.perf_counter() - mark
                    mark = time.perf_counter()
                    output = cli.run_inference(model, inp, device)
                    cuda_sync()
                    components['inference_seconds'] += time.perf_counter() - mark
                    mark = time.perf_counter()
                    pdb = cli.generate_pdb(output, inp)
                    components['pdb_seconds'] += time.perf_counter() - mark
                    # Preserve the five distinct predictions, outside the timed window.
                    if first < len(fixtures):
                        results.append(compact_prediction(item['name'], output, inp, pdb))
                    del output, inp
                else:
                    pairs = [(x['heavy'], x['light']) for x in group]
                    batch, inputs = strings_to_batch_input(pairs, device)
                    if args.mode == 'dev_language_adapter':
                        from torch.nn.utils.rnn import pad_sequence
                        prepared = [cli.prepare_input_with_embeddings(h, l, embedder, device) for h, l in pairs]
                        batch['single'] = pad_sequence([x['single'][0] for x in prepared], batch_first=True)
                        del prepared
                    cuda_sync()
                    components['prepare_seconds'] += time.perf_counter() - mark
                    mark = time.perf_counter()
                    with torch.no_grad():
                        output = model({'single': batch['single'], 'pair': batch['pair']},
                                       batch['aatype'], batch['seq_mask'])
                        outputs = batch_add_atom37_to_output(output, batch['aatype'], [len(x['aatype']) for x in inputs])
                    cuda_sync()
                    components['inference_seconds'] += time.perf_counter() - mark
                    mark = time.perf_counter()
                    pdbs = batch_output_to_pdbs(outputs, inputs)
                    components['pdb_seconds'] += time.perf_counter() - mark
                    for i, (item, out, inp, pdb) in enumerate(zip(group, outputs, inputs, pdbs)):
                        if first + i < len(fixtures):
                            results.append(compact_prediction(item['name'], out, inp, pdb))
                    del output, outputs, batch, out, inp, inputs
            cuda_sync()
            seconds = time.perf_counter() - start
        row = {'replicate': rep, 'warmup': rep == 0, 'model': args.model, 'mode': args.mode,
               'count': args.count, 'batch_size': 1 if args.mode == 'sequential' else args.batch_size,
               'processing_seconds': seconds, 'antibodies_per_minute': 60 * args.count / seconds,
               'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
               'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
               **components, **mem.result()}
        if len(results) != min(len(fixtures), args.count):
            raise ValueError('Missing predictions')
        for name, output, inp, pdb in results:
            save_prediction(args.output / f'run{rep}', name, output, inp, pdb)
        del results, output, inp
        records.append(row)
        (args.output / 'measurements.json').write_text(json.dumps({
            'loading': loading, 'measurements': records, 'completed': rep == 2,
            'torch': torch.__version__, 'allocator': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
        }, indent=2) + '\n')
        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
