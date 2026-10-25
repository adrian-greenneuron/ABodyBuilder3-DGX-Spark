"""Optional per-process CUDA accounting for the multiprocessing CLI benchmark."""
import atexit
import csv
import json
import os
from pathlib import Path
import threading
import time

if os.environ.get('ABB3_RESOURCE_DIR'):
    import torch
    directory = Path(os.environ['ABB3_RESOURCE_DIR'])
    directory.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    pid = os.getpid()
    samples = []

    def sample():
        if torch.cuda.is_initialized():
            samples.append([time.monotonic(), torch.cuda.memory_allocated(), torch.cuda.memory_reserved()])

    def monitor():
        while not stop.wait(.1):
            sample()

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()

    def finish():
        stop.set()
        thread.join()
        sample()
        if torch.cuda.is_initialized():
            result = {'pid': pid, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                      'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()}
            (directory / f'{pid}.json').write_text(json.dumps(result) + '\n')
            # A zero endpoint lets the host reconstruct a concurrent sum.
            samples.append([time.monotonic(), 0, 0])
            with (directory / f'{pid}.csv').open('w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['monotonic_seconds', 'cuda_allocated_bytes', 'cuda_reserved_bytes'])
                writer.writerows(samples)

    atexit.register(finish)
