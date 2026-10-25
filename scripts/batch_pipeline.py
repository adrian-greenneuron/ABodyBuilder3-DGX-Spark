"""Bounded scheduling for inference; no model or GPU imports at module import."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice


@dataclass(frozen=True)
class BatchPolicy:
    max_size: int = 32
    pair_budget: int = 2_000_000
    window: int = 512
    length_bin: int = 8

    def __post_init__(self):
        if min(self.max_size, self.pair_budget, self.window, self.length_bin) < 1:
            raise ValueError('Batch limits must be positive.')


def plan_batches(items, policy):
    """Sort bounded windows by length; preserve item identities, not execution order."""
    source = iter(items)
    while True:
        window = list(islice(source, policy.window))
        if not window:
            return
        window.sort(key=lambda item: len(item[1]) + len(item[2]))
        batch = []
        longest = 0
        for item in window:
            length = len(item[1]) + len(item[2])
            if length < 1:
                raise ValueError('An antibody cannot have zero residues.')
            next_longest = max(longest, length)
            same_bin = not batch or (length // policy.length_bin == longest // policy.length_bin)
            if batch and (not same_bin or len(batch) >= policy.max_size or
                          (len(batch) + 1) * next_longest**2 > policy.pair_budget):
                yield batch
                batch, longest = [], 0
            batch.append(item)
            longest = max(longest, length)
        if batch:
            yield batch


def run_pipeline(batches, prepare, infer, write, *, prefetch=2, writers=2):
    """One caller owns the GPU; bounded CPU preparation and output overlap it.

    Exceptions propagate. OOM recovery belongs to infer(), which can distinguish
    it from unrelated errors. No executor.map submits an unbounded iterable.
    """
    if min(prefetch, writers) < 1:
        raise ValueError('Pipeline queue limits must be positive.')
    source = iter(batches)
    pending = deque()
    outputs = deque()
    with ThreadPoolExecutor(max_workers=prefetch) as prep_pool, \
            ThreadPoolExecutor(max_workers=writers) as write_pool:
        def submit_next():
            try:
                items = next(source)
            except StopIteration:
                return False
            pending.append((items, prep_pool.submit(prepare, items)))
            return True

        for _ in range(prefetch):
            if not submit_next():
                break
        while pending:
            items, future = pending.popleft()
            prepared = future.result()
            # Queue stays bounded even while output backpressure stalls inference.
            submit_next()
            result = infer(items, prepared)
            del prepared, future
            outputs.append(write_pool.submit(write, result))
            if len(outputs) >= writers:
                outputs.popleft().result()
            del result
        while outputs:
            outputs.popleft().result()
