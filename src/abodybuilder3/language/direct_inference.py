"""Keep the released encoder on its selected GPU between exact batch-1 calls.

Only the Trainer/DataLoader lifecycle is replaced. The original predict_step
performs the encoder call and token masking; the parent preserves chain order
and concatenation. No embedding batching, mixed precision or caching of outputs.
"""
import torch
from torch.utils.data import DataLoader
from abodybuilder3.language.model import ProtT5 as ReferenceProtT5


class ProtT5(ReferenceProtT5):
    def __post_init__(self):
        super().__post_init__()
        devices = {parameter.device for parameter in self.embedding_module.model.parameters()}
        if len(devices) != 1 or next(iter(devices)).type != 'cuda':
            raise ValueError('This candidate requires the explicitly selected single CUDA device.')
        self.inference_device = next(iter(devices))
        self.embedding_module.to(self.inference_device).eval()

    def _embed_sequences(self, sequences):
        if self.batch_size != 1:
            raise ValueError('This candidate preserves the original batch-size-1 encoder calls.')
        loader = DataLoader(sequences, collate_fn=self.collate_fn, batch_size=1)
        outputs = []
        with torch.inference_mode():
            for index, batch in enumerate(loader):
                tensors = {key: value.to(self.inference_device) for key, value in batch.items()}
                outputs.append(self.embedding_module.predict_step(tensors, index).cpu())
        return outputs
