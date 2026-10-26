"""Reuse immutable model/package objects for consecutive local CLI requests.

This is an isolated process-local experiment, not a network service. Request
settings, statistics and output directories remain separate. Package/checkpoint
files in the pinned image are immutable for the lifetime of the worker.
"""
import copy
from pathlib import Path


class ResidentSession:
    def __init__(self, cli, runtime):
        self.cli, self.runtime = cli, runtime
        self.original_load = cli.load_model
        self.original_aot = runtime.AOTForward
        self.models, self.packages = {}, {}
        self.hits = {'model_loads': 0, 'model_reuses': 0, 'package_loads': 0, 'package_reuses': 0}

    def load_model(self, model_type, device, checkpoint=None):
        if checkpoint is not None or model_type not in ('base', 'plddt'):
            raise ValueError('Resident experiment supports only immutable released base/plddt checkpoints.')
        key = (model_type, str(device))
        if key not in self.models:
            model = self.original_load(model_type, device)
            self.models[key] = (model, model.forward)
            self.hits['model_loads'] += 1
        else:
            self.hits['model_reuses'] += 1
        model, original_forward = self.models[key]
        # The CLI attaches a new request-local adapter. Never stack its fallback
        # on a previous request's adapter or retain earlier request statistics.
        model.forward = original_forward
        return model

    def load_packages(self, root, name, checkpoint, fallback):
        key = (str(Path(root).resolve()), name, checkpoint)
        reused = key in self.packages
        if key not in self.packages:
            template = self.original_aot(root, name, checkpoint, fallback)
            self.packages[key] = template
            self.hits['package_loads'] += 1
        else:
            self.hits['package_reuses'] += 1
        template = self.packages[key]
        adapter = self.original_aot.__new__(self.original_aot)
        adapter.models, adapter.bounds = template.models, template.bounds
        adapter.fallback = fallback
        adapter.stats = copy.deepcopy(template.stats)
        adapter.stats.update(aot_batches=0, eager_shape_fallback_batches=0, fallback_reasons={})
        adapter.stats['resident_packages_reused'] = reused
        if reused:
            for record in adapter.stats.get('packages', {}).values():
                record['load_seconds'] = 0.
                record['resident_reuse'] = True
        return adapter

    def install(self):
        self.cli.load_model = self.load_model
        self.runtime.AOTForward = self.load_packages

    def restore(self):
        self.cli.load_model = self.original_load
        self.runtime.AOTForward = self.original_aot
