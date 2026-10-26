"""Publish compact deployment evidence and plots; no coordinates or weights."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    root = args.study
    repo = Path(__file__).resolve().parents[1]
    assets, data = repo / 'docs/assets', repo / 'docs/data'
    assets.mkdir(exist_ok=True); data.mkdir(exist_ok=True)
    records = json.loads((root / 'performance_measurements_261026.json').read_text())
    accuracy = json.loads((root / 'paired_accuracy_261026.json').read_text())
    colours = {'eager': '#687888', 'aot': '#16729e', 'base': '#16729e', 'plddt': '#a2508c'}
    plt.rcParams.update({'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})

    def save(fig, stem):
        for suffix in ('png', 'svg'):
            path = assets / f'{stem}_261026.{suffix}'
            fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white')
            if suffix == 'svg':
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        plt.close(fig)

    groups = [(count, model) for count in (100, 1000) for model in ('base', 'plddt')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    for axis, metric, title in zip(axes,
            ('container_seconds', 'peak_cuda_allocated_bytes', 'sampled_process_tree_rss_peak_bytes'),
            ('Complete job (seconds)', 'Peak CUDA allocated (GiB)', 'Peak process-tree RSS (GiB)')):
        for index, (count, model) in enumerate(groups):
            for mode, offset in (('eager', -.18), ('aot', .18)):
                rows = [row for row in records if (row['count'], row['model'], row['execution']) == (count, model, mode)]
                values = np.array([row[metric] if metric == 'container_seconds' else row['metrics'][metric] / 2**30 for row in rows])
                value = statistics.median(values) if metric == 'container_seconds' else max(values)
                axis.barh(index + offset, value, .33, color=colours[mode], label=mode if index == 0 else None)
                axis.plot(values, [index + offset] * len(values), '|', color='black', markersize=8)
                axis.text(value + .02 * max(1, axis.get_xlim()[1]), index + offset,
                          f'{value:.2f}', va='center', fontsize=10)
        axis.set_yticks(range(4), [f'{count:,} · {model}' for count, model in groups])
        axis.invert_yaxis(); axis.set_xlabel(title); axis.grid(axis='x', alpha=.18)
        axis.set_xlim(0, axis.get_xlim()[1] * 1.16)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, ['Strict eager + Python writer', 'Strict AOT + native writer'],
               loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(.5, 1.02))
    fig.suptitle('Installed CLI: startup, prediction and every PDB write', y=1.10, fontweight='bold')
    fig.text(.5, -.025, 'Two fresh runs per condition. Time bars: median; memory bars: higher peak. Black marks: individual runs.\n'
             'RSS and CUDA allocation overlap on unified memory; do not add them.', ha='center', fontsize=11)
    fig.tight_layout(); save(fig, 'abodybuilder3_dev_aot_resources')

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for model in ('base', 'plddt'):
        rows = [row for row in accuracy['rows'] if row['model'] == model and row['replicate'] == 1]
        delta = np.array([row['aot_H3_RMSD_to_experiment_A'] - row['reference_H3_RMSD_to_experiment_A'] for row in rows])
        axes[0].scatter([row['h3_length'] + (-.09 if model == 'base' else .09) for row in rows], delta * 1e6,
                        s=25, alpha=.6, color=colours[model], label=model)
        axes[1].hist([row['max_atom_displacement_A'] for row in rows], bins=np.linspace(0, .0004, 22),
                     histtype='step', linewidth=2, color=colours[model], label=model)
    axes[0].axhline(0, color='black', linewidth=1)
    axes[0].set(xlabel='CDR-H3 residues', ylabel='Change in experimental H3 RMSD (10⁻⁶ Å)',
                title='Paired accuracy changes; positive is worse')
    axes[1].set(xlabel='Largest atom displacement per antibody (Å)', ylabel='Antibodies',
                title='Local numerical differences')
    axes[1].ticklabel_format(axis='x', style='plain'); axes[1].tick_params(axis='x', labelrotation=30)
    for ax in axes: ax.grid(alpha=.18); ax.legend(frameon=False)
    fig.suptitle('Separate validation panel: 150 antibodies from 149 PDB entries', fontweight='bold')
    fig.text(.5, -.03, 'One point per antibody/checkpoint. Strict FP32 in both implementations.\n'
             'This tests implementation agreement; the panel is the authors’ validation split.', ha='center', fontsize=11)
    fig.tight_layout(); save(fig, 'abodybuilder3_dev_aot_accuracy')

    with (root / 'sustained/memory_samples_261026.csv').open() as stream:
        samples = list(csv.DictReader(stream))
    times = np.array([float(row['seconds_since_script_start']) for row in samples])
    rss = np.array([int(row['process_tree_rss_bytes']) / 2**30 for row in samples])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(times, rss, color=colours['aot'], linewidth=1.7)
    ax.set(xlabel='Elapsed seconds since measurement script start', ylabel='Process-tree RSS (GiB)',
           title='20,000 outputs: sampled memory through the complete job')
    ax.grid(alpha=.2)
    fig.text(.5, -.015, '150 validation sequences repeated; includes measurement fingerprints and retained CLI metadata.\n'
             'CPU RSS only; CUDA allocation and container accounting are reported separately.', ha='center', fontsize=10)
    fig.tight_layout(); save(fig, 'abodybuilder3_dev_aot_sustained')

    shutil.copy2(root / 'paired_accuracy_261026.csv', data / 'abodybuilder3_dev_aot_paired_accuracy_261026.csv')
    compact = {'performance': records, 'accuracy_summaries': accuracy['summaries'],
               'shape_qualification': json.loads((root / 'shape_qualification_261026.json').read_text()),
               'writer_qualification': json.loads((root / 'native_writer_qualification_261026.json').read_text()),
               'interruption': json.loads((root / 'interruption_check_261026.json').read_text()),
               'deployment_checks': json.loads((root / 'deployment_checks_261026.json').read_text()),
               'sustained': json.loads((root / 'sustained/metrics_261026.json').read_text())}
    (data / 'abodybuilder3_dev_aot_evidence_261026.json').write_text(json.dumps(compact, indent=2) + '\n')
    archive = data / 'abodybuilder3_dev_aot_evidence_261026.zip'
    members = {}
    for name in ('protocol_261026.json', 'paired_accuracy_261026.json', 'paired_accuracy_261026.csv',
                 'shape_qualification_261026.json', 'native_writer_qualification_261026.json',
                 'interruption_check_261026.json', 'deployment_checks_261026.json',
                 'measurement_sources_261026.json', 'independent_validation/selection_261026.json',
                 'independent_validation/validation_261026.json', 'sustained/metrics_261026.json',
                 'sustained/batching.json', 'sustained/memory_samples_261026.csv'):
        members[name] = (root / name).read_bytes()
    # Package identities contain checksums/configuration, never the executables.
    members['aot_package_manifest.json'] = (root / 'build_packages/packages/manifest.json').read_bytes()
    checksums = ''.join(f'{hashlib.sha256(value).hexdigest()}  {name}\n' for name, value in sorted(members.items()))
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, value in members.items(): bundle.writestr(name, value)
        bundle.writestr('SHA256SUMS', checksums)
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None: raise ValueError('Evidence archive CRC check failed.')
    (data / 'abodybuilder3_dev_aot_checksums_261026.txt').write_text(
        f'{hashlib.sha256(archive.read_bytes()).hexdigest()}  {archive.name}\n')
    print(json.dumps({'completed': True, 'plots': 3, 'archive_bytes': archive.stat().st_size}), flush=True)


if __name__ == '__main__':
    main()
