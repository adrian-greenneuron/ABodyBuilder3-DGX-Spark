"""Write compact results and a static figure from an existing comparison study."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in sorted((args.study_root / 'throughput').glob('*/predictions/measurements.json')):
        data = json.loads(path.read_text())
        if not data['completed']:
            raise ValueError(f'Incomplete case: {path}')
        measured = [x for x in data['measurements'] if not x['warmup']]
        condition = path.parent.parent.name.split('_', 1)[0]
        loading = data['loading']
        r = measured[0]
        rows.append({
            'condition': condition, 'model': r['model'], 'mode': r['mode'], 'count': r['count'],
            'batch_size': r['batch_size'], 'repetitions': len(measured),
            'seconds_median': statistics.median(x['processing_seconds'] for x in measured),
            'seconds_min': min(x['processing_seconds'] for x in measured),
            'seconds_max': max(x['processing_seconds'] for x in measured),
            'cuda_allocated_peak_gib': max(loading['peak_cuda_allocated_bytes'], *(x['peak_cuda_allocated_bytes'] for x in measured)) / 2**30,
            'cuda_reserved_peak_gib': max(loading['peak_cuda_reserved_bytes'], *(x['peak_cuda_reserved_bytes'] for x in measured)) / 2**30,
            'process_rss_peak_gib': max(loading['sampled_process_tree_rss_peak_bytes'], *(x['sampled_process_tree_rss_peak_bytes'] for x in measured)) / 2**30,
            'container_accounted_peak_gib': max(x['container_lifetime_memory_peak_bytes'] for x in measured) / 2**30,
            'model_load_seconds': loading['model_load_seconds'],
            'prott5_load_seconds': loading['prott5_load_seconds'],
            'raw_repetitions': measured,
        })
    cli = []
    for path in sorted((args.study_root / 'cli_throughput').glob('*/measurements.json')):
        measured = [x for x in json.loads(path.read_text()) if not x['warmup']]
        r = measured[0]
        cli.append({
            'condition': r['condition'], 'model': r['model'], 'count': r['count'], 'workers': r['workers'],
            'seconds_median': statistics.median(x['complete_process_seconds'] for x in measured),
            'seconds_min': min(x['complete_process_seconds'] for x in measured),
            'seconds_max': max(x['complete_process_seconds'] for x in measured),
            'sampled_concurrent_cuda_allocated_peak_gib': max(x['sampled_concurrent_cuda_allocated_peak_bytes'] for x in measured) / 2**30,
            'sampled_concurrent_cuda_reserved_peak_gib': max(x['sampled_concurrent_cuda_reserved_peak_bytes'] for x in measured) / 2**30,
            'summed_process_rss_peak_gib': max(x['sampled_process_tree_rss_peak_bytes'] for x in measured) / 2**30,
            'container_accounted_peak_gib': max(x['container_lifetime_memory_peak_bytes'] for x in measured) / 2**30,
            'raw_repetitions': measured,
        })
    result = {'processing': rows, 'cli': cli}
    (args.output / 'abodybuilder3_benchmark_measurements_251026.json').write_text(json.dumps(result, indent=2) + '\n')
    for name, data in [('processing', rows), ('cli', cli)]:
        fields = [k for k in data[0] if k != 'raw_repetitions']
        with (args.output / f'abodybuilder3_{name}_251026.csv').open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore', lineterminator='\n')
            writer.writeheader()
            writer.writerows(data)
    main_rows = [x for x in rows if x['count'] == (20 if x['model'] == 'language' else 1000)]
    labels = {'original': 'Original', 'rebuilt': 'Rebuilt', 'modernised': 'Modernised', 'dev': 'Dev batching'}
    lines = ['# ABodyBuilder3 deployment benchmarks — 25 October 2026', '',
             'This historical comparison preceded promotion of the modernised runtime to main. It uses the frozen earlier CLI; see the [current preparation benchmarks](abodybuilder3_preparation_benchmarks_251026.md) for the deployed CLI. All images, containers, checkpoints, inputs and outputs were retained.', '',
             '## Warmed processing', '',
             'One warm-up workload followed by two measured repetitions in the same loaded process. '
             'Times include feature/embedding preparation, inference and PDB string generation; imports, model loading and disk output are excluded. '
             'The workloads cycle five public sequence pairs. Language includes live ProtT5 embedding generation, with no embedding cache.', '',
             '| Model | Inputs per repetition | Condition | Time, median (range), s | CUDA allocated peak, GiB | CUDA reserved peak, GiB | Sampled process RSS peak, GiB | Container lifetime accounted peak, GiB |',
             '|---|---:|---|---:|---:|---:|---:|---:|']
    for model in ['base', 'plddt', 'language']:
        for condition in ['original', 'rebuilt', 'modernised', 'dev']:
            r = next(x for x in main_rows if x['model'] == model and x['condition'] == condition)
            lines.append(f'| {model} | {r["count"]:,} | {labels[condition]} | {r["seconds_median"]:.2f} ({r["seconds_min"]:.2f}–{r["seconds_max"]:.2f}) | {r["cuda_allocated_peak_gib"]:.3f} | {r["cuda_reserved_peak_gib"]:.3f} | {r["process_rss_peak_gib"]:.3f} | {r["container_accounted_peak_gib"]:.3f} |')
    lines += ['', 'CUDA and RSS peaks are the higher measured peak or loading peak. The container column is lifetime cgroup accounting, including warm-up. '
              'RSS is sampled every 50 ms and includes shared pages; container memory is cgroup-accounted memory and is a different measure. '
              'CUDA allocated/reserved memory must not be added to RSS to estimate physical unified-memory use.', '',
              'Base and pLDDT dev results use batch size 32, selected from the 100-input screen (16, 32 and 64). '
              'Language uses a benchmark-only adapter that inserts unchanged ProtT5 embeddings into the dev batch representation, with batch size 3. '
              'That adapter is not an existing CLI feature.', '',
              'Language timings vary strongly between repetitions, especially the modernised result. '
              'These data do not establish a reliable language-model speed ranking or an embedding-speed improvement. '
              'The language adapter was slower in these measurements and is not recommended for deployment.', '',
              '## Complete eight-worker CLI jobs', '',
              'Each measurement is a fresh container, including startup, model loading, prediction and writing all 1,000 PDB files. '
              'One whole-process warm-up precedes two measured fresh-container runs. '
              'These timings have a different scope from the warmed-processing table.', '',
              '| Model | Condition | Complete time, median (range), s | Sampled concurrent CUDA allocated peak, GiB | Summed process RSS peak, GiB | Container lifetime accounted peak, GiB |',
              '|---|---|---:|---:|---:|---:|']
    for model in ['base', 'plddt']:
        for condition in ['original', 'rebuilt', 'modernised']:
            r = next(x for x in cli if x['model'] == model and x['condition'] == condition)
            lines.append(f'| {model} | {labels[condition]} | {r["seconds_median"]:.2f} ({r["seconds_min"]:.2f}–{r["seconds_max"]:.2f}) | {r["sampled_concurrent_cuda_allocated_peak_gib"]:.3f} | {r["summed_process_rss_peak_gib"]:.3f} | {r["container_accounted_peak_gib"]:.3f} |')
    lines += ['', 'Worker CUDA usage is sampled every 100 ms and reconstructed as a concurrent sum. '
              'This can miss short allocation peaks; exact individual worker peaks are also retained in the measurement JSON. '
              'Summed RSS double-counts shared pages. Cgroup memory helps assess aggregate container use; neither column is total server memory.', '',
              '## Prediction agreement', '',
              '- All 88 captured source files, 322 installed Python distributions, three checkpoint hashes, OpenMM revision, Reduce binary and Reduce dictionary match in the original rebuild.',
              '- All 54 source files under `src/` and all three checkpoints match between original and modernised.',
              '- Rebuilt and modernised raw coordinates, atom masks, sequence identities and confidence arrays match the original exactly on the five saved public pairs in both measured repetitions, at every tested stage.',
              '- All 19 larger-stage acceptance checks passed. The largest dev Cα coordinate difference was 0.00578 Å, below the pre-existing dev tolerance of 0.01 Å. Cα local-distance agreement was 1.0 on the evaluated pairs.',
              '- All 18,000 PDB files from the 1,000-input CLI jobs were checked for correct heavy/light sequences and finite coordinates. For each model/input pair, all repeats and runtimes gave identical PDB bytes.',
              '- The checks assess deployment agreement, not accuracy against experimental structures. The five-pair workload is a throughput test, not a diverse accuracy benchmark.', '',
              '## Optional hydrogen addition', '',
              'The frozen historical CLI option named `--relaxation` called hydrogen addition; it did not perform energy minimisation. The current CLI rejects that option; see the [preparation guide](abodybuilder3_preparation_251026.md) for its replacements. '
              'Both supported hydrogen paths were checked separately on the 229-residue public example.', '',
              'Legacy Reduce 4.16 emits an otherwise usable structure but exits with code 1 for this example, causing the CLI to fall back to PDBFixer. '
              'Reduce 4.17 exits with code 0, so the modernised CLI uses its Reduce output. '
              'The former path produced 3,407 atoms (1,688 hydrogens); Reduce produced 3,399 atoms (1,682 hydrogens). '
              'Direct Reduce atom counts match between releases; the CLI difference is the fallback behaviour. '
              'PDBFixer succeeds in all three images. Unrelaxed prediction identity does not imply identical optional hydrogen-added files.', '',
              '## Interpretation', '',
              'The original rebuild is reproducible from pinned public sources. Modernisation removes the custom OpenMM/PDBFixer builds while preserving the tested unrelaxed predictions. '
              'Its base/pLDDT speed is similar to the original. Dev GPU batching improves warmed structure processing by about 2.5× versus sequential processing, at higher CUDA allocation. '
              'It has not been wired into the existing dev CLI: that CLI currently parses `--batch-size` without dispatching the batch helper. '
              'No default was changed and no resources were deleted.', '',
              'Only two timed repetitions were used. Language timings are not stable enough for a firm speed claim. '
              'Additional diverse sequences, longer loops and broader experimental-structure evaluation would be needed before making accuracy or general scaling claims.', '',
              '![Runtime and CUDA allocation](assets/abodybuilder3_benchmarks_251026.png)', '']
    derived = args.study_root / 'dev_prediction_summary_251026.json'
    if derived.exists():
        extra = ['', '## Dev prediction differences by model', '',
                 'Worst differences across the evaluated distinct pairs, stages, batch sizes and both measured repetitions. '
                 'All atom masks and residue identities match. The base model has no confidence head.', '',
                 '| Model | Maximum Cα coordinate difference, Å | Maximum atom coordinate difference, Å | Maximum aligned Cα RMSD, Å | Minimum Cα local-distance agreement | Maximum pLDDT difference, points |',
                 '|---|---:|---:|---:|---:|---:|']
        for row in json.loads(derived.read_text()):
            confidence = '—' if row['maximum_plddt_score_difference'] is None else f'{row["maximum_plddt_score_difference"]:.4f}'
            extra.append(f'| {row["model"]} | {row["max_ca_coordinate_difference_angstrom"]:.6f} | {row["max_atom_coordinate_difference_angstrom"]:.6f} | {row["max_ca_aligned_rmsd_angstrom"]:.6f} | {row["minimum_ca_lddt_agreement"]:.3f} | {confidence} |')
        extra += ['', 'The model output called `plddt` contains 50-bin confidence logits. '
                  'The pLDDT differences above use softmax probabilities and the upstream bin centres (1, 3, …, 99), '
                  'rather than interpreting the logits as scores.', '',
                  'An existing CLI limitation was identified: its JSON/confidence-filter path flattens these logits, '
                  'and its PDB writer uses zero B-factors. Those annotations should not be interpreted as per-residue pLDDT scores. '
                  'This comparison leaves the original CLI unchanged; the default coordinate predictions do not use a confidence threshold.', '']
        insert = lines.index('## Optional hydrogen addition')
        lines[insert:insert] = extra
    (args.output / 'abodybuilder3_benchmarks_251026.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    colours = ['#183a60', '#6699cc', '#289b8b', '#9d61b2']
    for col, model in enumerate(['base', 'plddt', 'language']):
        selected = [next(x for x in main_rows if x['model'] == model and x['condition'] == c) for c in ['original', 'rebuilt', 'modernised', 'dev']]
        x = np.arange(4)
        values = [r['seconds_median'] for r in selected]
        axes[0, col].bar(x, values, color=colours)
        axes[0, col].errorbar(x, values, yerr=[[r['seconds_median']-r['seconds_min'] for r in selected], [r['seconds_max']-r['seconds_median'] for r in selected]], fmt='none', color='black', capsize=5)
        axes[0, col].set_title(f'{model}: {selected[0]["count"]:,} inputs', fontweight='bold')
        axes[0, col].set_ylabel('Warmed processing time (seconds)')
        axes[1, col].bar(x, [r['cuda_allocated_peak_gib'] for r in selected], color=colours)
        axes[1, col].set_ylabel('Peak CUDA allocation (GiB)')
        for row in range(2):
            axes[row, col].set_xticks(x, ['Original', 'Rebuilt', 'Modernised', 'Dev batch'], rotation=20)
            axes[row, col].grid(axis='y', alpha=.2)
            axes[row, col].set_axisbelow(True)
    fig.suptitle('ABodyBuilder3 on DGX Spark: runtime and GPU allocation\nFive public sequence pairs repeated; medians of two runs; error bars show runtime range', fontsize=15)
    assets = args.output / 'assets'
    assets.mkdir(exist_ok=True)
    fig.savefig(assets / 'abodybuilder3_benchmarks_251026.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'processing_cases': len(rows), 'cli_cases': len(cli), 'output': str(args.output)}))


if __name__ == '__main__':
    main()
