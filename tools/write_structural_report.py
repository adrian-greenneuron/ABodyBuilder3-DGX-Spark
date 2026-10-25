"""Build the dated review report and compact evidence from completed results."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np

LABEL={'accepted':'Current dev','attention':'Attention, TF32 permitted',
       'point_ieee':'Attention, local FP32','compiled':'Compiled, TF32 permitted'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--docs',type=Path,required=True);args=p.parse_args();root=args.study;docs=args.docs
    docs.mkdir(exist_ok=True,parents=True);data=docs/'data';data.mkdir(exist_ok=True)
    assets=docs/'assets';assets.mkdir(exist_ok=True)
    result=json.loads((root/'structural_summary_251026.json').read_text())
    ranks=json.loads((root/'antibody_rankings_251026.json').read_text())
    side=json.loads((root/'sidechain_assessment_251026.json').read_text())
    pair=json.loads((root/'pairwise_summary_251026.json').read_text())
    scores={(r['model'],r['variant']):r for r in result['summaries']}
    timings={(r['model'],r['variant']):r for r in result['timings']}
    assert all(x['reference_accepted_control_exact'] for x in result['summaries'])
    assert pair['total_unordered_pairs']==12000
    directions={(r['model'],r['variant']):r for r in ranks['paired_experimental_directions']}
    lines=['# Attention and compiler structure checks — 25 October 2026','',
        'The optimisation closest to the existing prediction varies by antibody. Against experimental structures, each candidate improves some antibodies and worsens others. None is a consistent accuracy winner. All CDR-backbone accuracy changes in this test were below 0.01 Å; local side-chain changes were larger and should not be dismissed as harmless numerical noise.','',
        '**Main and dev prediction code are unchanged. Neither attention replacement nor compilation is enabled on dev.** This branch contains experimental component code and the review evidence.','',
        '## Experimental set and controls','',
        'We used all **100 test antibodies from 100 PDB entries** in the authors’ published ABodyBuilder3 split. The split file matches upstream byte for byte. Experimental coordinates, observed-atom masks and IMGT-derived CDR/framework labels came from the published [Zenodo dataset](https://zenodo.org/records/11354577), whose 3,020,370,695-byte archive matched its published MD5. All 100 structures passed chain-order, sequence, coordinate and residue-index checks; none was excluded. Total variable-region lengths were 219–242 residues; CDR-H3 lengths were 4–22. Unknown residues retain the upstream atom masks and have no assigned side-chain torsion.','',
        'Each base and pLDDT checkpoint was tested in eight conditions: frozen original reference, current accepted dev, point attention with the normal TF32-permitted settings, point attention with only its replacement products restricted to IEEE FP32, compiled accepted dev, and three globally strict FP32 controls (reference, point attention, compiled). Weights, eight learned backbone updates, input order and matched batch shapes were retained. The LM checkpoint was not tested.','',
        'The earlier 15 engineering fixtures were also retained: five public sequence pairs, two artificial length changes and eight single-residue substitutions. They test numerical behaviour, not experimental accuracy. Each condition saved **115 unrounded predictions**, giving **1,840 predictions across 16 conditions**, plus warm/timed/repeatability forwards. The first 16 targets repeated exactly within every condition. The current dev shortcuts matched the untouched original exactly on all 115 targets for both checkpoints.','',
        '## Does one approach consistently improve accuracy?','',
        'No. The following counts use unrounded CDR-H3 backbone RMSD to the experimental structure, after fitting that chain’s framework Cα atoms. Better/worse counts describe tiny numerical differences, not meaningful biological wins. All candidates and current dev tie within a **0.01 Å reporting tolerance** for both whole-Fv Cα RMSD and CDR-H3 backbone RMSD on every target. This tolerance is a reporting convention, not a universal bound on biological effects.','',
        '| Checkpoint | Candidate | Closer to experiment | Further from experiment | Largest absolute CDR-H3 accuracy change |',
        '|---|---|---:|---:|---:|']
    for model in ['base','plddt']:
        for variant in ['attention','point_ieee','compiled']:
            x=directions[model,variant]['experimental_accuracy']['cdrh3']
            lines.append(f"| {model} | {LABEL[variant]} | {x['better_unrounded']}/100 | {x['worse_unrounded']}/100 | {x['largest_absolute_change']:.6f} Å |")
    lines+=['','Current dev’s mean CDR-H3 backbone RMSD was **2.9138 Å** for base and **2.7991 Å** for pLDDT. Candidate changes in those means were smaller than 0.0003 Å. All six CDRs and both frameworks were scored separately. Whole-Fv Cα accuracy changes were below 0.001 Å for the three candidates; mean Cα-lDDT changes were below 0.001 percentage point. Full paired results, region counts, PDB-level bootstrap intervals and per-antibody scores accompany this report.','',
        'At unrounded precision the four-way CDR-H3 winner counts (current dev / attention / local-FP32 attention / compiled) were **20 / 39 / 23 / 18** for base and **22 / 28 / 24 / 26** for pLDDT. These rankings disappear at the 0.01 Å reporting tolerance; they do not establish that attention or compilation improves accuracy.','',
        '## All pairwise comparisons and clustering','',
        'For each antibody we compared all **16 predictions**, including both checkpoints: **120 unordered pairs per antibody, 12,000 pairs in total**. These are comparisons of predictions of the *same antibody*, not structural clustering of unrelated antibodies. Distances were whole-Fv aligned Cα RMSD, aligned backbone RMSD, heavy-framework-aligned CDR-H3 backbone RMSD and local Cα distance RMSD. Every antibody’s matrices and average-linkage trees are retained. No fixed number of clusters was imposed.','',
        'The aggregate tree separates checkpoint weights first. The median cross-checkpoint Cα RMSD was about **0.572 Å**, while numerical variants within a checkpoint typically differed by thousandths of an ångström. For **every one of the 100 antibodies, in both checkpoints and all four distance measures**, the largest distance within the fully strict FP32 trio was smaller than its nearest distance to any normal-precision condition. Thus arithmetic settings have a clear, consistent effect on the clustering.','',
        '![Prediction clustering](assets/abodybuilder3_prediction_clusters_251026.png)','',
        'The displayed tree uses median pair distances across antibodies. The accompanying archive contains all individual trees; the aggregate does not substitute for them.','',
        'Among the three candidates using the surrounding runtime’s normal settings, the candidate closest to the current prediction varied as follows:','',
        '| Checkpoint and comparison | Attention | Attention, local FP32 | Compiled |',
        '|---|---:|---:|---:|']
    for model in ['base','plddt']:
        for metric,label in [('ca_aligned_rmsd','whole-Fv Cα'),('cdrh3_framework_backbone_rmsd','CDR-H3 backbone')]:
            x=next(r for r in ranks['agreement_with_existing_prediction'] if r['model']==model and r['metric']==metric)['closest_candidate_to_current_prediction_counts']
            lines.append(f"| {model}, {label} | {x['attention']} | {x['point_ieee']} | {x['compiled']} |")
    lines+=['','Closeness to the current prediction is agreement with another prediction. It does not identify the structure closest to experiment.','',
        '![Local CDR-H3 comparisons](assets/abodybuilder3_pairwise_cdrh3_251026.png)','',
        '## What caused the numerical changes?','',
        'The weighted-point change replaces an explicit product/reduction with matrix multiplication. In a same-input first-layer diagnostic for each checkpoint, the original broadcast sum differed from a float64 sum by roughly 0.0000001 in the intermediate point representation. Strict FP32 matrix multiplication differed by roughly 0.000001; TF32-permitted multiplication differed by 0.0003–0.0005. These are arithmetic checks on model intermediates, including padded queries, not ångström accuracy measurements against a crystal structure.','',
        'Restricting only the replacement products to FP32 reduced that direct arithmetic error, but did not eliminate final prediction changes: the rest of the model still used the original TF32-permitted settings. Small input perturbations can affect later rounded matrix products. Layer-output diagnostics support propagation through subsequent blocks; they do not isolate every later kernel.','',
        'When the *whole model* used strict FP32, point attention and compilation were much closer to their **precision-matched strict reference**:','',
        '| Checkpoint | Strict candidate | Largest Cα displacement, 115 cases | Largest change in experimental CDR-H3 RMSD, 100 targets |',
        '|---|---|---:|---:|']
    for model in ['base','plddt']:
        for variant,label in [('strict_attention','Point attention'),('strict_compiled','Compiled')]:
            x=scores[model,variant]
            lines.append(f"| {model} | {label} | {x['max_raw_ca_displacement']:.7f} Å | {x['precision_matched_cdrh3_accuracy_change']['max_absolute']:.8f} Å |")
    lines+=['','They still are not bitwise identical. Different summation order, fused operations and intermediate rounding remain plausible residual causes. The controls establish a strong precision effect; no claim is made that a particular compiler kernel alone caused all normal-runtime differences. Switching the whole model to FP32 also changes the original prediction, and did not produce a consistent accuracy improvement.','',
        '## Local side chains and geometry','',
        'Backbone agreement does not imply identical local chemistry. We checked heavy-atom displacement, χ torsions, observed side-chain accuracy with equivalent atom names handled per residue, backbone bond lengths, peptide continuity and a heavy-atom clash proxy. Framework fitting was done separately for each chain.','',
        '| Checkpoint | Candidate | Largest atom displacement | Largest χ change | Worst per-residue side-chain accuracy increase |',
        '|---|---|---:|---:|---:|']
    for model in ['base','plddt']:
        for variant in ['attention','point_ieee','compiled']:
            x=next(r for r in side['summaries'] if r['model']==model and r['variant']==variant)
            lines.append(f"| {model} | {LABEL[variant]} | {x['largest_atom_displacements'][0]['raw_displacement']:.4f} Å | {x['largest_chi_changes'][0]['degrees']:.2f}° | {x['largest_residue_sidechain_accuracy_worsenings'][0]['change_negative_is_better']:.4f} Å |")
    lines+=['','The largest displacement was histidine H66 CE1 in **6dc4**, about **0.50 Å** with local-FP32 point attention. That residue became slightly closer to its experimental side chain; other residues became less accurate. For example, pLDDT point attention increased the side-chain RMSD of glutamate L105 in **7k8o** from **1.653 to 1.763 Å**. Residue positions here are one-based input-chain positions, not IMGT numbers.','',
        'Across all observed side-chain atoms, the largest per-antibody RMSD change was below 0.0042 Å. Within CDR-H3 alone it reached about 0.0185 Å in magnitude; the largest worsening was about 0.0115 Å. These averages can hide larger individual-residue effects, which is why the local cases are retained. Neither a docking result nor functional equivalence has been tested.','',
        'The clash proxy counts non-neighbouring heavy-atom pairs with a van der Waals overlap above 0.6 Å, excluding intra-residue pairs, immediate same-chain neighbours and short S–S bonds. A few pairs crossed that cutoff in either direction; new counted overlaps reached about 0.664 Å. This is not a MolProbity score. There was no substantial new backbone-continuity defect; small cutoff-count changes are reported in the data. Hydrogenation and relaxation were not used.','',
        'The largest pLDDT score difference among the three normal-runtime candidates was **0.326 on the 0–100 scale**. The base checkpoint has no confidence head; its absent confidence output is not evidence of confidence agreement.','',
        '## Speed and memory screen','',
        'Each condition ran in a fresh offline container with fixed batch size 16 and padded length 272. The pass contains 115 real inputs plus 13 dummy rows, discarded from every score. Values are medians of two warmed passes. They include prepared-feature transfer, projection and model execution, but **exclude** imports, checkpoint loading, atom conversion, PDB writing, scoring and compilation. These are not full production-CLI throughput results.','',
        '| Checkpoint | Condition | Warm pass | Speed vs current dev | Peak CUDA allocated | Sampled process RSS |',
        '|---|---|---:|---:|---:|---:|']
    for model in ['base','plddt']:
        for variant in LABEL:
            x=timings[model,variant];base=timings[model,'accepted']['median_model_pass_seconds']
            bold='**' if variant=='compiled' else ''
            lines.append(f"| {model} | {LABEL[variant]} | {bold}{x['median_model_pass_seconds']:.3f} s{bold} | {bold}{base/x['median_model_pass_seconds']:.2f}×{bold} | {bold}{x['peak_cuda_allocated_gib']:.3f} GiB{bold} | {x['sampled_process_rss_gib']:.3f} GiB |")
    lines+=['','Bold marks the best measured warm runtime and CUDA allocation in each checkpoint group, not statistical significance. Compilation’s first forward took about **24 seconds**, compared with under one second for the uncompiled conditions. Its sampled process RSS was higher despite lower CUDA allocator peaks. Do not add RSS and CUDA columns to estimate physical unified memory. Only two timed repetitions were run, with conditions sequential rather than a large randomised performance study.','',
        'The globally strict controls used the original dense-feature path to keep their arithmetic reference explicit; their RSS includes those retained CPU features. Their resource measurements are not a benchmark of a production compact-feature FP32 option. No attention-plus-compiler combination or dynamic production workload has been qualified.','',
        '## Decision and limitations','',
        'The tested numerical differences do **not** demonstrate a general gain or loss of backbone accuracy. They are small enough that the whole-Fv and CDR-backbone results are effectively tied at the stated reporting precision. They are large enough at a few side chains that universal atomic or functional equivalence should not be claimed. Compilation is the strongest warmed speed candidate, subject to startup, CPU-memory and production-throughput qualification. Both candidates remain disabled pending review.','',
        'This uses the authors’ released held-out set, not a newly independent audit of model training or a universal antibody benchmark. All 100 targets are reported; the artificial cases are kept separate from accuracy. The paired bootstrap resamples PDB entries; it does not turn these tiny differences into biological improvements. No new checkpoint was trained or selected using these results.','',
        '## Evidence and reproduction','',
        '- [Per-antibody accuracy changes (CSV)](data/per_antibody_accuracy_changes_251026.csv)',
        '- [Accuracy, resource and precision summaries](data/structural_summary_251026.json)',
        '- [All-pairs summaries and clustering](data/pairwise_summary_251026.json)',
        '- [Per-antibody rankings](data/antibody_rankings_251026.json)',
        '- [Side-chain assessment](data/sidechain_assessment_251026.json)',
        '- [Full comparison archive](data/abodybuilder3_structure_comparisons_251026.zip)',
        '- [Checksums](data/abodybuilder3_structure_checksums_251026.txt)','',
        'The archive contains every antibody’s 16 × 16 matrices/trees, all per-target structural scores, the test IDs and tensor hashes, frozen protocols/source hashes, and the point-reduction probes. Original tensors, raw predictions, PDBs, compiler caches, logs and failed startup/export records remain in the local study. The startup failure preceded model inference; later scorer corrections handled unknown-residue torsions and NumPy integer export. Existing successful inference was preserved.','',
        '`tools/prepare_experimental_data.py` obtains the published archive and verifies the exact split and checksum, or reuses an existing archive. `tools/check_structures.py` runs one isolated condition; `tools/run_structural_study.py` queues conditions using a retained image or `--image` for a verified rebuild. The source/runtime foundation is dev `07bfe1e`, with the same weights and dependency versions as the accepted runtime. The reference image used here is `sha256:0a3ffb4edcef8d608d65eca49787fcb4ea820098a63e6dfdca853baee0527afa`.','',
        'Mount a frozen copy of this component branch at `/component`, prepared data/results at `/study`, and the pinned dev tools at `/measurement`; use `PYTHONPATH=/measurement:/component/tools` and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before CUDA initialisation. Record the actual rebuild digest. The controller accepts `--component` and `--variants`; include `strict_compiled` for the final eight-condition set. Run `analyse_structures.py`, then `assess_sidechains.py`, `pairwise_structures.py` and `antibody_rankings.py` against the same study. `pairwise_structures.py --render-only` reuses saved matrices. `write_structural_report.py` packages the completed evidence.','',
        'Seven focused precision/scoring tests and the two existing component interface/guard tests passed. The runtime was PyTorch 2.10.0a0 NGC 25.11, CUDA 13.0, NVIDIA GB10, driver 580.178.04. The normal runtime permits TF32 (`float32_matmul_precision=high`); globally strict controls use `highest` and disallow TF32. PyTorch’s [numerical-accuracy guidance](https://docs.pytorch.org/docs/2.10/notes/numerical_accuracy.html) explains why equivalent floating-point calculations can differ.','']
    (docs/'abodybuilder3_attention_compiler_accuracy_251026.md').write_text('\n'.join(lines))
    filenames=['structural_summary_251026.json','pairwise_summary_251026.json','antibody_rankings_251026.json',
               'sidechain_assessment_251026.json','per_antibody_accuracy_changes_251026.csv']
    for name in filenames:shutil.copy2(root/name,data/name)
    for name in ['abodybuilder3_prediction_clusters_251026','abodybuilder3_pairwise_ca_251026','abodybuilder3_pairwise_cdrh3_251026']:
        for ext in ['png','svg']:shutil.copy2(root/(name+'.'+ext),assets/(name+'.'+ext))
    archive=data/'abodybuilder3_structure_comparisons_251026.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted((root/'pairwise').glob('*.json')):z.write(path,'pairwise/'+path.name)
        for name in filenames+['structural_per_target_251026.json','protocol_251026.json','followup_protocol_251026.json',
                              'frozen_source_v2_251026.json','frozen_source_v3_251026.json','analysis_source_v6_251026.json',
                              'analysis_source_sidechain_v7_251026.json']:
            z.write(root/name,name)
        for name in ['test_ids_251026.json','experimental_manifest_251026.json','upstream_split_verification_251026.json']:
            z.write(root/'data'/name,'dataset/'+name)
        for model in ['base','plddt']:
            meta=json.loads((root/model/'point_ieee/metadata_251026.json').read_text())
            z.writestr(f'precision/{model}_point_reduction.json',json.dumps(meta['weighted_point_reduction_probe'],indent=2)+'\n')
        if (root/'final_validation_251026.json').exists():
            z.write(root/'final_validation_251026.json','final_validation_251026.json')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    paths=[data/name for name in filenames]+[archive]
    (data/'abodybuilder3_structure_checksums_251026.txt').write_text(''.join(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.name+'\n' for path in paths))
    print(json.dumps({'completed':True,'report':str(docs/'abodybuilder3_attention_compiler_accuracy_251026.md'),
                      'archive_bytes':archive.stat().st_size}),flush=True)


if __name__=='__main__':main()
