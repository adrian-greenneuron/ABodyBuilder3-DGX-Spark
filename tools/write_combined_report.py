"""Package the controlled attention-plus-compiler comparison for review."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np

VARIANTS=['compiled','combined_compiled','strict_compiled','strict_combined_compiled']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--docs',type=Path,required=True)
    args=p.parse_args();root=args.study;docs=args.docs
    data=docs/'data/combined_compilation_251026';data.mkdir(parents=True,exist_ok=True)
    assets=docs/'assets';assets.mkdir(exist_ok=True)
    summary=json.loads((root/'structural_summary_251026.json').read_text())
    rows=json.loads((root/'structural_per_target_251026.json').read_text())
    side=json.loads((root/'sidechain_assessment_251026.json').read_text())
    scores={(x['model'],x['variant']):x for x in summary['summaries']}
    timings={(x['model'],x['variant']):x for x in summary['timings']}
    controls={};statistics=[];changes=[]
    for model in ['base','plddt']:
        for precision,base,candidate in [('Normal','compiled','combined_compiled'),('Strict FP32','strict_compiled','strict_combined_compiled')]:
            a=timings[model,base];b=timings[model,candidate]
            control={r['name']:r for r in rows if r['model']==model and r['variant']==base}
            actual=[r for r in rows if r['model']==model and r['variant']==candidate]
            delta=np.array([r['accuracy']['regions']['cdrh3']['framework_aligned_backbone_rmsd']-
                control[r['name']]['accuracy']['regions']['cdrh3']['framework_aligned_backbone_rmsd'] for r in actual])
            statistics.append({'model':model,'precision':precision,'reference':base,'candidate':candidate,
                'runtime_change_percent':100*(b['median_model_pass_seconds']/a['median_model_pass_seconds']-1),
                'speedup':a['median_model_pass_seconds']/b['median_model_pass_seconds'],
                'cdrh3_better_unrounded':int((delta < -1e-10).sum()),'cdrh3_worse_unrounded':int((delta > 1e-10).sum()),
                'cdrh3_ties_1e_10':int((abs(delta)<=1e-10).sum()),'cdrh3_ties_within_0_01_A':int((abs(delta)<=.01).sum())})
            for r in actual:
                original=control[r['name']]
                for region in ['fwh','fwl','cdrh1','cdrh2','cdrh3','cdrl1','cdrl2','cdrl3']:
                    value=r['accuracy']['regions'][region]['framework_aligned_backbone_rmsd']
                    reference=original['accuracy']['regions'][region]['framework_aligned_backbone_rmsd']
                    changes.append({'model':model,'precision':precision,'target':r['name'],'metric':region+'_backbone_rmsd',
                        'compiler_only':reference,'combined':value,'combined_minus_compiler':value-reference})
        for variant in VARIANTS:
            m=json.loads((root/model/variant/'metadata_251026.json').read_text())
            controls[model+'/'+variant]={k:v for k,v in m.items() if k not in ['targets']}
            controls[model+'/'+variant]['target_ids']=list(m['targets'])
    (data/'comparison_summary_251026.json').write_text(json.dumps({'comparisons':statistics,'condition_metadata':controls},indent=2)+'\n')
    with (data/'per_antibody_accuracy_changes_251026.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(changes[0]),lineterminator='\n');writer.writeheader();writer.writerows(changes)
    lines=['# Attention plus compilation — 25 October 2026','',
        '**Normal precision:** adding the point-attention rewrite reduced warmed model-pass time by about **4–5%** beyond compilation alone. **Strict FP32:** the combination was about **1–2% slower** than compilation alone. Memory use was almost unchanged. Neither combination is enabled on dev; main and dev are unchanged.','',
        '## Controlled comparison','',
        'All **100 experimental antibodies from the earlier published test set** were included, for both base and pLDDT checkpoints. No new targets were selected and the earlier 15 engineering fixtures were excluded. Each checkpoint ran compilation alone and attention-plus-compilation, each with normal TF32-permitted settings and globally strict FP32 settings: **8 conditions and 800 saved predictions**.','',
        'The controls and combinations used the same candidate module, weights, eight backbone updates, compact features, final-only side-chain outputs and pair-buffer reuse. Within each precision pair, the only model option changed was enabling the weighted-point matrix-multiplication rewrite. Strict FP32 was set before model loading and compilation. Normal settings were `high` with TF32 permitted; strict settings were `highest` with TF32 disabled. Both store FP32 weights and tensors.','',
        'Normal precision permits accelerated matrix multiplication with lower internal input precision and FP32 accumulation. Strict FP32 requests full FP32 matrix multiplication. This changes arithmetic, not the model or its weights, and does not double stored tensor memory. Strict FP32 still rounds results, and compilation can change operation order, so it does not guarantee bit-identical coordinates or better agreement with experiment.','',
        'Unlike the earlier strict-arithmetic diagnostic, **every condition here used compact features**. The old strict controls retained dense CPU features and should not be mixed with these resource results. Matching dataset hashes, observed-atom masks, feature mode, precision flags and model options were checked after execution.','',
        '## Speed','',
        'Values are medians of **four warmed passes** over the 100 antibodies. Negative runtime changes mean faster. Bold marks the lower runtime within each matched checkpoint/precision pair, not statistical significance.','',
        '| Checkpoint | Precision | Compilation alone | Attention + compilation | Runtime change |',
        '|---|---|---:|---:|---:|']
    for s in statistics:
        a=timings[s['model'],s['reference']]['median_model_pass_seconds'];b=timings[s['model'],s['candidate']]['median_model_pass_seconds']
        aa=f'**{a:.4f} s**' if a<b else f'{a:.4f} s';bb=f'**{b:.4f} s**' if b<a else f'{b:.4f} s'
        lines.append(f"| {s['model']} | {s['precision']} | {aa} | {bb} | {s['runtime_change_percent']:+.2f}% |")
    lines+=['','The normal-precision gain is modest: attention and compilation do not multiply their separate speedups. The benchmark used fixed **B=16, padded length 272**, with 100 real targets plus 12 dummy rows excluded from every score. It includes prepared-feature transfer, projection and model execution; it excludes imports, checkpoint loading, atom conversion, PDB writing, scoring and compilation. These are **model-stage timings, not complete CLI throughput**.','',
        'Every condition ran in a fresh offline container with a fresh compiler-cache directory. Condition order was reversed for the second checkpoint. The first forward, including compilation and kernel startup, took **24.1–25.1 seconds**. Four timed passes in one process are not four independent process replicates. The raw timing ranges and startup measurements are retained; a mixed-length production workload has not been qualified.','',
        '## Memory','',
        '| Checkpoint | Precision | Condition | Warm peak CUDA allocated | Sampled process-tree RSS peak | Container lifetime peak |',
        '|---|---|---|---:|---:|---:|']
    for model in ['base','plddt']:
        for variant in VARIANTS:
            m=controls[model+'/'+variant];measure=m['measurements'];t=timings[model,variant]
            container=max(x['container_lifetime_memory_peak_bytes'] for x in measure)/2**30
            precision='Strict FP32' if variant.startswith('strict_') else 'Normal'
            name='Combined' if 'combined' in variant else 'Compiler only'
            lines.append(f"| {model} | {precision} | {name} | {t['peak_cuda_allocated_gib']:.3f} GiB | {t['sampled_process_rss_gib']:.3f} GiB | {container:.3f} GiB |")
    lines+=['','Adding attention used approximately **4.8 MiB more peak CUDA allocation**. Sampled process-tree RSS stayed around **4.12–4.13 GiB**; container lifetime accounting stayed around **2.29–2.31 GiB**. Strict FP32 does not require larger stored tensors, so it did not materially increase memory here.','',
        'CUDA peaks are PyTorch allocator peaks during warmed passes; startup CUDA peaks were not collected. RSS sums the main process and its children and can count shared pages more than once. Container accounting is a separate lifetime measure, including startup. **Do not add these columns** to estimate physical unified memory. The data also retains main-process lifetime RSS, sampled-RSS definitions, reserved CUDA memory and all repetitions.','',
        '## Experimental accuracy','',
        'Each combined condition is compared with **compilation alone at the same precision**, using the original experimental coordinates and observed-atom masks. CDRs are scored after fitting that chain’s framework Cα atoms. All six CDRs, both frameworks, whole-Fv Cα/backbone RMSD, Cα-lDDT and confidence were assessed. No target was excluded.','',
        '| Checkpoint | Precision | Mean CDR-H3 RMSD, compiler → combined | Largest absolute CDR-H3 accuracy change | Closer / further from experiment |',
        '|---|---|---:|---:|---:|']
    for s in statistics:
        x=scores[s['model'],s['candidate']]['experimental_region_changes']['cdrh3']
        lines.append(f"| {s['model']} | {s['precision']} | {x['baseline_mean_backbone_rmsd']:.6f} → {x['candidate_mean_backbone_rmsd']:.6f} Å | {x['max_absolute_paired_change']:.8f} Å | {s['cdrh3_better_unrounded']} / {s['cdrh3_worse_unrounded']} |")
    lines+=['','Better/worse counts use unrounded values with a 1e-10 numerical tie tolerance. They are not biological wins: **all 100 targets tie within a 0.01 Å reporting tolerance** for CDR-H3 backbone accuracy in every comparison. The normal-precision changes vary in direction across antibodies; the strict changes are smaller still. Tiny mean or bootstrap differences do not establish improved structural accuracy.','',
        'With normal precision, the maximum absolute CDR-H3 accuracy change was about **0.0051 Å**. With strict FP32 it was at most **0.000011 Å**. Strict combinations had exactly the same Cα-lDDT scores as their matched compiler-only controls on all 100 antibodies. These results provide no meaningful backbone-accuracy reason to prefer the combined approach.','',
        '## Local side chains','',
        '| Checkpoint | Precision | Largest atom displacement | Largest χ change | Largest per-residue side-chain accuracy worsening |',
        '|---|---|---:|---:|---:|']
    for model in ['base','plddt']:
        for variant in ['combined_compiled','strict_combined_compiled']:
            x=next(r for r in side['summaries'] if r['model']==model and r['variant']==variant)
            precision='Strict FP32' if variant.startswith('strict_') else 'Normal'
            lines.append(f"| {model} | {precision} | {x['largest_atom_displacements'][0]['raw_displacement']:.6f} Å | {x['largest_chi_changes'][0]['degrees']:.4f}° | {x['largest_residue_sidechain_accuracy_worsenings'][0]['change_negative_is_better']:.6f} Å |")
    lines+=['','The normal-precision base model’s largest change was **histidine H66 CE1 in 6dc4**, moving **0.775 Å**, with about **12.7°** change in χ1. Its symmetry-aware side-chain RMSD to experiment increased from **5.209 to 5.386 Å** after framework fitting. For pLDDT, glutamate L105 in **7k8o** worsened from **1.607 to 1.764 Å**. These are one-based input-chain positions, not IMGT residue numbers. Other residues improved. These local effects should not be hidden by whole-antibody averages or called universally harmless.','',
        'Across all observed side-chain atoms, the largest per-antibody RMSD change was about **0.0034 Å**. CDR-H3 side-chain accuracy changes reached about **0.012 Å** in magnitude. Strict-FP32 combinations stayed much closer locally: atom displacements were below **0.0007 Å**, and the largest per-residue accuracy worsening was below **0.00009 Å**. Equivalent side-chain atom names are handled per residue; unknown residues have no assigned χ torsions.','',
        'The same backbone-bond, peptide-continuity and heavy-atom clash screens were retained. The clash screen is a geometric proxy, not MolProbity. Normal conditions moved a few atom pairs across the cutoff in either direction; the strict combinations left clash counts unchanged for every target. No hydrogenation or relaxation was performed. The largest pLDDT score change was **0.243/100** with normal precision and below **0.0005/100** with strict FP32. The base checkpoint has no confidence head.','',
        '## Pairwise structure comparisons','',
        'All eight predictions of each same antibody were compared, giving **28 pairs per antibody and 2,800 pairs**. Four distances were retained: aligned Cα RMSD, aligned backbone RMSD, framework-aligned CDR-H3 backbone RMSD and local Cα distance RMSD. Individual matrices and average-linkage trees are in the evidence archive. The strict compiler/combined pair remained closer to each other than to either normal condition for all 100 antibodies in both checkpoints.','',
        '![Combined CDR-H3 comparisons](assets/abodybuilder3_combined_pairwise_cdrh3_251026.png)','',
        'Each cell is a median of same-antibody comparisons across the 100 targets, not an alignment of unrelated antibodies. In the figure, “FP32” denotes strict arithmetic; all conditions store FP32 tensors. Zero cells are white; colours are logarithmic.','',
        '## Decision','',
        '**Strict FP32:** compilation alone is the better measured choice here: faster, slightly less CUDA allocation, and effectively the same accuracy. **Normal precision:** combining attention adds a small warmed compute-stage gain but introduces additional local side-chain changes. A production throughput comparison would be needed to determine whether that small gain survives CPU work, output writing and varying lengths. Neither combination has been enabled or merged into dev.','',
        'The scope is the authors’ published 100-antibody test set and the base/pLDDT checkpoints. The LM checkpoint, other GPUs, dynamic production batches, functional binding effects and docking were not tested. This is a controlled engineering screen, not a new universal accuracy claim.','',
        '## Evidence and reproduction','',
        '- [Timing, resource metadata and paired directions](data/combined_compilation_251026/comparison_summary_251026.json)',
        '- [All region/accuracy summaries](data/combined_compilation_251026/structural_summary_251026.json)',
        '- [Per-antibody accuracy changes](data/combined_compilation_251026/per_antibody_accuracy_changes_251026.csv)',
        '- [Local side-chain assessment](data/combined_compilation_251026/sidechain_assessment_251026.json)',
        '- [Pairwise matrices and cluster summary](data/combined_compilation_251026/pairwise_summary_251026.json)',
        '- [Full comparison archive](data/combined_compilation_251026/abodybuilder3_combined_comparisons_251026.zip)',
        '- [Checksums](data/combined_compilation_251026/abodybuilder3_combined_checksums_251026.txt)','',
        'The original tensors, 800 raw predictions/PDBs, compiler caches, logs and frozen source remain in the local study. The existing dataset was reused read-only, with every input tensor hash and observed-atom mask matched. All first-batch repeated coordinates and confidence outputs matched exactly within each condition (16 targets × 8 conditions).','',
        'Use the unchanged pinned runtime from the earlier study, or a verified rebuild supplied with `--image`. The reusable controller was run with `--experimental-only --feature-mode compact --implementation candidate --variants compiled combined_compiled strict_compiled strict_combined_compiled --repeats 4 --reference compiled --strict-reference strict_compiled --matched-precision --reverse-second-model`, a fresh study directory/frozen component copy and the verified dataset supplied through `--data`. The same parameters for reference/precision selection are supported by the structural and side-chain scorers. The pairwise tool accepts the four variants explicitly. Protocol, source hashes and validation records are in the archive.','',
        'Model execution and all stored-output validations passed. The existing scoring/model tests remain applicable: no model equations or scoring formulas changed in this follow-up; the experiment composes the already tested attention and compiler paths and adds explicit dataset/feature/reference selection. Main remains `a5deeb1`, dev remains `bd8a12e`. The full source/runtime identities are retained in the data.','']
    (docs/'abodybuilder3_combined_compilation_251026.md').write_text('\n'.join(lines))
    copied=['structural_summary_251026.json','sidechain_assessment_251026.json','pairwise_summary_251026.json','protocol_251026.json','validation_251026.json']
    for name in copied:shutil.copy2(root/name,data/name)
    for name in ['pairwise_ca','pairwise_cdrh3','prediction_clusters']:
        for extension in ['png','svg']:
            source=root/f'abodybuilder3_{name}_251026.{extension}'
            target=assets/f'abodybuilder3_combined_{name}_251026.{extension}'
            if extension=='svg':
                target.write_text('\n'.join(line.rstrip() for line in source.read_text().splitlines())+'\n')
            else:shutil.copy2(source,target)
    archive=data/'abodybuilder3_combined_comparisons_251026.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for path in sorted((root/'pairwise').glob('*.json')):z.write(path,'pairwise/'+path.name)
        for name in copied+['frozen_source_251026.json','analysis_source_251026.json','structural_per_target_251026.json']:
            z.write(root/name,name)
        for name in ['test_ids_251026.json','experimental_manifest_251026.json','upstream_split_verification_251026.json']:
            z.write(root/'data'/name,'dataset/'+name)
        for path in data.glob('*.csv'):z.write(path,path.name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    paths=[path for path in data.iterdir() if path.is_file() and not path.name.endswith('checksums_251026.txt')]
    (data/'abodybuilder3_combined_checksums_251026.txt').write_text(''.join(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.name+'\n' for path in sorted(paths)))
    print(json.dumps({'completed':True,'archive_bytes':archive.stat().st_size,'report':str(docs/'abodybuilder3_combined_compilation_251026.md')}),flush=True)


if __name__=='__main__':main()
