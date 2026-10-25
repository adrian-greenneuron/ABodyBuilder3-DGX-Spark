"""All within-antibody prediction pairs; exploratory hierarchical clustering.

Experimental coordinates define the CDR labels and the CA neighbour selection.
This clusters predictions of the SAME antibody, never unrelated antibodies.
Checkpoint weights remain separate when interpreting numerical changes.
"""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from structural_metrics import fit, rmsd, BACKBONE

VARIANTS=['reference','accepted','attention','point_ieee','compiled','strict_reference','strict_attention','strict_compiled']
TITLES=['Reference','Current dev','Attention','Attention: local FP32','Compiled',
        'Reference: FP32','Attention: FP32','Compiled: FP32']
TITLE_MAP=dict(zip(VARIANTS,TITLES)) | {'combined_compiled':'Attention + compiled',
    'strict_combined_compiled':'Attention + compiled: FP32'}


def tree(matrix):
    distance=squareform(matrix,checks=True)
    z=linkage(distance,method='average',optimal_ordering=True)
    return {'method':'average linkage; no forced number of clusters',
            'linkage':z.tolist(),'leaf_order':leaves_list(z).tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--render-only',action='store_true',help='Reuse saved matrices without repeating comparisons.')
    parser.add_argument('--variants',nargs='+',default=VARIANTS)
    args=parser.parse_args();root=args.study
    if args.render_only:
        render(json.loads((root/'pairwise_summary_251026.json').read_text()),root)
        return
    out=root/'pairwise';out.mkdir(exist_ok=False)
    ids=json.loads((root/'data/test_ids_251026.json').read_text())
    variants=args.variants;nvar=len(variants)
    labels=[f'{model}/{variant}' for model in ['base','plddt'] for variant in variants]
    metrics=['ca_aligned_rmsd','backbone_aligned_rmsd','cdrh3_framework_backbone_rmsd','local_ca_distance_rmsd']
    stack={name:[] for name in metrics};coherence={name:[] for name in metrics}
    case_results=[]
    for name in ids:
        arrays=[np.load(root/label/(name+'.npz')) for label in labels]
        coordinates=[d['positions'].astype(np.float64) for d in arrays]
        mask=np.logical_and.reduce([d['masks'] for d in arrays])
        heavy=arrays[0]['heavy'];regions=arrays[0]['regions'];ca=mask[:,1]
        framework=heavy & np.char.startswith(regions,'fw') & ca
        h3=regions=='cdrh3';h3mask=h3[:,None]&mask[:,BACKBONE]
        bbmask=mask[:,BACKBONE]
        truth=arrays[0]['truth_positions'];true_valid=arrays[0]['truth_masks'][:,1] & ca
        distances=np.linalg.norm(truth[:,None,1]-truth[None,:,1],axis=-1)
        near=np.triu((distances<15)&true_valid[:,None]&true_valid[None,:],1)
        ai,bi=np.where(near)
        local=[np.linalg.norm(p[ai,1]-p[bi,1],axis=-1) for p in coordinates]
        matrices={metric:np.zeros((len(labels),len(labels)),np.float64) for metric in metrics}
        for i,j in itertools.combinations(range(len(labels)),2):
            x,y=coordinates[i],coordinates[j]
            rotation,translation=fit(x[ca,1],y[ca,1]);aligned=x @ rotation+translation
            hrot,hshift=fit(x[framework,1],y[framework,1]);h_aligned=x @ hrot+hshift
            values=[rmsd(aligned[ca,1],y[ca,1]),
                    rmsd(aligned[:,BACKBONE][bbmask],y[:,BACKBONE][bbmask]),
                    rmsd(h_aligned[:,BACKBONE][h3mask],y[:,BACKBONE][h3mask]),
                    float(np.sqrt(np.mean((local[i]-local[j])**2)))]
            for metric,value in zip(metrics,values):
                # Rigid superposition of bitwise-identical arrays has SVD roundoff.
                if value<1e-12:value=0.
                matrices[metric][i,j]=matrices[metric][j,i]=value
        entry={'target':name,'labels':labels,'matrices':{k:v.tolist() for k,v in matrices.items()},
               'trees':{k:tree(v) for k,v in matrices.items()}}
        case_results.append(entry)
        (out/(name+'.json')).write_text(json.dumps(entry,indent=2)+'\n')
        for metric,matrix in matrices.items():
            stack[metric].append(matrix)
            for model,offset in [('base',0),('plddt',nvar)]:
                strict=[offset+i for i,v in enumerate(variants) if v.startswith('strict_')]
                other=[offset+i for i,v in enumerate(variants) if not v.startswith('strict_')]
                if len(strict)<2 or not other:continue
                within=max(matrix[i,j] for i,j in itertools.combinations(strict,2))
                outward=min(matrix[i,j] for i in strict for j in other)
                coherence[metric].append({'target':name,'model':model,'strict_group_max_internal':float(within),
                    'strict_group_min_external':float(outward),'strict_group_separated':bool(within<outward)})
    summary={'targets':len(ids),'predictions_per_antibody':len(labels),
        'unordered_pairs_per_antibody':len(labels)*(len(labels)-1)//2,
        'total_unordered_pairs':len(ids)*len(labels)*(len(labels)-1)//2,
        'labels':labels,'variants':variants,'scope':'All pairs of predictions of each same antibody; the 100 antibodies are not aligned to each other.',
        'zero_roundoff_threshold_A':1e-12,'metrics':{}}
    for metric,values in stack.items():
        a=np.stack(values);median=np.median(a,axis=0)
        summary['metrics'][metric]={'median_matrix':median.tolist(),'max_matrix':a.max(0).tolist(),
            'median_matrix_tree':tree(median),
            'strict_group_separated_count':{model:sum(r['strict_group_separated'] for r in coherence[metric] if r['model']==model) for model in ['base','plddt']},
            'strict_group_checks':coherence[metric]}
    (root/'pairwise_summary_251026.json').write_text(json.dumps(summary,indent=2)+'\n')
    render(summary,root)
    print(json.dumps({'completed':True,'targets':len(ids),'unordered_pairs':summary['total_unordered_pairs']}),flush=True)


def render(summary,root):
    variants=summary.get('variants',VARIANTS);nvar=len(variants)
    titles=[TITLE_MAP.get(v,v) for v in variants]
    # Publish compact, readable static figures; all individual matrices/trees
    # remain machine-readable rather than choosing a convenient example alone.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    for metric,title in [('ca_aligned_rmsd','Whole-antibody Cα RMSD'),
                         ('cdrh3_framework_backbone_rmsd','CDR-H3 backbone RMSD after framework alignment')]:
        matrix=np.array(summary['metrics'][metric]['median_matrix'])
        fig,axes=plt.subplots(1,2,figsize=(15,7.8),layout='constrained')
        vmax=max(float(matrix[:nvar,:nvar].max()),float(matrix[nvar:,nvar:].max()))
        for ax,model,start in zip(axes,['Base checkpoint','pLDDT checkpoint'],[0,nvar]):
            m=matrix[start:start+nvar,start:start+nvar]
            ax.set_facecolor('#fafafa')
            im=ax.imshow(np.ma.masked_less_equal(m,0),norm=LogNorm(vmin=1e-6,vmax=max(vmax,1e-5)),cmap='viridis')
            for i in range(nvar):
                for j in range(nvar):
                    value=m[i,j]
                    label='0' if value==0 else (f'{value:.6f}' if value<.00001 else f'{value:.5f}')
                    ax.text(j,i,label,ha='center',va='center',fontsize=8,
                            color='black' if value==0 or value>vmax/3 else 'white')
            ax.set_xticks(range(nvar),titles,rotation=55,ha='right',fontsize=9)
            ax.set_yticks(range(nvar),titles,fontsize=9);ax.set_title(model,fontsize=14)
        fig.colorbar(im,ax=axes,shrink=.65,label='Å; logarithmic colour scale; 0 is white')
        fig.suptitle(title+'\nMedian of 100 experimental-test antibodies; each cell compares the same antibody',fontsize=15)
        basename='abodybuilder3_pairwise_ca_251026' if metric=='ca_aligned_rmsd' else 'abodybuilder3_pairwise_cdrh3_251026'
        fig.savefig(root/(basename+'.png'),dpi=180);fig.savefig(root/(basename+'.svg'));plt.close(fig)
    from scipy.cluster.hierarchy import dendrogram
    from matplotlib.ticker import FuncFormatter
    matrix=np.array(summary['metrics']['ca_aligned_rmsd']['median_matrix'])
    fig,ax=plt.subplots(figsize=(13,8),layout='constrained')
    names=[f'{model} | {title}' for model in ['Base','pLDDT'] for title in titles]
    dendrogram(np.array(summary['metrics']['ca_aligned_rmsd']['median_matrix_tree']['linkage']),
               labels=names,orientation='left',color_threshold=0,above_threshold_color='#465568',ax=ax)
    ax.set_xscale('symlog',linthresh=1e-5)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:'0' if x==0 else f'{x:.5f}'.rstrip('0').rstrip('.')))
    ax.set_xlabel('Average-linkage distance from median Cα RMSD (Å); expanded scale near zero')
    ax.set_title('Prediction clustering across checkpoints and arithmetic settings\nMedian distances across 100 antibodies; individual trees also retained')
    fig.savefig(root/'abodybuilder3_prediction_clusters_251026.png',dpi=180)
    fig.savefig(root/'abodybuilder3_prediction_clusters_251026.svg');plt.close(fig)


if __name__=='__main__':main()
