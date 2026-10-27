"""Geometry-gated rescue after an incompatible fixed-heavy planar rejection."""
from dataclasses import asdict
import hashlib
import random
import time


def prepare_rejected(raw,options,policy):
    import numpy as np
    import openmm as mm
    from openmm import app
    import structure_preparation as sp
    import hydrogen_geometry as hg
    import heavy_atom_rescue as rescue
    from rescue_geometry import disulfides, heavy_gate, displacement, failing_centres, heavy_metrics
    started=time.perf_counter();random.seed(options.seed);np.random.seed(options.seed)
    report={'status':'started','options':asdict(options),'warnings':[],
        'raw_sha256':hashlib.sha256(raw.encode()).hexdigest(),
        'forcefield':['amber14/protein.ff14SB.xml','implicit/gbn2.xml'],
        'fallback':'Explicit geometry-gated rescue; no alternate backend, topology or protonation substitution.',
        'rescue':{'policy':policy,'applied':False,'heavy_attempts':[]}}
    report['versions']={'openmm':mm.version.short_version,'pdbfixer':sp.importlib.metadata.version('pdbfixer')}
    report['hydrogen_runtime']={'policy':'gpu_mixed','platform':'CUDA',
                                'requested_properties':{'Precision':'mixed','DeterministicForces':'true'}}
    try:
        original,fixer=sp._repair(raw,options,report);top,positions=fixer.topology,fixer.positions
        if options.hydrogen_method=='reduce':top,positions=sp._reduce(fixer,options,report)
        modeller=app.Modeller(top,positions)
        variants=modeller.addHydrogens(sp._forcefield(),pH=7,platform=mm.Platform.getPlatformByName('CPU'),
            platformProperties={'Threads':'1','DeterministicForces':'true'},hydrogenOptimizer=lambda s,t,p,m,pl,pr:p)
        top,before=modeller.topology,modeller.positions
        system=sp._forcefield().createSystem(top,nonbondedMethod=app.NoCutoff,constraints=None)
        bad=failing_centres(system,top,before)
        if not bad:raise ValueError('Rescue preflight found no incompatible frozen heavy motif; no repair performed.')
        detail=report['rescue'];detail.update(failing_centres_before=len(bad),heavy_before=heavy_metrics(system,top,before),
            seeded_reference_pdb=sp._pdb(top,before),reference_frame='After common atom repair and optional explicitly requested Reduce orientation.')
        if policy=='local':
            accepted=False
            for radius,steric in [(1,100000),(2,500000)]:
                positions,info=rescue.repair(system,top,before,[m['parent_index'] for m in bad],radius=radius,
                    restraint_k=10000,bond_scale=10,preserve_unrelated=True,steric_k=steric)
                gate=heavy_gate(system,top,before,positions,bad)
                detail['heavy_attempts'].append({'radius':radius,'steric_k':steric,'minimisation':info,'geometry_gate':gate,
                                               'intermediate_pdb':sp._pdb(top,positions)})
                if gate['passed']:accepted=True;break
            if not accepted:raise ValueError('Local rescue exhausted its bounded search; no whole-protein relaxation was substituted.')
        elif policy=='restrained':
            positions,prepass=rescue.repair(system,top,before,[],selection='all',restraint_k=10000,bond_scale=10,max_iterations=2000)
            positions=hg.initial_positions(system,top,positions,[a.index for a in top.atoms() if a.element.symbol=='H'])
            opts=sp.Options(seed=options.seed,relax='restrained',restraint_k=1000,max_iterations=3000,relax_platform='CUDA')
            positions,first=sp._minimise(top,positions,opts,freeze_heavy=True)
            positions,second=rescue.restrained_with_guard(system,top,positions,restraint_k=1000,max_iterations=3000,
                reference=before,chirality_strength=1e8)
            gate=heavy_gate(system,top,before,positions,bad)
            detail['heavy_attempts'].append({'geometry_prepass':prepass,'physical_H_stage':first,
                'restrained_stage':second,'geometry_gate':gate,'intermediate_pdb':sp._pdb(top,positions)})
            if not(gate['passed'] and first['converged'] and second['converged']):
                raise ValueError('Restrained rescue did not converge or failed its heavy geometry gate.')
        else:raise ValueError('Unknown rescue policy.')
        final=app.Modeller(top,positions);final.delete([a for a in top.atoms() if a.element.symbol=='H'])
        final.addHydrogens(sp._forcefield(),pH=7,variants=variants,platform=mm.Platform.getPlatformByName('CUDA'),
            platformProperties={'Precision':'mixed','DeterministicForces':'true'},hydrogenOptimizer=hg.optimise)
        if sp._keys(top)!=sp._keys(final.topology) or disulfides(top)!=disulfides(final.topology):
            raise ValueError('Rescue changed atom identities or disulfide topology.')
        heavy=[a.index for a in top.atoms() if a.element.symbol!='H']
        if np.max(abs(sp._positions(positions)[heavy]-sp._positions(final.positions)[heavy]))>=1e-12:
            raise ValueError('Final hydrogen placement moved rescued heavy atoms.')
        detail.update(heavy_after=heavy_metrics(system,top,final.positions),
            displacement=displacement(top,before,final.positions,set(sp._keys(original.topology))),
            core_seconds=time.perf_counter()-started,geometry_gate=heavy_gate(system,top,before,final.positions,bad))
        report.update(effective_ph=7.,hydrogen_backend='R8 fixed-heavy GPU H placement after explicit heavy rescue',
            protonation_variants=list(variants),hydrogen_geometry_optimisation=dict(hg.optimise.last_report),
            preparation_changes=sp._changes(original.topology,original.positions,final.topology,final.positions),
            prepared_geometry=sp._geometry(final.topology,final.positions))
        report['warnings'].append('Explicit '+policy+' rescue changed heavy coordinates; raw prediction remains separate.')
        if policy=='local':report['warnings'].append('Local rescue does not repair unrelated pre-existing heavy geometry defects.')
        report.update(status='completed_with_warnings',seconds=time.perf_counter()-started)
        return sp._pdb(final.topology,final.positions),None,report
    except Exception as exc:
        report.update(status='failed',error=str(exc),seconds=time.perf_counter()-started)
        raise sp.PreparationError(str(exc),report) from exc
