"""Experimental local heavy-atom valence repair; not a deployment default.

This is a geometry objective, not full Amber energy minimisation. Only the
selected residue neighbourhood moves; fixed-only and H-dependent force terms
are omitted. A separate full restrained Amber calculation is the comparator.
"""
import math
import time

import numpy as np
import openmm as mm
from openmm import unit


def neighbourhood(topology, centres, radius):
    if radius not in (0,1,2):
        raise ValueError('Local radius must be zero, one or two residues.')
    atoms=list(topology.atoms());selected=set()
    for centre in centres:
        residue=atoms[centre].residue
        chain=list(residue.chain.residues());index=chain.index(residue)
        selected.update(chain[max(0,index-radius):index+radius+1])
    return {a.index for a in atoms if a.residue in selected and a.element.symbol!='H'}


def bond_shell(topology,centres,radius):
    if radius not in (1,2,3):raise ValueError('Bond shell must be one, two or three.')
    atoms=list(topology.atoms());adj={a.index:set() for a in atoms}
    for a,b in topology.bonds():
        if a.element.symbol!='H' and b.element.symbol!='H':adj[a.index].add(b.index);adj[b.index].add(a.index)
    selected=set(centres);front=set(centres)
    for _ in range(radius):front=set().union(*(adj[i] for i in front))-selected;selected|=front
    return selected


def chiral_volumes(topology, xyz,biological_only=False):
    atoms=list(topology.atoms());adj={a.index:[] for a in atoms};result={}
    for a,b in topology.bonds():adj[a.index].append(b.index);adj[b.index].append(a.index)
    for a in atoms:
        neighbours=[i for i in adj[a.index] if atoms[i].element.symbol!='H']
        is_biological=(a.name=='CA' and a.residue.name!='GLY') or (a.name=='CB' and a.residue.name in ('ILE','THR'))
        if a.element.symbol=='C' and len(neighbours)==3 and len(adj[a.index])==4 and (not biological_only or is_biological):
            value=float(np.linalg.det(xyz[neighbours]-xyz[a.index]))
            if abs(value)<1e-10:
                raise ValueError('Degenerate tetrahedral heavy-atom centre; local repair unqualified.')
            result[(a.index,*neighbours)]=value
    return result


def orientation_guard(topology,xyz,selected=None,biological_only=False,strength=1e10):
    volume='(x2-x1)*((y3-y1)*(z4-z1)-(z3-z1)*(y4-y1))-(y2-y1)*((x3-x1)*(z4-z1)-(z3-z1)*(x4-x1))+(z2-z1)*((x3-x1)*(y4-y1)-(y3-y1)*(x4-x1))'
    guard=mm.CustomCompoundBondForce(4,f'0.5*{strength}*max(0,floor-sign*volume)^2;volume='+volume)
    guard.addPerBondParameter('sign');guard.addPerBondParameter('floor')
    for indices,value in chiral_volumes(topology,xyz,biological_only).items():
        if selected is None or set(selected).intersection(indices):guard.addBond(indices,[float(np.sign(value)),.25*abs(value)])
    return guard


def local_system(original,topology,positions,mobile,restraint_k,angle_scale=10,preserve_boundary=False,preserve_unrelated=False,core_centres=(),bond_scale=1,steric_k=100000):
    if not math.isfinite(restraint_k) or restraint_k<=0:
        raise ValueError('Restraint must be finite and positive.')
    mobile=set(mobile);atoms=list(topology.atoms());heavy={a.index for a in atoms if a.element.symbol!='H'}
    if not mobile or not mobile<=heavy:
        raise ValueError('A nonempty heavy-only mobile set is required.')
    xyz=np.asarray(positions.value_in_unit(unit.nanometer));system=mm.System()
    # addHydrogens temporarily makes existing atoms massless. Restore physical
    # element masses for the requested mobile set, irrespective of that input.
    for i in range(original.getNumParticles()):system.addParticle(atoms[i].element.mass if i in mobile else 0)
    for force in original.getForces():
        if isinstance(force,mm.HarmonicBondForce):
            new=mm.HarmonicBondForce()
            for i in range(force.getNumBonds()):
                a,b,r,k=force.getBondParameters(i)
                if {a,b}<=heavy and mobile.intersection((a,b)):
                    preserve=(preserve_boundary and not {a,b}<=mobile) or (preserve_unrelated and not set(core_centres).intersection((a,b)))
                    if preserve and abs(np.linalg.norm(xyz[a]-xyz[b])-r.value_in_unit(unit.nanometer))>.005:
                        r=float(np.linalg.norm(xyz[a]-xyz[b]))*unit.nanometer
                    new.addBond(a,b,r,k*bond_scale)
            system.addForce(new)
        elif isinstance(force,mm.HarmonicAngleForce):
            new=mm.HarmonicAngleForce()
            for i in range(force.getNumAngles()):
                a,b,c,t,k=force.getAngleParameters(i)
                if {a,b,c}<=heavy and mobile.intersection((a,b,c)):
                    u,v=xyz[a]-xyz[b],xyz[c]-xyz[b]
                    actual=math.acos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1))
                    preserve=(preserve_boundary and not {a,b,c}<=mobile) or (preserve_unrelated and b not in core_centres)
                    if preserve and abs(actual-t.value_in_unit(unit.radian))>math.radians(10):t=actual*unit.radian
                    new.addAngle(a,b,c,t,k*angle_scale)
            system.addForce(new)
        elif isinstance(force,mm.PeriodicTorsionForce):
            new=mm.PeriodicTorsionForce()
            for i in range(force.getNumTorsions()):
                values=force.getTorsionParameters(i)
                if set(values[:4])<=heavy and mobile.intersection(values[:4]):new.addTorsion(*values)
            system.addForce(new)
        elif isinstance(force,mm.NonbondedForce):
            new=mm.CustomNonbondedForce(f'0.5*{float(steric_k)}*min(0,r-0.8*(radius1+radius2))^2')
            new.addPerParticleParameter('radius');radii={'H':0,'C':.17,'N':.155,'O':.152,'S':.18}
            for a in atoms:new.addParticle([radii[a.element.symbol]])
            for i in range(force.getNumExceptions()):
                a,b,*_=force.getExceptionParameters(i);new.addExclusion(a,b)
            new.addInteractionGroup(mobile,heavy);new.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic);new.setCutoffDistance(1.)
            system.addForce(new)
    restraint=mm.CustomExternalForce('0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.addGlobalParameter('k',restraint_k)
    for name in ['x0','y0','z0']:restraint.addPerParticleParameter(name)
    for i in sorted(mobile):restraint.addParticle(i,xyz[i])
    system.addForce(restraint)
    system.addForce(orientation_guard(topology,xyz,mobile))
    return system


def repair(original,topology,positions,centres,radius=1,restraint_k=10000,max_iterations=2000,selection='residue',preserve_boundary=False,preserve_unrelated=False,bond_scale=1,steric_k=100000):
    started=time.perf_counter()
    if selection=='residue':mobile=neighbourhood(topology,centres,radius)
    elif selection=='bond_shell':mobile=bond_shell(topology,centres,radius)
    elif selection=='all':mobile={a.index for a in topology.atoms() if a.element.symbol!='H'}
    else:raise ValueError('Unknown local selection.')
    xyz=np.asarray(positions.value_in_unit(unit.nanometer));system=local_system(original,topology,positions,mobile,restraint_k,preserve_boundary=preserve_boundary,preserve_unrelated=preserve_unrelated,core_centres=centres,bond_scale=bond_scale,steric_k=steric_k)
    integrator=mm.VerletIntegrator(0)
    context=mm.Context(system,integrator,mm.Platform.getPlatformByName('CPU'),{'Threads':'1','DeterministicForces':'true'})
    try:
        context.setPositions(positions)
        initial=context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        mm.LocalEnergyMinimizer.minimize(context,1.,max_iterations)
        state=context.getState(getPositions=True,getEnergy=True,getForces=True)
        out=np.asarray(state.getPositions(asNumpy=True).value_in_unit(unit.nanometer))
        energy=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        force=state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
        assert np.isfinite(out).all() and math.isfinite(energy) and energy<=initial+max(1,abs(initial))*1e-6
        fixed=sorted(set(range(len(xyz)))-mobile)
        assert np.max(abs(out[fixed]-xyz[fixed]),initial=0)<1e-12
        before=chiral_volumes(topology,xyz);after=chiral_volumes(topology,out)
        inversions=sum(before[k]*after[k]<=0 for k in before)
        if inversions:raise ValueError('Local repair inverted heavy-atom chirality.')
        return out*unit.nanometer,{'kind':'local heavy-only geometry objective','selection':selection,'radius':radius,'mobile_heavy_atoms':len(mobile),'preserve_preexisting_bad_boundary_geometry':preserve_boundary,'preserve_preexisting_bad_noncore_valence':preserve_unrelated,
            'restraint_k_kj_mol_nm2':restraint_k,'angle_scale':10,'bond_scale':bond_scale,'steric_k':steric_k,'max_iterations':max_iterations,
            'initial_geometry_objective':initial,'final_geometry_objective':energy,'heavy_chiral_inversions':inversions,
            'rms_force_kj_mol_nm':float(np.sqrt(np.sum(force[list(mobile)]**2)/len(xyz))),
            'seconds':time.perf_counter()-started,'platform':'CPU','fixed_atoms_unchanged':True}
    finally:del context,integrator


def restrained_with_guard(system,topology,positions,restraint_k=1000,max_iterations=1000,reference=None,chirality_strength=1e12):
    """Full physical Amber forces plus explicit positional/chirality restraints.

    The caller supplies a fresh ordinary force-field System. The positive-volume
    wall protects true protein stereocentres, not equivalent methyl labels.
    """
    started=time.perf_counter();xyz=np.asarray((positions if reference is None else reference).value_in_unit(unit.nanometer))
    restraint=mm.CustomExternalForce('0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)')
    restraint.addGlobalParameter('k',restraint_k)
    for p in ['x0','y0','z0']:restraint.addPerParticleParameter(p)
    for atom in topology.atoms():
        if atom.element.symbol!='H':restraint.addParticle(atom.index,xyz[atom.index])
    restraint.setForceGroup(1);system.addForce(restraint)
    guard=orientation_guard(topology,xyz,biological_only=True,strength=chirality_strength);guard.setForceGroup(2);system.addForce(guard)
    integrator=mm.VerletIntegrator(.001)
    context=mm.Context(system,integrator,mm.Platform.getPlatformByName('CUDA'),{'Precision':'mixed','DeterministicForces':'true'})
    try:
        context.setPositions(positions);initial=context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        mm.LocalEnergyMinimizer.minimize(context,10,max_iterations)
        state=context.getState(getPositions=True,getEnergy=True,getForces=True)
        out=state.getPositions(asNumpy=True);energy=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        force=state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
        rms=float(np.sqrt(np.sum(force**2)/topology.getNumAtoms()))
        assert np.isfinite(out.value_in_unit(unit.nanometer)).all() and math.isfinite(energy) and energy<=initial+max(1,abs(initial))*1e-6
        return out,{'kind':'Full Amber ff14SB/GBn2 plus heavy restraints and biological chirality wall',
                    'restraint_k_kj_mol_nm2':restraint_k,'chirality_wall_k':chirality_strength,'max_iterations':max_iterations,
                    'initial_energy_kj_mol':initial,'final_energy_kj_mol':energy,
                    'final_physical_energy_kj_mol':context.getState(getEnergy=True,groups={0}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                    'final_guard_energy_kj_mol':context.getState(getEnergy=True,groups={2}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
                    'rms_force_kj_mol_nm':rms,'converged':rms<=10,'seconds':time.perf_counter()-started,'platform':'CUDA mixed'}
    finally:del context,integrator
