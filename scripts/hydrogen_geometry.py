"""Fixed-heavy hydrogen geometry placement for optional pH-7 preparation.

The diagnostic conditioned mode removes only H-independent energies, keeping
GB coupling. Robust mode is a different geometry objective: fixed X-H lengths,
Amber valence targets, bounded H-bond-aware sterics and chirality guards, without
Coulomb or GB energy. It is not an Amber relaxation or an energy-equivalent fix.
"""
import itertools
import math
import time

import numpy as np
import openmm as mm
from openmm import unit


def conditioned_system(original, mobile, constrain=False, robust=False, topology=None, positions=None, steric_k=100000):
    mobile=set(mobile);system=mm.System()
    for i in range(original.getNumParticles()):system.addParticle(original.getParticleMass(i))
    for i in range(original.getNumConstraints()):system.addConstraint(*original.getConstraintParameters(i))
    for force in original.getForces():
        if isinstance(force,mm.HarmonicBondForce):
            new=mm.HarmonicBondForce()
            for i in range(force.getNumBonds()):
                a,b,r,k=force.getBondParameters(i)
                if a in mobile or b in mobile:
                    new.addBond(a,b,r,k)
                    if constrain:system.addConstraint(a,b,r)
            system.addForce(new)
        elif isinstance(force,mm.HarmonicAngleForce):
            new=mm.HarmonicAngleForce()
            guard=mm.CustomAngleForce('0.5*100000*max(0,abs(theta-t)-0.2617993877991494)^2');guard.addPerAngleParameter('t')
            for i in range(force.getNumAngles()):
                values=force.getAngleParameters(i)
                if any(a in mobile for a in values[:3]):
                    a,b,c,t,k=values;new.addAngle(a,b,c,t,k*(10 if robust else 1))
                    if robust:guard.addAngle(a,b,c,[t])
            system.addForce(new)
            if robust:system.addForce(guard)
        elif isinstance(force,mm.PeriodicTorsionForce):
            new=mm.PeriodicTorsionForce()
            for i in range(force.getNumTorsions()):
                values=force.getTorsionParameters(i)
                if any(a in mobile for a in values[:4]):new.addTorsion(*values)
            system.addForce(new)
        elif isinstance(force,mm.NonbondedForce):
            if force.getNonbondedMethod()!=mm.NonbondedForce.CutoffNonPeriodic:
                raise ValueError('Conditioned H experiment requires CutoffNonPeriodic.')
            if force.getNumParticleParameterOffsets() or force.getNumExceptionParameterOffsets():raise ValueError('Parameter offsets unsupported.')
            if force.getUseSwitchingFunction():raise ValueError('Switched nonbonded potentials are unqualified.')
            cutoff=force.getCutoffDistance().value_in_unit(unit.nanometer);dielectric=force.getReactionFieldDielectric()
            krf=(dielectric-1)/(2*dielectric+1)/cutoff**3;crf=3*dielectric/(2*dielectric+1)/cutoff
            if robust:
                steric_scale = 1.0
                new=mm.CustomNonbondedForce(f'0.5*{steric_k}*min(0,r-d)^2;d=select(h1*a2+h2*a1,0.18,{steric_scale}*(radius1+radius2))')
                for key in ['radius','h','a']:new.addPerParticleParameter(key)
                radii={'H':.12,'C':.17,'N':.155,'O':.152,'S':.18}
                adjacent={a.index:[] for a in topology.atoms()}
                for a,b in topology.bonds():adjacent[a.index].append(b);adjacent[b.index].append(a)
                for atom in topology.atoms():
                    symbol=atom.element.symbol
                    donor=symbol=='H' and any(a.element.symbol in ['N','O','S'] for a in adjacent[atom.index])
                    acceptor=symbol in ['O','S'] or (symbol=='N' and atom.residue.name=='HIS' and not any(a.element.symbol=='H' for a in adjacent[atom.index]))
                    new.addParticle([radii[symbol],int(donor),int(acceptor)])
            else:
                new=mm.CustomNonbondedForce(f'138.93545764438198*q1*q2*(1/r+{krf}*r^2-{crf})+4*sqrt(e1*e2)*(s^12-s^6);s=(s1+s2)/(2*r)')
                for name in ['q','s','e']:new.addPerParticleParameter(name)
                for i in range(force.getNumParticles()):new.addParticle(force.getParticleParameters(i))
            new.addInteractionGroup(mobile,set(range(original.getNumParticles())))
            new.setNonbondedMethod(mm.CustomNonbondedForce.CutoffNonPeriodic);new.setCutoffDistance(force.getCutoffDistance())
            new.setUseSwitchingFunction(force.getUseSwitchingFunction());new.setSwitchingDistance(force.getSwitchingDistance())
            exceptions=mm.CustomBondForce('138.93545764438198*q/r+4*e*((s/r)^12-(s/r)^6)')
            for name in ['q','s','e']:exceptions.addPerBondParameter(name)
            for i in range(force.getNumExceptions()):
                a,b,q,s,e=force.getExceptionParameters(i);new.addExclusion(a,b)
                if not robust and (a in mobile or b in mobile):exceptions.addBond(a,b,[q,s,e])
            system.addForce(new)
            if not robust:system.addForce(exceptions)
        elif isinstance(force,mm.CMMotionRemover):pass
        elif isinstance(force,mm.CustomGBForce):
            if not robust:system.addForce(mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(force)))
        else:raise ValueError('Unqualified force in hydrogen system: '+type(force).__name__)
    if robust:
        xyz=np.array(positions.value_in_unit(unit.nanometer));atoms=list(topology.atoms());adj={a.index:[] for a in atoms}
        for a,b in topology.bonds():adj[a.index].append(b.index);adj[b.index].append(a.index)
        wall=mm.CustomCompoundBondForce(2,'0.5*10000000*min(0,nx*(x1-x2)+ny*(y1-y2)+nz*(z1-z2)-0.015)^2')
        for key in ['nx','ny','nz']:wall.addPerBondParameter(key)
        for a in atoms:
            hs=[i for i in adj[a.index] if i in mobile];heavy=[i for i in adj[a.index] if i not in mobile]
            if a.element.symbol=='C' and len(hs)==1 and len(heavy)==3:
                vectors=xyz[heavy]-xyz[a.index];normal=np.cross(vectors[0],vectors[1]);normal/=np.linalg.norm(normal)
                normal*=-np.sign(np.dot(normal,vectors[2]))
                wall.addBond([hs[0],a.index],normal.tolist())
        system.addForce(wall)
    return system


def initial_positions(system,topology,positions,mobile):
    xyz=np.array(positions.value_in_unit(unit.nanometer),copy=True);mobile=set(mobile);atoms=list(topology.atoms())
    neighbours={i:[] for i in range(len(atoms))}
    for a,b in topology.bonds():neighbours[a.index].append(b.index);neighbours[b.index].append(a.index)
    lengths={};angle_targets={}
    for force in system.getForces():
        if isinstance(force,mm.HarmonicBondForce):
            for j in range(force.getNumBonds()):
                a,b,r,k=force.getBondParameters(j)
                if a in mobile or b in mobile:lengths[a if a in mobile else b]=r.value_in_unit(unit.nanometer)
        if isinstance(force,mm.HarmonicAngleForce):
            for j in range(force.getNumAngles()):
                a,b,c,t,k=force.getAngleParameters(j);angle_targets[(a,b,c)]=t.value_in_unit(unit.radian)
    for parent in range(len(atoms)):
        hs=sorted(i for i in neighbours[parent] if i in mobile)
        if not hs:continue
        heavy=[i for i in neighbours[parent] if i not in mobile]
        if not heavy:raise ValueError('Hydrogen parent without a heavy-atom reference.')
        u=np.array([xyz[i]-xyz[parent] for i in heavy]);u/=np.linalg.norm(u,axis=1)[:,None]
        axis=-u.sum(axis=0);axis/=np.linalg.norm(axis)
        normal=np.cross(u[0],u[1]) if len(heavy)>1 else np.cross(axis,np.eye(3)[np.argmin(abs(axis))])
        if np.linalg.norm(normal)<1e-8:normal=np.cross(axis,np.eye(3)[np.argmin(abs(axis))])
        normal/=np.linalg.norm(normal)
        if len(hs)==1:
            direction=axis
            if len(heavy)==1:
                theta=angle_targets.get((hs[0],parent,heavy[0]),angle_targets.get((heavy[0],parent,hs[0]),math.radians(109.5)))
                direction=math.cos(theta)*u[0]+math.sin(theta)*normal
            if len(heavy)==3 and np.dot(normal,direction)*np.dot(normal,u[2])>=0:direction-=2*np.dot(direction,normal)*normal
            directions=[direction]
        elif len(hs)==2:
            if len(heavy)>1:
                theta=[angle_targets.get((hs[0],parent,i),angle_targets.get((i,parent,hs[0]),math.radians(109.5))) for i in heavy]
                projection=np.linalg.lstsq(u,np.cos(theta),rcond=None)[0]
                projection/=max(1,np.linalg.norm(projection)/.95)
                width=math.sqrt(max(0,1-np.dot(projection,projection)))
                directions=[projection+width*normal,projection-width*normal]
            else:
                theta=angle_targets.get((hs[0],parent,heavy[0]),angle_targets.get((heavy[0],parent,hs[0]),math.radians(120)))
                directions=[math.cos(theta)*u[0]+s*math.sin(theta)*normal for s in [-1,1]]
        elif len(hs)==3:
            other=np.cross(axis,normal)
            directions=[axis/3+math.sqrt(8/9)*(math.cos(t)*normal+math.sin(t)*other) for t in [0,2*math.pi/3,4*math.pi/3]]
        else:raise ValueError('Unsupported hydrogen valence.')
        for h,direction in zip(hs,directions):xyz[h]=xyz[parent]+lengths[h]*direction/np.linalg.norm(direction)
    return xyz*unit.nanometer


def optimise(system,topology,positions,mobile,platform,properties):
    """Bounded steric back-off; geometry remains the acceptance gate.

    Only the bounded steric weight changes. Topology, protonation, fixed heavy
    coordinates, platform and geometry thresholds remain the same. Every
    attempted weight and rejected geometry is recorded; no silent fallback.
    """
    weights = [500000., 200000., 100000.]
    attempts = []
    started = time.perf_counter()
    for weight in weights:
        optimise.last_report = {}
        try:
            result = _optimise_once(system, topology, positions, mobile, platform, properties, steric_k=weight)
        except ValueError as exc:
            attempts.append({'steric_k': weight, 'accepted': False, 'error': str(exc),
                             'quality': optimise.last_report.get('quality')})
            if not str(exc).startswith('Hydrogen geometry did not meet'):
                raise
        else:
            attempts.append({'steric_k': weight, 'accepted': True, 'quality': optimise.last_report['quality']})
            optimise.last_report.update(steric_schedule=weights, steric_attempts=attempts,
                                        seconds=time.perf_counter()-started)
            return result
    optimise.last_report.update(steric_schedule=weights, steric_attempts=attempts)
    raise ValueError('All configured steric weights failed the unchanged hydrogen geometry checks.')


def _optimise_once(system,topology,positions,mobile,platform,properties, *, steric_k=500000.):
    started=time.perf_counter();mode='robust_planar'
    if platform is None:raise ValueError('An explicit CPU or CUDA platform is required.')
    # OpenMM disallows constraints to massless parents. Fixed-bond mode instead
    # optimises angular coordinates, with an analytic Cartesian chain rule.
    robust=mode in ('robust','robust_planar')
    reduced=conditioned_system(system,mobile,robust=robust,topology=topology,positions=positions,steric_k=steric_k)
    positions=initial_positions(system,topology,positions,mobile)
    integrator=mm.VerletIntegrator(0)
    if properties is None:properties={'Threads':'1','DeterministicForces':'true'} if platform.getName()=='CPU' else {}
    context=mm.Context(reduced,integrator,platform,properties)
    try:
        context.setPositions(positions)
        if mode in ('constrained','robust','robust_planar'):
            from scipy.optimize import minimize
            xyz=np.array(positions.value_in_unit(unit.nanometer));mobile=np.array(mobile,dtype=int)
            parent={}
            for a,b in topology.bonds():
                if a.index in mobile:parent[a.index]=b.index
                elif b.index in mobile:parent[b.index]=a.index
            parents=np.array([parent[h] for h in mobile]);delta=xyz[mobile]-xyz[parents]
            lengths=np.linalg.norm(delta,axis=1);u=delta/lengths[:,None]
            angles=np.column_stack([np.arccos(np.clip(u[:,2],-1,1)),np.arctan2(u[:,1],u[:,0])])
            frames=planar_frames(topology,xyz,mobile) if mode=='robust_planar' else {}
            angle_bounds=[(None,None)]*angles.size
            for index,(e1,e2) in frames.items():
                angles[index]=[math.atan2(np.dot(u[index],e2),np.dot(u[index],e1)),0]
            for parent_index in set(parents):
                group=[i for i,p in enumerate(parents) if p==parent_index and i in frames]
                if len(group)==2:
                    # Planar NH2 groups start with distinct 120-degree rays;
                    # projecting arbitrary 3D initial Hs can collapse both rays.
                    angles[group[0]]=[-2*math.pi/3,0];angles[group[1]]=[2*math.pi/3,0]
            if mode=='robust_planar':
                sectors=planar_sectors(system,topology,xyz,mobile,frames)
                for i,(lo,hi) in sectors.items():
                    angle_bounds[2*i]=(lo,hi);angles[i,0]=np.clip(angles[i,0]%(2*math.pi),lo,hi)
            def evaluate(vector):
                mapped,dt,dp=coordinate_map(vector,xyz[parents],lengths,frames);xyz[mobile]=mapped
                context.setPositions(xyz*unit.nanometer);state=context.getState(getEnergy=True,getForces=True)
                energy=state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                forces=state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)[mobile]
                gradient=-np.column_stack([np.sum(forces*dt,axis=1),np.sum(forces*dp,axis=1)])
                if not np.isfinite(energy) or not np.isfinite(gradient).all():raise ValueError('Non-finite hydrogen objective.')
                return energy,gradient.ravel()
            maxiter=2000
            gradient_tolerance=5.0
            if not math.isfinite(gradient_tolerance) or gradient_tolerance<=0:raise ValueError('Gradient tolerance must be finite and positive.')
            result=minimize(evaluate,angles.ravel(),method='L-BFGS-B',jac=True,bounds=angle_bounds if mode=='robust_planar' else None,options={'maxiter':maxiter,'maxls':50,'ftol':1e-12,'gtol':gradient_tolerance})
            evaluate(result.x)
            optimise.last_report={'mode':mode,'iterations':result.nit,'evaluations':result.nfev,'success':bool(result.success),'message':str(result.message),'max_angular_gradient':float(max(abs(result.jac)))}
        else:
            mm.LocalEnergyMinimizer.minimize(context,1.0,1000)
            optimise.last_report={'mode':mode,'max_iterations':1000}
        result=context.getState(getPositions=True).getPositions()
        optimise.last_report['seconds']=time.perf_counter()-started
        optimise.last_report['parameters']={'steric_scale':1.0,'steric_k':steric_k,'iteration_limit':2000,'gradient_tolerance':5.0}
        optimise.last_report['quality']=validate_geometry(system,topology,result,mobile)
        optimise.last_report['objective']='Amber valence angles x10 and torsions, exact X-H lengths, bounded sterics, explicit chiral wall; Coulomb/GB omitted' if robust else 'H-dependent full Amber objective, constant heavy-only terms removed'
        if mode=='robust_planar':optimise.last_report['planar_H_count']=len(frames)
        if robust and not optimise.last_report['quality']['passed']:
            raise ValueError('Hydrogen geometry did not meet the fixed-heavy preparation checks: '+str(optimise.last_report['quality']))
        return result
    finally:del context,integrator


def spherical_map(vector, origins, lengths):
    theta,phi=np.asarray(vector).reshape(-1,2).T;st,ct,sp,cp=np.sin(theta),np.cos(theta),np.sin(phi),np.cos(phi)
    positions=origins+lengths[:,None]*np.column_stack([st*cp,st*sp,ct])
    dt=np.column_stack([ct*cp,ct*sp,-st])*lengths[:,None]
    dp=np.column_stack([-st*sp,st*cp,np.zeros(len(st))])*lengths[:,None]
    return positions,dt,dp


def coordinate_map(vector,origins,lengths,frames):
    positions,dt,dp=spherical_map(vector,origins,lengths);theta=np.asarray(vector).reshape(-1,2)[:,0]
    for index,(e1,e2) in frames.items():
        c,s=math.cos(theta[index]),math.sin(theta[index])
        positions[index]=origins[index]+lengths[index]*(c*e1+s*e2)
        dt[index]=lengths[index]*(-s*e1+c*e2);dp[index]=0
    return positions,dt,dp


def planar_frames(topology,xyz,mobile):
    atoms=list(topology.atoms());adj={a.index:[] for a in atoms};indices={h:i for i,h in enumerate(mobile)}
    for a,b in topology.bonds():adj[a.index].append(b.index);adj[b.index].append(a.index)
    aromatic={'PHE':{'CD1','CD2','CE1','CE2','CZ'},'TYR':{'CD1','CD2','CE1','CE2'},'TRP':{'CD1','CE3','CZ2','CZ3','CH2','NE1'},'HIS':{'CD2','CE1','ND1','NE2'}}
    frames={}
    for a in atoms:
        hs=[h for h in adj[a.index] if h in indices];heavy=[i for i in adj[a.index] if i not in indices]
        if not hs:continue
        is_ring=a.name in aromatic.get(a.residue.name,set())
        backbone=a.name=='N' and len(hs)==1 and any(atoms[i].name=='C' and atoms[i].residue!=a.residue for i in heavy)
        side=(a.residue.name=='ARG' and a.name in ('NE','NH1','NH2')) or (a.residue.name=='ASN' and a.name=='ND2') or (a.residue.name=='GLN' and a.name=='NE2')
        if not (is_ring or backbone or side):continue
        if len(heavy)==2:
            u,v=xyz[heavy]-xyz[a.index]
        elif len(heavy)==1:
            first=heavy[0];refs=[i for i in adj[first] if i!=a.index and i not in indices]
            refs.sort(key=lambda i:(atoms[i].element.symbol!='O',atoms[i].name))
            if not refs:raise ValueError('No heavy-atom reference for planar amide hydrogens.')
            u=xyz[first]-xyz[a.index];v=xyz[refs[0]]-xyz[first]
        else:raise ValueError('Unsupported planar H coordination.')
        e1=u/np.linalg.norm(u);normal=np.cross(u,v)
        if np.linalg.norm(normal)<1e-8:raise ValueError('Degenerate heavy-atom plane; hydrogen placement cannot repair its geometry.')
        normal/=np.linalg.norm(normal);e2=np.cross(normal,e1)
        for h in hs:frames[indices[h]]=(e1,e2)
    return frames


def planar_sector(delta,theta1,theta2,limit=math.radians(29.5)):
    """Intersection in the outward trigonal sector, avoiding inward H rays."""
    lo=max(math.pi,2*math.pi-theta1-limit,delta+theta2-limit)
    hi=min(math.pi+delta,2*math.pi-theta1+limit,delta+theta2+limit)
    if lo>hi+1e-10:raise ValueError('Frozen heavy-atom geometry is incompatible with a planar, outward hydrogen and the angle-error limit; explicit heavy-atom repair is required.')
    return lo,hi


def planar_sectors(system,topology,xyz,mobile,frames):
    atoms=list(topology.atoms());mobile=list(mobile);mobile_set=set(mobile);adj={a.index:[] for a in atoms}
    for a,b in topology.bonds():adj[a.index].append(b.index);adj[b.index].append(a.index)
    targets={}
    for f in system.getForces():
        if isinstance(f,mm.HarmonicAngleForce):
            for j in range(f.getNumAngles()):
                a,b,c,t,k=f.getAngleParameters(j);targets[(a,b,c)]=t.value_in_unit(unit.radian)
    result={}
    for i in frames:
        h=mobile[i];p=adj[h][0];heavy=[j for j in adj[p] if j not in mobile_set];hs=[j for j in adj[p] if j in mobile_set]
        if len(heavy)==2 and len(hs)==1:
            u,v=xyz[heavy]-xyz[p];delta=math.acos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1))
            ts=[targets.get((h,p,j),targets.get((j,p,h))) for j in heavy]
            result[i]=planar_sector(delta,*ts)
    return result


def validate_geometry(original,topology,positions,mobile):
    xyz=np.array(positions.value_in_unit(unit.nanometer));mobile=set(mobile);bond_error=[];angle_error=[]
    for force in original.getForces():
        if isinstance(force,mm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                a,b,r,k=force.getBondParameters(i)
                if a in mobile or b in mobile:bond_error.append(abs(np.linalg.norm(xyz[a]-xyz[b])-r.value_in_unit(unit.nanometer))*10)
        elif isinstance(force,mm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                a,b,c,t,k=force.getAngleParameters(i)
                if a in mobile or c in mobile:
                    u,v=xyz[a]-xyz[b],xyz[c]-xyz[b]
                    angle_error.append(abs(math.degrees(math.acos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1)))-t.value_in_unit(unit.degree)))
    atoms=list(topology.atoms());adj={a.index:[] for a in atoms}
    for a,b in topology.bonds():adj[a.index].append(b.index);adj[b.index].append(a.index)
    wrong=0
    for a in atoms:
        hs=[i for i in adj[a.index] if i in mobile];heavy=[i for i in adj[a.index] if i not in mobile]
        if a.element.symbol=='C' and len(hs)==1 and len(heavy)==3:
            u=xyz[heavy]-xyz[a.index];n=np.cross(u[0],u[1]);wrong+=int(np.dot(n,u[2])*np.dot(n,xyz[hs[0]]-xyz[a.index])>=0)
    result={'max_H_bond_error_A':float(max(bond_error,default=0)),'max_H_angle_error_deg':float(max(angle_error,default=0)),
            'H_bond_errors_over_0_10_A':int(np.sum(np.array(bond_error)>.1)), 'H_angle_errors_over_30_deg':int(np.sum(np.array(angle_error)>30)),
            'wrong_side_tetrahedral_H':wrong}
    # Bounds may land exactly on 29.5 degrees; do not reject round-off at
    # 29.500000000000014. The independent stored-PDB gate remains 30 degrees.
    result['passed']=geometry_passes(result, np.isfinite(xyz).all())
    return result


def geometry_passes(result, finite):
    return bool(finite and result['max_H_bond_error_A'] < .05
                and result['max_H_angle_error_deg'] <= 29.5 + 1e-8
                and result['wrong_side_tetrahedral_H'] == 0)
