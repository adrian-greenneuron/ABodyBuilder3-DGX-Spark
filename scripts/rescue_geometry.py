"""Paired geometry gates used by rejection-only heavy-atom rescue."""


import math
import numpy as np
import openmm as mm
from openmm import unit
from scipy.spatial import cKDTree
import hydrogen_geometry as hg
import structure_preparation as sp


def heavy_metrics(system, topology, positions):
    xyz=sp._positions(positions)*10
    atoms=list(topology.atoms());hs={a.index for a in atoms if a.element.symbol=='H'}
    bonds=[];angles=[]
    for force in system.getForces():
        if isinstance(force,mm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                a,b,r,k=force.getBondParameters(i)
                if a in hs or b in hs:continue
                error=np.linalg.norm(xyz[a]-xyz[b])-r.value_in_unit(unit.angstrom)
                bonds.append(float(error))
        elif isinstance(force,mm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                a,b,c,t,k=force.getAngleParameters(i)
                if hs.intersection((a,b,c)):continue
                u,v=xyz[a]-xyz[b],xyz[c]-xyz[b]
                actual=math.degrees(math.acos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1)))
                angles.append(actual-t.value_in_unit(unit.degree))
    return {'heavy_bonds':len(bonds),'heavy_bond_errors_over_0_10_A':sum(abs(e)>.1 for e in bonds),
            'max_heavy_bond_error_A':max(map(abs,bonds),default=0),
            'heavy_angles':len(angles),'heavy_angle_errors_over_20_deg':sum(abs(e)>20 for e in angles),
            'max_heavy_angle_error_deg':max(map(abs,angles),default=0)}


def failing_centres(system, topology, positions):
    xyz=sp._positions(positions);atoms=list(topology.atoms());keys=sp._keys(topology)
    mobile=[a.index for a in atoms if a.element.symbol=='H'];ms=set(mobile)
    frames=hg.planar_frames(topology,xyz,mobile)
    adjacent={a.index:[] for a in atoms}
    for a,b in topology.bonds():adjacent[a.index].append(b.index);adjacent[b.index].append(a.index)
    targets={};bond_targets={}
    for force in system.getForces():
        if isinstance(force,mm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                a,b,r,k=force.getBondParameters(i);bond_targets[tuple(sorted((a,b)))]=r.value_in_unit(unit.angstrom)
        if isinstance(force,mm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                a,b,c,t,k=force.getAngleParameters(i);targets[a,b,c]=t.value_in_unit(unit.radian)
    records=[]
    for index in frames:
        h=mobile[index];p=adjacent[h][0];heavy=[i for i in adjacent[p] if i not in ms]
        if len(heavy)!=2 or len([i for i in adjacent[p] if i in ms])!=1:continue
        u,v=xyz[heavy]-xyz[p];delta=math.acos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1))
        ts=[targets.get((h,p,i),targets.get((i,p,h))) for i in heavy]
        try:hg.planar_sector(delta,*ts)
        except ValueError:
            records.append({'parent_index':p,'neighbour_indices':heavy,'parent':keys[p],
                            'neighbours':[keys[i] for i in heavy], 'heavy_angle_deg':math.degrees(delta),
                            'H_angle_targets_deg':list(map(math.degrees,ts)),
                            'heavy_angle_target_deg':math.degrees(targets.get((heavy[0],p,heavy[1]),targets.get((heavy[1],p,heavy[0])))),
                            'heavy_bond_targets_A':[bond_targets[tuple(sorted((p,i)))] for i in heavy],
                            'heavy_distances_A':[float(np.linalg.norm(xyz[i]-xyz[p])*10) for i in heavy]})
    return records


def heavy_error_sets(system,topology,positions):
    xyz=sp._positions(positions)*10;heavy={a.index for a in topology.atoms() if a.element.symbol!='H'};bonds=set();angles=set()
    for force in system.getForces():
        if isinstance(force,mm.HarmonicBondForce):
            for i in range(force.getNumBonds()):
                a,b,r,k=force.getBondParameters(i)
                if {a,b}<=heavy and abs(np.linalg.norm(xyz[a]-xyz[b])-r.value_in_unit(unit.angstrom))>.1:bonds.add(tuple(sorted((a,b))))
        elif isinstance(force,mm.HarmonicAngleForce):
            for i in range(force.getNumAngles()):
                a,b,c,t,k=force.getAngleParameters(i)
                if not {a,b,c}<=heavy:continue
                u,v=xyz[a]-xyz[b],xyz[c]-xyz[b];actual=math.degrees(math.acos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1)))
                if abs(actual-t.value_in_unit(unit.degree))>20:angles.add(min((a,b,c),(c,b,a)))
    return bonds,angles


def assess_original_motifs(bad,positions):
    xyz=sp._positions(positions)*10;rows=[]
    for motif in bad:
        p=motif['parent_index'];a,b=motif['neighbour_indices'];u,v=xyz[a]-xyz[p],xyz[b]-xyz[p]
        angle=math.degrees(math.acos(np.clip(np.dot(u,v)/np.linalg.norm(u)/np.linalg.norm(v),-1,1)))
        bond_errors=[float(np.linalg.norm(xyz[i]-xyz[p])-target) for i,target in zip([a,b],motif['heavy_bond_targets_A'])]
        error=angle-motif['heavy_angle_target_deg']
        rows.append({'heavy_angle_deg':angle,'angle_error_deg':error,'bond_errors_A':bond_errors,'restored':abs(error)<=10 and max(map(abs,bond_errors))<=.05})
    return rows


def heavy_contacts(topology,positions):
    atoms=list(topology.atoms());xyz=sp._positions(positions)*10;adj={a.index:set() for a in atoms}
    for a,b in topology.bonds():adj[a.index].add(b.index);adj[b.index].add(a.index)
    excluded={}
    for a in atoms:
        if a.element.symbol=='H':continue
        near={a.index};front={a.index}
        for _ in range(3):front=set().union(*(adj[i] for i in front))-near;near|=front
        excluded[a.index]=near
    radii={'C':1.7,'N':1.55,'O':1.52,'S':1.8};pairs=set()
    for a,b in cKDTree(xyz).query_pairs(2.6):
        if a not in excluded or b not in excluded or b in excluded[a]:continue
        if np.linalg.norm(xyz[a]-xyz[b])<.7*(radii[atoms[a].element.symbol]+radii[atoms[b].element.symbol]):pairs.add((a,b))
    return pairs


def biological_volumes(topology,positions):
    xyz=sp._positions(positions);values={}
    for residue in topology.residues():
        atoms={a.name:xyz[a.index] for a in residue.atoms()}
        sites=[('CA',['N','C','CB'])] if residue.name!='GLY' else []
        if residue.name=='ILE':sites.append(('CB',['CA','CG1','CG2']))
        if residue.name=='THR':sites.append(('CB',['CA','OG1','CG2']))
        for centre,others in sites:
            if all(a in atoms for a in [centre]+others):
                values[(residue.chain.id,residue.id,residue.name,centre)]=float(np.linalg.det(np.array([atoms[a]-atoms[centre] for a in others])))
    return values


def disulfides(top):
    keys=sp._keys(top)
    return sorted(tuple(sorted((keys[a.index],keys[b.index]))) for a,b in top.bonds() if a.name==b.name=='SG')


def heavy_gate(system,top,before,after,bad):
    bb,ba=heavy_error_sets(system,top,before);ab,aa=heavy_error_sets(system,top,after)
    bc=heavy_contacts(top,before);ac=heavy_contacts(top,after)
    bv=biological_volumes(top,before);av=biological_volumes(top,after)
    motifs=assess_original_motifs(bad,after)
    result={'new_heavy_bond_flags':len(ab-bb),'new_heavy_angle_flags':len(aa-ba),
            'new_heavy_contact_pairs':len(ac-bc),'resolved_heavy_contact_pairs':len(bc-ac),
            'biological_stereocentre_inversions':sum(bv[k]*av[k]<=0 for k in bv),
            'original_motifs':motifs,'original_motifs_restored':all(m['restored'] for m in motifs)}
    result['passed']=result['original_motifs_restored'] and not any(result[k] for k in (
        'new_heavy_bond_flags','new_heavy_angle_flags','new_heavy_contact_pairs','biological_stereocentre_inversions'))
    return result


def displacement(top,before,after,keys):
    atoms=list(top.atoms());tk=sp._keys(top)
    heavy=[a.index for a in atoms if a.element.symbol!='H' and tk[a.index] in keys]
    ca=[i for i in heavy if atoms[i].name=='CA']
    delta=np.linalg.norm((sp._positions(after)-sp._positions(before))*10,axis=1)
    return {'original_heavy_count':len(heavy),'heavy_rms_A':float(np.sqrt(np.mean(delta[heavy]**2))),
            'heavy_max_A':float(max(delta[heavy])),'heavy_moved_over_0_01_A':int(np.sum(delta[heavy]>.01)),
            'CA_rms_A':float(np.sqrt(np.mean(delta[ca]**2))),'CA_max_A':float(max(delta[ca]))}
