"""Independent stored-PDB heavy geometry, contact and stereochemistry checks."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from openmm import app,unit

sys.path.insert(0,'/opt/abodybuilder3/scripts')
import structure_preparation as sp
from diagnose_heavy_rescue import heavy_metrics,heavy_error_sets


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


def compare(before,after):
    a,b=app.PDBFile(str(before)),app.PDBFile(str(after));assert sp._keys(a.topology)==sp._keys(b.topology)
    av=biological_volumes(a.topology,a.positions);bv=biological_volumes(b.topology,b.positions)
    ac=heavy_contacts(a.topology,a.positions);bc=heavy_contacts(b.topology,b.positions)
    system=sp._forcefield().createSystem(a.topology,nonbondedMethod=app.NoCutoff,constraints=None)
    ab,aa=heavy_error_sets(system,a.topology,a.positions);bb,ba=heavy_error_sets(system,b.topology,b.positions)
    return {'before_heavy':heavy_metrics(system,a.topology,a.positions),'after_heavy':heavy_metrics(system,b.topology,b.positions),
            'new_heavy_bond_flags':len(bb-ab),'new_heavy_angle_flags':len(ba-aa),
            'biological_stereocentre_inversions':sum(av[k]*bv[k]<=0 for k in av),
            'before_heavy_contact_pairs':len(ac),'after_heavy_contact_pairs':len(bc),
            'new_heavy_contact_pairs':len(bc-ac),'resolved_heavy_contact_pairs':len(ac-bc)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--screen',default='screen_v4');args=parser.parse_args();rows=[]
    d=json.loads((args.root/(args.screen+'_271026.json')).read_text())
    for row in d['rows']:
        path=args.root/args.screen/row['target']/row['condition']/'rescued.prepared.pdb'
        if not path.exists():continue
        value=compare(args.root/'diagnosis'/(row['target']+'.initial_hydrogen_topology.pdb'),path)
        rows.append({'target':row['target'],'group':row['group'],'condition':row['condition'],**value})
        if row['group']=='rejected':print(row['target'],row['condition'],'new bond/angle/contact',value['new_heavy_bond_flags'],value['new_heavy_angle_flags'],value['new_heavy_contact_pairs'],'bio inv',value['biological_stereocentre_inversions'],flush=True)
    (args.root/(args.screen+'_stored_geometry_271026.json')).write_text(json.dumps({'completed':True,'protocol':__doc__,'rows':rows},indent=2)+'\n')


if __name__=='__main__':main()
