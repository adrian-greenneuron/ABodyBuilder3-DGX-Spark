"""Score fixed prepared coordinates against bonds/angles and local geometry.

No minimisation, protonation changes, output rewriting or energy ranking.
Detailed per-residue lists stay local; published summaries contain counts.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np


def alpha_h_geometry(positions):
    """An alpha H should oppose CB across the N-CA-C plane."""
    n,c,cb,h=(np.asarray(positions[key],dtype=float)-np.asarray(positions["CA"],dtype=float) for key in ("N","C","CB","HA"))
    normal=np.cross(n,c);length=np.linalg.norm(normal)
    if length<1e-8:return {"assessable":False}
    normal/=length;cb_side=float(np.dot(normal,cb));h_side=float(np.dot(normal,h))
    if abs(cb_side)<.05:return {"assessable":False,"cb_height_angstrom":cb_side,"h_height_angstrom":h_side}
    return {"assessable":True,"wrong_side":cb_side*h_side>=0,
            "near_plane":abs(h_side)<.05,"cb_height_angstrom":cb_side,"h_height_angstrom":h_side}


def score(path):
    from scipy.spatial import cKDTree
    import openmm as mm
    from openmm import app,unit
    sys.path.insert(0,"/opt/abodybuilder3/scripts")
    import structure_preparation as sp
    pdb=app.PDBFile(str(path));top=pdb.topology;atoms=list(top.atoms());keys=sp._keys(top)
    xyz=sp._positions(pdb.positions)*10;assert np.isfinite(xyz).all()
    hydrogens={atom.index for atom in atoms if atom.element.symbol=="H"}
    system=sp._forcefield().createSystem(top,nonbondedMethod=app.CutoffNonPeriodic,constraints=None,rigidWater=False)
    bonds=[];angles=[]
    for force in system.getForces():
        if isinstance(force,mm.HarmonicBondForce):
            for index in range(force.getNumBonds()):
                a,b,target,k=force.getBondParameters(index)
                if a in hydrogens or b in hydrogens:
                    actual=float(np.linalg.norm(xyz[a]-xyz[b]));error=actual-target.value_in_unit(unit.angstrom)
                    bonds.append({"atoms":[keys[a],keys[b]],"error_angstrom":float(error),"length_angstrom":actual})
        elif isinstance(force,mm.HarmonicAngleForce):
            for index in range(force.getNumAngles()):
                a,b,c,target,k=force.getAngleParameters(index)
                if any(x in hydrogens for x in [a,b,c]):
                    u=xyz[a]-xyz[b];v=xyz[c]-xyz[b];cosine=np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v));actual=math.degrees(math.acos(np.clip(cosine,-1,1)))
                    angles.append({"atoms":[keys[a],keys[b],keys[c]],"error_degrees":actual-target.value_in_unit(unit.degree)})
    alpha=[]
    for residue in top.residues():
        group={a.name:xyz[a.index] for a in residue.atoms()}
        if residue.name!="GLY" and all(key in group for key in ["N","CA","C","CB","HA"]):
            alpha.append({"residue":[residue.chain.id,residue.id,residue.name],**alpha_h_geometry(group)})
    neighbours={a.index:set() for a in atoms}
    for a,b in top.bonds():neighbours[a.index].add(b.index);neighbours[b.index].add(a.index)
    exclusions={}
    for atom in hydrogens:
        near={atom};frontier={atom}
        for _ in range(3):frontier=set().union(*(neighbours[x] for x in frontier))-near;near|=frontier
        exclusions[atom]=near
    radii={"H":1.2,"C":1.7,"N":1.55,"O":1.52,"S":1.8};clashes=[]
    for a,b in cKDTree(xyz).query_pairs(2.1):
        if not (a in hydrogens or b in hydrogens):continue
        h=a if a in hydrogens else b;other=b if h==a else a
        if other in exclusions[h]:continue
        threshold=.7*(radii.get(atoms[a].element.symbol,1.7)+radii.get(atoms[b].element.symbol,1.7));distance=float(np.linalg.norm(xyz[a]-xyz[b]))
        if distance<threshold:clashes.append({"atoms":[keys[a],keys[b]],"distance_angstrom":distance,"threshold_angstrom":threshold})
    errors=np.array([abs(x["error_angstrom"]) for x in bonds]);angle_errors=np.array([abs(x["error_degrees"]) for x in angles])
    summary={"atoms":len(atoms),"hydrogens":len(hydrogens),"H_bonds":len(bonds),
             "H_bond_errors_over_0_05_A":int(np.sum(errors>.05)),"H_bond_errors_over_0_10_A":int(np.sum(errors>.1)),
             "max_H_bond_error_A":float(errors.max(initial=0)),"H_angles":len(angles),
             "H_angle_errors_over_20_deg":int(np.sum(angle_errors>20)),"H_angle_errors_over_30_deg":int(np.sum(angle_errors>30)),
             "max_H_angle_error_deg":float(angle_errors.max(initial=0)),
             "alpha_H_assessable":sum(x["assessable"] for x in alpha),"alpha_H_wrong_side":sum(x.get("wrong_side",False) for x in alpha),
             "alpha_H_near_plane":sum(x.get("near_plane",False) for x in alpha),"severe_H_clashes":len(clashes)}
    return {"summary":summary,"alpha_H":alpha,"bad_bonds":[x for x in bonds if abs(x["error_angstrom"])>.05],
            "bad_angles":[x for x in angles if abs(x["error_degrees"])>20],"severe_H_clashes":clashes}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--study",type=Path,required=True);parser.add_argument("--replicates",type=int,nargs="+",default=[1]);args=parser.parse_args()
    d=json.loads((args.study/"cli_measurements_261026.json").read_text());rows=[]
    for case in d["rows"]:
        if case["replicate"] not in args.replicates:continue
        directory=args.study/"cli"/case["label"]/"pdbs"
        for path in sorted(directory.glob("*.prepared.pdb")):
            value=score(path);record={"label":case["label"],"route":case["route"],"condition":case["condition"],"target":path.name,"replicate":case["replicate"],**value["summary"]};rows.append(record)
            (path.parent/(path.stem+".geometry.json")).write_text(json.dumps(value,indent=2)+"\n")
        print("GEOMETRY",case["label"],flush=True)
    (args.study/"chemical_geometry_261026.json").write_text(json.dumps({"completed":True,"rows":rows,"protocol":"Fixed PDB geometry against Amber bond/angle targets, alpha-H opposite-CB check and H-involving severe overlaps; see source for thresholds."},indent=2)+"\n")


if __name__=="__main__":main()
