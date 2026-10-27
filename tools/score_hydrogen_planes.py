"""Independent checks of stored protein H planarity and trigonal coordination.

Coordinates are never changed. A 1-degree allowance covers PDB coordinate
rounding; this is a screening tolerance, not a claim of force-field accuracy.
This checker does not import the candidate optimiser or its coordinate map.
"""
import math

import numpy as np


def angle(a, b):
    return math.degrees(math.acos(np.clip(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b), -1, 1)))


def plane_deviation(vector, first, second):
    normal = np.cross(first, second)
    if np.linalg.norm(normal) < 1e-10:
        raise ValueError("Degenerate reference plane")
    return math.degrees(math.asin(np.clip(abs(np.dot(vector, normal)) / np.linalg.norm(vector) / np.linalg.norm(normal), 0, 1)))


def trigonal_gap(vectors):
    """Positive deficit means three bond rays occupy less than a full circle."""
    return 360 - sum(angle(vectors[i], vectors[j]) for i, j in ((0, 1), (1, 2), (2, 0)))


def signed_volume_inversion(heavy, hydrogen):
    """Compare oriented heavy/centre and full four-substituent tetrahedra."""
    first = np.linalg.det(heavy)
    second = np.linalg.det(heavy - hydrogen)
    if abs(first) < 1e-8:
        raise ValueError("Degenerate tetrahedral heavy-atom reference")
    return bool(first * second <= 0)


def score_planes(path):
    from openmm import app, unit

    pdb = app.PDBFile(str(path))
    atoms = list(pdb.topology.atoms())
    xyz = np.asarray(pdb.positions.value_in_unit(unit.angstrom))
    adjacent = {a.index: [] for a in atoms}
    for a, b in pdb.topology.bonds():
        adjacent[a.index].append(b.index)
        adjacent[b.index].append(a.index)
    rings = {
        "PHE": {"CD1", "CD2", "CE1", "CE2", "CZ"},
        "TYR": {"CD1", "CD2", "CE1", "CE2"},
        "TRP": {"CD1", "CE3", "CZ2", "CZ3", "CH2", "NE1"},
        "HIS": {"CD2", "CE1", "ND1", "NE2"},
    }
    records = []
    tetrahedral_wrong_side = 0
    signed_inversions = 0
    for atom in atoms:
        hs = [i for i in adjacent[atom.index] if atoms[i].element.symbol == "H"]
        heavy = [i for i in adjacent[atom.index] if atoms[i].element.symbol != "H"]
        if atom.element.symbol == "C" and len(hs) == 1 and len(heavy) == 3:
            vectors = xyz[heavy] - xyz[atom.index]
            normal = np.cross(vectors[0], vectors[1])
            tetrahedral_wrong_side += int(np.dot(normal, vectors[2]) * np.dot(normal, xyz[hs[0]] - xyz[atom.index]) >= 0)
            signed_inversions += int(signed_volume_inversion(vectors, xyz[hs[0]] - xyz[atom.index]))
        if not hs:
            continue
        if atom.name in rings.get(atom.residue.name, set()):
            kind = "aromatic"
        elif atom.name == "N" and len(hs) == 1 and any(atoms[i].name == "C" and atoms[i].residue != atom.residue for i in heavy):
            kind = "backbone_amide"
        elif ((atom.residue.name == "ARG" and atom.name in ("NE", "NH1", "NH2"))
              or (atom.residue.name == "ASN" and atom.name == "ND2")
              or (atom.residue.name == "GLN" and atom.name == "NE2")):
            kind = "sidechain_planar_N"
        else:
            continue
        if len(heavy) == 2:
            first, second = xyz[heavy] - xyz[atom.index]
        elif len(heavy) == 1:
            parent = heavy[0]
            refs = [i for i in adjacent[parent] if i != atom.index and atoms[i].element.symbol != "H"]
            refs.sort(key=lambda i: (atoms[i].element.symbol != "O", atoms[i].name))
            if not refs:
                raise ValueError("Missing planar-group reference")
            first = xyz[parent] - xyz[atom.index]
            second = xyz[refs[0]] - xyz[parent]
        else:
            raise ValueError("Unrecognised planar coordination")
        vectors = xyz[heavy + hs] - xyz[atom.index]
        gap = trigonal_gap(vectors) if len(vectors) == 3 else None
        for h in hs:
            records.append({"kind": kind, "parent": [atom.residue.chain.id, atom.residue.id, atom.name],
                            "hydrogen": atoms[h].name,
                            "plane_deviation_deg": plane_deviation(xyz[h] - xyz[atom.index], first, second),
                            "trigonal_angle_deficit_deg": gap})
    return {"summary": {
        "planar_H_assessed": len(records),
        "planar_H_over_1_deg": sum(r["plane_deviation_deg"] > 1 for r in records),
        "max_planar_H_deviation_deg": max((r["plane_deviation_deg"] for r in records), default=0),
        "inward_planar_H_over_1_deg": sum(r["trigonal_angle_deficit_deg"] is not None and r["trigonal_angle_deficit_deg"] > 1 for r in records),
        "tetrahedral_H_wrong_side": tetrahedral_wrong_side,
        "tetrahedral_H_signed_volume_inversions": signed_inversions,
    }, "planar_H": records}
