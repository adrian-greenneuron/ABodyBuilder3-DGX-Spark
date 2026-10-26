"""Check hydrogen permutations and fixed-protocol energies without minimising.

Match interchangeable H atoms only when bonded to the same heavy atom. This
diagnostic does not change names, coordinates, protonation or any saved output.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import openmm as mm
from openmm import app, unit

sys.path.insert(0, "/opt/abodybuilder3/scripts")
import structure_preparation as sp


def load(path):
    pdb = app.PDBFile(str(path))
    keys = sp._keys(pdb.topology)
    coordinates = sp._positions(pdb.positions) * 10
    atoms = list(pdb.topology.atoms())
    hydrogen_groups = {}
    for first, second in pdb.topology.bonds():
        if first.element.symbol == "H":
            first, second = second, first
        if second.element.symbol == "H":
            assert first.element.symbol != "H"
            hydrogen_groups.setdefault(keys[first.index], []).append(keys[second.index])
    assert sum(map(len, hydrogen_groups.values())) == sum(atom.element.symbol == "H" for atom in atoms)
    return pdb, dict(zip(keys, coordinates)), hydrogen_groups


def energy(pdb):
    system = sp._forcefield().createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None)
    integrator = mm.VerletIntegrator(.001*unit.picoseconds)
    platform = mm.Platform.getPlatformByName("CUDA")
    context = mm.Context(system, integrator, platform,
                         {"Precision":"double", "DeterministicForces":"true"})
    try:
        context.setPositions(pdb.positions)
        state = context.getState(getEnergy=True, getForces=True)
        value = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole/unit.nanometer)
        hydrogens = [atom.index for atom in pdb.topology.atoms() if atom.element.symbol == "H"]
        assert np.isfinite(value) and np.isfinite(forces).all()
        return {"potential_energy_kj_mol":value,
                "hydrogen_force_rms_kj_mol_nm":float(np.sqrt(np.mean(np.sum(forces[hydrogens]**2,axis=1))))}
    finally:
        del context, integrator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    args = parser.parse_args()
    root = args.study
    measurements = json.loads((root/"cli_measurements_261026.json").read_text())
    records = []
    energy_cache = {}
    for route in ("openmm", "reduce"):
        reference = next(row for row in measurements["rows"] if row["route"]==route and row["condition"]=="reference_serial" and row["replicate"]==1)
        reference_directory = root/"cli"/reference["label"]/"pdbs"
        names = [name[:-4] for name in reference["outputs"] if not name.endswith(".prepared.pdb")]
        for condition in ("cpu_serial", "double_serial"):
            candidate = next(row for row in measurements["rows"] if row["route"]==route and row["condition"]==condition and row["replicate"]==1)
            directory = root/"cli"/candidate["label"]/"pdbs"
            for name in names:
                sources = [reference_directory/f"{name}.prepared.pdb",directory/f"{name}.prepared.pdb"]
                left, a, groups_a = load(sources[0]);right, b, groups_b = load(sources[1])
                assert set(a)==set(b) and groups_a.keys()==groups_b.keys()
                identity_deltas = [];matched_deltas = [];group_records = []
                for parent, members in groups_a.items():
                    assert set(members)==set(groups_b[parent]) and len(members)<=3
                    aa = np.array([a[key] for key in members]);bb = np.array([b[key] for key in members])
                    identity_deltas.extend(np.linalg.norm(aa-bb,axis=1))
                    permutations = list(itertools.permutations(range(len(members))))
                    best = min(permutations,key=lambda order:float(np.sum((aa-bb[list(order)])**2)))
                    delta = np.linalg.norm(aa-bb[list(best)],axis=1)
                    matched_deltas.extend(delta)
                    group_records.append({"parent":parent,"hydrogens":members,"permutation":best,
                                          "max_after_matching_angstrom":float(max(delta)),
                                          "max_by_atom_name_angstrom":float(max(np.linalg.norm(aa-bb,axis=1)))})
                energies = []
                for path,pdb in zip(sources,[left,right]):
                    key = hashlib.sha256(path.read_bytes()).hexdigest()
                    if key not in energy_cache:energy_cache[key]=energy(pdb)
                    energies.append(energy_cache[key])
                row = {"route":route,"condition":condition,"target":name,
                       "hydrogen_max_by_name_angstrom":float(max(identity_deltas)),
                       "hydrogen_max_after_matching_angstrom":float(max(matched_deltas)),
                       "hydrogen_rms_by_name_angstrom":float(np.sqrt(np.mean(np.square(identity_deltas)))),
                       "hydrogen_rms_after_matching_angstrom":float(np.sqrt(np.mean(np.square(matched_deltas)))),
                       "reference_energy":energies[0],"candidate_energy":energies[1],
                       "delta_energy_kj_mol":energies[1]["potential_energy_kj_mol"]-energies[0]["potential_energy_kj_mol"],
                       "hydrogen_groups":group_records}
                records.append(row)
                print(route,condition,name,"H max after matching",row["hydrogen_max_after_matching_angstrom"],"energy delta",row["delta_energy_kj_mol"],flush=True)
                (root/"hydrogen_differences_261026.json").write_text(json.dumps({
                    "completed":len(records)==32,"protocol":"Same-parent H permutation matching; fixed-coordinate Amber ff14SB/GBn2 energy, NoCutoff, CUDA double deterministic; no minimisation. PDB-rounded coordinates.",
                    "rows":records},indent=2)+"\n")


if __name__ == "__main__":
    main()
