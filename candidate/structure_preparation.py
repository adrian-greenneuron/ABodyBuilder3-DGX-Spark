"""Optional chemical preparation. Imported only when explicitly requested.

No neural-network code lives here. PDBFixer repairs atoms, OpenMM adds hydrogens
and optionally minimises, and Reduce is an explicitly selected alternative.
"""
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import importlib.metadata
import io
import math
import os
import random
import subprocess
import time


@dataclass(frozen=True)
class Options:
    hydrogen_method: str = "openmm"
    ph: float | None = None
    sidechain_flips: bool = False
    relax: str = "none"
    restraint_k: float = 1000.0  # kJ mol^-1 nm^-2
    tolerance: float = 10.0  # RMS force, kJ mol^-1 nm^-1
    max_iterations: int = 1000
    relax_platform: str = "CPU"
    seed: int = 0

    def validate(self):
        if self.hydrogen_method not in ("openmm", "reduce"):
            raise ValueError("--hydrogen-method must be 'openmm' or 'reduce'.")
        if self.relax not in ("none", "restrained"):
            raise ValueError("--relax must be 'none' or 'restrained'.")
        if self.ph is not None and (not math.isfinite(self.ph) or not 0 <= self.ph <= 14):
            raise ValueError("--ph must be finite and between 0 and 14.")
        if self.sidechain_flips and self.hydrogen_method != "reduce":
            raise ValueError("--sidechain-flips requires --hydrogen-method reduce.")
        if not all(math.isfinite(v) and v > 0 for v in (self.restraint_k, self.tolerance)):
            raise ValueError("Relaxation restraint strength and force tolerance must be finite and positive.")
        if self.max_iterations < 1:
            raise ValueError("--relax-max-iterations must be positive (unbounded minimisation is disabled).")
        if self.relax_platform not in ("CPU", "CUDA"):
            raise ValueError("--relax-platform must be cpu or cuda; unavailable platforms fail without fallback.")
        if self.relax == "none" and (self.restraint_k != 1000. or self.tolerance != 10. or
                                     self.max_iterations != 1000 or self.relax_platform != "CPU"):
            raise ValueError("Minimisation settings require --relax restrained.")


class PreparationError(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def _runtime():
    # Bound preparation CPU use independently of inference workers.
    os.environ.setdefault("OPENMM_CPU_THREADS", "1")
    import numpy as np
    import openmm as mm
    from openmm import app, unit
    from pdbfixer import PDBFixer
    return np, mm, app, unit, PDBFixer


@lru_cache(maxsize=1)
def _forcefield():
    _, _, app, _, _ = _runtime()
    return app.ForceField("amber14/protein.ff14SB.xml", "implicit/gbn2.xml")


def _keys(topology):
    return [(a.residue.chain.id, a.residue.id, a.residue.insertionCode,
             a.residue.name, a.name) for a in topology.atoms()]


def _positions(positions):
    np, _, _, unit, _ = _runtime()
    return np.asarray(positions.value_in_unit(unit.nanometer))


def _pdb(topology, positions):
    _, _, app, _, _ = _runtime()
    result = io.StringIO()
    app.PDBFile.writeFile(topology, positions, result, keepIds=True)
    return result.getvalue()


def _geometry(topology, positions):
    np, _, _, _, _ = _runtime()
    xyz = _positions(positions) * 10
    if not np.isfinite(xyz).all():
        raise ValueError("Non-finite atom coordinates.")
    atoms = list(topology.atoms())
    peptide = []; disulfides = []
    for a, b in topology.bonds():
        distance = float(np.linalg.norm(xyz[a.index] - xyz[b.index]))
        if {a.name, b.name} == {"C", "N"} and a.residue != b.residue:
            peptide.append(distance)
        if a.name == b.name == "SG":
            disulfides.append({"residues": [_keys(topology)[a.index][0:4], _keys(topology)[b.index][0:4]],
                               "distance_angstrom": distance})
    # Severe heavy-atom overlaps; exclude atoms within three covalent bonds.
    from scipy.spatial import cKDTree
    neighbours = {a.index: set() for a in atoms}
    for a, b in topology.bonds():
        neighbours[a.index].add(b.index); neighbours[b.index].add(a.index)
    heavy = [a.index for a in atoms if a.element.symbol != "H"]
    radii = {"C": 1.70, "N": 1.55, "O": 1.52, "S": 1.80}
    overlaps = 0
    for i, j in cKDTree(xyz[heavy]).query_pairs(2.6):
        a, b = heavy[i], heavy[j]
        near = {a}; frontier = {a}
        for _ in range(3):
            frontier = set().union(*(neighbours[k] for k in frontier)) - near
            near |= frontier
        if b not in near:
            threshold = .7 * (radii.get(atoms[a].element.symbol, 1.7) + radii.get(atoms[b].element.symbol, 1.7))
            overlaps += int(np.linalg.norm(xyz[a] - xyz[b]) < threshold)
    return {"atoms": len(atoms), "hydrogens": sum(a.element.symbol == "H" for a in atoms),
            "residues": topology.getNumResidues(), "severe_heavy_atom_overlaps": overlaps,
            "overlap_definition": "distance < 0.7*(vdW radii sum); excludes <=3 covalent bonds",
            "max_peptide_cn_angstrom": max(peptide, default=None),
            "peptide_cn_over_1_8_angstrom": sum(d > 1.8 for d in peptide),
            "disulfides": disulfides}


def _changes(before_topology, before_positions, after_topology, after_positions):
    np, _, _, _, _ = _runtime()
    original = dict(zip(_keys(before_topology), _positions(before_positions)))
    candidate = dict(zip(_keys(after_topology), _positions(after_positions)))
    heavy_keys = [k for k in original if not k[-1].startswith("H")]
    missing = [k for k in heavy_keys if k not in candidate]
    if missing:
        raise ValueError(f"Preparation lost original heavy atoms: {missing[:5]}")
    moved = [(k, float(np.linalg.norm(original[k] - candidate[k]) * 10)) for k in heavy_keys]
    ca = [d for k, d in moved if k[-1] == "CA"]
    return {"added_atoms": [k for k in candidate if k not in original],
            "moved_heavy_atoms": [{"atom": k, "distance_angstrom": d} for k,d in moved if d > 1e-5],
            "max_original_heavy_displacement_angstrom": max((d for _,d in moved), default=0),
            "ca_rms_displacement_angstrom": float(np.sqrt(np.mean(np.square(ca)))) if ca else None}


def _repair(raw, options, report):
    np, _, app, _, PDBFixer = _runtime()
    original = app.PDBFile(io.StringIO(raw))
    if len(set(_keys(original.topology))) != original.topology.getNumAtoms():
        raise ValueError("Duplicate chain/residue/atom identities in input.")
    if any(a.element.symbol == "H" for a in original.topology.atoms()):
        raise ValueError("Preparation expects the raw heavy-atom prediction; input already contains hydrogens.")
    fixer = PDBFixer(pdbfile=io.StringIO(raw))
    # Never infer or rebuild missing sequence stretches or replace residues.
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    report["repair"] = {"missing_atoms": [{"chain": r.chain.id, "residue": r.id,
        "atoms": [a.name for a in names]} for r,names in fixer.missingAtoms.items()],
        "missing_terminal_atoms": [{"chain": r.chain.id, "residue": r.id,
        "atoms": list(names)} for r,names in fixer.missingTerminals.items()],
        "missing_sequence_reconstruction": False}
    policy = os.environ.get("AB3_EXPERIMENTAL_REPAIR", "reference")
    from repair_policy import repair_properties
    platform_name, properties = repair_properties(policy)
    if platform_name is None:
        fixer.addMissingAtoms(seed=options.seed)
    else:
        _, mm, _, _, _ = _runtime()
        fixer.platform = mm.Platform.getPlatformByName(platform_name)
        fixer.addMissingAtoms(seed=options.seed, platformProperties=properties)
    report["repair_runtime"] = {"policy": policy, "platform": platform_name,
                                "requested_properties": properties}
    if list((r.chain.id,r.id,r.name) for r in fixer.topology.residues()) != list((r.chain.id,r.id,r.name) for r in original.topology.residues()):
        raise ValueError("Repair changed chain/residue identities.")
    changes = _changes(original.topology, original.positions, fixer.topology, fixer.positions)
    if changes["max_original_heavy_displacement_angstrom"] > 1e-5:
        raise ValueError("Atom repair moved original heavy atoms.")
    report["raw_geometry"] = _geometry(original.topology, original.positions)
    return original, fixer


def _reduce(fixer, options, report):
    _, _, app, _, _ = _runtime()
    # Reduce uses residue numbers when recognising N-termini. Renumber each
    # chain internally, then restore every original ID after preparation.
    original_ids = {}; mapping = []
    for chain in fixer.topology.chains():
        original_ids[chain.id] = []
        for n, residue in enumerate(chain.residues(), 1):
            original_ids[chain.id].append((residue.id, residue.insertionCode))
            mapping.append({"chain": chain.id, "reduce_residue": str(n), "original_residue": residue.id,
                            "original_insertion_code": residue.insertionCode})
            residue.id = str(n); residue.insertionCode = ""
    prepared = _pdb(fixer.topology, fixer.positions)
    command = ["reduce", "-FLIP" if options.sidechain_flips else "-NOFLIP", "-Quiet", "-"]
    result = subprocess.run(command, input=prepared, text=True, capture_output=True, timeout=60)
    report["reduce"] = {"command": command, "exit_code": result.returncode,
                        "stderr": result.stderr, "residue_mapping": mapping,
                        "version": subprocess.run(["reduce", "-Version"], text=True, capture_output=True).stderr.strip()}
    if result.returncode or not result.stdout.strip():
        raise ValueError(f"Reduce failed (exit {result.returncode}); no alternative method was used. {result.stderr[-500:]}")
    output = app.PDBFile(io.StringIO(result.stdout))
    for chain in output.topology.chains():
        residues = list(chain.residues())
        ids = original_ids.get(chain.id, [])
        if len(residues) != len(ids):
            raise ValueError("Reduce changed chain/residue count.")
        for residue, (rid, code) in zip(residues, ids):
            residue.id = rid; residue.insertionCode = code
    # Restore the input topology IDs too, for subsequent comparisons.
    for chain in fixer.topology.chains():
        for residue, (rid, code) in zip(chain.residues(), original_ids[chain.id]):
            residue.id = rid; residue.insertionCode = code
    # Reduce infers bonds from distances; strained predicted peptide/disulfide
    # bonds can be misidentified. Keep its explicitly requested heavy-atom
    # orientations, but rebuild hydrogens against the authoritative topology.
    # This is a declared pipeline, never a failure-triggered backend switch.
    report["reduce"]["hydrogen_policy"] = "discard Reduce hydrogens; OpenMM rebuilds all H using repaired topology and requested pH"
    report["reduce"]["initial_hydrogens"] = sum(a.element.symbol == "H" for a in output.topology.atoms())
    reduced_atoms = dict(zip(_keys(output.topology), _positions(output.positions)))
    repaired_keys = _keys(fixer.topology)
    if any(k not in reduced_atoms for k in repaired_keys):
        raise ValueError("Reduce lost repaired heavy atoms.")
    _, _, _, unit, _ = _runtime()
    return fixer.topology, [reduced_atoms[k] for k in repaired_keys] * unit.nanometer


def _minimise(topology, positions, options, freeze_heavy=False):
    started = time.perf_counter()
    np, mm, app, unit, _ = _runtime()
    system = _forcefield().createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=None)
    original = _positions(positions)
    restraint = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    restraint.addGlobalParameter("k", options.restraint_k)
    for parameter in ("x0", "y0", "z0"):
        restraint.addPerParticleParameter(parameter)
    restraint.setForceGroup(1)
    for atom in topology.atoms():
        if atom.element.symbol != "H":
            if freeze_heavy:
                system.setParticleMass(atom.index, 0)
            else:
                restraint.addParticle(atom.index, original[atom.index])
    system.addForce(restraint)
    integrator = mm.VerletIntegrator(.001 * unit.picoseconds)
    properties = {"Threads": "1"} if options.relax_platform == "CPU" else {"Precision": "mixed", "DeterministicForces": "true"}
    context = mm.Context(system, integrator, mm.Platform.getPlatformByName(options.relax_platform), properties)
    try:
        context.setPositions(positions)
        initial = context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        mm.LocalEnergyMinimizer.minimize(context, options.tolerance, options.max_iterations)
        state = context.getState(getEnergy=True, getPositions=True, getForces=True)
        final = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        physical = context.getState(getEnergy=True, groups={0}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        xyz = state.getPositions(asNumpy=True)
        forces = state.getForces(asNumpy=True).value_in_unit(unit.kilojoule_per_mole / unit.nanometer)
        mobile = [a.index for a in topology.atoms() if not freeze_heavy or a.element.symbol == "H"]
        # OpenMM's criterion is |gradient| / sqrt(number of particles),
        # with zero gradients for immobile particles, not RMS over 3N components.
        rms = float(np.sqrt(np.sum(np.square(forces[mobile])) / topology.getNumAtoms()))
        if not np.isfinite(_positions(xyz)).all() or not all(math.isfinite(v) for v in (initial,final,rms)):
            raise ValueError("Minimisation produced non-finite coordinates, energy or forces.")
        if final > initial + max(1.,abs(initial)) * 1e-6:
            raise ValueError("Minimisation increased the restrained system energy.")
        return xyz, {"initial_energy_kj_mol": initial, "final_energy_kj_mol": final,
                     "final_physical_energy_kj_mol": physical,
                     "rms_force_kj_mol_nm": rms, "converged": rms <= options.tolerance,
                     "force_norm_definition": "sqrt(sum(mobile force vector components squared) / total particles)",
                     "force_tolerance_kj_mol_nm": options.tolerance,
                     "max_iterations": options.max_iterations, "platform": options.relax_platform,
                     "platform_properties": properties, "seconds": time.perf_counter()-started}
    finally:
        del context, integrator


def prepare_structure(raw, options):
    """Return (prepared PDB, optional relaxed PDB, machine-readable record).

    Failure carries the partial record and is never converted into raw success.
    """
    options.validate()
    started = time.perf_counter()
    report = {"status": "started", "options": asdict(options),
              "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
              "forcefield": ["amber14/protein.ff14SB.xml", "implicit/gbn2.xml"],
              "fallback": "disabled", "warnings": []}
    try:
        np, mm, app, _, _ = _runtime()
        random.seed(options.seed); np.random.seed(options.seed)
        report["versions"] = {"openmm": mm.version.short_version,
                              "pdbfixer": importlib.metadata.version("pdbfixer")}
        original, fixer = _repair(raw, options, report)
        topology, positions = fixer.topology, fixer.positions
        if options.hydrogen_method == "reduce":
            topology, positions = _reduce(fixer, options, report)
        ph = options.ph if options.ph is not None else 7.0
        report["effective_ph"] = ph
        report["hydrogen_backend"] = "OpenMM (after optional Reduce orientation)"
        modeller = app.Modeller(topology, positions)
        geometry_policy = os.environ.get("AB3_EXPERIMENTAL_GEOMETRY", "off")
        try:
            from hydrogen_policy import hydrogen_properties
            policy = os.environ.get("AB3_EXPERIMENTAL_HYDROGEN", "cpu1")
            platform_name, properties = hydrogen_properties(policy)
            report["hydrogen_runtime"] = {"policy": policy, "platform": platform_name,
                                          "requested_properties": properties}
            kwargs = {} if properties is None else {"platformProperties": properties}
            if geometry_policy != "off":
                if options.relax != "none":
                    raise ValueError("Experimental fixed-heavy hydrogen placement is qualified only without structural relaxation.")
                from hydrogen_geometry import optimise
                kwargs["hydrogenOptimizer"] = optimise
            variants = modeller.addHydrogens(_forcefield(), pH=ph,
                platform=mm.Platform.getPlatformByName(platform_name), **kwargs)
            if geometry_policy != "off":
                report["hydrogen_geometry_optimisation"] = dict(optimise.last_report)
                report["hydrogen_backend"] = "Experimental fixed-heavy geometry placement (after optional Reduce orientation)"
                if not optimise.last_report.get("success", True):
                    report["warnings"].append("Hydrogen placement reached its finite optimiser budget; geometric checks passed, optimisation convergence is not claimed.")
        except ValueError as exc:
            if geometry_policy != "off":
                raise ValueError(f"Experimental hydrogen placement failed; no alternate method was used. {exc}") from exc
            raise ValueError(f"OpenMM could not parameterise preparation at pH {ph}. "
                             f"Some protonation/terminal combinations lack Amber templates; no different state was substituted. {exc}") from exc
        topology, positions = modeller.topology, modeller.positions
        report["protonation_variants"] = list(variants)
        # Parameterisation must succeed: finite coordinates and a nonzero H
        # count alone do not demonstrate a chemically complete topology.
        _forcefield().createSystem(topology, nonbondedMethod=app.NoCutoff, constraints=None)
        report["preparation_changes"] = _changes(original.topology, original.positions, topology, positions)
        movements = report["preparation_changes"]["moved_heavy_atoms"]
        if movements and (not options.sidechain_flips or any(x["atom"][-1] in ("N", "CA", "C", "O") for x in movements)):
            raise ValueError("Hydrogen preparation moved original atoms beyond the requested side-chain policy.")
        report["prepared_geometry"] = _geometry(topology, positions)
        if not report["prepared_geometry"]["hydrogens"]:
            raise ValueError("No hydrogens were added.")
        if report["prepared_geometry"]["peptide_cn_over_1_8_angstrom"]:
            report["warnings"].append("Long peptide C-N bonds remain in the predicted geometry; no chain was split or capped at an internal gap.")
        prepared = _pdb(topology, positions)
        relaxed = None
        if options.relax == "restrained":
            positions, hydrogen_min = _minimise(topology, positions, options, freeze_heavy=True)
            report["hydrogen_minimisation"] = hydrogen_min
            positions, restrained_min = _minimise(topology, positions, options)
            report["restrained_minimisation"] = restrained_min
            report["relaxed_geometry"] = _geometry(topology, positions)
            report["relaxation_changes"] = _changes(original.topology, original.positions, topology, positions)
            if not restrained_min["converged"]:
                report["warnings"].append("Restrained minimisation reached its finite budget without meeting the force tolerance.")
            relaxed = _pdb(topology, positions)
        report["status"] = "completed_with_warnings" if report["warnings"] else "completed"
        report["seconds"] = time.perf_counter() - started
        return prepared, relaxed, report
    except Exception as exc:
        report.update(status="failed", error=str(exc), seconds=time.perf_counter()-started)
        raise PreparationError(str(exc), report) from exc
