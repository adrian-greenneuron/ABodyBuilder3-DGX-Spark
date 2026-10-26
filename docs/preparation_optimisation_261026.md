# Optional preparation: profiling and concurrency — 26 October 2026

Independent preparation processes gave a substantial speed gain, but final
qualification found small differences in prepared hydrogen coordinates. The
candidate therefore **remains isolated for review**. Dev retains its existing
serial chemical preparation; main is unchanged. No raw neural-network prediction
or chemistry engine has been replaced.

![Optional-stage profile, latency/memory and candidate speedups](assets/preparation_optimisation_261026.png)

## H–I: the dominant cost is hydrogen optimisation

Profiles of two retained public pLDDT predictions per route put OpenMM's
`addHydrogens` at roughly 8–9 seconds per antibody. The Reduce executable took
0.68–0.76 seconds; the complete Reduce wrapper added about 0.8 seconds. A
separate version subprocess cost about 0.002 seconds. Missing-atom detection
took about 0.003 seconds; repair, including its minimisation and checks, took
roughly 0.6–0.9 seconds after the first call.

These are diagnostic wall-clock boundaries, not additive CPU utilisation
measurements. Nested `createSystem` calls are already inside their parent
stages. cProfile function totals were not used as a causal time decomposition
around OpenMM's native/threaded work. The ForceField was already cached per
process in the reference; that cache is not a new improvement.

The measured Reduce fraction does not justify rewriting its chemistry or
dictionary handling. Even removing its entire subprocess cost would save less
than a second from a roughly ten-second preparation. Instead, the candidate
processes antibodies independently, with bounded pending work and one OpenMM
CPU thread per worker. It preserves repair, pH, protonation decisions, the
optional Reduce orientation pass, force fields and validation. Its lightweight
controller avoids importing the neural-network CLI in each worker.

## Complete CLI performance

The pLDDT model used the same strict-FP32 AOT packages, native writer, eight
distinct targets, pH 7 and seed 42. Each condition had two fresh-container runs,
with order reversed on the second repetition. Times include imports, prediction,
preparation and output writing. The later four-versus-eight-worker screen was
interleaved; the serial comparison comes from the preceding matched screen.

| Route | Serial, two runs (s) | Four workers, two runs (s) | Eight workers, two runs (s) |
|---|---:|---:|---:|
| OpenMM hydrogens | 79.57, 79.66 | 26.62, 26.58 | 17.88, 18.03 |
| Reduce flips + OpenMM hydrogens | 84.33, 83.80 | 26.54, 26.60 | 18.15, 18.41 |

The observed eight-worker median speedups are 4.43× and 4.60×. Higher measured
container peaks rose from 1.19 to 2.20 GiB for OpenMM hydrogens, and from 1.29
to 2.99 GiB with Reduce. Higher summed process-tree RSS peaks were 1.52/3.55 GiB
and 2.93/4.31 GiB respectively. Shared pages can be counted in more than one
process; these quantities must not be added. PyTorch inference allocation was
unchanged at about 0.316 GiB, but **does not measure OpenMM's CUDA allocations**.

The initial serial/four-worker installed-CLI comparisons matched all 68 paired
raw/prepared PDB files, including a two-input eager-path check. All 64 paired
PDB files in the later four/eight-worker comparison also matched. Those passing
comparisons were insufficient to establish exact repeatability, as the final
check below demonstrated.

## Why the concurrency candidate was not enabled

A final independent eight-worker run changed eleven hydrogen coordinate fields
by 0.001 Å in one antibody. Its raw predicted PDB, heavy-atom fields, atom
identities and protonation choices were unchanged.

A focused trace then prepared the same raw antibody repeatedly using one,
four and eight processes, saving full floating-point arrays after repair and
after hydrogen addition. Four serial calls were identical. The four-worker
group contained two repair/output variants; the eight-worker group happened to
be identical in that trace. The first difference was in **PDBFixer's added-atom
repair**, before hydrogen addition: the maximum Cartesian coordinate difference
was 1.49 × 10⁻⁷ Å. It propagated to a maximum hydrogen coordinate difference of
0.000212 Å, sufficient to cross some PDB rounding boundaries.

The unchanged repair actually selected OpenMM CUDA **single precision**, with
`DeterministicForces=false`; the hydrogen stage explicitly used CPU with one
thread. That places the first observed difference in the existing repair
engine. Variation in GPU force accumulation is a plausible explanation, but
the experiment does not prove which operation introduced it. The strict-FP32
setting controls the neural network, not these separate OpenMM protocols.

This is not evidence of worse neural-network accuracy or changed protonation,
and its scale is much smaller than the model's structural errors. It still
prevents claiming byte-identical prepared output or silently changing the
default under the agreed policy. The result needs review before enabling
parallel chemistry. A future deterministic-repair comparison should change
that setting explicitly and compare serial/concurrent controls; it must not be
introduced as an invisible performance patch.

## Other CPU/GPU candidates also remain isolated

| Candidate | Measured performance | Observed differences |
|---|---|---|
| More CPU threads inside one OpenMM hydrogen engine | Two-input jobs: about 19.77 s at one thread versus 7.24 s at four | Atom identities and original heavy atoms match, but hydrogen displacements reach 2.21 Å across settings; repeats within a setting also vary. This is more than a PDB rounding difference. |
| Two preparation-plus-CUDA-relaxation workers | Five-input groups: median 56.63 s serial versus 31.31 s concurrent | Largest relaxed atom displacement 0.069 Å across worker counts, versus 0.046 Å between within-condition repeats. Whole-atom coordinate RMS difference reaches 0.00553 Å. |

All 20 preparation-plus-relaxation outputs in the worker comparison completed;
both minimisation stages met their force criterion and relaxed severe-heavy-
overlap counts were zero. Final restrained-energy differences reached
2.45 kJ/mol between worker counts, versus 2.12 kJ/mol between same-condition
repeats. These findings do not establish equal experimental accuracy. Neither
extra engine threads nor concurrent relaxation is enabled.

The existing Amber ff14SB/GBn2 protocol, pH, seeds, restraints, tolerances,
iteration limits and CUDA mixed-precision minimisation were retained. This
screen did not reuse contexts across different topologies, change a minimiser,
skip a chemistry check or replace Reduce/PDBFixer/OpenMM internals.

## Failure behaviour, fast prediction and reproducibility

Four policy/dispatch tests passed, along with the existing four preparation
policy checks and ten CLI regression checks. A real worker failure preserved
its failed audit and raw input while an independent valid job completed. The
first test attempted to use a duplicate atom record as an invalid fixture;
OpenMM merged it, so that test correctly failed its expectation. It was
replaced with an already hydrogenated input, explicitly forbidden by the
existing contract. Both attempts and the correction are retained.

Two fresh 1,000-input runs per base/pLDDT condition compared the candidate image
with current dev. All **4,000 paired raw fingerprints and PDBs were exact**;
median complete-job times differed by less than 1% and CUDA allocation was
identical at about 2.390 GiB. This supports the absence of a plain-prediction
regression in the candidate, not promotion of its chemical outputs.

The reference is dev `49080c3`, image
`sha256:a4d29734d1b2ce98a09c4508e9047ad714a6da94be610fd5b8dda6b15285b590`.
Experimental images `parallel-preparation-candidate-261026` and
`parallel-preparation-261026` are retained; neither replaces dev's `latest`.
The latter's automatic eight-worker qualification was the one that exposed
the rounding difference. Its name is a local experiment identifier, not an
acceptance decision.

The [measurements](data/preparation_optimisation_measurements_261026.json) and
[checksummed archive](data/preparation_optimisation_evidence_261026.zip) retain
individual timings, numerical comparisons, platform properties, protocols,
source hashes and the qualification limitation. Raw profiler traces, structures
and float-array captures remain local. `reference/` and `candidate/` preserve
the original wrapper and experimental scheduler; the integration patch is not
applied to dev. All previous images and data remain intact.
