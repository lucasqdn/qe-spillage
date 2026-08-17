# qe-spillage

DFT spin–orbit spillage from Quantum ESPRESSO wavefunctions, without VASP.

The **spin–orbit spillage** of Liu and Vanderbilt measures how much a material's occupied
electronic subspace changes when spin–orbit coupling is switched on. A large value flags a
possible band inversion, which makes it a cheap high-throughput screen for topological
candidates — the basis of the JARVIS spillage database.

    gamma(k) = N_occ - sum_{m,n} |<psi^SOC_m(k) | psi^nonSOC_n(k)>|^2
    eta      = max_k gamma(k)

The reference implementation in [`jarvis-tools`](https://github.com/usnistgov/jarvis) reads VASP
`WAVECAR` files, so reproducing a published spillage requires a VASP licence. This repository
computes the same quantity from the HDF5 wavefunctions written by Quantum ESPRESSO's `pw.x`,
which is free software.

## Validation

Reproduces published JARVIS (VASP) values to within 1% on clean-gap insulators.

| Material | JARVIS ID | Type | k-mesh | e⁻ | This work | JARVIS (VASP) | Δ |
|---|---|---|---|---|---|---|---|
| BaMg₂Bi₂ | JVASP-4053 | insulator | 6×6×4 | 60 | **2.094** | 2.075 | +0.9% |
| Bi₂Se₃ | JVASP-1067 | insulator | 6×6×6 | 78 | **2.1146** | 2.098 | +0.8% |
| Bi₂Te₃ | JVASP-25 | insulator | 6×6×6 | 78 | **2.1116** | 2.094 | +0.8% |
| PbTe | JVASP-1103 | insulator | 6×6×6 | 30 | **2.010** | none published | — |
| Ba₃BiSb | JVASP-36485 | semimetal | 4×4×4 | 60 | 2.016 | 2.267 | −11% |
| Ba₃Bi₂ | JVASP-36513 | semimetal | 4×4×4 | 60 | 2.78 | 4.097 | −32% |

All three insulators land on the same side of VASP by the same amount, which reads as a small
systematic QE-NC vs VASP-PAW offset rather than scatter. The two semimetals are a different
story — see [Scope](#scope) below.

The Bi₂Se₃ result also agrees to 0.3% with the 2.12 that Liu and Vanderbilt originally reported
at Γ, computed in QE with a different pseudopotential table, cutoff and mesh.

## Requirements

- **Quantum ESPRESSO ≥ 7.0, built with HDF5.** The post-processor reads `wfc*.hdf5`, which a
  non-HDF5 build will not write. Check with `ldd $(command -v pw.x) | grep hdf5`.
- **Python 3.9+** with `numpy` and `h5py`.

```sh
pip install numpy h5py
```

## Pseudopotentials

Not vendored — download them yourself into `pseudo_nc/`:

- Source: [PseudoDojo](http://www.pseudo-dojo.org) → norm-conserving (NC), **v0.4**, **PBE**,
  standard accuracy.
- Take the **SR** (scalar-relativistic) files for the non-SOC run and the **FR** (fully
  relativistic) files for the SOC run, named `<El>_SR.upf` and `<El>_FR.upf`.

Norm-conserving is not optional. With PAW or ultrasoft pseudopotentials the stored smooth
wavefunctions are orthonormal under the augmentation operator *S*, not under the plane-wave dot
product, and *S* is not saved in the wavefunction file. Using PAW here produced a raw
γ ≈ −17 before any correction. With NC pseudopotentials the ordinary coefficient dot product
*is* the correct overlap.

## Quick start

Worked end to end on BaMg₂Bi₂. Run from the repository root.

```sh
# 1. Two SCF runs: identical cell, cutoffs and k-mesh, differing only in SOC.
export OMP_NUM_THREADS=1
mpirun -np 8 pw.x -npool 8 -in inputs/BaMg2Bi2.scf.nosoc.nc.in > nosoc.out   # ~2 min
mpirun -np 8 pw.x -npool 8 -in inputs/BaMg2Bi2.scf.soc.nc.in   > soc.out     # ~13 min

# 2. Overlap post-processing.
SPILLAGE_NELEC=60 SPILLAGE_REFERENCE=2.075 python compute_spillage.py \
    out_nosoc_nc_BaMg2Bi2/BaMg2Bi2_nosoc_nc.save \
    out_soc_nc_BaMg2Bi2/BaMg2Bi2_soc_nc.save
# -> max gamma (lowdin) = 2.0941   <-- reported spillage
```

Timings are for an Apple M2 Pro. The script prints γ(k) at every k-point and the maximum.

On a SLURM cluster, `slurm/fir_spillage.sbatch` chains both SCF runs and the post-processing:

```sh
sbatch --export=ALL,MAT=Bi2Se3 slurm/fir_spillage.sbatch
```

### `SPILLAGE_NELEC` must be set

It is the number of valence electrons in the cell, and it defaults to **50** — a value correct
only for the original Ba₃BiSb PAW run. Setting it wrong gives a wrong answer silently, with no
error. Values for the shipped inputs:

| Material | `SPILLAGE_NELEC` |
|---|---|
| PbTe | 30 |
| BaMg₂Bi₂, Ba₃BiSb (NC), Ba₃Bi₂ | 60 |
| Bi₂Se₃, Bi₂Te₃ | 78 |
| Ba₃BiSb (PAW decks) | 50 |

The two SLURM scripts hardcode `SPILLAGE_NELEC=78` for the Bi₂X₃ pair they were written for.
Change it before running them on anything else.

## Scope

`compute_spillage.py` fixes N_occ across the whole Brillouin zone. That is **exact for a
clean-gap insulator** and unreliable otherwise.

The condition to check is that the gap above the last filled band never closes at any sampled
k-point, in *both* runs. It was verified explicitly for the three validated insulators — 0 of
144 k-points for BaMg₂Bi₂, 0 of 216 for each of Bi₂Se₃ and Bi₂Te₃.

For a **semimetal** the occupied count genuinely varies with k, and the spillage becomes
ill-conditioned with respect to it. Both antiperovskites here have a degenerate manifold sitting
on E_F at Γ, which is exactly where γ peaks. Sweeping the assumed count over the same
wavefunctions:

| electrons assumed occupied | 56 | 58 | **60** | 62 | 64 |
|---|---|---|---|---|---|
| γ(Γ), Ba₃BiSb | 2.12 | 1.43 | **2.02** | 0.08 | 1.63 |
| γ(Γ), Ba₃Bi₂ | 2.10 | 1.44 | **2.78** | 4.12 | 4.22 |

Two bands either way moves γ from 0.08 to 4.22. This is a limitation of the spillage definition
for gapless systems, not of this implementation. Treat semimetal numbers here as diagnostic.

`compute_spillage_kocc.py` implements the `jarvis-tools` alternative — counting occupied bands
per k-point from the non-SOC occupations with a 0.5 threshold (`SPILLAGE_OCC_THR`). It is not a
fix. On Ba₃Bi₂ it selects 62 electrons and returns 4.12 against the 4.097 reference; applied
unchanged to Ba₃BiSb it also selects 62 and returns 0.08 against 2.267.

A high spillage flags a candidate, not a result. PbTe scores 2.010 at L yet has normal band
ordering and is topologically trivial at ambient pressure, unlike band-inverted SnTe. Confirming
topology needs invariants or surface states.

## Repository layout

```
compute_spillage.py          fixed N_occ (use this for clean-gap insulators)
compute_spillage_kocc.py     k-dependent occupied count, jarvis-tools convention
inputs/                      QE decks, {material}.scf.{nosoc,soc}.{nc,fx,paw}.in
slurm/                       SLURM submission scripts
reference/scf-logs/          QE SCF logs for every run in the table above
reference/spillage/          per-k gamma(k) tables produced by the post-processors
```

Input suffixes: `nc` norm-conserving with smearing, `fx` norm-conserving with
`occupations='fixed'`, `paw` the superseded PAW run kept for comparison.

Wavefunction directories (`out_*/`) are not tracked — the runs behind this table are about 33 GB.

## References

1. J. Liu and D. Vanderbilt, *Spin-orbit spillage as a measure of band inversion in insulators*,
   [Phys. Rev. B **90**, 125133 (2014)](https://doi.org/10.1103/PhysRevB.90.125133).
2. K. Choudhary, K. F. Garrity and F. Tavazza, *High-throughput discovery of topologically
   non-trivial materials using spin-orbit spillage*,
   [Sci. Rep. **9**, 8534 (2019)](https://doi.org/10.1038/s41598-019-45028-y).
3. K. Choudhary *et al.*, *High-throughput search for magnetic topological materials using
   spin-orbit spillage, machine learning, and experiments*,
   [Phys. Rev. B **103**, 155131 (2021)](https://doi.org/10.1103/PhysRevB.103.155131).
4. P. Giannozzi *et al.*, *Advanced capabilities for materials modelling with Quantum ESPRESSO*,
   [J. Phys.: Condens. Matter **29**, 465901 (2017)](https://doi.org/10.1088/1361-648X/aa8f79).
5. M. J. van Setten *et al.*, *The PseudoDojo*,
   [Comput. Phys. Commun. **226**, 39 (2018)](https://doi.org/10.1016/j.cpc.2018.01.012).
