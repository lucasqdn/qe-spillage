#!/usr/bin/env python
"""Spin-orbit spillage from two Quantum ESPRESSO SCF runs (HDF5 wavefunctions).

Reproduces the Liu-Vanderbilt / JARVIS spillage used by Choudhary et al. 2021:

    gamma(k) = N_occ  -  sum_{m in occ(SOC)} sum_{n in occ(nonSOC)} |<psi^SOC_m | psi^nonSOC_n>|^2

and the material-level spillage is max_k gamma(k). A large value flags band inversion
(topological candidate). Set SPILLAGE_REFERENCE to print an optional comparison value.

Inputs are the two .save directories:
  - nonSOC: spinless (npol=1), scalar-relativistic pseudos, N_occ_spatial = nelec/2 doubly-occupied
  - SOC   : spinor  (npol=2), fully-relativistic pseudos,  N_occ = nelec spinor states

The non-SOC occupied subspace is built as {phi_a (x) up, phi_a (x) down} for the lowest
nelec/2 spatial orbitals, so both projectors have dimension N_occ = nelec.

CAVEAT (PAW/ultrasoft): the smooth pseudo-wavefunctions are orthonormal under the PAW
overlap operator S, not under the plain plane-wave dot product. We approximate S by
Loewdin-orthonormalizing each occupied set in the smooth metric (so both projectors are
genuine orthogonal projectors). This neglects the cross-state augmentation term, so the
number is approximate. With norm-conserving pseudos the ordinary coefficient overlap is
the appropriate metric, and this numerical orthonormalization should be negligible.
"""
import sys, glob, os, re
import numpy as np
import h5py


def ik_of(path):
    return int(re.match(r"wfc(\d+)\.hdf5", os.path.basename(path)).group(1))

NELEC = int(os.environ.get("SPILLAGE_NELEC", "50"))   # PAW run: 50; NC run (Sb 4d semicore): 60
N_OCC_SPATIAL = NELEC // 2   # non-SOC spatial occupied (spin-degenerate)
N_OCC = NELEC                # SOC spinor occupied states


def read_wfc(path):
    with h5py.File(path, "r") as f:
        a = dict(f.attrs)
        igwx = int(a["igwx"]); npol = int(a["npol"]); nbnd = int(a["nbnd"])
        mill = f["MillerIndices"][:].astype(np.int64)          # (igwx, 3)
        evc = f["evc"][:]                                       # (nbnd, 2*npol*igwx)
    c = evc[:, 0::2] + 1j * evc[:, 1::2]                        # (nbnd, npol*igwx)
    return dict(xk=np.array(a["xk"], float), ik=int(a["ik"]),
                igwx=igwx, npol=npol, nbnd=nbnd, mill=mill, c=c)


def align_soc_to_ns(ns, soc):
    """Return SOC coeffs reordered so column g matches the non-SOC G-vector g."""
    if ns["igwx"] == soc["igwx"] and np.array_equal(ns["mill"], soc["mill"]):
        perm = np.arange(ns["igwx"])
    else:
        idx = {tuple(m): i for i, m in enumerate(soc["mill"])}
        perm = np.array([idx[tuple(m)] for m in ns["mill"]])   # KeyError => bases differ
    up = soc["c"][:, :soc["igwx"]][:, perm]                    # (nbnd_soc, g)
    dn = soc["c"][:, soc["igwx"]:2 * soc["igwx"]][:, perm]     # (nbnd_soc, g)
    return up, dn


def loewdin(C):
    """Orthonormalize rows of C (states x pw) under the plain dot product."""
    S = C @ C.conj().T
    w, V = np.linalg.eigh(S)
    w = np.clip(w.real, 1e-12, None)
    return (V * (1.0 / np.sqrt(w))) @ V.conj().T @ C


def spillage_at_k(ns, soc, orthonormalize=True):
    Cns = ns["c"][:N_OCC_SPATIAL]                              # (25, g)  spatial occupied
    up, dn = align_soc_to_ns(ns, soc)
    up, dn = up[:N_OCC], dn[:N_OCC]                            # (50, g) each

    if orthonormalize:
        Cns = loewdin(Cns)
        Csoc = np.concatenate([up, dn], axis=1)               # (50, 2g)
        Csoc = loewdin(Csoc)
        up, dn = Csoc[:, :ns["igwx"]], Csoc[:, ns["igwx"]:]

    # <psi^SOC_m | phi_a (x) up> = sum_G conj(up_m) phi_a ; likewise for down
    A_up = up.conj() @ Cns.T                                   # (50, 25)
    A_dn = dn.conj() @ Cns.T                                   # (50, 25)
    trace = np.sum(np.abs(A_up) ** 2) + np.sum(np.abs(A_dn) ** 2)
    return N_OCC - trace


def main(ns_dir, soc_dir):
    ns_files = sorted(glob.glob(os.path.join(ns_dir, "wfc*.hdf5")), key=ik_of)
    print(f"{'ik':>3} {'kx':>7} {'ky':>7} {'kz':>7} {'gamma_raw':>11} {'gamma_lowdin':>13}")
    g_raw_all, g_low_all = [], []
    for nf in ns_files:
        ik = ik_of(nf)
        sf = os.path.join(soc_dir, f"wfc{ik}.hdf5")
        ns, soc = read_wfc(nf), read_wfc(sf)
        assert np.allclose(ns["xk"], soc["xk"], atol=1e-6), (ns["xk"], soc["xk"])
        g_raw = spillage_at_k(ns, soc, orthonormalize=False).real
        g_low = spillage_at_k(ns, soc, orthonormalize=True).real
        g_raw_all.append(g_raw); g_low_all.append(g_low)
        kx, ky, kz = ns["xk"]
        print(f"{ik:3d} {kx:7.3f} {ky:7.3f} {kz:7.3f} {g_raw:11.4f} {g_low:13.4f}")
    print("-" * 48)
    print(f"max gamma (raw)    = {max(g_raw_all):.4f}")
    print(f"max gamma (lowdin) = {max(g_low_all):.4f}   <-- reported spillage")
    reference = os.environ.get("SPILLAGE_REFERENCE")
    if reference:
        print(f"reference spillage = {reference}")


if __name__ == "__main__":
    ns_dir = sys.argv[1] if len(sys.argv) > 1 else "out_nosoc/Ba3BiSb_nosoc.save"
    soc_dir = sys.argv[2] if len(sys.argv) > 2 else "out_soc/Ba3BiSb_soc.save"
    main(ns_dir, soc_dir)
