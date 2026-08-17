#!/usr/bin/env python
"""Spin-orbit spillage with the JARVIS k-DEPENDENT occupied-band count.

This is the faithful version of `compute_spillage.py`. The original uses a fixed
N_occ = SPILLAGE_NELEC at every k-point, which is correct only for a clean-gap
material (occupied count = nelec at all k). For a SEMIMETAL (band overlap, e.g.
Ba3Bi2 with a Fermi pocket at Gamma) the number of occupied bands VARIES with k,
and the spillage peaks exactly where it does. jarvis.analysis.topological.spillage
handles this by counting, per k-point, the occupied bands from the non-SOC
occupations:

    cup(k) = index of the first non-SOC band with occupation < 0.5   (per spin)
    nelec_tot(k) = cup(k) + cdn(k)      (= 2*cup for a non-magnetic cell)
    gamma(k) = nelec_tot(k) - sum_{m,n in occ(k)} |<psi^SOC_m | psi^noSOC_n>|^2
    spillage = max_k gamma(k)

We read the per-k occupations from the non-SOC QE data-file-schema.xml (occupations
are on a [0,1] per-spin scale for nspin=1, so the 0.5 threshold matches JARVIS's
VASP convention), then build k-dependent occupied projectors exactly as before.

With norm-conserving pseudos the plane-wave dot product IS the true overlap, so the
Loewdin step is a no-op (raw == lowdin), same as the fixed-N_occ script.
"""
import sys, glob, os, re
import numpy as np
import h5py
import xml.etree.ElementTree as ET

OCC_THR = float(os.environ.get("SPILLAGE_OCC_THR", "0.5"))


def ik_of(path):
    return int(re.match(r"wfc(\d+)\.hdf5", os.path.basename(path)).group(1))


def read_occupations(save_dir):
    """Map rounded k-vector (tpiba) -> non-SOC occupation array (per band)."""
    xml = glob.glob(os.path.join(save_dir, "*.xml"))[0]
    bs = ET.parse(xml).getroot().find(".//output/band_structure")
    out = {}
    for kp in bs.findall("ks_energies"):
        k = tuple(round(float(x), 4) for x in kp.find("k_point").text.split())
        occ = np.array([float(x) for x in kp.find("occupations").text.split()])
        out[k] = occ
    return out


def cup_from_occ(occ, thr=OCC_THR):
    """JARVIS prescription: first band with occ < thr -> that 0-based index = #occupied."""
    below = np.where(occ < thr)[0]
    return int(below[0]) if len(below) else len(occ)


def read_wfc(path):
    with h5py.File(path, "r") as f:
        a = dict(f.attrs)
        igwx = int(a["igwx"]); npol = int(a["npol"]); nbnd = int(a["nbnd"])
        mill = f["MillerIndices"][:].astype(np.int64)
        evc = f["evc"][:]
    c = evc[:, 0::2] + 1j * evc[:, 1::2]
    return dict(xk=np.array(a["xk"], float), ik=int(a["ik"]),
                igwx=igwx, npol=npol, nbnd=nbnd, mill=mill, c=c)


def align_soc_to_ns(ns, soc):
    if ns["igwx"] == soc["igwx"] and np.array_equal(ns["mill"], soc["mill"]):
        perm = np.arange(ns["igwx"])
    else:
        idx = {tuple(m): i for i, m in enumerate(soc["mill"])}
        perm = np.array([idx[tuple(m)] for m in ns["mill"]])
    up = soc["c"][:, :soc["igwx"]][:, perm]
    dn = soc["c"][:, soc["igwx"]:2 * soc["igwx"]][:, perm]
    return up, dn


def loewdin(C):
    S = C @ C.conj().T
    w, V = np.linalg.eigh(S)
    w = np.clip(w.real, 1e-12, None)
    return (V * (1.0 / np.sqrt(w))) @ V.conj().T @ C


def spillage_at_k(ns, soc, n_occ_spatial, orthonormalize=True):
    n_occ = 2 * n_occ_spatial                                  # spinor occupied (= nelec_tot)
    Cns = ns["c"][:n_occ_spatial]                              # (cup, g) spatial occupied
    up, dn = align_soc_to_ns(ns, soc)
    up, dn = up[:n_occ], dn[:n_occ]                            # lowest nelec_tot spinor bands

    if orthonormalize:
        Cns = loewdin(Cns)
        Csoc = np.concatenate([up, dn], axis=1)
        Csoc = loewdin(Csoc)
        up, dn = Csoc[:, :ns["igwx"]], Csoc[:, ns["igwx"]:]

    A_up = up.conj() @ Cns.T                                   # (nelec_tot, cup)
    A_dn = dn.conj() @ Cns.T
    trace = np.sum(np.abs(A_up) ** 2) + np.sum(np.abs(A_dn) ** 2)
    return n_occ - trace


def main(ns_dir, soc_dir):
    occ_map = read_occupations(ns_dir)
    ns_files = sorted(glob.glob(os.path.join(ns_dir, "wfc*.hdf5")), key=ik_of)
    print(f"{'ik':>3} {'kx':>7} {'ky':>7} {'kz':>7} {'nocc':>5} {'g_raw':>9} {'g_lowdin':>9}")
    g_raw_all, g_low_all, info = [], [], []
    for nf in ns_files:
        ik = ik_of(nf)
        sf = os.path.join(soc_dir, f"wfc{ik}.hdf5")
        ns, soc = read_wfc(nf), read_wfc(sf)
        assert np.allclose(ns["xk"], soc["xk"], atol=1e-6), (ns["xk"], soc["xk"])
        key = tuple(round(float(x), 4) for x in ns["xk"])
        if key not in occ_map:                                # fall back to nearest
            kk = min(occ_map, key=lambda q: sum((a - b) ** 2 for a, b in zip(q, key)))
            occ = occ_map[kk]
        else:
            occ = occ_map[key]
        cup = cup_from_occ(occ)                                # occupied spatial bands at this k
        g_raw = spillage_at_k(ns, soc, cup, orthonormalize=False).real
        g_low = spillage_at_k(ns, soc, cup, orthonormalize=True).real
        g_raw_all.append(g_raw); g_low_all.append(g_low)
        kx, ky, kz = ns["xk"]
        info.append((ik, kx, ky, kz, 2 * cup, g_raw, g_low))
        print(f"{ik:3d} {kx:7.3f} {ky:7.3f} {kz:7.3f} {2*cup:5d} {g_raw:9.4f} {g_low:9.4f}")
    print("-" * 52)
    imax = int(np.argmax(g_low_all))
    print(f"max gamma (raw)    = {max(g_raw_all):.4f}")
    print(f"max gamma (lowdin) = {max(g_low_all):.4f}   <-- reported spillage")
    ik, kx, ky, kz, nocc, _, _ = info[imax]
    print(f"  at ik={ik}  k=({kx:.3f},{ky:.3f},{kz:.3f})  nelec_tot={nocc}")


if __name__ == "__main__":
    ns_dir = sys.argv[1]
    soc_dir = sys.argv[2]
    main(ns_dir, soc_dir)
