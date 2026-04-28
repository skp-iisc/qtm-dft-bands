import numpy as np
from qtm.constants import RYDBERG, ELECTRONVOLT
from qtm.lattice import RealLattice
from qtm.crystal import BasisAtoms, Crystal
from qtm.pseudo import UPFv2Data
from qtm.kpts import gen_monkhorst_pack_grid, KList
from qtm.gspace import GSpace
from qtm.mpi import QTMComm
from qtm.dft import DFTCommMod, scf

from qtm.io_utils.dft_printers import print_eigenvalues, print_scf_status

from qtm.logger import qtmlogger

from qtm.config import MPI4PY_INSTALLED
if MPI4PY_INSTALLED:
    from mpi4py.MPI import COMM_WORLD
else:
    COMM_WORLD = None

comm_world = QTMComm(COMM_WORLD)

# Only k-pt parallelization:
dftcomm = DFTCommMod(comm_world, comm_world.size, 1)
# Only band parallelization:
# dftcomm = DFTCommMod(comm_world, 1, 1)

# Lattice
reallat = RealLattice.from_alat(alat=10.2,
    a1=[-0.5, 0.0, 0.5], a2=[0.0, 0.5, 0.5], a3=[-0.5, 0.5, 0.0],  # Bohr
)

# Atom Basis
pseudopot = UPFv2Data.from_file("../pseudo/Si_ONCV_PBE-1.2.upf")
si_atoms = BasisAtoms(
    "silicon",
    pseudopot,
    28.086,
    reallat,
    # np.array([[0.875, 0.875, 0.875], [0.125, 0.125, 0.125]]).T,
    np.array([[0., 0., 0.], [0.25, 0.25, 0.25]]).T,
)

crystal = Crystal(reallat, [si_atoms])  # Represents the crystal

# Generating k-points from a Monkhorst Pack grid (reduced to the crystal's IBZ)
mpgrid_shape = (4, 4, 4)
mpgrid_shift = (True, True, True)
kpts = gen_monkhorst_pack_grid(crystal, mpgrid_shape, mpgrid_shift)

# -----Setting up G-Space of calculation-----
ecut_wfn = 25 * RYDBERG
ecut_rho = 4 * ecut_wfn
grho = GSpace(crystal.recilat, ecut_rho)
gwfn = grho

numbnd = crystal.numel // 2  # Ensure adequate # of bands if system is not an insulator
conv_thr = 1e-8 * RYDBERG
diago_thr_init = 1e-2 * RYDBERG

out = scf(
    dftcomm,
    crystal,
    kpts,
    grho,
    gwfn,
    numbnd,
    is_spin=False,
    is_noncolin=False,
    symm_rho=True,
    rho_start=None,
    occ_typ="fixed",
    conv_thr=conv_thr,
    diago_thr_init=diago_thr_init,
    iter_printer=print_scf_status,
)

scf_converged, rho, l_wfn_kgrp, en = out

# np.save('si4_rho.npy', rho.data)

# print_eigenvalues(l_wfn_kgrp)

if comm_world.rank == 0:
    print("\nSCF Routine has exited")
    print(qtmlogger)

# High-symmetry k-points in crystal coordinates
L = np.array([0.0, 0.5, 0.0])
G = np.array([0.0, 0.0, 0.0])
X = np.array([-0.5, 0.0, -0.5])
K = np.array([-0.375, 0.25, -0.375])

# Use fixed number of k-points per segment (matching QE convention)
N = 50  # Number of points per segment
kpts_L_G = np.linspace(L, G, N, endpoint=False)
kpts_G_X = np.linspace(G, X, N, endpoint=False)
kpts_X_K = np.linspace(X, K, N, endpoint=False)
kpts_K_G = np.linspace(K, G, N, endpoint=True)

# Combine all segments
k_path = np.vstack([kpts_L_G, kpts_G_X, kpts_X_K, kpts_K_G])

# Create weights
k_weights = np.ones(len(k_path)) / len(k_path)

# Create KList
kpts1 = KList(recilat=crystal.recilat, 
              k_coords=k_path.T,
              k_weights=k_weights,
              coords_typ="cryst")

numbnd1 = 12
out1 = scf(dftcomm, crystal, kpts1, grho, gwfn, 
            numbnd=numbnd1,
            is_spin=False,
            is_noncolin=False,
            symm_rho=True,
            rho_start=rho,
            occ_typ="fixed",
            conv_thr=conv_thr,
            diago_thr_init=diago_thr_init,
            maxiter=1,
            iter_printer=print_scf_status,
)

scf_converged1, rho1, l_wfn_kgrp1, en1 = out1

band_data = []

# High-symmetry point labels
hsym_labels = ['L', r'$\Gamma$', 'X', 'K', r'$\Gamma$']

# Use QE reference segment boundaries (k-distances in Angstrom^-1)
hsym_x = [0.0, 0.866, 1.866, 2.2196, 3.2802]

# Compute k-distances in reciprocal space for band data mapping
b1 = crystal.recilat.recvec[:, 0]
b2 = crystal.recilat.recvec[:, 1]
b3 = crystal.recilat.recvec[:, 2]

k_path_cart = np.array([k[0]*b1 + k[1]*b2 + k[2]*b3 for k in k_path])
k_distances = np.zeros(len(k_path))
for i in range(1, len(k_path)):
    k_distances[i] = k_distances[i-1] + np.linalg.norm(k_path_cart[i] - k_path_cart[i-1])

# Rescale k_distances to match QE convention
if k_distances[-1] > 0:
    k_distances = k_distances * (hsym_x[-1] / k_distances[-1])

# Collect and process band data
local_band_data = []
for kgrp in l_wfn_kgrp1:
    for kswfn in kgrp:
        k_point = np.array(kswfn.k_cryst)
        distances = np.linalg.norm(k_path.T - k_point.reshape(-1, 1), axis=0)
        global_k_index = np.argmin(distances)
        
        for band_idx in range(kswfn.numbnd):
            eigenvalue_ev = kswfn.evl[band_idx] / ELECTRONVOLT
            local_band_data.append([band_idx + 1, k_distances[global_k_index], eigenvalue_ev])

# MPI Gather
if MPI4PY_INSTALLED:
    all_band_data = comm_world.comm.gather(local_band_data, root=0)
else:
    all_band_data = [local_band_data]

if comm_world.rank == 0:
    band_data = []
    for data in all_band_data:
        band_data.extend(data)
    
    # Sort by band index, then by k-distance
    band_data.sort(key=lambda x: (x[0], x[1]))

    e_fermi = en1.HO_level / ELECTRONVOLT

    # Write output file
    with open("si_bands.dat", 'w') as f:
        f.write(f"# Fermi Energy: {e_fermi:.6f}\n")
        f.write(f"# Segment boundaries (k-distance): {' '.join(f'{x:.6f}' for x in hsym_x)}\n")
        f.write(f"# Segment labels: {' '.join(hsym_labels)}\n")
        
        current_band = None
        for band_idx, k_dist, ev in band_data:
            if current_band is not None and band_idx != current_band:
                f.write('\n')
            f.write(f"{k_dist:10.4f} {ev:10.4f}\n")
            current_band = band_idx

    print('='*60+f"\nBand structure data saved to: si_bands.dat")

