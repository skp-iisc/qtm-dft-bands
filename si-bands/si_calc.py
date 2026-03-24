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
pseudopot = UPFv2Data.from_file("../pseudo_dir/Si_ONCV_PBE-1.2.upf")
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

# Define high-symmetry points for Si (FCC structure)
# High-symmetry k-points in crystal coordinates
L = np.array([0.0, 0.5, 0.0])      # L point
G = np.array([0.0, 0.0, 0.0])      # Gamma point
X = np.array([-0.5, 0.0, -0.5])      # X point
K = np.array([-0.375, 0.25, -0.375]) # K point

del_k = 0.01
N1 = int(np.linalg.norm(L-G)/del_k)
N2 = int(np.linalg.norm(G-X)/del_k)
N3 = int(np.linalg.norm(X-K)/del_k)
N4 = int(np.linalg.norm(K-G)/del_k) + 1  # +1 so final G is included

# endpoint=False on all but the last segment avoids duplicate boundary points
kpts_L_G = np.linspace(L, G, N1, endpoint=False)
kpts_G_X = np.linspace(G, X, N2, endpoint=False)
kpts_X_K = np.linspace(X, K, N3, endpoint=False)
kpts_K_G = np.linspace(K, G, N4, endpoint=True)

# Combine all segments (no duplicate boundary points)
k_path = np.vstack([kpts_L_G, kpts_G_X, kpts_X_K, kpts_K_G])

# Create weights for band structure calculation
# Weights should sum to 1 and represent path distances
k_weights = np.ones(len(k_path)) / len(k_path)

# Create kpts1 using KList with crystal coordinates
kpts1 = KList(recilat=crystal.recilat, 
              k_coords=k_path.T,  # Transpose to (3, nkpts) format
              k_weights=k_weights,
              coords_typ="cryst")

numbnd1 = 8
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

# np.save('si4_rho1.npy', rho1.data)

band_data = []
k_index = 0

# High-symmetry point labels along the path (using LaTeX math strings)
hsym_labels = ['L', r'$\Gamma$', 'X', 'K', r'$\Gamma$']

# Segment boundaries marking the start index of each segment
segment_boundaries = [0, N1, N1+N2, N1+N2+N3, N1+N2+N3+N4-1]

# True cumulative distance across the whole path (never resets)
k_distances_cumulative = np.zeros(len(k_path))
for i in range(1, len(k_path)):
    k_distances_cumulative[i] = (
        k_distances_cumulative[i-1] + np.linalg.norm(k_path[i] - k_path[i-1])
    )

# x-positions of each high-symmetry point along the cumulative path
hsym_x = [k_distances_cumulative[idx] for idx in segment_boundaries]

# Collect k-point data from all MPI processes
local_band_data = []
local_k_indices = []

for kgrp in l_wfn_kgrp1:
    for kswfn in kgrp:
        k_point = np.array(kswfn.k_cryst)
        # Find the global k-index by finding matching k-point in k_path
        # Since k-points are distributed, we need to find which one this is
        k_point_cryst_2d = k_point.reshape(-1, 1)
        
        # Find closest k-point in k_path (to match due to potential numerical differences)
        distances = np.linalg.norm(k_path.T - k_point_cryst_2d, axis=0)
        global_k_index = np.argmin(distances)
        
        # Determine which segment this k-point belongs to
        k_dist = k_distances_cumulative[global_k_index]
        segment_idx = 0
        for seg_idx, boundary in enumerate(segment_boundaries[1:]):
            if global_k_index >= boundary:
                segment_idx = seg_idx + 1
        
        # Determine path labels for this segment
        path_start = hsym_labels[segment_idx]
        path_end = hsym_labels[segment_idx + 1] if segment_idx + 1 < len(hsym_labels) else hsym_labels[-1]

        for band_idx in range(kswfn.numbnd):
            eigenvalue_ev = kswfn.evl[band_idx] / ELECTRONVOLT
            local_band_data.append([global_k_index, k_dist, band_idx + 1, eigenvalue_ev, path_start, path_end])
        
        local_k_indices.append(global_k_index)

# MPI Gather: collect all band data from all processes
if MPI4PY_INSTALLED:
    all_band_data_list = comm_world.comm.gather(local_band_data, root=0)
else:
    all_band_data_list = [local_band_data]

if comm_world.rank == 0:
    # Combine data from all processes
    band_data = []
    for band_data_from_process in all_band_data_list:
        band_data.extend(band_data_from_process)
    
    # Sort by k-index then band index to ensure proper ordering
    # x[0] is k-index, x[2] is band index
    band_data.sort(key=lambda x: (x[0], x[2]))

    # Calculate Fermi energy (Valence Band Maximum)
    # Si has 4 filled bands. We find the max eigenvalue for band 4.
    n_val = 4
    val_energies = [row[3] for row in band_data if row[2] == n_val]
    e_fermi = max(val_energies) if val_energies else 0.0

    output_file = "si_bands.dat"
    
    # Write custom file with mixed types (floats and strings)
    with open(output_file, 'w') as f:
        f.write(f'# Fermi Energy: {e_fermi:.6f}\n')
        f.write('# k-index  k-cumul-dist  band  eigenvalue(eV)  path_start  path_end\n')
        for row in band_data:
            k_idx, k_dist, band_idx, ev, p_start, p_end = row
            # Ensure raw strings are written plainly
            f.write(f"{k_idx:4d} {k_dist:14.8f} {band_idx:4d} {ev:14.8f} {p_start} {p_end}\n")

    print('='*60+"\nNSCF Routine has exited\n"+'='*60)
    print("Path for bandstructure: L-G-X-K-G\n"+'-'*60)

    print(f"\nBand structure data saved to: {output_file}.")
    print(f"\nTotal k-points collected: {len(band_data) // numbnd1}")

