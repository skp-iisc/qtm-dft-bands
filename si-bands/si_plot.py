"""
Script to plot band structure from the .dat file
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_band_structure(dat_file):
    """
    Plot band structure from the .dat file.
    Reads Fermi energy from the first line and path info from the data columns:
    k-index  k-cumul-dist  band  eigenvalue(eV) path_start path_end
    """
    
    e_fermi = 0.0
    k_dist_list = []
    band_idx_list = []
    eigenvalues_list = []
    
    ticks_x = []
    ticks_labels = []
    
    current_segment_start = None
    last_x = 0.0
    last_end_label = None
    
    with open(dat_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Read Fermi Energy from the first line
            if line.startswith('# Fermi Energy:'):
                e_fermi = float(line.split(':')[1].strip())
                continue
            
            # Skip other headers
            if line.startswith('#'):
                continue
                
            parts = line.split()
            if len(parts) >= 6:
                x = float(parts[1])
                band = int(parts[2])
                ev = float(parts[3])
                p_start = parts[4]
                p_end = parts[5]
                
                k_dist_list.append(x)
                band_idx_list.append(band)
                eigenvalues_list.append(ev)
                
                if p_start != current_segment_start:
                    if x not in ticks_x:
                        ticks_x.append(x)
                        ticks_labels.append(p_start)
                    current_segment_start = p_start
                
                last_x = x
                last_end_label = p_end

    # Add the final boundary based on the last point's path_end
    if ticks_x and abs(ticks_x[-1] - last_x) > 1e-5:
        ticks_x.append(last_x)
        ticks_labels.append(last_end_label)

    k_dist = np.array(k_dist_list)
    band_idx = np.array(band_idx_list)
    eigenvalues = np.array(eigenvalues_list)

    print(f"Fermi level (VBM) read from file: {e_fermi:.4f} eV")

    # ------------------------------------------------------------------ #
    # Plot
    # ------------------------------------------------------------------ #
    fig, ax = plt.subplots(figsize=(8, 7))

    for band in np.unique(band_idx):
        mask = band_idx == band
        # Sort within each band by k_dist
        order = np.argsort(k_dist[mask])
        ax.plot(k_dist[mask][order], eigenvalues[mask][order], 'b-', linewidth=1.5)

    # Vertical lines at segment boundaries
    for x_pos in ticks_x:
        ax.axvline(x=x_pos, color='k', linestyle='-', linewidth=0.8)

    # Fermi level at 0
    ax.axhline(y=e_fermi, color='r', linestyle='--', linewidth=1.0, label=f'$E_F$ = {e_fermi:.2f} eV')

    # X-axis ticks
    if ticks_x and ticks_labels:
        ax.set_xticks(ticks_x)
        ax.set_xticklabels(ticks_labels, fontsize=13)
        ax.set_xlim(ticks_x[0], ticks_x[-1])

    ax.set_ylabel('Energy − $E_F$ (eV)', fontsize=12)
    path_str = '–'.join(ticks_labels) if ticks_labels else 'Band Structure'
    ax.set_title(f'Silicon Band Structure: {path_str}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)

    output_file = dat_file.replace('.dat', '.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Band structure plot saved to: {output_file}")
    plt.show()

if __name__ == "__main__":
    plot_band_structure("si_bands.dat")

