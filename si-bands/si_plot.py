import numpy as np
import matplotlib.pyplot as plt

dat_file = "si_bands.dat"

# Read metadata from first 3 lines
with open(dat_file, 'r') as f:
    line1 = f.readline()
    line2 = f.readline()
    line3 = f.readline()

EF = float(line1.split(':')[1].strip())
kvals = [float(x) for x in line2.split(':')[1].strip().split()]
klabels = line3.split(':')[1].strip().split()

# Read band data
kpts, ens = [], []
kpt, en = [], []

with open(dat_file, 'r') as f:
    # Skip the first 3 metadata lines
    f.readline()
    f.readline()
    f.readline()
    
    for line in f:
        if line.strip():    # not empty
            parts = line.split()
            kpt.append(float(parts[0]))
            en.append(float(parts[1]) - EF)
        else:               # empty line => change band
            kpts.append(kpt)
            ens.append(en)
            kpt, en = [], []

# Add the last band if it has data
if kpt:
    kpts.append(kpt)
    ens.append(en)

nbnd = len(kpts)
print(f"Plotting {nbnd} bands...")

plt.figure(figsize=(5, 5))
for i in range(nbnd):
    plt.plot(kpts[i], ens[i], color='blue')

for k in kvals:
    plt.axvline(k, color='black', linestyle='-', linewidth=0.8)
plt.axhline(0.0, linestyle='--', color='red', linewidth=1.0)

plt.xticks(kvals, klabels)
plt.xlim(min(kvals), max(kvals))
plt.ylim(-15, 10)
plt.ylabel(r"Energy − $E_F$ (eV)")
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(dat_file.replace('.dat', '.png'), dpi=300)

print(f"Band structure plot saved to: {dat_file.replace('.dat', '.png')}")
