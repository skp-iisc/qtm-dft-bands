import numpy as np
import matplotlib.pyplot as plt

data = "qe-bands/bandx.dat.gnu"

EF = 6.457
kvals = [0.0, 0.866, 1.866, 2.2196, 3.2802]
klabels = ['L', r'$\Gamma$', 'X', 'U', r'$\Gamma$']

kpts, ens = [], []
kpt, en = [], []

with open(data, 'r') as f:
    for line in f:
        if line.strip():    # not empty
            parts = line.split()
            kpt.append(float(parts[0]))
            en.append(float(parts[1]) - EF)
        else:               # empty line => change band
            kpts.append(kpt)
            ens.append(en)
            kpt, en = [], []

nbnd = len(kpts)
print(f"Plotting {nbnd} bands...")

plt.figure(figsize=(8, 6))
for i in range(nbnd):
    plt.plot(kpts[i], ens[i], color='blue')

for k in kvals:
    plt.axvline(k, color='black', linestyle='-', linewidth=0.5)
plt.axhline(0.0, linestyle='--', color='black')

plt.xticks(kvals, klabels)
plt.xlim(min(kvals), max(kvals))
# plt.ylim(-10, 15)
plt.ylabel(r"Energy - $E_F$ (eV)")
plt.tight_layout()
plt.savefig("qe-bands/si_bands.png", dpi=300)

print("Band structure saved as qe-bands/si_bands.png")
