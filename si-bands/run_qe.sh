#!/bin/bash

if [[ -d qe-bands ]];
then
	rm -r qe-bands
fi
mkdir qe-bands
cd qe-bands

a=10.2
k=4
ecut=25.0

cat > si.scf.in << EOF
&control
	calculation = 'scf'
	restart_mode = 'from_scratch'
	prefix = 'Si'
	verbosity = 'medium'
	outdir = './tmp/'
	pseudo_dir = '../../pseudo/'
/
&system
	ibrav=2, celldm(1)=$a, nat=2, ntyp=1, ecutwfc=$ecut
/
&electrons
	mixing_beta = 0.7
	conv_thr = 1.0e-8
	diagonalization = 'david'
/
ATOMIC_SPECIES
 Si 28.086 Si_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS {alat}
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_POINTS {automatic}
 $k $k $k 0 0 0
EOF

# bands calculation
cat > si.bands.in << EOF
&control
	calculation = 'bands'
	prefix = 'Si'
	outdir = './tmp/'
	pseudo_dir = '../../pseudo/'
/
&system
	ibrav=2, celldm(1)=$a, nat=2, ntyp=1, ecutwfc=$ecut, nbnd=12
/
&electrons
	mixing_beta = 0.7
	conv_thr = 1.0e-7
	diagonalization = 'cg'
/
ATOMIC_SPECIES
 Si 28.086 Si_ONCV_PBE-1.2.upf
ATOMIC_POSITIONS {alat}
 Si 0.00 0.00 0.00
 Si 0.25 0.25 0.25
K_points {crystal_b}
5
0.000  0.500  0.000  50 ! L
0.000  0.000  0.000  50 ! GAMMA
-0.500  0.000  -0.500  50 ! X
-0.375  0.250  -0.375  50 ! K
0.000  0.000  0.000  0 ! GAMMA
EOF

cat > si.bandx.in << EOF
&bands
prefix = 'Si'
outdir = './tmp/'
filband = 'bandx.dat'
/
EOF

echo "Running the scf calculation..."
mpirun -np 4 pw.x < si.scf.in > si.scf.out

echo "Running the bands calculation..."
mpirun -np 4 pw.x < si.bands.in > si.bands.out

bands.x < si.bandx.in > si.bandx.out

rm -r tmp/		# removing temporary folder to free up space

cd ../

echo "Plotting QE bands in qe-bands/..."
python plot_qe_bands.py
