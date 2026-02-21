.venv/bin/python3 simulations/single_crystal_tensile/single_crystal_tensile.py --element Ag  --lattice 4.09 --potential potentials/Ag_u3.eam --pair-style eam --miller 1 0 0 --max-strain 1.0 --target-size 24.54 --runpod --gpu --label extended_100 --no-halt

.venv/bin/python3 simulations/single_crystal_tensile/single_crystal_tensile.py --element Au --lattice 4.078 --potential potentials/Au_u3.eam --pair-style eam --miller 1 0 0 --max-strain 1.0 --target-size 24.47 --runpod --gpu --label extended_100 --no-halt
