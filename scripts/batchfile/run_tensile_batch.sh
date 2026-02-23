#!/bin/bash

# ミラー指数のリスト
MILLERS=("1 1 0" "1 1 1")

# Cu の実行
for m in "${MILLERS[@]}"; do
  .venv/bin/python3 simulations/single_crystal_tensile/single_crystal_tensile.py \
    --element Cu --lattice 3.615 --potential potentials/Cu_zhou.eam.alloy --pair-style eam/alloy \
    --miller $m --target-size 50 --erate 0.0005 --max-strain 0.30 \
    --triclinic --halt-drop 2.0 
done

