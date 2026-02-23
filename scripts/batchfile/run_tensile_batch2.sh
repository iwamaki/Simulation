#!/bin/bash

# ミラー指数のリスト
MILLERS=("1 0 0" "1 1 0" "1 1 1")

# Au の実行
for m in "${MILLERS[@]}"; do
  .venv/bin/python3 simulations/single_crystal_tensile/single_crystal_tensile.py \
    --element Au --lattice 4.078 --potential potentials/Au_u3.eam --pair-style eam \
    --miller $m --target-size 50 --erate 0.0005 --max-strain 0.30 \
    --triclinic --halt-drop 2.0
done

