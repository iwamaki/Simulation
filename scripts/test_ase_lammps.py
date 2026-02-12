"""ASE + LAMMPS 動作確認テスト
Cu単体のFCC結晶でエネルギー計算ができることを確認する
"""
from ase.build import bulk
from ase.calculators.lammpsrun import LAMMPS
import os

# ポテンシャルファイルのパス
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POTENTIAL_DIR = os.path.join(PROJECT_DIR, "potentials")

# Cu FCC結晶を作成
cu = bulk("Cu", "fcc", a=3.615, cubic=True)
print(f"Cu unit cell: {len(cu)} atoms")
print(f"Cell:\n{cu.cell}")

# LAMMPS calculator の設定
LAMMPS_CMD = os.environ.get("ASE_LAMMPSRUN_COMMAND", "/home/iwash/lammps/build/lmp")

calc = LAMMPS(
    command=LAMMPS_CMD,
    pair_style="eam/alloy",
    pair_coeff=[f"* * {os.path.join(POTENTIAL_DIR, 'AlCu.eam.alloy')} Cu"],
    mass=["1 63.546"],
    keep_tmp_dir=False,
)
cu.calc = calc

# エネルギー計算
energy = cu.get_potential_energy()
forces = cu.get_forces()

print(f"\nPotential energy: {energy:.4f} eV")
print(f"Energy per atom:  {energy / len(cu):.4f} eV/atom")
print(f"Max force:        {abs(forces).max():.6f} eV/Å")
print(f"\n✓ ASE + LAMMPS connection works!")
