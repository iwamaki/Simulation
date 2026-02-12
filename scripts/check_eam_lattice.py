"""
EAMポテンシャルの平衡格子定数を確認するスクリプト

AlCu.eam.alloy ポテンシャルでのCu/Alの
0K平衡格子定数を最小化で求める。
"""

import os
import subprocess
import tempfile
import re


def check_lattice_constant(element, a_guess, pot_path, lmp_cmd):
    """EAMポテンシャルの平衡格子定数を求める"""

    # AlCu.eam.alloy の元素順序: Al Cu
    if element == 'Al':
        pair_coeff_line = f"pair_coeff * * {pot_path} Al"
        mass = "1 26.982"
    else:
        pair_coeff_line = f"pair_coeff * * {pot_path} NULL Cu"
        pair_coeff_line = f"pair_coeff * * {pot_path} Cu"  # 1原子種ならこれでOK?
        # 2元素ポテンシャルで1元素だけ使う場合:
        # pair_coeff * * AlCu.eam.alloy NULL Cu → Type1をCuに（Alをスキップ）
        # でも1 atom typeならType1=Cu
        pair_coeff_line = f"pair_coeff * * {pot_path} Cu"
        mass = "1 63.546"

    with tempfile.TemporaryDirectory() as tmpdir:
        input_content = f"""
units metal
boundary p p p
atom_style atomic
lattice fcc {a_guess}
region box block 0 4 0 4 0 4
create_box 1 box
create_atoms 1 box
mass {mass}
pair_style eam/alloy
{pair_coeff_line}
thermo 1
thermo_style custom step pe lx ly lz press pxx pyy pzz

# 箱も原子も同時に緩和（box/relax iso 0.0）
fix relax all box/relax iso 0.0 vmax 0.001
minimize 1.0e-10 1.0e-12 10000 100000
unfix relax

variable a_eq equal lx/4
print "EQUILIBRIUM_A ${{a_eq}}"
print "FINAL_PE $(pe/count(all))"
print "FINAL_PRESS $(press)"
"""
        input_file = os.path.join(tmpdir, "in.check")
        with open(input_file, 'w') as f:
            f.write(input_content)

        result = subprocess.run(
            f"{lmp_cmd} < {input_file}",
            shell=True, capture_output=True, text=True, cwd=tmpdir
        )

        output = result.stdout + result.stderr

        a_eq = None
        pe = None
        press = None

        for line in output.split('\n'):
            if 'EQUILIBRIUM_A' in line:
                match = re.search(r'EQUILIBRIUM_A\s+([\d.]+)', line)
                if match:
                    a_eq = float(match.group(1))
            if 'FINAL_PE' in line:
                match = re.search(r'FINAL_PE\s+([-\d.]+)', line)
                if match:
                    pe = float(match.group(1))
            if 'FINAL_PRESS' in line:
                match = re.search(r'FINAL_PRESS\s+([-\d.e+]+)', line)
                if match:
                    press = float(match.group(1))

        return a_eq, pe, press


def main():
    project_root = os.getcwd()
    pot_path = os.path.abspath(os.path.join(project_root, "potentials", "AlCu.eam.alloy"))
    lmp_cmd = os.environ.get("ASE_LAMMPSRUN_COMMAND", "/home/iwash/lammps/build/lmp")

    print("=" * 50)
    print("EAMポテンシャル平衡格子定数の確認")
    print(f"ポテンシャル: AlCu.eam.alloy")
    print("=" * 50)

    for element, a_guess, a_exp in [('Cu', 3.61, 3.615), ('Al', 4.05, 4.050)]:
        a_eq, pe, press = check_lattice_constant(element, a_guess, pot_path, lmp_cmd)
        print(f"\n--- {element} ---")
        print(f"  入力値 (ドライバ): a = {a_guess:.4f} Å")
        print(f"  実験値:            a = {a_exp:.4f} Å")
        if a_eq:
            print(f"  EAMポテンシャル:   a = {a_eq:.4f} Å")
            print(f"  凝集エネルギー:    {pe:.4f} eV/atom")
            print(f"  残留圧力:          {press:.2f} bar")
            diff = (a_guess - a_eq) / a_eq * 100
            print(f"  入力値との差:      {diff:+.3f}%")
            if abs(diff) > 0.5:
                print(f"  ⚠ ドライバの格子定数をEAM値に更新すべき！")
        else:
            print(f"  ERROR: 格子定数を取得できませんでした")


if __name__ == "__main__":
    main()
