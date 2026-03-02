"""
弾性定数の計算（LAMMPSドライバ）

各EAMポテンシャルからC11, C12, C44を中心差分法で求め、
結果をCSVに保存する。

使い方:
  python simulations/Zener_ratio_calculate/elastic_constants.py
"""

import csv
import os
import subprocess
import tempfile
import numpy as np

PROJECT_ROOT = os.getcwd()
LAMMPS = os.environ.get("ASE_LAMMPSRUN_COMMAND",
                         "/home/iwash/lammps/build/lmp")
DELTA = 1e-6  # 有限差分のひずみ振幅
DATA_DIR = os.path.join("simulations", "Zener_ratio_calculate", "data")

# 各元素のポテンシャル設定
MATERIALS = {
    'Cu': {'potential': 'potentials/Cu_zhou.eam.alloy',
           'pair_style': 'eam/alloy', 'a': 3.615},
    'Ag': {'potential': 'potentials/Ag_u3.eam',
           'pair_style': 'eam', 'a': 4.09},
    'Au': {'potential': 'potentials/Au_u3.eam',
           'pair_style': 'eam', 'a': 4.078},
}


def pair_coeff_line(element, style, pot_path):
    if style in ('eam/alloy', 'eam/fs'):
        return f"pair_coeff * * {pot_path} {element}"
    return f"pair_coeff 1 1 {pot_path}"


def make_lammps_input(element, mat, strain_type, strain_sign):
    """1つのひずみ摂動に対するLAMMPS入力を生成。

    strain_type: 'xx' (垂直ひずみ) or 'xy' (せん断ひずみ)
    strain_sign: +1 or -1
    """
    pot_path = os.path.abspath(os.path.join(PROJECT_ROOT, mat['potential']))
    delta = DELTA * strain_sign
    a = mat['a']

    if strain_type == 'xx':
        deform_cmd = f"change_box all x scale {1.0 + delta} remap"
    elif strain_type == 'xy':
        # change_box xy delta は lattice 単位で解釈される（内部で × a_lattice）
        # → delta を渡せば実tilt = delta * a, γ = delta * a / Ly ≈ delta
        deform_cmd = (f"change_box all triclinic\n"
                      f"change_box all xy delta {delta} remap")

    script = f"""# 弾性定数計算: {element} {strain_type} {'+'if strain_sign>0 else '-'}δ
units metal
boundary p p p
atom_style atomic

lattice fcc {a}
region box block 0 1 0 1 0 1
create_box 1 box
create_atoms 1 box
mass 1 1.0

pair_style {mat['pair_style']}
{pair_coeff_line(element, mat['pair_style'], pot_path)}

neighbor 2.0 bin
neigh_modify delay 0

# 構造緩和（平衡格子定数を決定）
fix relax all box/relax aniso 0.0 vmax 0.001
minimize 1.0e-12 1.0e-14 10000 100000
unfix relax

# ひずみ印加
{deform_cmd}

# 原子位置のみ緩和（FCC Bravais格子なので実質不要だが念のため）
minimize 1.0e-12 1.0e-14 1000 10000

# ボックス寸法確認
print "BOX lx $(lx) ly $(ly) lz $(lz) xy $(xy) xz $(xz) yz $(yz)"

# 応力テンソル出力
print "STRESS pxx $(pxx) pyy $(pyy) pzz $(pzz) pxy $(pxy) pxz $(pxz) pyz $(pyz)"
"""
    return script


def run_lammps(script, workdir):
    """LAMMPSを実行して出力を返す。"""
    input_file = os.path.join(workdir, "in.elastic")
    with open(input_file, 'w') as f:
        f.write(script)

    result = subprocess.run(
        f"{LAMMPS} -in in.elastic -log none",
        shell=True, cwd=workdir, capture_output=True, text=True, timeout=60)

    if result.returncode != 0:
        print(f"  LAMMPS error:\n{result.stderr[-500:]}")
        raise RuntimeError("LAMMPS failed")

    return result.stdout


def parse_stress(output):
    """LAMMPS出力から応力テンソルを抽出。[pxx, pyy, pzz, pxy, pxz, pyz] (bar)"""
    for line in output.split('\n'):
        if line.startswith('STRESS pxx'):
            parts = line.split()
            vals = {}
            for i in range(1, len(parts), 2):
                vals[parts[i]] = float(parts[i + 1])
            return [vals['pxx'], vals['pyy'], vals['pzz'],
                    vals['pxy'], vals['pxz'], vals['pyz']]
    raise ValueError("STRESS行が見つからない")


def compute_elastic_constants(element, mat):
    """C11, C12, C44を中心差分法で計算。"""
    results = {}

    with tempfile.TemporaryDirectory() as tmpdir:
        for strain_type in ['xx', 'xy']:
            for sign in [+1, -1]:
                script = make_lammps_input(element, mat, strain_type, sign)
                output = run_lammps(script, tmpdir)
                # BOX情報を表示
                for line in output.split('\n'):
                    if line.startswith('BOX lx'):
                        print(f"  [DEBUG] {strain_type} sign={sign:+d}: {line}")
                        break
                stress = parse_stress(output)
                results[(strain_type, sign)] = stress
                print(f"  [DEBUG] {strain_type} sign={sign:+d}: "
                      f"pxx={stress[0]:.6f} pyy={stress[1]:.6f} pzz={stress[2]:.6f} "
                      f"pxy={stress[3]:.6f} pxz={stress[4]:.6f} pyz={stress[5]:.6f}")

    # 中心差分: dσ/dε = -(P(+δ) - P(-δ)) / (2δ)
    # 単位変換: bar → GPa (÷10000)
    s_plus = results[('xx', +1)]
    s_minus = results[('xx', -1)]

    C11 = -(s_plus[0] - s_minus[0]) / (2 * DELTA) / 10000  # GPa
    C12 = -(s_plus[1] - s_minus[1]) / (2 * DELTA) / 10000  # GPa

    s_plus = results[('xy', +1)]
    s_minus = results[('xy', -1)]

    print(f"  [DEBUG] xy pxy差分: {s_plus[3]:.6f} - {s_minus[3]:.6f} = {s_plus[3] - s_minus[3]:.6f}")

    # せん断ひずみ γ₆ = DELTA, σ₆ = -pxy
    C44 = -(s_plus[3] - s_minus[3]) / (2 * DELTA) / 10000  # GPa

    return C11, C12, C44


def save_csv(all_results):
    """弾性定数をCSVに保存。"""
    out_dir = os.path.join(PROJECT_ROOT, DATA_DIR)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "elastic_constants.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["element", "C11", "C12", "C44"])
        for element, (C11, C12, C44) in all_results.items():
            writer.writerow([element, f"{C11:.4f}", f"{C12:.4f}", f"{C44:.4f}"])

    print(f"\nCSV saved: {csv_path}")
    return csv_path


def main():
    print("=" * 70)
    print("弾性定数の計算（LAMMPS中心差分法）")
    print("=" * 70)

    all_results = {}

    for element, mat in MATERIALS.items():
        print(f"\n--- {element} (a = {mat['a']} Å, {mat['pair_style']}) ---")

        C11, C12, C44 = compute_elastic_constants(element, mat)

        print(f"  C11 = {C11:.1f} GPa")
        print(f"  C12 = {C12:.1f} GPa")
        print(f"  C44 = {C44:.1f} GPa")

        all_results[element] = (C11, C12, C44)

    save_csv(all_results)


if __name__ == '__main__':
    main()
