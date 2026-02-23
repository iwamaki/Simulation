"""
検証A: 弾性定数と方位別ヤング率の計算

各EAMポテンシャルからC11, C12, C44を求め、
方位別ヤング率 E[100], E[110], E[111] を算出する。
応力-ひずみ曲線の弾性域の傾きと比較し、疑問1の仮説を検証する。

使い方:
  python simulations/single_crystal_tensile/verify_elastic.py
"""

import os
import subprocess
import tempfile
import glob
import numpy as np

PROJECT_ROOT = os.getcwd()
LAMMPS = os.environ.get("ASE_LAMMPSRUN_COMMAND",
                         "/home/iwash/lammps/build/lmp")
DELTA = 1e-6  # 有限差分のひずみ振幅

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
        # 工学せん断ひずみ γ = tilt / Ly = (delta * Ly) / Ly = delta
        deform_cmd = (f"change_box all triclinic\n"
                      f"change_box all xy delta {delta * a} remap")

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
            # "STRESS pxx VAL pyy VAL pzz VAL pxy VAL pxz VAL pyz VAL"
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
                stress = parse_stress(output)
                results[(strain_type, sign)] = stress

    # 中心差分: dσ/dε = -(P(+δ) - P(-δ)) / (2δ)
    # 単位変換: bar → GPa (÷10000)
    s_plus = results[('xx', +1)]
    s_minus = results[('xx', -1)]

    C11 = -(s_plus[0] - s_minus[0]) / (2 * DELTA) / 10000  # GPa
    C12 = -(s_plus[1] - s_minus[1]) / (2 * DELTA) / 10000  # GPa

    s_plus = results[('xy', +1)]
    s_minus = results[('xy', -1)]

    # せん断ひずみ γ₆ = DELTA, σ₆ = -pxy
    C44 = -(s_plus[3] - s_minus[3]) / (2 * DELTA) / 10000  # GPa

    return C11, C12, C44


def youngs_moduli(C11, C12, C44):
    """立方晶の弾性コンプライアンスから方位別ヤング率を計算。"""
    # コンプライアンステンソル
    S11 = (C11 + C12) / ((C11 - C12) * (C11 + 2 * C12))
    S12 = -C12 / ((C11 - C12) * (C11 + 2 * C12))
    S44 = 1.0 / C44

    def E_hkl(h, k, l):
        """E[hkl] = 1 / (S11 - 2(S11 - S12 - S44/2) * Γ)"""
        norm = np.sqrt(h**2 + k**2 + l**2)
        l1, l2, l3 = h / norm, k / norm, l / norm
        gamma = l1**2 * l2**2 + l2**2 * l3**2 + l3**2 * l1**2
        return 1.0 / (S11 - 2 * (S11 - S12 - S44 / 2) * gamma)

    E100 = E_hkl(1, 0, 0)
    E110 = E_hkl(1, 1, 0)
    E111 = E_hkl(1, 1, 1)
    A = 2 * C44 / (C11 - C12)  # Zener異方性比

    return E100, E110, E111, A


def measure_elastic_slope(element, hkl):
    """応力-ひずみデータの弾性域から傾き（ヤング率）を測定。"""
    sim_dir = os.path.join(PROJECT_ROOT, "simulations", "single_crystal_tensile",
                           "data", "02_本番実験", "L50")
    # ファイルを探す
    pattern = os.path.join(sim_dir, "*",
                           f"{element}{hkl}_tension_*/stress_strain.txt")
    files = glob.glob(pattern)
    if not files:
        return None

    # 読み込み（ヘッダ行 "# strain stress" をスキップ）
    data = np.loadtxt(files[0], comments='#')
    if data.ndim != 2 or data.shape[1] < 2:
        return None

    strain, stress = data[:, 0], data[:, 1]

    # ε < 0.02 の範囲で線形フィット
    mask = (strain > 0.001) & (strain < 0.02)
    if mask.sum() < 3:
        return None

    coeffs = np.polyfit(strain[mask], stress[mask], 1)
    return coeffs[0]  # 傾き = ヤング率 (GPa)


def main():
    print("=" * 70)
    print("検証A: 弾性定数と方位別ヤング率")
    print("=" * 70)

    all_results = {}

    for element, mat in MATERIALS.items():
        print(f"\n--- {element} (a = {mat['a']} Å, {mat['pair_style']}) ---")

        C11, C12, C44 = compute_elastic_constants(element, mat)
        E100, E110, E111, A = youngs_moduli(C11, C12, C44)
        B = (C11 + 2 * C12) / 3  # 体積弾性率

        print(f"  弾性定数:")
        print(f"    C11 = {C11:.1f} GPa")
        print(f"    C12 = {C12:.1f} GPa")
        print(f"    C44 = {C44:.1f} GPa")
        print(f"    体積弾性率 B = {B:.1f} GPa")
        print(f"    Zener比 A = 2C44/(C11-C12) = {A:.3f}")
        print(f"  方位別ヤング率 (0K, ポテンシャル):")
        print(f"    E[100] = {E100:.1f} GPa")
        print(f"    E[110] = {E110:.1f} GPa")
        print(f"    E[111] = {E111:.1f} GPa")
        print(f"    E[110]/E[100] = {E110 / E100:.3f}")
        print(f"    E[111]/E[100] = {E111 / E100:.3f}")

        # 応力-ひずみ曲線の傾きと比較
        print(f"  応力-ひずみ曲線の弾性傾き (300K MD):")
        for hkl in ['100', '110', '111']:
            slope = measure_elastic_slope(element, hkl)
            if slope is not None:
                print(f"    E[{hkl}]_MD = {slope:.1f} GPa")
            else:
                print(f"    E[{hkl}]_MD = (データなし)")

        all_results[element] = {
            'C11': C11, 'C12': C12, 'C44': C44,
            'E100': E100, 'E110': E110, 'E111': E111,
            'A': A, 'B': B,
        }

    # 比較表
    print("\n" + "=" * 70)
    print("比較表: 方位別ヤング率")
    print("=" * 70)
    print(f"{'元素':>4} | {'E[100]':>8} {'E[110]':>8} {'E[111]':>8} | "
          f"{'E110/E100':>9} {'E111/E100':>9} | {'Zener A':>8}")
    print("-" * 70)
    for element, r in all_results.items():
        print(f"{element:>4} | {r['E100']:8.1f} {r['E110']:8.1f} {r['E111']:8.1f} | "
              f"{r['E110'] / r['E100']:9.3f} {r['E111'] / r['E100']:9.3f} | "
              f"{r['A']:8.3f}")

    print("\n※ ヤング率は0Kでの値。300K MDの弾性傾きは熱揺らぎにより5-10%低い可能性あり。")
    print("※ A > 1 のとき E[111] > E[110] > E[100]（弾性域で{110}の方が{100}より硬い）")


if __name__ == '__main__':
    main()
