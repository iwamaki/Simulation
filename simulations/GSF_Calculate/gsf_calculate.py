"""
検証B: 一般化積層欠陥エネルギー（GSF）の計算

{111}面を<112>方向にずらしたときのエネルギー曲線を計算し、
不安定積層欠陥エネルギー γ_us を Au, Ag, Cu で比較する。

LAMMPSの lattice orient で直接{111}スラブを構築。
  x: [1,-1,0], y: [1,1,-2], z: [1,1,1]

使い方:
  python simulations/GSF_Calculate/gsf_calculate.py
"""

import csv
import os
import shutil
import subprocess
import tempfile
import numpy as np

# ASEモジュールを追加
from ase.build import fcc111
from ase.io import write
from ase.io.lammpsdata import write_lammps_data

PROJECT_ROOT = os.getcwd()
LAMMPS = os.environ.get("ASE_LAMMPSRUN_COMMAND",
                         "/home/iwash/lammps/build/lmp")

# 各元素のポテンシャル設定
MATERIALS = {
    'Cu': {'potential': 'potentials/Cu_zhou.eam.alloy',
           'pair_style': 'eam/alloy', 'a': 3.615},
    'Ag': {'potential': 'potentials/Ag_u3.eam',
           'pair_style': 'eam', 'a': 4.09},
    'Au': {'potential': 'potentials/Au_u3.eam',
           'pair_style': 'eam', 'a': 4.078},
}

# スラブパラメータ
# lattice orient → block (NX, NY, NZ) in lattice units
# x: [1,-1,0] repeat = a√2/2, y: [1,1,-2] repeat = a√6/2, z: [1,1,1] repeat = a√3
NX, NY, NZ = 3, 6, 8  # NY=2(約4.4A)だとカットオフより小さいので6(約13A)に拡大
N_STEPS = 40
DATA_DIR = os.path.join("simulations", "GSF_Calculate", "data")


def pair_coeff_line(element, style, pot_path):
    if style in ('eam/alloy', 'eam/fs'):
        return f"pair_coeff * * {pot_path} {element}"
    return f"pair_coeff 1 1 {pot_path}"


def run_lammps(script, workdir):
    """LAMMPSを実行して出力を返す。"""
    input_file = os.path.join(workdir, "in.gsf")
    with open(input_file, 'w') as f:
        f.write(script)

    result = subprocess.run(
        f"{LAMMPS} -in in.gsf -log none",
        shell=True, cwd=workdir, capture_output=True, text=True, timeout=120)

    if result.returncode != 0:
        raise RuntimeError(f"LAMMPS failed:\n{result.stderr[-500:]}")

    return result.stdout


def compute_gsf(element, mat):
    """GSF曲線を計算（ASEで構造構築 → 確認用保存 → LAMMPS実行）。"""
    a = mat['a']
    pot_path = os.path.abspath(os.path.join(PROJECT_ROOT, mat['potential']))

    # 1. ASEで構造を作成
    # fcc111(orthogonal=True) で x=[1-10], y=[11-2], z=[111] の直交セルを作成
    slab = fcc111(element, size=(NX, NY, NZ), a=a, vacuum=0.0, orthogonal=True)
    slab.pbc = [True, True, False]  # z方向は非周期(Slab)

    # 2. 確認用に構造を保存（ここを確認してください！）
    viz_dir = os.path.join(PROJECT_ROOT, DATA_DIR, "structures")
    os.makedirs(viz_dir, exist_ok=True)
    check_file = os.path.join(viz_dir, f"{element}_initial.xyz")
    write(check_file, slab)
    print(f"  [Check] Structure saved: {check_file}")

    # 上下分割位置の決定 (原子のZ座標分布から層間を見つける)
    z_coords = slab.positions[:, 2]
    z_unique = np.unique(z_coords.round(decimals=5))
    z_unique.sort()
    
    # 中央付近の層間で分割
    mid_idx = len(z_unique) // 2
    z_split_box = (z_unique[mid_idx-1] + z_unique[mid_idx]) / 2.0

    # 部分バーガースベクトル = a/√6（y方向）
    b_partial = a / np.sqrt(6)
    # 変位範囲: 0 → b_partial (0 → γ_us → γ_isf)
    max_disp = b_partial
    displacements = np.linspace(0, max_disp, N_STEPS + 1)

    energies = []
    area = None

    with tempfile.TemporaryDirectory() as tmpdir:
        # 3. LAMMPS用データファイルを作成
        data_file = os.path.join(tmpdir, "system.data")
        write_lammps_data(data_file, slab, atom_style='atomic')

        for i, dy in enumerate(displacements):
            # 変位がゼロの場合は displace_atoms をスキップ
            displace_cmd = ""
            if dy > 0:
                # a/6[-1-12]方向: (0, -dy, 0)
                # +y方向だとLayer4原子がLayer3原子と重なるため、-y方向に滑らせる
                displace_cmd = f"displace_atoms upper move 0 {-dy:.10f} 0 units box"

            script = f"""# GSF計算: {element} step {i}/{N_STEPS}, dy={dy:.4f} Å
units metal
boundary p p s
atom_style atomic

# ASEで作った構造データを読み込み
read_data system.data

pair_style {mat['pair_style']}
{pair_coeff_line(element, mat['pair_style'], pot_path)}

neighbor 2.0 bin
neigh_modify delay 0

# Phase 1: セルは固定したまま、初期応力を抜くための軽い座標緩和のみ行う
minimize 1.0e-12 1.0e-14 1000 10000

# 上半分のグループ定義（z方向 = [111]方向）
# ASE側で決定した分割位置を使用
region upper block INF INF INF INF {z_split_box:.6f} INF units box
group upper region upper
group lower subtract all upper

# Phase 2: 変位印加 + z方向のみ緩和
{displace_cmd}
fix zonly all setforce 0.0 0.0 NULL
minimize 1.0e-10 1.0e-12 10000 100000

print "TOTAL_PE $(pe) NATOMS $(atoms) N_UPPER $(count(upper)) N_LOWER $(count(lower)) LX $(lx) LY $(ly) LZ $(lz)"

# アニメーション用: minimize後にタイムステップを設定し、フレーム順序を保証
reset_timestep {i}
write_dump all custom dump.lammpstrj id type x y z modify append yes
"""
            output = run_lammps(script, tmpdir)

            # エネルギー抽出
            pe = None
            for line in output.split('\n'):
                if line.startswith('TOTAL_PE'):
                    parts = line.split()
                    pe = float(parts[1])
                    if i == 0:
                        natoms = int(float(parts[3]))
                        n_upper = int(float(parts[5]))
                        n_lower = int(float(parts[7]))
                        lx = float(parts[9])
                        ly = float(parts[11])
                        lz = float(parts[13])
                        area = lx * ly
                        print(f"  Box: {lx:.2f} x {ly:.2f} x {lz:.2f} Å")
                        print(f"  Area = {area:.1f} Å²")
                        print(f"  原子数: {natoms}, 上半分: {n_upper}, 下半分: {n_lower}")
                        if abs(n_upper - n_lower) > natoms * 0.05:
                            print(f"  WARNING: 上下の分割が不均等！")
            if pe is None:
                raise ValueError(f"TOTAL_PE not found at step {i}")

            energies.append(pe)

            if i % 10 == 0:
                print(f"  {element}: step {i}/{N_STEPS}, dy={dy:.4f} Å, PE={pe:.4f} eV")

        # dumpファイルを出力先にコピー
        dump_src = os.path.join(tmpdir, "dump.lammpstrj")
        if os.path.exists(dump_src):
            dump_dir = os.path.join(PROJECT_ROOT, DATA_DIR, element)
            os.makedirs(dump_dir, exist_ok=True)
            shutil.copy2(dump_src, os.path.join(dump_dir,
                                                 f"dump_{element}.lammpstrj"))
            print(f"  Dump saved: {DATA_DIR}/{element}/dump_{element}.lammpstrj")

    energies = np.array(energies)

    # GSFエネルギー: (E - E0) / 面積 — 周期境界ではないので1つの欠陥面のみ
    # eV/Å² → mJ/m² の変換: × 16021.7663
    gsf = (energies - energies[0]) / area * 16021.7663

    # 変位比（0~1 = 0~b_partial）
    disp_frac = displacements / max_disp

    # γ_us: 曲線全体の最大値（不安定積層欠陥）
    gamma_us = np.max(gsf)

    # γ_isf: 終端の値（変位 = b_partial = 固有積層欠陥）
    gamma_isf = gsf[-1]

    return disp_frac, gsf, gamma_us, gamma_isf


def save_csv(results):
    """計算結果をCSVに保存。"""
    out_dir = os.path.join(PROJECT_ROOT, DATA_DIR)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "gsf_results.csv")

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # ヘッダー: disp_frac, Cu, Ag, Au, ...
        elements = list(results.keys())
        writer.writerow(["disp_frac"] + elements)
        # 各行: 変位比, 各元素のGSFエネルギー
        n = len(results[elements[0]][0])
        for j in range(n):
            row = [f"{results[elements[0]][0][j]:.6f}"]
            for elem in elements:
                row.append(f"{results[elem][1][j]:.4f}")
            writer.writerow(row)

    print(f"\nCSV saved: {csv_path}")
    return csv_path


def main():
    print("=" * 70)
    print("検証B: 一般化積層欠陥エネルギー（GSF曲線）")
    print("=" * 70)

    results = {}

    for element, mat in MATERIALS.items():
        print(f"\n--- {element} (a = {mat['a']} Å) ---")
        disp_frac, gsf, gamma_us, gamma_isf = compute_gsf(element, mat)
        results[element] = (disp_frac, gsf, gamma_us, gamma_isf)

        print(f"  γ_us  = {gamma_us:.1f} mJ/m² (unstable SF)")
        print(f"  γ_isf = {gamma_isf:.1f} mJ/m² (intrinsic SF)")

    # 比較表
    print("\n" + "=" * 70)
    print("比較表: 積層欠陥エネルギー")
    print("=" * 70)
    print(f"{'元素':>4} | {'γ_us (mJ/m²)':>14} | {'γ_isf (mJ/m²)':>15} | {'γ_us/γ_isf':>11}")
    print("-" * 55)
    for element, (_, _, gamma_us, gamma_isf) in results.items():
        ratio = gamma_us / gamma_isf if gamma_isf > 0 else float('inf')
        print(f"{element:>4} | {gamma_us:14.1f} | {gamma_isf:15.1f} | {ratio:11.2f}")

    # Au と Ag の比較
    if 'Au' in results and 'Ag' in results:
        au_us = results['Au'][2]
        ag_us = results['Ag'][2]
        diff_pct = abs(au_us - ag_us) / ((au_us + ag_us) / 2) * 100
        print(f"\nAu γ_us / Ag γ_us = {au_us / ag_us:.3f} (diff: {diff_pct:.1f}%)")
        if diff_pct < 15:
            print("→ Au ≈ Ag: γ_us is similar → explains {111} ideal strength match")
        else:
            print("→ Au ≠ Ag: γ_us differs significantly")

    # CSV保存
    save_csv(results)


if __name__ == '__main__':
    main()
