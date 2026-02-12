"""
平衡化後の格子ひずみ分析スクリプト

NPT平衡化後のセル寸法から、Cu/Al各領域の格子ひずみを推定する。
異種金属界面では、熱膨張係数の差によりCu/Alが同じ箱寸法を
共有することで、一方が引張、他方が圧縮のひずみを受ける。

使い方:
    python scripts/analyze_lattice_strain.py
"""

import os
import sys
import re
import numpy as np


def parse_log_equilibrium(log_file):
    """ログファイルから引張試験開始時のセル寸法と圧力を取得"""
    with open(log_file, 'r') as f:
        content = f.read()

    # 'variable L0 equal <value>' の行からLzを取得
    match = re.search(r'variable L0 equal (\d+\.\d+)', content)
    lz_eq = float(match.group(1)) if match else None

    # 引張試験開始時のthermo行を取得
    # 'Tensile Test' の後の最初の数値行
    lines = content.split('\n')
    tensile_start = False
    header_found = False
    lx_eq = ly_eq = None
    pxx = pyy = pzz = None

    for line in lines:
        if '--- Tensile Test ---' in line:
            tensile_start = True
            continue
        if tensile_start and 'Step' in line and 'Temp' in line:
            header_found = True
            continue
        if tensile_start and header_found:
            parts = line.strip().split()
            if len(parts) >= 9:
                try:
                    step = int(parts[0])
                    if step == 0:
                        pxx = float(parts[3])
                        pyy = float(parts[4])
                        pzz = float(parts[5])
                        lx_eq = float(parts[6])
                        ly_eq = float(parts[7])
                        lz_eq_thermo = float(parts[8])
                        if lz_eq is None:
                            lz_eq = lz_eq_thermo
                        break
                except (ValueError, IndexError):
                    continue

    return {
        'lx': lx_eq, 'ly': ly_eq, 'lz': lz_eq,
        'pxx': pxx, 'pyy': pyy, 'pzz': pzz
    }


def analyze_job(job_dir, job_name):
    """1つのジョブの格子ひずみを分析"""
    print(f"\n{'='*60}")
    print(f"格子ひずみ分析: {job_name}")
    print(f"{'='*60}")

    log_file = os.path.join(job_dir, "log.lammps")
    data_file = os.path.join(job_dir, "data.interface")

    if not os.path.exists(log_file):
        print("  log.lammps が見つかりません")
        return

    # 初期セル寸法（データファイルから）
    lx_init = ly_init = lz_init = None
    with open(data_file, 'r') as f:
        for line in f:
            if 'xlo xhi' in line:
                parts = line.split()
                lx_init = float(parts[1]) - float(parts[0])
            elif 'ylo yhi' in line:
                parts = line.split()
                ly_init = float(parts[1]) - float(parts[0])
            elif 'zlo zhi' in line:
                parts = line.split()
                lz_init = float(parts[1]) - float(parts[0])

    # 平衡化後のセル寸法
    eq = parse_log_equilibrium(log_file)

    if eq['lx'] is None:
        print("  平衡化後のセル寸法を取得できませんでした")
        return

    print(f"\n--- セル寸法の変化 ---")
    print(f"  初期 (0K構造):  Lx={lx_init:.4f}  Ly={ly_init:.4f}  Lz={lz_init:.4f} Å")
    print(f"  平衡後 (300K):  Lx={eq['lx']:.4f}  Ly={eq['ly']:.4f}  Lz={eq['lz']:.4f} Å")

    dx = (eq['lx'] - lx_init) / lx_init * 100
    dy = (eq['ly'] - ly_init) / ly_init * 100
    dz = (eq['lz'] - lz_init) / lz_init * 100
    print(f"  変化率:         ΔLx={dx:+.3f}%  ΔLy={dy:+.3f}%  ΔLz={dz:+.3f}%")

    if eq['pxx'] is not None:
        print(f"\n--- 引張試験開始時のグローバル応力 ---")
        print(f"  Pxx = {eq['pxx']:+.1f} bar ({eq['pxx']/10000:+.4f} GPa)")
        print(f"  Pyy = {eq['pyy']:+.1f} bar ({eq['pyy']/10000:+.4f} GPa)")
        print(f"  Pzz = {eq['pzz']:+.1f} bar ({eq['pzz']/10000:+.4f} GPa)")

    # 理論的な格子ひずみ推定
    # 0K格子定数
    a_cu_0K = 3.61  # Å
    a_al_0K = 4.05  # Å

    # 熱膨張（線形近似）
    alpha_cu = 16.5e-6  # /K
    alpha_al = 23.1e-6  # /K
    T = 300.0  # K

    a_cu_300K = a_cu_0K * (1 + alpha_cu * T)
    a_al_300K = a_al_0K * (1 + alpha_al * T)

    print(f"\n--- 理論的な格子定数 (線形熱膨張) ---")
    print(f"  Cu: a(0K) = {a_cu_0K:.4f} → a(300K) ≈ {a_cu_300K:.4f} Å (Δ={alpha_cu*T*100:.3f}%)")
    print(f"  Al: a(0K) = {a_al_0K:.4f} → a(300K) ≈ {a_al_300K:.4f} Å (Δ={alpha_al*T*100:.3f}%)")

    # 方位に応じた理想的な表面セル寸法
    orientations = {
        'Cu100_Al100': {
            'cu_ideal_x': lambda a: a / np.sqrt(2),
            'al_ideal_x': lambda a: a / np.sqrt(2),
            'cu_rep_x': 9, 'al_rep_x': 8,
        },
        'Cu110_Al110': {
            'cu_ideal_x': lambda a: a,
            'al_ideal_x': lambda a: a,
            'cu_rep_x': 9, 'al_rep_x': 8,
        },
        'Cu111_Al111': {
            'cu_ideal_x': lambda a: a / np.sqrt(2),
            'al_ideal_x': lambda a: a / np.sqrt(2),
            'cu_rep_x': 9, 'al_rep_x': 8,
        },
    }

    for key in orientations:
        if key in job_name:
            info = orientations[key]

            # 300Kでの理想的なsupercell寸法
            cu_Lx_ideal = info['cu_rep_x'] * info['cu_ideal_x'](a_cu_300K)
            al_Lx_ideal = info['al_rep_x'] * info['al_ideal_x'](a_al_300K)

            # 実際の箱寸法（Cu/Al共通）
            actual_Lx = eq['lx']

            # 各材料の格子ひずみ（箱寸法と理想値の差）
            cu_strain = (actual_Lx - cu_Lx_ideal) / cu_Lx_ideal * 100
            al_strain = (actual_Lx - al_Lx_ideal) / al_Lx_ideal * 100

            print(f"\n--- X方向の格子ひずみ推定 (300K基準) ---")
            print(f"  Cu理想 ({info['cu_rep_x']}unit): {cu_Lx_ideal:.4f} Å")
            print(f"  Al理想 ({info['al_rep_x']}unit): {al_Lx_ideal:.4f} Å")
            print(f"  実際の箱:           {actual_Lx:.4f} Å")
            print(f"  Cu側ひずみ: {cu_strain:+.3f}% ({'引張' if cu_strain > 0 else '圧縮'})")
            print(f"  Al側ひずみ: {al_strain:+.3f}% ({'引張' if al_strain > 0 else '圧縮'})")

            print(f"\n  → CuとAlが同じ箱を共有するため、")
            if cu_strain > 0 and al_strain < 0:
                print(f"    Cu: 引張ひずみ, Al: 圧縮ひずみ（熱膨張差による残留応力）")
            elif cu_strain < 0 and al_strain > 0:
                print(f"    Cu: 圧縮ひずみ, Al: 引張ひずみ")
            else:
                print(f"    ※ EAMポテンシャルの平衡格子定数が教科書値と異なる可能性あり")
            print(f"    グローバル応力≈0 は、両者の応力が打ち消し合った結果")
            break


def main():
    sim_dir = os.path.join("simulations", "CuAl_interface_strength")

    if not os.path.exists(sim_dir):
        print(f"ERROR: {sim_dir} が見つかりません")
        sys.exit(1)

    print("=" * 60)
    print("Cu-Al界面 格子ひずみ分析")
    print("=" * 60)

    job_dirs = sorted([
        d for d in os.listdir(sim_dir)
        if os.path.isdir(os.path.join(sim_dir, d)) and d.endswith('K')
    ])

    for job_name in job_dirs:
        job_dir = os.path.join(sim_dir, job_name)
        analyze_job(job_dir, job_name)


if __name__ == "__main__":
    main()
