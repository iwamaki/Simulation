"""
ヤング率・Zener比の解析

elastic_constants.py が出力した elastic_constants.csv を読み込み、
方位別ヤング率・Zener異方性比を算出する。
応力-ひずみ曲線の弾性傾きとの比較も行う。

使い方:
  python simulations/Zener_ratio_calculate/analyze_youngs.py
"""

import csv
import glob
import os
import numpy as np

PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join("simulations", "Zener_ratio_calculate", "data")


def load_csv(csv_path):
    """弾性定数CSVを読み込み、{element: (C11, C12, C44)} の辞書を返す。"""
    results = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            elem = row['element']
            results[elem] = (float(row['C11']), float(row['C12']),
                             float(row['C44']))
    return results


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
    csv_path = os.path.join(PROJECT_ROOT, DATA_DIR, "elastic_constants.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} が見つかりません。")
        print("先に elastic_constants.py を実行してください。")
        return

    elastic = load_csv(csv_path)

    print("=" * 70)
    print("検証A: 弾性定数と方位別ヤング率")
    print("=" * 70)

    all_results = {}

    for element, (C11, C12, C44) in elastic.items():
        E100, E110, E111, A = youngs_moduli(C11, C12, C44)
        B = (C11 + 2 * C12) / 3  # 体積弾性率

        print(f"\n--- {element} ---")
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
