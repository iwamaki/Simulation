#!/usr/bin/env python3
"""
DAMASK CPFEM 結果可視化スクリプト

応力-ひずみ線図の描画と、ParaView用VTKファイルのエクスポートを行う。

使用例:
    # 応力-ひずみ線図
    python3 simulations/DAMASK/single_crystal_tensile/visualize.py --job Cu_100_erate1e-03 --plot

    # ParaView用VTKエクスポート（応力・ひずみ・IPFカラー等）
    python3 simulations/DAMASK/single_crystal_tensile/visualize.py --job Cu_100_erate1e-03 --vtk

    # 両方
    python3 simulations/DAMASK/single_crystal_tensile/visualize.py --job Cu_100_erate1e-03 --plot --vtk

    # 複数ジョブの応力-ひずみ線図を重ねて比較
    python3 simulations/DAMASK/single_crystal_tensile/visualize.py --job Cu_100_erate1e-03 Cu_110_erate1e-03 Cu_111_erate1e-03 --plot
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_stress_strain(job_dirs, output_path=None):
    """応力-ひずみ線図を描画（複数ジョブの重ね描き対応）"""
    fig, ax = plt.subplots(figsize=(8, 6))

    for job_dir in job_dirs:
        csv_path = job_dir / "stress_strain.csv"
        if not csv_path.exists():
            print(f"  警告: {csv_path} が見つかりません、スキップ")
            continue

        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        strain = data[:, 0] * 100  # → %
        stress = data[:, 1]        # MPa

        label = job_dir.name
        ax.plot(strain, stress, linewidth=2, label=label)

    ax.set_xlabel("Engineering Strain (%)", fontsize=14)
    ax.set_ylabel("Stress (MPa)", fontsize=14)
    ax.set_title("Single Crystal Tensile - Stress-Strain Curve", fontsize=15)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=12)
    fig.tight_layout()

    if output_path is None:
        if len(job_dirs) == 1:
            output_path = job_dirs[0] / "stress_strain.png"
        else:
            output_path = job_dirs[0].parent / "stress_strain_comparison.png"

    fig.savefig(output_path, dpi=150)
    print(f"  応力-ひずみ線図保存: {output_path}")
    plt.close(fig)
    return output_path


def export_vtk(job_dir):
    """HDF5からParaView用VTKファイルをエクスポート"""
    import damask

    hdf5_files = list(job_dir.glob("*.hdf5"))
    if not hdf5_files:
        print(f"  エラー: HDF5ファイルが見つかりません: {job_dir}")
        return

    hdf5_path = hdf5_files[0]
    print(f"  HDF5読み込み: {hdf5_path}")

    r = damask.Result(str(hdf5_path))

    # --- 各種物理量を追加 ---
    print("  物理量を計算中...")

    # 応力・ひずみ（既に追加済みの場合はスキップ）
    try:
        r.add_stress_Cauchy()
    except ValueError:
        pass  # 既にHDF5に書き込み済み

    try:
        r.add_strain()
    except ValueError:
        pass

    # Mises相当応力・相当ひずみ
    try:
        r.add_equivalent_Mises("sigma")
    except (ValueError, KeyError):
        pass

    try:
        r.add_equivalent_Mises("epsilon_V^0.0(F)")
    except (ValueError, KeyError):
        pass

    # IPFカラー（逆極点図カラーマップ: 引張方向=z軸）
    try:
        r.add_IPF_color([0, 0, 1])
    except (ValueError, KeyError):
        pass

    # --- VTKエクスポート ---
    vtk_dir = job_dir / "vtk"
    vtk_dir.mkdir(exist_ok=True)

    print(f"  VTKエクスポート先: {vtk_dir}")
    r.export_VTK(target_dir=str(vtk_dir), parallel=False)
    print(f"  VTKエクスポート完了")

    # エクスポートされたファイル数を表示
    vti_files = list(vtk_dir.glob("*.vti"))
    print(f"  生成ファイル数: {len(vti_files)} 個")
    if vti_files:
        print(f"  例: {vti_files[0].name}")

    return vtk_dir


def main():
    parser = argparse.ArgumentParser(
        description="CPFEM 結果可視化（応力-ひずみ線図 / ParaView用VTK）"
    )
    parser.add_argument(
        "--job", nargs="+", required=True,
        help="ジョブディレクトリ名（例: Cu_100_erate1e-03）"
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="応力-ひずみ線図を描画"
    )
    parser.add_argument(
        "--vtk", action="store_true",
        help="ParaView用VTKファイルをエクスポート"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="応力-ひずみ線図の出力パス（--plot時）"
    )

    args = parser.parse_args()

    if not args.plot and not args.vtk:
        print("--plot または --vtk を指定してください")
        return

    sim_dir = Path("simulations/DAMASK/single_crystal_tensile")
    job_dirs = []
    for job in args.job:
        jd = sim_dir / job
        if not jd.exists():
            print(f"エラー: ジョブディレクトリが見つかりません: {jd}")
            return
        job_dirs.append(jd)

    if args.plot:
        print("\n=== 応力-ひずみ線図 ===")
        out = Path(args.output) if args.output else None
        plot_stress_strain(job_dirs, output_path=out)

    if args.vtk:
        for jd in job_dirs:
            print(f"\n=== VTKエクスポート: {jd.name} ===")
            export_vtk(jd)


if __name__ == "__main__":
    main()
