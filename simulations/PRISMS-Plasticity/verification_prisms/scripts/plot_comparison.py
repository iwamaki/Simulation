"""
MD vs PRISMS-Plasticity 結果比較プロット

MDの生データ・スムース曲線とCP-FEM曲線を同一グラフに描画し、
弾性域の拡大パネルと定量比較テキストを含む図を生成する。
--orientations を指定すると、下段に変形段階ごとの極点図パネルを追加する。

Usage:
  python simulations/verification_prisms/scripts/plot_comparison.py \
    --md <stress_strain.txt> --prisms <prisms_output_dir> --output <comparison.png>

  # 極点図付き統合プロット
  python simulations/verification_prisms/scripts/plot_comparison.py \
    --md <stress_strain.txt> --prisms <prisms_output_dir> \
    --orientations <orientations.txt> --output <comparison.png>
"""

import argparse
import os
import sys
import glob

import numpy as np

sys.path.append(os.getcwd())

from simulations.verification_prisms.scripts.extract_md_params import (
    load_stress_strain,
    smooth_data,
    fit_elastic_modulus,
    find_yield_stress_offset,
)

from scripts.plot_pole_figure import (
    parse_multi_frame_orientations,
    quaternion_to_rotation_matrix_vectorized,
    get_symmetry_variants,
    stereographic_projection,
    draw_pole_figure_axes,
)

# plot_pole_figure が TkAgg を設定するため、インポート後に Agg に上書きする
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_prisms_results(prisms_dir):
    """
    PRISMS-Plasticity出力ディレクトリから体積平均応力ひずみデータを読み込む

    VTUファイルがある場合はpyvistaで体積平均を計算、
    なければCSV/テキストファイルを探す。

    Parameters
    ----------
    prisms_dir : str
        PRISMS出力ディレクトリ

    Returns
    -------
    strain : ndarray
    stress : ndarray (GPa)
    """
    # まずテキスト形式の結果を探す（PRISMS-Plasticity の標準出力）
    txt_candidates = [
        os.path.join(prisms_dir, 'stress_strain.txt'),
        os.path.join(prisms_dir, 'results.csv'),
        os.path.join(prisms_dir, 'stressstrain.txt'),
    ]

    for path in txt_candidates:
        if os.path.exists(path):
            data = np.loadtxt(path, comments='#', delimiter=None)
            if data.ndim == 2 and data.shape[1] >= 2:
                strain = data[:, 0]
                stress = data[:, 1]  # MPa → GPa
                if np.max(np.abs(stress)) > 100:
                    stress = stress / 1000.0  # MPa → GPa
                return strain, stress

    # VTUファイルから読み込み（pyvistaが必要）
    vtu_files = sorted(glob.glob(os.path.join(prisms_dir, '*.vtu')))
    if vtu_files:
        try:
            import pyvista as pv
            return _extract_from_vtu(vtu_files)
        except ImportError:
            print("[警告] pyvistaがインストールされていません。VTU読み込みをスキップします。")
            print("  pip install pyvista でインストールしてください。")

    raise FileNotFoundError(
        f"PRISMS結果ファイルが見つかりません: {prisms_dir}\n"
        f"  検索したファイル: {txt_candidates}"
    )


def _extract_from_vtu(vtu_files):
    """
    VTUファイル群から体積平均の応力ひずみを抽出

    Parameters
    ----------
    vtu_files : list of str
        時系列順のVTUファイルパス

    Returns
    -------
    strain : ndarray
    stress : ndarray (GPa)
    """
    import pyvista as pv

    strains = []
    stresses = []

    for vtu_path in vtu_files:
        mesh = pv.read(vtu_path)

        # 体積を重みとした平均
        volumes = mesh.compute_cell_sizes()['Volume']
        total_vol = np.sum(volumes)

        # 応力テンソルのzz成分（引張方向）
        # PRISMS-Plasticity の変数名候補
        stress_keys = ['stress', 'Cauchy_stress', 'sigma', 'S']
        stress_zz = None
        for key in stress_keys:
            if key in mesh.cell_data:
                stress_tensor = mesh.cell_data[key]
                # テンソル形式: (n_cells, 6) Voigt: xx,yy,zz,yz,xz,xy
                if stress_tensor.ndim == 2 and stress_tensor.shape[1] >= 3:
                    stress_zz = stress_tensor[:, 2]  # zz成分
                    break

        # ひずみテンソルのzz成分
        strain_keys = ['strain', 'total_strain', 'epsilon', 'E']
        strain_zz = None
        for key in strain_keys:
            if key in mesh.cell_data:
                strain_tensor = mesh.cell_data[key]
                if strain_tensor.ndim == 2 and strain_tensor.shape[1] >= 3:
                    strain_zz = strain_tensor[:, 2]
                    break

        # 等価応力・ひずみをフォールバック
        if stress_zz is None:
            for key in ['equivalent_stress', 'vonMises_stress']:
                if key in mesh.cell_data:
                    stress_zz = mesh.cell_data[key]
                    break
        if strain_zz is None:
            for key in ['equivalent_strain', 'vonMises_strain']:
                if key in mesh.cell_data:
                    strain_zz = mesh.cell_data[key]
                    break

        if stress_zz is not None and strain_zz is not None:
            avg_stress = np.sum(stress_zz * volumes) / total_vol
            avg_strain = np.sum(strain_zz * volumes) / total_vol
            stresses.append(avg_stress)
            strains.append(avg_strain)

    strain = np.array(strains)
    stress = np.array(stresses)

    # MPa → GPa（PRISMS出力がMPaの場合）
    if np.max(np.abs(stress)) > 100:
        stress = stress / 1000.0

    return strain, stress


def load_orientations(path):
    """
    OVITO方位データファイルを読み込む

    Parameters
    ----------
    path : str
        orientations.txt のパス

    Returns
    -------
    n_frames : int
        フレーム数
    frames_data : list of ndarray
        各フレームの四元数配列 (N, 4)
    """
    timesteps, frames_data = parse_multi_frame_orientations(path)
    return len(frames_data), frames_data


def select_key_frames(n_frames, strain_max, yield_strain):
    """
    応力ひずみ曲線上の特徴点に対応するフレームインデックスを選択

    Parameters
    ----------
    n_frames : int
        総フレーム数
    strain_max : float
        最大ひずみ
    yield_strain : float
        降伏ひずみ

    Returns
    -------
    frames : list of int
        選択された4フレームのインデックス
    strains : list of float
        各フレームに対応するひずみ値
    labels : list of str
        各点のラベル
    """
    # フレーム→ひずみ対応: frame_i の strain ≈ i * (strain_max / (n_frames - 1))
    frame_to_strain = lambda i: i * (strain_max / (n_frames - 1)) if n_frames > 1 else 0.0

    # (1) 初期状態
    f0 = 0

    # (2) 降伏ひずみに最も近いフレーム
    best_f1 = 0
    best_diff = abs(frame_to_strain(0) - yield_strain)
    for i in range(n_frames):
        diff = abs(frame_to_strain(i) - yield_strain)
        if diff < best_diff:
            best_diff = diff
            best_f1 = i
    f1 = best_f1

    # (3) 塑性変形中間点 (ε ≈ 0.15)
    target_mid = 0.15
    best_f2 = 0
    best_diff = abs(frame_to_strain(0) - target_mid)
    for i in range(n_frames):
        diff = abs(frame_to_strain(i) - target_mid)
        if diff < best_diff:
            best_diff = diff
            best_f2 = i
    f2 = best_f2

    # (4) 最終状態
    f3 = n_frames - 1

    frames = [f0, f1, f2, f3]
    strains = [frame_to_strain(f) for f in frames]
    labels = [
        f'① ε={strains[0]:.3f}',
        f'② ε={strains[1]:.3f}',
        f'③ ε={strains[2]:.3f}',
        f'④ ε={strains[3]:.3f}',
    ]

    return frames, strains, labels


def plot_pole_panel(ax, quaternions, family, title):
    """
    1つの極点図を描画

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        描画先の軸
    quaternions : ndarray (N, 4)
        四元数データ (x, y, z, w)
    family : str
        結晶面族 ('{100}', '{110}', '{111}')
    title : str
        パネルタイトル
    """
    # 背景描画
    draw_pole_figure_axes(ax, title)

    if quaternions.shape[0] == 0:
        ax.text(0, 0, "No Data", ha='center', va='center', color='red')
        return

    # 回転行列を計算
    R = quaternion_to_rotation_matrix_vectorized(quaternions)

    # 基準ベクトル
    v_ref = get_symmetry_variants(family)  # (M, 3)

    # 試料座標系でのベクトル: R @ v_ref^T → (N, 3, M)
    sample_vectors_t = np.einsum('nij,jk->nik', R, v_ref.T)
    # (N, M, 3) に転置
    sample_vectors = sample_vectors_t.transpose(0, 2, 1)

    # ステレオ投影
    px, py = stereographic_projection(sample_vectors)

    # 散布図
    ax.scatter(px.flatten(), py.flatten(), s=1, alpha=0.05,
               color='blue', edgecolors='none', zorder=3)


def plot_comparison(md_path, prisms_dir, output_path,
                    orientations_path=None, family='{111}'):
    """
    MD vs CP-FEM 比較プロットを生成

    Parameters
    ----------
    md_path : str
        MD stress_strain.txt パス
    prisms_dir : str
        PRISMS 出力ディレクトリ
    output_path : str
        保存先パス
    orientations_path : str, optional
        OVITO方位データファイルパス（指定時に極点図パネルを追加）
    family : str
        極点図の結晶面族（デフォルト: '{111}'）
    """
    # MD データ読み込み
    md_strain, md_stress = load_stress_strain(md_path)
    md_strain_s, md_stress_s = smooth_data(md_strain, md_stress)

    # MD パラメータ
    E_md, r2 = fit_elastic_modulus(md_strain_s, md_stress_s)
    sigma_y_md, eps_y_md = find_yield_stress_offset(md_strain_s, md_stress_s, E_md)

    # PRISMS データ読み込み
    try:
        pr_strain, pr_stress = load_prisms_results(prisms_dir)
        has_prisms = True

        # PRISMS パラメータ
        pr_strain_s, pr_stress_s = smooth_data(pr_strain, pr_stress)
        E_pr, _ = fit_elastic_modulus(pr_strain_s, pr_stress_s)
        sigma_y_pr, _ = find_yield_stress_offset(pr_strain_s, pr_stress_s, E_pr)
    except (FileNotFoundError, ValueError) as e:
        print(f"[警告] PRISMS結果の読み込みに失敗: {e}")
        print("  MDデータのみプロットします。")
        has_prisms = False

    # 方位データ読み込み
    has_orientations = False
    if orientations_path and os.path.exists(orientations_path):
        n_frames, frames_data = load_orientations(orientations_path)
        if n_frames > 0:
            has_orientations = True
            strain_max = float(md_strain[-1])
            key_frames, key_strains, key_labels = select_key_frames(
                n_frames, strain_max, eps_y_md)
            print(f"[極点図] フレーム選択: {list(zip(key_frames, key_labels))}")

    # --- プロット作成 ---
    if has_orientations:
        # 上段: 応力ひずみ (2列) + 下段: 極点図 (4列)
        fig = plt.figure(figsize=(14, 11))
        gs = GridSpec(2, 4, height_ratios=[1.2, 1], hspace=0.45, wspace=0.3)

        # 上段左: 全体の応力ひずみ曲線
        ax1 = fig.add_subplot(gs[0, :3])
        # 上段右: 弾性域拡大
        ax2 = fig.add_subplot(gs[0, 3])
    else:
        fig = plt.figure(figsize=(14, 6))
        gs = GridSpec(1, 2, width_ratios=[2, 1], wspace=0.3)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

    # 左パネル: 全体の応力ひずみ曲線
    ax1.plot(md_strain * 100, md_stress, color='C0', alpha=0.25, linewidth=0.5,
             label='MD (raw)')
    ax1.plot(md_strain_s * 100, md_stress_s, color='C0', linewidth=2,
             label=f'MD (smooth), E={E_md:.1f} GPa')

    if has_prisms:
        ax1.plot(pr_strain * 100, pr_stress, color='C1', linewidth=2,
                 linestyle='--', label=f'CP-FEM, E={E_pr:.1f} GPa')

    # 特徴点マーカーを描画
    if has_orientations:
        marker_colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#984ea3']
        for i, (strain_val, label) in enumerate(zip(key_strains, key_labels)):
            # スムース曲線上で最も近い点の応力を取得
            idx = np.argmin(np.abs(md_strain_s - strain_val))
            stress_val = md_stress_s[idx]
            ax1.plot(strain_val * 100, stress_val, 'o', color=marker_colors[i],
                     markersize=8, zorder=5, markeredgecolor='black',
                     markeredgewidth=0.8)
            ax1.annotate(label[:1],  # ①②③④のみ
                         xy=(strain_val * 100, stress_val),
                         xytext=(5, 10), textcoords='offset points',
                         fontsize=11, fontweight='bold', color=marker_colors[i])

    ax1.set_xlabel('Strain (%)')
    ax1.set_ylabel('Stress (GPa)')
    ax1.set_title('MD vs CP-FEM: Stress-Strain Comparison')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # 右パネル: 弾性域拡大
    elastic_mask = md_strain_s < 0.02  # ε < 2%
    ax2.plot(md_strain[md_strain < 0.02] * 100, md_stress[md_strain < 0.02],
             color='C0', alpha=0.25, linewidth=0.5)
    ax2.plot(md_strain_s[elastic_mask] * 100, md_stress_s[elastic_mask],
             color='C0', linewidth=2, label='MD')

    if has_prisms:
        pr_mask = pr_strain < 0.02
        if np.any(pr_mask):
            ax2.plot(pr_strain[pr_mask] * 100, pr_stress[pr_mask],
                     color='C1', linewidth=2, linestyle='--', label='CP-FEM')

    ax2.set_xlabel('Strain (%)')
    ax2.set_ylabel('Stress (GPa)')
    ax2.set_title('Elastic Region (ε < 2%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # テキストボックス: 定量比較
    text_lines = [
        f'MD:  E = {E_md:.1f} GPa, σ_y = {sigma_y_md:.3f} GPa',
    ]
    if has_prisms:
        E_err = abs(E_pr - E_md) / E_md * 100
        sy_err = abs(sigma_y_pr - sigma_y_md) / sigma_y_md * 100
        text_lines.extend([
            f'FEM: E = {E_pr:.1f} GPa, σ_y = {sigma_y_pr:.3f} GPa',
            f'誤差: ΔE = {E_err:.1f}%, Δσ_y = {sy_err:.1f}%',
        ])

    textstr = '\n'.join(text_lines)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.98, 0.05, textstr, transform=ax1.transAxes, fontsize=9,
             verticalalignment='bottom', horizontalalignment='right',
             bbox=props, family='monospace')

    # 下段: 極点図パネル
    if has_orientations:
        for i, (frame_idx, label) in enumerate(zip(key_frames, key_labels)):
            ax_pole = fig.add_subplot(gs[1, i])
            plot_pole_panel(ax_pole, frames_data[frame_idx], family,
                            f'{family}  {label}')
            # タイトルを上にオフセットして「TD」ラベルとの重なりを回避
            ax_pole.set_title(ax_pole.get_title(), pad=15)

    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"プロット保存: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="MD vs CP-FEM 結果比較プロット")
    parser.add_argument('--md', required=True, help='MD stress_strain.txt パス')
    parser.add_argument('--prisms', required=True, help='PRISMS出力ディレクトリ')
    parser.add_argument('--output', required=True, help='出力画像パス')
    parser.add_argument('--orientations', default=None,
                        help='OVITO方位データファイル（省略時は極点図なし）')
    parser.add_argument('--family', default='{111}',
                        choices=['{100}', '{110}', '{111}'],
                        help='極点図の結晶面族（デフォルト: {111}）')
    args = parser.parse_args()

    plot_comparison(args.md, args.prisms, args.output,
                    orientations_path=args.orientations,
                    family=args.family)


if __name__ == '__main__':
    main()
