#!/usr/bin/env python3
"""
DAMASK HDF5結果ファイルから極点図をプロットするスクリプト。

塑性変形による結晶方位の回転を、各インクリメントの極点図で可視化する。
スライダーでインクリメントを切り替え、ラジオボタンで結晶面族を選択できる。

使用例:
    python scripts/plot_pole_figure_damask.py path/to/single_crystal.hdf5
    python scripts/plot_pole_figure_damask.py path/to/single_crystal.hdf5 --family "{111}"
    python scripts/plot_pole_figure_damask.py path/to/single_crystal.hdf5 --save output.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons

import damask


def load_orientations(hdf5_path):
    """
    DAMASK HDF5結果ファイルから全インクリメントの方位データを読み込む。

    Returns:
        increments: インクリメント名のリスト
        orientations_per_inc: 各インクリメントの damask.Rotation のリスト
    """
    print(f"HDF5ファイルを読み込んでいます: {hdf5_path}")
    r = damask.Result(str(hdf5_path))

    increments = r.increments

    # Oが出力に含まれているか確認
    r_test = r.view(increments=increments[0])
    O_test = r_test.get('O')

    if O_test is None:
        print("  エラー: HDF5に方位データ 'O' が含まれていません。")
        print("  material.yaml の mechanical output に 'O' を追加して再実行してください。")
        print("  例: output: [F, P, O]")
        sys.exit(1)

    orientations_per_inc = []
    for inc in increments:
        r_inc = r.view(increments=inc)
        O = r_inc.get('O')
        rot = damask.Rotation(O.reshape(-1, 4))
        orientations_per_inc.append(rot)

    print(f"  {len(increments)} インクリメント読み込み完了 ({O.reshape(-1,4).shape[0]} 材料点/inc)")
    return increments, orientations_per_inc


def get_symmetry_variants(family):
    """結晶面族に属する法線方向ベクトルを返す（立方晶）"""
    if family == '{100}':
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    elif family == '{110}':
        s = 1.0 / np.sqrt(2)
        variants = np.array([
            [s, s, 0], [s, 0, s], [0, s, s],
            [s, -s, 0], [s, 0, -s], [0, s, -s],
        ])
        return np.vstack([variants, -variants])
    elif family == '{111}':
        s = 1.0 / np.sqrt(3)
        variants = np.array([
            [s, s, s], [s, s, -s], [s, -s, s], [-s, s, s],
        ])
        return np.vstack([variants, -variants])
    else:
        raise ValueError(f"未対応の面族: {family}")


def stereographic_projection(v):
    """
    3Dベクトルをステレオ投影する（北半球）。
    v: (..., 3) array
    """
    v = v.copy()
    # z < 0 のベクトルは反転して北半球に写像
    mask = v[..., 2] < 0
    v[mask] *= -1

    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm[norm == 0] = 1.0
    v = v / norm

    denom = 1 + v[..., 2]
    denom[denom == 0] = 1e-10
    x = v[..., 0] / denom
    y = v[..., 1] / denom
    return x, y


def precompute_projections(orientations_per_inc, families):
    """
    全インクリメント・全面族の投影座標を事前計算する。

    Returns:
        cached: dict[inc_idx][family] = (x_array, y_array)
    """
    print("投影データを事前計算しています...")
    family_vectors = {fam: get_symmetry_variants(fam) for fam in families}
    cached = {}

    for inc_idx, rot in enumerate(orientations_per_inc):
        cached[inc_idx] = {}
        # damask.Rotation で結晶→試料座標系への変換
        # rot @ v_crystal = v_sample
        R = rot.as_matrix()  # (N, 3, 3)

        for fam in families:
            v_ref = family_vectors[fam]  # (M, 3)
            # (N, 3, 3) @ (3, M) -> (N, 3, M) -> transpose -> (N, M, 3)
            sample_vectors = np.einsum('nij,jk->nik', R, v_ref.T).transpose(0, 2, 1)
            px, py = stereographic_projection(sample_vectors)
            cached[inc_idx][fam] = (px.flatten(), py.flatten())

        if inc_idx % 10 == 0:
            print(f"  インクリメント {inc_idx} 完了")

    print("事前計算完了。")
    return cached


def draw_pole_figure_axes(ax, title):
    """極点図の背景（円、十字線、ラベル）を描画"""
    ax.cla()
    circle = plt.Circle((0, 0), 1, edgecolor='black', facecolor='none', zorder=1)
    ax.add_artist(circle)
    ax.plot([-1, 1], [0, 0], 'k-', lw=0.5, zorder=2)
    ax.plot([0, 0], [-1, 1], 'k-', lw=0.5, zorder=2)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0, 1.08, "TD", ha='center', va='bottom', fontsize=9)
    ax.text(1.08, 0, "RD", ha='left', va='center', fontsize=9)

    def format_coord(x, y):
        r = np.sqrt(x**2 + y**2)
        if r > 1.05:
            return f"外側 (x={x:.2f}, y={y:.2f})"
        tilt_deg = np.degrees(2 * np.arctan(r))
        azimuth_deg = np.degrees(np.arctan2(y, x))
        if azimuth_deg < 0:
            azimuth_deg += 360
        return f"傾斜: {tilt_deg:.1f}°, 方位角: {azimuth_deg:.1f}° (x={x:.2f}, y={y:.2f})"

    ax.format_coord = format_coord


def main():
    parser = argparse.ArgumentParser(
        description="DAMASK HDF5結果から極点図をプロット（塑性変形による結晶回転の可視化）"
    )
    parser.add_argument('input', type=str, help='DAMASK HDF5結果ファイル (*.hdf5)')
    parser.add_argument('--family', type=str, default='{100}',
                        help='初期表示する結晶面族 (デフォルト: {100})')
    parser.add_argument('--save', type=str, default=None,
                        help='指定すると最終インクリメントの極点図を画像保存してGUI表示しない')
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: ファイルが見つかりません: {input_path}")
        sys.exit(1)

    # --- データ読み込み ---
    increments, orientations_per_inc = load_orientations(input_path)
    if not orientations_per_inc:
        print("エラー: 方位データを読み込めませんでした。")
        sys.exit(1)

    # --- 事前計算 ---
    families = ['{100}', '{110}', '{111}']
    cached = precompute_projections(orientations_per_inc, families)

    # --- 静的保存モード ---
    if args.save:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        last_idx = len(increments) - 1
        for ax, fam in zip(axes, families):
            draw_pole_figure_axes(ax, f"{fam} Pole Figure (inc {last_idx})")
            px, py = cached[last_idx][fam]
            if len(px) > 0:
                ax.scatter(px, py, s=10, alpha=0.5, color='blue', edgecolors='none', zorder=3)
        fig.suptitle(f"Pole Figure — {input_path.name} (final increment)", fontsize=12)
        plt.tight_layout()
        plt.savefig(args.save, dpi=150)
        print(f"保存完了: {args.save}")
        return

    # --- インタラクティブモード ---
    fig = plt.figure(figsize=(9, 8))

    radio_ax = fig.add_axes([0.05, 0.7, 0.15, 0.15])
    pole_ax = fig.add_axes([0.25, 0.2, 0.7, 0.75])
    slider_ax = fig.add_axes([0.25, 0.05, 0.6, 0.03])

    radio = RadioButtons(radio_ax, families, active=families.index(args.family))

    frame_slider = Slider(
        ax=slider_ax,
        label='Inc ',
        valmin=0,
        valmax=len(increments) - 1,
        valinit=0,
        valstep=1,
    )

    current_state = {'family': args.family}

    def update(val=None):
        inc_idx = int(frame_slider.val)
        family = current_state['family']
        px, py = cached[inc_idx][family]

        draw_pole_figure_axes(pole_ax, f"{family} Pole Figure (inc {inc_idx})")

        if len(px) > 0:
            pole_ax.scatter(px, py, s=10, alpha=0.5, color='blue', edgecolors='none', zorder=3)
        else:
            pole_ax.text(0, 0, "No Data", ha='center', va='center', color='red')

        fig.canvas.draw_idle()

    def change_family(label):
        current_state['family'] = label
        update()

    frame_slider.on_changed(update)
    radio.on_clicked(change_family)

    update()
    plt.show()


if __name__ == '__main__':
    main()
