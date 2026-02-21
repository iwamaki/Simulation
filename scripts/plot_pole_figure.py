import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # GUIバックエンドを明示的に指定
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import argparse
import sys
from pathlib import Path
import io

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # GUIバックエンドを明示的に指定
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import argparse
import sys
from pathlib import Path
import io

def parse_multi_frame_orientations(filepath):
    """
    OVITOのダンプファイル形式（LAMMPS dump formatに類似）をパースする。
    
    Format structure per frame:
    Line 1: N (number of particles)
    Line 2: Lattice="..." Properties=...
    Line 3 to N+2: Data lines (id, qx, qy, qz, qw)
    """
    print("方位データファイルを読み込んでいます...")
    frames_data = []
    timesteps = []
    
    with open(filepath, 'r') as f:
        lines = f.readlines()

    num_lines = len(lines)
    current_line_idx = 0
    frame_count = 0

    while current_line_idx < num_lines:
        line = lines[current_line_idx].strip()
        if not line: # 空行スキップ
            current_line_idx += 1
            continue
            
        # 1. 粒子数の読み取り (フレーム開始)
        try:
            num_particles = int(line)
        except ValueError:
            # 整数に変換できない場合は、予期せぬ行かファイル末尾の改行など
            # print(f"警告: 行 {current_line_idx+1} で粒子数の読み取りに失敗しました: '{line}'。スキップします。")
            current_line_idx += 1
            continue

        # 2. ヘッダー行 (Lattice...) の読み飛ばし
        current_line_idx += 1
        if current_line_idx >= num_lines:
            break
            
        header_line = lines[current_line_idx]
        if 'Lattice' not in header_line:
             print(f"警告: フレーム {frame_count} のヘッダー行が期待通りではありません。")
        
        # 3. データ行の読み込み
        start_data_idx = current_line_idx + 1
        end_data_idx = start_data_idx + num_particles
        
        if end_data_idx > num_lines:
             print(f"エラー: フレーム {frame_count} のデータが途中で切れています。")
             break
             
        data_lines = lines[start_data_idx : end_data_idx]
        
        # 次のフレームのためにインデックスを進める
        current_line_idx = end_data_idx
        timesteps.append(frame_count)

        # 4. データ解析
        if data_lines:
            try:
                # 文字列リストを結合して一括変換
                string_io = io.StringIO('\n'.join(data_lines))
                data = np.loadtxt(string_io)
                
                # データ形状の検証と整形
                if data.ndim == 1 and data.shape[0] == 5:
                    data = data.reshape(1, 5)
                elif data.ndim == 1 and data.size == 0: # 粒子数0の場合
                    data = np.empty((0, 5))

                if data.size > 0:
                    # id, qx, qy, qz, qw -> qx, qy, qz, qw
                    quaternions = data[:, 1:5]
                    # (0,0,0,0) 除外
                    valid_mask = np.any(quaternions != 0, axis=1)
                    valid_quaternions = quaternions[valid_mask]
                    frames_data.append(valid_quaternions)
                    print(f"フレーム {frame_count}: {valid_quaternions.shape[0]} 個の有効なデータを読み込みました。")
                else:
                    frames_data.append(np.empty((0, 4)))
                    print(f"フレーム {frame_count}: データなし")

            except Exception as e:
                print(f"エラー: フレーム {frame_count} のパース中に例外が発生: {e}")
                frames_data.append(np.empty((0, 4)))
        else:
             frames_data.append(np.empty((0, 4)))
             print(f"フレーム {frame_count}: 粒子数0")

        frame_count += 1
            
    print(f"合計 {len(frames_data)} 個のタイムフレームを読み込みました。")
    return timesteps, frames_data

def quaternion_to_rotation_matrix_vectorized(Q):
    """
    四元数 (x, y, z, w) の配列から回転行列の配列への変換 (ベクトル化)
    Q: (N, 4) array
    Returns: (N, 3, 3) array of rotation matrices
    """
    if Q.size == 0:
        return np.empty((0, 3, 3))
        
    # 正規化
    norms = np.linalg.norm(Q, axis=1, keepdims=True)
    # ゼロ除算回避
    norms[norms == 0] = 1.0
    Q = Q / norms
    
    x, y, z, w = Q[:, 0], Q[:, 1], Q[:, 2], Q[:, 3]
    
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, xw = x*y, x*z, x*w
    yz, yw, zw = y*z, y*w, z*w
    
    R = np.empty((len(Q), 3, 3))
    
    R[:, 0, 0] = 1 - 2*(yy + zz)
    R[:, 0, 1] = 2*(xy - zw)
    R[:, 0, 2] = 2*(xz + yw)
    
    R[:, 1, 0] = 2*(xy + zw)
    R[:, 1, 1] = 1 - 2*(xx + zz)
    R[:, 1, 2] = 2*(yz - xw)
    
    R[:, 2, 0] = 2*(xz - yw)
    R[:, 2, 1] = 2*(yz + xw)
    R[:, 2, 2] = 1 - 2*(xx + yy)
    
    return R

def get_symmetry_variants(family):
    """
    指定された結晶面の族に属する方位のリストを返す (立方晶)
    """
    if family == '{100}':
        return np.array([[1,0,0], [0,1,0], [0,0,1]])
    elif family == '{110}':
        variants = np.array([[1,1,0], [1,0,1], [0,1,1], [1,-1,0], [1,0,-1], [0,1,-1]])
        return np.vstack([variants, -variants])
    elif family == '{111}':
        variants = np.array([[1,1,1], [1,1,-1], [1,-1,1], [-1,1,1]])
        return np.vstack([variants, -variants])
    else:
        raise ValueError("Unsupported plane family. Choose from {100}, {110}, {111}")

def stereographic_projection(v):
    """
    3Dベクトルをステレオ投影する (北半球)
    v: (N, 3) or (N, M, 3) array
    """
    # z<0 のベクトルは反対側（北半球）に写像して等価として扱う
    v = v.copy()
    if v.ndim == 2:
        v[v[:, 2] < 0] *= -1
    elif v.ndim == 3:
        v[v[:, :, 2] < 0] *= -1

    # ノルム計算 (ゼロ除算回避)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    norm[norm == 0] = 1
    v = v / norm
    
    # 投影: X = x / (1 + z), Y = y / (1 + z)
    # v[..., 2] (z成分) が -1 に近いと発散するが、上記で z>=0 にしてるので安全
    denom = 1 + v[..., 2]
    x = v[..., 0] / denom
    y = v[..., 1] / denom
    return x, y

def precompute_projections(frames_data, families):
    """
    全フレーム、全ファミリーの投影座標を事前計算する。
    Returns:
        cached_data: dict[frame_idx][family] = (x_array, y_array)
    """
    print("表示用データを事前計算しています（これには数秒かかる場合があります）...")
    cached_data = {}
    
    # ファミリーごとの基準ベクトルを取得
    family_vectors = {fam: get_symmetry_variants(fam) for fam in families}
    
    for frame_idx, quaternions in enumerate(frames_data):
        cached_data[frame_idx] = {}
        
        if quaternions.shape[0] == 0:
            for fam in families:
                cached_data[frame_idx][fam] = ([], [])
            continue

        # 回転行列を一括計算 (N, 3, 3)
        R = quaternion_to_rotation_matrix_vectorized(quaternions)
        
        for fam in families:
            v_ref = family_vectors[fam] # (M, 3)
            
            # ベクトル計算: V_sample = R * V_ref
            # R: (N, 3, 3), v_ref.T: (3, M) -> (N, 3, M)
            # einsumの結果は (粒子数, xyz成分, バリアントID)
            sample_vectors_t = np.einsum('nij,jk->nik', R, v_ref.T)
            
            # 投影関数は最後の次元がxyz成分であることを期待しているため、
            # (粒子数, バリアントID, xyz成分) に軸を入れ替える
            sample_vectors = sample_vectors_t.transpose(0, 2, 1)
            
            # 投影 (N, M)
            px, py = stereographic_projection(sample_vectors)
            
            # フラット化して格納
            cached_data[frame_idx][fam] = (px.flatten(), py.flatten())
            
        # 進捗表示（簡易）
        if frame_idx % 5 == 0:
            print(f"  事前計算: フレーム {frame_idx} 完了")
            
    print("事前計算完了。")
    return cached_data

def draw_pole_figure_axes(ax, title):
    """
    極点図の背景（円、十字線、ラベル）を描画
    """
    # 既に描画済みの固定要素は消さず、タイトルだけ更新するのが高速だが、
    # 簡易実装としてクリアする方針をとる
    ax.cla()
    circle = plt.Circle((0, 0), 1, edgecolor='black', facecolor='none', zorder=1)
    ax.add_artist(circle)
    ax.plot([-1, 1], [0, 0], 'k-', lw=0.5, zorder=2)
    ax.plot([0, 0], [-1, 1], 'k-', lw=0.5, zorder=2)
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(0, 1.05, "TD", ha='center', va='bottom')
    ax.text(1.05, 0, "RD", ha='left', va='center')

    # カーソル座標のフォーマットをカスタマイズ
    def format_coord(x, y):
        r = np.sqrt(x**2 + y**2)
        if r > 1.05:
            return f"Outside (x={x:.2f}, y={y:.2f})"
        
        # ステレオ投影の逆変換: r = tan(alpha/2) -> alpha = 2 * arctan(r)
        tilt_deg = np.degrees(2 * np.arctan(r))
        azimuth_deg = np.degrees(np.arctan2(y, x))
        if azimuth_deg < 0:
            azimuth_deg += 360
            
        return f"Tilt: {tilt_deg:.1f}°, Azimuth: {azimuth_deg:.1f}° (x={x:.2f}, y={y:.2f})"

    ax.format_coord = format_coord

def main():
    parser = argparse.ArgumentParser(description="OVITOからエクスポートした方位データを用いてインタラクティブな極点図をプロットするスクリプト")
    parser.add_argument('input', type=str, help='OVITOからエクスポートした方位データファイル (orientations.txt)')
    parser.add_argument('--family', type=str, default='{100}', help='初期表示する結晶面の族 (例: {100})')
    args = parser.parse_args()

    # --- データの読み込み ---
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"エラー: 入力ファイルが見つかりません: {input_path}")
        sys.exit(1)
        
    timesteps, frames_data = parse_multi_frame_orientations(input_path)
    if not frames_data:
        print("エラー: ファイルから有効なフレームを読み込めませんでした。")
        sys.exit(1)

    # --- 事前計算 ---
    families = ['{100}', '{110}', '{111}']
    cached_data = precompute_projections(frames_data, families)

    # --- インタラクティブプロットの準備 ---
    fig = plt.figure(figsize=(9, 8)) # 横幅を少し広げる
    
    # レイアウト調整
    # 左: 操作パネル (RadioButtons)
    # 中央〜右: 極点図
    # 下: スライダー
    
    radio_ax = fig.add_axes([0.05, 0.7, 0.15, 0.15]) # [left, bottom, width, height]
    pole_ax = fig.add_axes([0.25, 0.2, 0.7, 0.75])
    slider_ax = fig.add_axes([0.25, 0.05, 0.6, 0.03])

    # ラジオボタン
    radio = RadioButtons(radio_ax, families, active=families.index(args.family))
    
    # スライダー
    frame_slider = Slider(
        ax=slider_ax,
        label='Frame ',
        valmin=0,
        valmax=len(timesteps) - 1,
        valinit=0,
        valstep=1
    )
    
    # 状態管理用の変数
    current_state = {'family': args.family}

    # --- 更新関数 ---
    def update(val=None): # val引数はSlider/RadioButtonsから来るが使わない
        frame_idx = int(frame_slider.val)
        family = current_state['family'] # ラジオボタンの現在の値
        
        # 事前計算データから取得
        px, py = cached_data[frame_idx][family]
        
        # 描画
        draw_pole_figure_axes(pole_ax, f"{family} Pole Figure (Frame {frame_idx})")
        
        if len(px) > 0:
            # 散布図の描画 (alphaを調整して点の重なりを見やすく)
            pole_ax.scatter(px, py, s=2, alpha=0.1, color='blue', edgecolors='none', zorder=3)
        else:
            pole_ax.text(0, 0, "No Data", ha='center', va='center', color='red')
            
        fig.canvas.draw_idle()

    # コールバック関数
    def change_family(label):
        current_state['family'] = label
        update()

    frame_slider.on_changed(update)
    radio.on_clicked(change_family)

    # 初期描画
    update()

    plt.show()

if __name__ == '__main__':
    main()


