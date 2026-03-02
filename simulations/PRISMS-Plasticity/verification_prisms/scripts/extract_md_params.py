"""
MD応力ひずみデータからPRISMS-Plasticity入力パラメータを抽出するモジュール

主要機能:
- stress_strain.txt の読み込み・スムージング
- 弾性率（ヤング率）の線形フィッティング
- 0.2%オフセット法による降伏応力の決定
- FCC {111}<110> すべり系のSchmid因子計算
- PRISMS用Rodrigues-Frankベクトルの算出
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


# FCC {111}<110> すべり系 12個
# 法線: {111} の 4面, 方向: <110> の各面3方向
FCC_SLIP_NORMALS = np.array([
    [ 1,  1,  1],
    [ 1,  1,  1],
    [ 1,  1,  1],
    [-1,  1,  1],
    [-1,  1,  1],
    [-1,  1,  1],
    [ 1, -1,  1],
    [ 1, -1,  1],
    [ 1, -1,  1],
    [ 1,  1, -1],
    [ 1,  1, -1],
    [ 1,  1, -1],
], dtype=float)

FCC_SLIP_DIRECTIONS = np.array([
    [ 0,  1, -1],
    [-1,  0,  1],
    [ 1, -1,  0],
    [ 0,  1, -1],
    [ 1,  0,  1],
    [-1, -1,  0],
    [ 0,  1,  1],
    [-1,  0, -1],
    [ 1, -1,  0],  # 修正: (1,-1,0) ← 正しいFCC系
    [ 0,  1,  1],
    [ 1,  0, -1],
    [-1, -1,  0],
], dtype=float)


def load_stress_strain(path):
    """
    stress_strain.txt を読み込み、strain配列とstress配列(GPa)を返す

    Parameters
    ----------
    path : str
        stress_strain.txt のパス

    Returns
    -------
    strain : ndarray
        ひずみ（無次元）
    stress : ndarray
        応力（GPa）
    """
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    strain = float(parts[0])
                    stress = float(parts[1])
                    data.append([strain, stress])
                except ValueError:
                    continue
    arr = np.array(data)
    return arr[:, 0], arr[:, 1]


def smooth_data(strain, stress, window_length=51, polyorder=3):
    """
    Savitzky-Golayフィルタでノイズ除去

    Parameters
    ----------
    strain : ndarray
    stress : ndarray
    window_length : int
        フィルタ窓幅（奇数）。データ数に応じて自動調整。
    polyorder : int
        多項式次数

    Returns
    -------
    strain_smooth : ndarray
    stress_smooth : ndarray
    """
    # データ数が窓幅より少ない場合は調整
    n = len(stress)
    if window_length > n:
        window_length = n if n % 2 == 1 else n - 1
    if window_length < polyorder + 2:
        return strain.copy(), stress.copy()

    stress_smooth = savgol_filter(stress, window_length, polyorder)
    return strain.copy(), stress_smooth


def fit_elastic_modulus(strain, stress, eps_min=0.002, eps_max=0.008):
    """
    弾性域の線形フィットでヤング率を算出

    Parameters
    ----------
    strain : ndarray
        ひずみ（スムージング済み推奨）
    stress : ndarray
        応力（GPa、スムージング済み推奨）
    eps_min : float
        フィット範囲の下限ひずみ（初期ノイズ回避）
    eps_max : float
        フィット範囲の上限ひずみ

    Returns
    -------
    E : float
        ヤング率（GPa）
    r_squared : float
        決定係数
    """
    mask = (strain >= eps_min) & (strain <= eps_max)
    if np.sum(mask) < 3:
        raise ValueError(f"フィット範囲 [{eps_min}, {eps_max}] にデータが不足しています")

    s = strain[mask]
    ss = stress[mask]

    # 線形フィット: stress = E * strain + b
    coeffs = np.polyfit(s, ss, 1)
    E = coeffs[0]  # 傾き = ヤング率 (GPa)

    # 決定係数
    ss_pred = np.polyval(coeffs, s)
    ss_res = np.sum((ss - ss_pred) ** 2)
    ss_tot = np.sum((ss - np.mean(ss)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return E, r_squared


def find_yield_stress_offset(strain, stress, E, offset=0.002):
    """
    0.2%オフセット法で降伏応力を決定

    Parameters
    ----------
    strain : ndarray
    stress : ndarray
    E : float
        ヤング率（GPa）
    offset : float
        オフセットひずみ（デフォルト0.2% = 0.002）

    Returns
    -------
    sigma_y : float
        降伏応力（GPa）
    eps_y : float
        降伏ひずみ
    """
    # オフセット直線: sigma_offset = E * (strain - offset)
    sigma_offset = E * (strain - offset)

    # 応力ひずみ曲線とオフセット直線の交点を探す
    diff = stress - sigma_offset

    # 最初にdiffが正→負に変わる点（または正の最小値）
    # 弾性域ではdiff > 0（曲線がオフセット線より上）
    # 降伏後にdiff < 0になる
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) > 0:
        # 最初の符号変化点で線形補間
        idx = sign_changes[0]
        # 補間
        d0, d1 = diff[idx], diff[idx + 1]
        frac = d0 / (d0 - d1) if (d0 - d1) != 0 else 0.5
        eps_y = strain[idx] + frac * (strain[idx + 1] - strain[idx])
        sigma_y = stress[idx] + frac * (stress[idx + 1] - stress[idx])
    else:
        # 交点が見つからない場合、最大応力を使用
        idx_max = np.argmax(stress)
        sigma_y = stress[idx_max]
        eps_y = strain[idx_max]

    return sigma_y, eps_y


def compute_schmid_factors(miller):
    """
    単結晶引張方位に対するFCC {111}<110> 12系のSchmid因子を計算

    Parameters
    ----------
    miller : tuple of int
        引張軸の結晶方位 (h, k, l)

    Returns
    -------
    schmid_factors : ndarray (12,)
        各すべり系のSchmid因子（絶対値）
    """
    # 引張軸方向の単位ベクトル
    t = np.array(miller, dtype=float)
    t = t / np.linalg.norm(t)

    schmid_factors = np.zeros(12)
    for i in range(12):
        n = FCC_SLIP_NORMALS[i].copy()
        n = n / np.linalg.norm(n)
        d = FCC_SLIP_DIRECTIONS[i].copy()
        d = d / np.linalg.norm(d)
        # Schmid因子 = |cos(φ) × cos(λ)| = |(t·n)(t·d)|
        schmid_factors[i] = abs(np.dot(t, n) * np.dot(t, d))

    return schmid_factors


def compute_tau0(sigma_y, miller):
    """
    降伏応力とSchmid因子から臨界分解せん断応力 τ₀ を算出

    Parameters
    ----------
    sigma_y : float
        降伏応力（GPa）
    miller : tuple of int
        引張軸方位

    Returns
    -------
    tau0 : float
        臨界分解せん断応力（GPa）
    max_schmid : float
        最大Schmid因子
    """
    schmid_factors = compute_schmid_factors(miller)
    max_schmid = np.max(schmid_factors)
    tau0 = sigma_y * max_schmid
    return tau0, max_schmid


def rodrigues_from_miller(miller):
    """
    Miller指数で指定された引張方位からPRISMS用Rodrigues-Frankベクトルを算出

    単結晶テスト: z軸=[miller]方向になる回転をRodrigues表現で返す。
    PRISMS-Plasticityのorientations.txtに書き込む形式。

    Parameters
    ----------
    miller : tuple of int
        z軸の結晶方位 (h, k, l)

    Returns
    -------
    rodrigues : ndarray (3,)
        Rodrigues-Frankベクトル (r1, r2, r3)
    """
    # single_crystal_tensile.py と同じ方位テーブルを使用
    ORIENTATIONS = {
        (1, 0, 0): {'a': (1, 0, 0),  'b': (0, 1, 0),  'c': (0, 0, 1)},
        (1, 1, 0): {'a': (-1, 1, 0), 'b': (0, 0, 1),  'c': (1, 1, 0)},
        (1, 1, 1): {'a': (1, -1, 0), 'b': (1, 1, -2), 'c': (1, 1, 1)},
    }

    miller_tuple = tuple(miller)
    if miller_tuple in ORIENTATIONS:
        ori = ORIENTATIONS[miller_tuple]
        a = np.array(ori['a'], dtype=float)
        b = np.array(ori['b'], dtype=float)
        c = np.array(ori['c'], dtype=float)
    else:
        # 汎用: z=[miller], x,yは適当に直交基底を構成
        c = np.array(miller, dtype=float)
        # xを適当に選ぶ（cと直交するベクトル）
        if abs(c[0]) < 0.9 * np.linalg.norm(c):
            a = np.cross(c, [1, 0, 0])
        else:
            a = np.cross(c, [0, 1, 0])
        b = np.cross(c, a)

    # 正規化して回転行列を構成
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    c = c / np.linalg.norm(c)

    # 回転行列: 結晶座標系 → 試料座標系
    # 列 = 結晶軸方向ベクトル（試料座標で表現）
    # PRISMS: 行列の行 = 試料軸を結晶座標で表現
    # R[i,j] = 試料i軸の結晶j成分
    # → x軸=a方向, y軸=b方向, z軸=c方向
    R = np.array([a, b, c])  # 各行が試料軸の結晶座標表現

    # scipy Rotation で Rodrigues-Frank ベクトルに変換
    rot = Rotation.from_matrix(R)
    rotvec = rot.as_rotvec()
    angle = np.linalg.norm(rotvec)
    if angle < 1e-10:
        return np.array([0.0, 0.0, 0.0])
    axis = rotvec / angle
    rodrigues = axis * np.tan(angle / 2)
    return rodrigues


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python extract_md_params.py <stress_strain.txt> [h k l]")
        sys.exit(1)

    path = sys.argv[1]
    miller = (1, 0, 0)
    if len(sys.argv) >= 5:
        miller = (int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))

    # データ読み込み・スムージング
    strain, stress = load_stress_strain(path)
    print(f"データ数: {len(strain)}")

    strain_s, stress_s = smooth_data(strain, stress)

    # 弾性率フィット
    E, r2 = fit_elastic_modulus(strain_s, stress_s)
    print(f"ヤング率 E = {E:.1f} GPa (R² = {r2:.4f})")

    # 降伏応力
    sigma_y, eps_y = find_yield_stress_offset(strain_s, stress_s, E)
    print(f"降伏応力 σ_y = {sigma_y:.3f} GPa (ε_y = {eps_y:.4f})")

    # Schmid因子・τ₀
    schmid = compute_schmid_factors(miller)
    tau0, max_sf = compute_tau0(sigma_y, miller)
    print(f"方位 {miller}: 最大Schmid因子 = {max_sf:.4f}")
    print(f"臨界分解せん断応力 τ₀ = {tau0:.3f} GPa = {tau0*1000:.1f} MPa")

    # Rodrigues-Frank ベクトル
    rod = rodrigues_from_miller(miller)
    print(f"Rodrigues-Frank ベクトル: {rod[0]:.6f} {rod[1]:.6f} {rod[2]:.6f}")
