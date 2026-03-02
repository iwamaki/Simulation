#!/usr/bin/env python3
"""
CPFEM 単結晶バリデーションスクリプト

HDF5結果から理論値との比較を行い、単結晶CPFEMが正しく動作しているか検証する。

検証項目:
  1. 活性すべり系（gamma_sl の時間微分で判定）
  2. 初期降伏応力（σ_y = CRSS / Schmid因子）
  3. 弾性率（応力-ひずみ初期勾配 vs Voigt方向依存弾性率）

使用例:
    python3 simulations/DAMASK/single_crystal_tensile/validate_single_crystal.py --job Cu_100_erate1e-03
    python3 simulations/DAMASK/single_crystal_tensile/validate_single_crystal.py --job Cu_110_erate1e-03 --plot
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

# --- 材料パラメータ（single_crystal_tensile.py と同じ） ---
MATERIALS = {
    "Cu": {
        "C_11": 168.4e9, "C_12": 121.4e9, "C_44": 75.4e9,
        "xi_0_sl": 31.0e6,
    },
    "Ag": {
        "C_11": 124.0e9, "C_12": 93.4e9, "C_44": 46.1e9,
        "xi_0_sl": 25.0e6,
    },
    "Au": {
        "C_11": 192.9e9, "C_12": 163.8e9, "C_44": 41.5e9,
        "xi_0_sl": 15.0e6,
    },
}

# --- FCC {111}<110> 12すべり系 ---
# すべり面法線 n と すべり方向 s（Miller指数、正規化前）
FCC_SLIP_SYSTEMS = [
    # {111} 面
    {"n": [ 1, 1, 1], "s": [ 0, 1,-1], "label": "(111)[0-11]"},
    {"n": [ 1, 1, 1], "s": [-1, 0, 1], "label": "(111)[-101]"},
    {"n": [ 1, 1, 1], "s": [ 1,-1, 0], "label": "(111)[1-10]"},
    # {-111} 面
    {"n": [-1, 1, 1], "s": [ 0, 1,-1], "label": "(-111)[0-11]"},
    {"n": [-1, 1, 1], "s": [ 1, 0, 1], "label": "(-111)[101]"},
    {"n": [-1, 1, 1], "s": [-1,-1, 0], "label": "(-111)[-1-10]"},
    # {1-11} 面
    {"n": [ 1,-1, 1], "s": [ 0,-1,-1], "label": "(1-11)[0-1-1]"},
    {"n": [ 1,-1, 1], "s": [-1, 0, 1], "label": "(1-11)[-101]"},
    {"n": [ 1,-1, 1], "s": [ 1, 1, 0], "label": "(1-11)[110]"},
    # {11-1} 面
    {"n": [ 1, 1,-1], "s": [ 0, 1, 1], "label": "(11-1)[011]"},
    {"n": [ 1, 1,-1], "s": [-1, 0,-1], "label": "(11-1)[-10-1]"},
    {"n": [ 1, 1,-1], "s": [ 1,-1, 0], "label": "(11-1)[1-10]"},
]


def compute_schmid_factors(tensile_axis):
    """
    引張軸に対する各すべり系のSchmid因子を計算。
    m = |cos(φ) * cos(λ)| = |(n·t)(s·t)| / (|n||s||t|^2)
    """
    t = np.array(tensile_axis, dtype=float)
    t /= np.linalg.norm(t)

    schmid_factors = []
    for slip in FCC_SLIP_SYSTEMS:
        n = np.array(slip["n"], dtype=float)
        s = np.array(slip["s"], dtype=float)
        n /= np.linalg.norm(n)
        s /= np.linalg.norm(s)
        m = abs(np.dot(n, t) * np.dot(s, t))
        schmid_factors.append(m)

    return np.array(schmid_factors)


def compute_youngs_modulus_theory(C_11, C_12, C_44, direction):
    """
    立方晶の方向依存ヤング率を計算。
    1/E(hkl) = S_11 - 2*(S_11 - S_12 - S_44/2) * (l1^2*l2^2 + l2^2*l3^2 + l3^2*l1^2)
    ここで S はコンプライアンステンソル、l_i は方向余弦。
    """
    # コンプライアンス行列の成分
    det = (C_11 - C_12) * (C_11 + 2 * C_12)
    S_11 = (C_11 + C_12) / det
    S_12 = -C_12 / det
    S_44 = 1.0 / C_44

    # 方向余弦
    d = np.array(direction, dtype=float)
    d /= np.linalg.norm(d)
    l1, l2, l3 = d

    aniso = l1**2 * l2**2 + l2**2 * l3**2 + l3**2 * l1**2
    inv_E = S_11 - 2.0 * (S_11 - S_12 - S_44 / 2.0) * aniso

    return 1.0 / inv_E


def parse_job_name(job_name):
    """ジョブ名から元素名とMiller指数を推定"""
    parts = job_name.split("_")
    element = parts[0]
    miller_str = parts[1]

    # "100" → [1,0,0], "110" → [1,1,0], "111" → [1,1,1]
    miller = [int(c) for c in miller_str]

    return element, miller


def find_hdf5(job_dir):
    """ジョブディレクトリ内のHDF5ファイルを探す"""
    hdf5_files = list(job_dir.glob("*.hdf5"))
    if not hdf5_files:
        raise FileNotFoundError(f"HDF5ファイルが見つかりません: {job_dir}")
    return hdf5_files[0]


def load_material_yaml(job_dir):
    """material.yamlから元素名を取得"""
    mat_path = job_dir / "material.yaml"
    if mat_path.exists():
        with open(mat_path) as f:
            mat = yaml.safe_load(f)
        phases = list(mat.get("phase", {}).keys())
        if phases:
            return phases[0]
    return None


def validate(job_dir, do_plot=True):
    """バリデーション実行"""
    import damask

    job_name = job_dir.name
    element, miller = parse_job_name(job_name)

    # material.yamlからも元素名を確認
    element_from_yaml = load_material_yaml(job_dir)
    if element_from_yaml and element_from_yaml != element:
        print(f"  警告: ジョブ名の元素({element}) と material.yaml({element_from_yaml}) が不一致")
        element = element_from_yaml

    if element not in MATERIALS:
        print(f"エラー: 未知の元素 {element}")
        return

    p = MATERIALS[element]
    crss = p["xi_0_sl"]

    print(f"\n{'='*70}")
    print(f"  バリデーション: {job_name}")
    print(f"  元素: {element}, 引張軸: {miller}")
    print(f"{'='*70}")

    # --- 1. Schmid因子テーブル ---
    schmid = compute_schmid_factors(miller)
    max_schmid = np.max(schmid)
    active_threshold = max_schmid - 1e-6  # 最大値とほぼ同じものを活性系とする

    print(f"\n--- Schmid因子テーブル ---")
    print(f"{'系番号':>6}  {'すべり系':<20}  {'Schmid因子':>10}  {'活性':>4}")
    print("-" * 50)
    n_active_theory = 0
    for i, (slip, sf) in enumerate(zip(FCC_SLIP_SYSTEMS, schmid)):
        is_active = sf >= active_threshold
        if is_active:
            n_active_theory += 1
        marker = " ***" if is_active else ""
        print(f"  {i+1:>4}  {slip['label']:<20}  {sf:>10.4f}{marker}")
    print(f"\n  最大Schmid因子: {max_schmid:.4f}")
    print(f"  理論活性系数: {n_active_theory}")

    # --- 2. 理論値計算 ---
    sigma_y_theory = crss / max_schmid
    E_theory = compute_youngs_modulus_theory(p["C_11"], p["C_12"], p["C_44"], miller)

    print(f"\n--- 理論予測値 ---")
    print(f"  CRSS:       {crss/1e6:.1f} MPa")
    print(f"  降伏応力:   σ_y = CRSS / m = {crss/1e6:.1f} / {max_schmid:.4f} = {sigma_y_theory/1e6:.1f} MPa")
    print(f"  弾性率:     E[{miller}] = {E_theory/1e9:.1f} GPa")

    # --- 3. HDF5からシミュレーション結果を読み取り ---
    hdf5_path = find_hdf5(job_dir)
    print(f"\n  HDF5読み込み: {hdf5_path}")

    r = damask.Result(str(hdf5_path))

    # 応力・ひずみを追加（既に追加済みの場合はスキップ）
    try:
        r.add_stress_Cauchy()
    except ValueError:
        pass
    try:
        r.add_strain()
    except ValueError:
        pass

    increments = r.increments
    times = r.times

    stress_zz = []
    strain_zz = []
    gamma_sl_all = []  # (n_inc, 12) 累積せん断ひずみ

    for inc in increments:
        r_inc = r.view(increments=inc)

        # 応力
        sigma = r_inc.get("sigma")
        if sigma is not None:
            sigma_avg = np.mean(sigma, axis=0)
            stress_zz.append(sigma_avg[2, 2])
        else:
            stress_zz.append(0.0)

        # ひずみ
        eps = r_inc.get("epsilon_V^0.0(F)")
        if eps is not None:
            eps_avg = np.mean(eps, axis=0)
            strain_zz.append(eps_avg[2, 2])
        else:
            strain_zz.append(0.0)

        # gamma_sl（各すべり系の累積せん断ひずみ）
        gs = r_inc.get("gamma_sl")
        if gs is not None:
            gs_avg = np.mean(gs, axis=0)  # (12,)
            gamma_sl_all.append(gs_avg)

    stress_zz = np.array(stress_zz)
    strain_zz = np.array(strain_zz)

    # gamma_sl の時間微分から dot_gamma_sl を計算
    dot_gamma_all = []
    has_gamma_sl = len(gamma_sl_all) > 0
    if has_gamma_sl:
        gamma_sl_all = np.array(gamma_sl_all)  # (n_inc, 12)
        for i in range(1, len(gamma_sl_all)):
            dt = times[i] - times[i-1]
            if dt > 0:
                dot_gamma_all.append((gamma_sl_all[i] - gamma_sl_all[i-1]) / dt)
        if dot_gamma_all:
            dot_gamma_all = np.array(dot_gamma_all)  # (n_inc-1, 12)

    # --- 4. シミュレーション弾性率の推定 ---
    # 弾性域のデータ点で原点通過の線形フィット
    # 降伏ひずみ（理論値）以下の範囲を使用
    yield_strain_theory = sigma_y_theory / E_theory
    elastic_mask = (strain_zz > 0) & (strain_zz < yield_strain_theory * 0.8)
    E_sim = np.nan
    e_note = ""
    if np.sum(elastic_mask) >= 2:
        # 原点通過の最小二乗法: E = Σ(σ*ε) / Σ(ε²)
        E_sim = (np.sum(stress_zz[elastic_mask] * strain_zz[elastic_mask])
                 / np.sum(strain_zz[elastic_mask]**2))
        e_note = f"（弾性域{np.sum(elastic_mask)}点でフィット）"
    elif np.sum(elastic_mask) == 1:
        idx = np.where(elastic_mask)[0][0]
        E_sim = stress_zz[idx] / strain_zz[idx]
        e_note = "（弾性域1点のみ）"
    else:
        # 弾性域にデータ点がない → 弾性率は測定不可能
        E_sim = np.nan
        e_note = "（弾性域にデータ点なし: --increments を増やして再実行推奨 例: --increments 200）"

    # --- 5. シミュレーション降伏応力の推定 ---
    # 弾性率が信頼できない場合は理論弾性率で0.2%オフセット法を適用
    E_for_offset = E_sim if not np.isnan(E_sim) else E_theory
    offset_strain = 0.002
    sigma_y_sim = np.nan
    if len(strain_zz) >= 3:
        offset_line = E_for_offset * (strain_zz - offset_strain)
        diff = stress_zz - offset_line
        # diffが正→負に変わる点を探す（= オフセット線と交差）
        for i in range(1, len(diff)):
            if diff[i-1] > 0 and diff[i] <= 0:
                # 線形補間
                frac = diff[i-1] / (diff[i-1] - diff[i])
                sigma_y_sim = stress_zz[i-1] + frac * (stress_zz[i] - stress_zz[i-1])
                break
        if np.isnan(sigma_y_sim):
            # 交差が見つからない場合: 応力が理論降伏応力付近の点を探す
            for i in range(1, len(stress_zz)):
                if stress_zz[i] >= sigma_y_theory * 0.9:
                    sigma_y_sim = stress_zz[i]
                    break

    # --- 6. 活性すべり系の判定（gamma_sl → dot_gamma_sl） ---
    n_active_sim = 0
    active_systems_sim = []
    dg_at_yield = None
    has_dot_gamma = isinstance(dot_gamma_all, np.ndarray) and len(dot_gamma_all) > 0

    if has_dot_gamma:
        # 降伏後のインクリメントのdot_gamma_slで判定
        # dot_gamma_allのインデックスは元のincrementから1ずれている（差分計算のため）
        yield_inc = -1
        for i in range(1, len(stress_zz)):
            if strain_zz[i] > offset_strain * 2:  # ひずみが0.4%以上
                yield_inc = i - 1  # dot_gamma_allのインデックスに変換
                break
        if yield_inc < 0:
            yield_inc = len(dot_gamma_all) - 1

        # yield_inc付近のdot_gamma_slを使用
        dg_at_yield = np.abs(dot_gamma_all[min(yield_inc, len(dot_gamma_all) - 1)])
        dg_max = np.max(dg_at_yield)
        active_threshold_sim = dg_max * 0.1  # 最大値の10%以上を活性とみなす

        for i in range(12):
            if dg_at_yield[i] >= active_threshold_sim:
                n_active_sim += 1
                active_systems_sim.append(i)
    else:
        print(f"\n  警告: gamma_sl データが HDF5 に見つかりません。")
        print(f"  → material.yaml の plastic output に 'gamma_sl' を追加して再実行してください:")
        print(f"    python simulations/CPFEM/single_crystal_tensile.py --element {element} --miller {' '.join(map(str, miller))}")

    # --- 7. 結果テーブル出力 ---
    print(f"\n{'='*70}")
    print(f"  検証結果サマリー")
    print(f"{'='*70}")
    print(f"{'項目':<20}  {'理論値':>15}  {'シミュレーション':>15}  {'誤差':>10}")
    print("-" * 65)

    # 弾性率
    E_err = abs(E_sim - E_theory) / E_theory * 100 if not np.isnan(E_sim) else np.nan
    E_sim_str = f"{E_sim/1e9:>15.1f}" if not np.isnan(E_sim) else f"{'N/A':>15}"
    E_err_str = f"{E_err:>9.1f}%" if not np.isnan(E_err) else f"{'N/A':>10}"
    print(f"{'弾性率 [GPa]':<20}  {E_theory/1e9:>15.1f}  {E_sim_str}  {E_err_str}")
    if e_note:
        print(f"  {e_note}")

    # 降伏応力
    sy_err = abs(sigma_y_sim - sigma_y_theory) / sigma_y_theory * 100 if not np.isnan(sigma_y_sim) else np.nan
    sy_sim_str = f"{sigma_y_sim/1e6:>15.1f}" if not np.isnan(sigma_y_sim) else f"{'N/A':>15}"
    sy_err_str = f"{sy_err:>9.1f}%" if not np.isnan(sy_err) else f"{'N/A':>10}"
    print(f"{'降伏応力 [MPa]':<20}  {sigma_y_theory/1e6:>15.1f}  {sy_sim_str}  {sy_err_str}")

    # 活性すべり系数
    if has_dot_gamma:
        print(f"{'活性すべり系数':<20}  {n_active_theory:>15d}  {n_active_sim:>15d}  {'OK' if n_active_theory == n_active_sim else 'NG':>10}")
    else:
        print(f"{'活性すべり系数':<20}  {n_active_theory:>15d}  {'N/A':>15}  {'N/A':>10}")
        print(f"  （gamma_sl データなし）")

    # 活性系の詳細
    if active_systems_sim and dg_at_yield is not None:
        print(f"\n  シミュレーション活性系:")
        for idx in active_systems_sim:
            slip = FCC_SLIP_SYSTEMS[idx]
            print(f"    系{idx+1:2d}: {slip['label']:<20}  Schmid因子={schmid[idx]:.4f}  |dot_gamma|={dg_at_yield[idx]:.3e}")

    # 判定
    print(f"\n--- 総合判定 ---")
    ok_count = 0
    total = 0

    # 弾性率（弾性域データがある場合のみ判定）
    if not np.isnan(E_err):
        total += 1
        if E_err < 5.0:
            print(f"  弾性率:     OK (誤差 {E_err:.1f}% < 5%)")
            ok_count += 1
        else:
            print(f"  弾性率:     NG (誤差 {E_err:.1f}%)")
    else:
        print(f"  弾性率:     SKIP (弾性域にデータ点なし → --increments 200 以上で再実行推奨)")

    # 降伏応力
    if not np.isnan(sy_err):
        total += 1
        if sy_err < 10.0:
            print(f"  降伏応力:   OK (誤差 {sy_err:.1f}% < 10%)")
            ok_count += 1
        else:
            print(f"  降伏応力:   NG (誤差 {sy_err:.1f}%)")
    else:
        print(f"  降伏応力:   SKIP (データ不足)")

    # 活性すべり系
    if has_dot_gamma:
        total += 1
        if n_active_theory == n_active_sim:
            print(f"  活性系:     OK ({n_active_sim}系)")
            ok_count += 1
        else:
            print(f"  活性系:     NG (理論{n_active_theory}系 vs 実測{n_active_sim}系)")
    else:
        print(f"  活性系:     SKIP (gamma_sl データなし → 再実行が必要)")

    print(f"\n  結果: {ok_count}/{total} 項目パス")

    # --- 8. プロット ---
    if do_plot and has_dot_gamma:
        plot_dot_gamma(job_dir, times, dot_gamma_all, schmid, miller)

    return ok_count == total


def plot_dot_gamma(job_dir, times, dot_gamma_all, schmid, miller):
    """各すべり系のせん断ひずみ速度の時間発展をプロット"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    n_inc = dot_gamma_all.shape[0]
    # dot_gamma_allは差分なのでtimesの中間点を使用
    t = 0.5 * (times[:n_inc] + times[1:n_inc+1])

    # 活性系の判定（Schmid因子が最大値に近いもの）
    max_schmid = np.max(schmid)
    active_mask = schmid >= max_schmid - 1e-6

    # 左パネル: 全すべり系のdot_gamma_sl
    ax = axes[0]
    for i in range(12):
        slip = FCC_SLIP_SYSTEMS[i]
        if active_mask[i]:
            ax.plot(t, np.abs(dot_gamma_all[:, i]), linewidth=2,
                    label=f"{i+1}: {slip['label']} (m={schmid[i]:.3f})")
        else:
            ax.plot(t, np.abs(dot_gamma_all[:, i]), linewidth=0.8,
                    alpha=0.4, color="gray")

    ax.set_xlabel("Time [s]", fontsize=12)
    ax.set_ylabel("|dot_gamma_sl| [1/s]", fontsize=12)
    ax.set_title(f"Shear Rate per Slip System - [{miller[0]}{miller[1]}{miller[2]}]", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.tick_params(labelsize=10)

    # 右パネル: Schmid因子の棒グラフ
    ax = axes[1]
    colors = ["#d62728" if active_mask[i] else "#7f7f7f" for i in range(12)]
    bars = ax.barh(range(12), schmid, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(12))
    ax.set_yticklabels([f"{i+1}: {FCC_SLIP_SYSTEMS[i]['label']}" for i in range(12)],
                       fontsize=9, fontfamily="monospace")
    ax.set_xlabel("Schmid Factor", fontsize=12)
    ax.set_title(f"Schmid Factors - [{miller[0]}{miller[1]}{miller[2]}]", fontsize=13)
    ax.axvline(x=max_schmid, color="red", linestyle="--", alpha=0.5, label=f"max = {max_schmid:.4f}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="x")
    ax.tick_params(labelsize=10)
    ax.invert_yaxis()

    fig.tight_layout()
    out_path = job_dir / "validation.png"
    fig.savefig(out_path, dpi=150)
    print(f"\n  バリデーションプロット保存: {out_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="CPFEM 単結晶バリデーション（理論値との比較検証）"
    )
    parser.add_argument(
        "--job", required=True,
        help="ジョブディレクトリ名（例: Cu_100_erate1e-03）"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="プロットを生成しない"
    )

    args = parser.parse_args()

    sim_dir = Path("simulations/CPFEM")
    job_dir = sim_dir / args.job

    if not job_dir.exists():
        print(f"エラー: ジョブディレクトリが見つかりません: {job_dir}")
        return

    validate(job_dir, do_plot=not args.no_plot)


if __name__ == "__main__":
    main()
