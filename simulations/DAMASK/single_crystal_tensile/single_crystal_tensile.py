#!/usr/bin/env python3
"""
CPFEM 単結晶引張シミュレーション（DAMASK_grid）

FCC単結晶（Cu, Ag, Au）の単軸引張をスペクトル法で実行する。
ジオメトリ・材料・荷重条件を自動生成し、DAMASK_gridを呼び出す。

使用例:
    python simulations/DAMASK/single_crystal_tensile/single_crystal_tensile.py --element Cu
    python simulations/DAMASK/single_crystal_tensile/single_crystal_tensile.py --element Au --miller 1 1 0
    python simulations/DAMASK/single_crystal_tensile/single_crystal_tensile.py --element Ag --max-strain 0.10 --increments 50
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import damask

# --- 材料パラメータ辞書 ---
# 弾性定数 [Pa]、塑性パラメータ（phenopowerlaw）
# 弾性定数はDAMASK公式例 (Overton & Gaffney 1955, Neighbours & Alers 1958) を参照
# 塑性パラメータはプラン指定値を使用（h_sl-slはFCC 7係数形式）
MATERIALS = {
    "Cu": {
        "C_11": 168.4e9, "C_12": 121.4e9, "C_44": 75.4e9,
        "xi_0_sl": [31.0e6], "xi_inf_sl": [63.0e6], "h_0_sl-sl": 500.0e6,
        "n_sl": [20], "a_sl": [2.25], "dot_gamma_0_sl": [0.001],
        "h_sl-sl": [1, 1, 1.4, 1.4, 1.4, 1.4, 1.4],
    },
    "Ag": {
        "C_11": 124.0e9, "C_12": 93.4e9, "C_44": 46.1e9,
        "xi_0_sl": [25.0e6], "xi_inf_sl": [55.0e6], "h_0_sl-sl": 200.0e6,
        "n_sl": [20], "a_sl": [2.25], "dot_gamma_0_sl": [0.001],
        "h_sl-sl": [1, 1, 1.4, 1.4, 1.4, 1.4, 1.4],
    },
    "Au": {
        "C_11": 192.9e9, "C_12": 163.8e9, "C_44": 41.5e9,
        "xi_0_sl": [15.0e6], "xi_inf_sl": [35.0e6], "h_0_sl-sl": 150.0e6,
        "n_sl": [20], "a_sl": [2.25], "dot_gamma_0_sl": [0.001],
        "h_sl-sl": [1, 1, 1.4, 1.4, 1.4, 1.4, 1.4],
    },
}

# FCC滑り系数（{111}<110>: 12系）
N_SL = [12]


def miller_to_quaternion(hkl):
    """
    Miller指数 [h,k,l] から、その方向をz軸に合わせる回転の四元数を返す。
    引張軸（z軸）を [h,k,l] 方向に揃えるための結晶回転。
    """
    hkl = np.array(hkl, dtype=float)
    hkl /= np.linalg.norm(hkl)

    z = np.array([0.0, 0.0, 1.0])

    if np.allclose(hkl, z):
        return damask.Rotation(np.array([1.0, 0.0, 0.0, 0.0]))
    if np.allclose(hkl, -z):
        return damask.Rotation.from_axis_angle(np.array([1.0, 0.0, 0.0, np.pi]))

    # z軸を [h,k,l] に回す回転を求める
    axis = np.cross(z, hkl)
    axis /= np.linalg.norm(axis)
    angle = np.arccos(np.clip(np.dot(z, hkl), -1.0, 1.0))

    return damask.Rotation.from_axis_angle(np.append(axis, angle))


def generate_geometry(grid_n, job_dir):
    """均一単結晶グリッドを生成（材料ID=0のみ）"""
    material = np.zeros((grid_n, grid_n, grid_n), dtype=int)
    size = np.array([1.0e-3, 1.0e-3, 1.0e-3])  # 1mm立方
    geom = damask.GeomGrid(material, size)
    vti_path = job_dir / "single_crystal.vti"
    geom.save(vti_path)
    print(f"  ジオメトリ生成: {vti_path} ({grid_n}x{grid_n}x{grid_n})")
    return vti_path


def generate_material(element, orientation, job_dir):
    """material.yaml を生成"""
    p = MATERIALS[element]

    material = {
        "homogenization": {
            "SX": {
                "N_constituents": 1,
                "mechanical": {"type": "pass"},
            }
        },
        "phase": {
            element: {
                "lattice": "cF",
                "mechanical": {
                    "output": ["F", "P", "O"],
                    "elastic": {
                        "type": "Hooke",
                        "C_11": p["C_11"],
                        "C_12": p["C_12"],
                        "C_44": p["C_44"],
                    },
                    "plastic": {
                        "type": "phenopowerlaw",
                        "N_sl": N_SL,
                        "xi_0_sl": p["xi_0_sl"],
                        "xi_inf_sl": p["xi_inf_sl"],
                        "h_0_sl-sl": p["h_0_sl-sl"],
                        "h_sl-sl": p["h_sl-sl"],
                        "n_sl": p["n_sl"],
                        "a_sl": p["a_sl"],
                        "dot_gamma_0_sl": p["dot_gamma_0_sl"],
                        "output": ["xi_sl", "gamma_sl"],
                    },
                },
            }
        },
        "material": [
            {
                "homogenization": "SX",
                "constituents": [
                    {
                        "phase": element,
                        "O": orientation.as_quaternion().tolist(),
                        "v": 1.0,
                    }
                ],
            }
        ],
    }

    mat_path = job_dir / "material.yaml"
    import yaml
    with open(mat_path, "w") as f:
        yaml.dump(material, f, default_flow_style=False, allow_unicode=True)
    print(f"  材料定義生成: {mat_path} (元素: {element})")
    return mat_path


def generate_load(erate, max_strain, increments, job_dir):
    """load.yaml を生成（z軸単軸引張）"""
    total_time = max_strain / erate
    dt = total_time / increments

    load = {
        "solver": {
            "mechanical": "spectral_basic",
        },
        "loadstep": [
            {
                "boundary_conditions": {
                    "mechanical": {
                        "dot_F": [["x", 0, 0],
                                  [0, "x", 0],
                                  [0, 0, erate]],
                        "P": [[0, "x", "x"],
                              ["x", 0, "x"],
                              ["x", "x", "x"]],
                    }
                },
                "discretization": {
                    "t": total_time,
                    "N": increments,
                },
            }
        ],
    }

    load_path = job_dir / "load.yaml"
    import yaml
    with open(load_path, "w") as f:
        yaml.dump(load, f, default_flow_style=None, allow_unicode=True)
    print(f"  荷重条件生成: {load_path} (ε̇={erate}/s, ε_max={max_strain}, {increments}ステップ)")
    return load_path


def run_damask(job_dir, geom_file):
    """DAMASK_grid を実行"""
    cmd = [
        "DAMASK_grid",
        "--geom", str(geom_file.name),
        "--load", "load.yaml",
        "--material", "material.yaml",
    ]
    print(f"\n  実行コマンド: {' '.join(cmd)}")
    print(f"  作業ディレクトリ: {job_dir}")
    print("=" * 60)

    result = subprocess.run(
        cmd, cwd=str(job_dir),
        stdout=sys.stdout, stderr=sys.stderr,
    )

    if result.returncode != 0:
        print(f"\nエラー: DAMASK_grid が終了コード {result.returncode} で終了しました")
        sys.exit(1)

    print("=" * 60)
    print("  DAMASK_grid 完了")


def postprocess(job_dir, geom_file):
    """HDF5結果から応力-ひずみ曲線を抽出"""
    # DAMASK_gridの出力ファイル名はジオメトリ名に基づく
    hdf5_name = geom_file.stem + ".hdf5"
    hdf5_path = job_dir / hdf5_name

    if not hdf5_path.exists():
        # ジョブディレクトリ内のhdf5ファイルを探す
        hdf5_files = list(job_dir.glob("*.hdf5"))
        if hdf5_files:
            hdf5_path = hdf5_files[0]
        else:
            print(f"エラー: HDF5ファイルが見つかりません: {job_dir}")
            return

    print(f"\n  後処理開始: {hdf5_path}")

    r = damask.Result(str(hdf5_path))

    # 応力・ひずみを追加（HDF5に書き込む）
    r.add_stress_Cauchy()
    r.add_strain()

    # 全インクリメントのデータを取得
    increments = r.increments
    times = r.times

    stress_zz_list = []
    strain_zz_list = []

    for inc in increments:
        r_inc = r.view(increments=inc)

        # Cauchy応力を取得 → (N_cells, 3, 3) の ndarray
        sigma = r_inc.get("sigma")
        if sigma is None:
            continue
        # 全セルの体積平均
        sigma_avg = np.mean(sigma, axis=0)
        stress_zz_list.append(sigma_avg[2, 2])

        # ひずみを取得
        eps = r_inc.get("epsilon_V^0.0(F)")
        if eps is None:
            continue
        eps_avg = np.mean(eps, axis=0)
        strain_zz_list.append(eps_avg[2, 2])

    # CSV出力
    csv_path = job_dir / "stress_strain.csv"
    header = "strain_zz,stress_zz_MPa,time_s"
    data = np.column_stack([
        strain_zz_list,
        np.array(stress_zz_list) / 1e6,  # Pa → MPa
        times[:len(strain_zz_list)],
    ])
    np.savetxt(csv_path, data, delimiter=",", header=header, comments="")
    print(f"  応力-ひずみCSV出力: {csv_path} ({len(strain_zz_list)} データ点)")

    # 弾性率の推定（第1インクリメントの勾配）
    # 注意: インクリメント数が少ない場合、第1ステップで既に降伏している可能性あり
    if len(strain_zz_list) >= 2 and strain_zz_list[1] > 0:
        E_est = stress_zz_list[1] / strain_zz_list[1]
        print(f"  初期勾配: {E_est/1e9:.1f} GPa（弾性率推定、降伏後は過小評価）")

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="CPFEM 単結晶引張シミュレーション（DAMASK_grid）"
    )
    parser.add_argument(
        "--element", choices=["Cu", "Ag", "Au"], default="Cu",
        help="元素（デフォルト: Cu）"
    )
    parser.add_argument(
        "--orientation", type=float, nargs=4, default=None,
        help="結晶方位（四元数 w x y z）"
    )
    parser.add_argument(
        "--miller", type=int, nargs=3, default=None,
        help="引張軸のMiller指数（例: 1 0 0）→ 自動で四元数に変換"
    )
    parser.add_argument(
        "--erate", type=float, default=1.0e-3,
        help="ひずみ速度 [/s]（デフォルト: 1e-3）"
    )
    parser.add_argument(
        "--max-strain", type=float, default=0.10,
        help="最大ひずみ（デフォルト: 0.10）"
    )
    parser.add_argument(
        "--increments", type=int, default=40,
        help="時間ステップ数（デフォルト: 40）"
    )
    parser.add_argument(
        "--grid", type=int, default=4,
        help="グリッド分割数（デフォルト: 4）"
    )
    parser.add_argument(
        "--no-run", action="store_true",
        help="入力ファイル生成のみ（DAMASK_gridを実行しない）"
    )

    args = parser.parse_args()

    # 結晶方位の決定
    if args.miller is not None:
        orientation = miller_to_quaternion(args.miller)
        orient_label = f"{''.join(map(str, args.miller))}"
        print(f"Miller指数 {args.miller} → 四元数: {orientation.as_quaternion()}")
    elif args.orientation is not None:
        orientation = damask.Rotation.from_quaternion(args.orientation)
        orient_label = "custom"
    else:
        orientation = damask.Rotation(np.array([1.0, 0.0, 0.0, 0.0]))
        orient_label = "100"

    # ジョブ名とディレクトリ
    job_name = f"{args.element}_{orient_label}_erate{args.erate:.0e}"
    sim_dir = Path("simulations/CPFEM")
    job_dir = sim_dir / job_name
    job_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  CPFEM 単結晶引張: {args.element} [{orient_label}]")
    print(f"  ジョブ: {job_dir}")
    print(f"{'='*60}")

    # 入力ファイル生成
    geom_file = generate_geometry(args.grid, job_dir)
    generate_material(args.element, orientation, job_dir)
    generate_load(args.erate, args.max_strain, args.increments, job_dir)

    if args.no_run:
        print("\n  --no-run: 入力ファイル生成のみ完了")
        return

    # DAMASK_grid 実行
    run_damask(job_dir, geom_file)

    # 後処理
    postprocess(job_dir, geom_file)


if __name__ == "__main__":
    main()
