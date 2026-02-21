"""
PRISMS-Plasticity 検証実行スクリプト

MDシミュレーションの応力ひずみ応答をCP-FEM（PRISMS-Plasticity）で検証する。
パラメータ抽出 → 入力ファイル生成 → Docker実行 → 結果比較 のパイプライン。

Usage:
  # Step 1: パラメータ抽出 + 入力生成のみ（Docker不要）
  python simulations/verification_prisms/run_verification.py --skip-md \
    --md-result simulations/single_crystal_tensile/Cu100_tension_300K_gpu_test/stress_strain.txt \
    --generate-only

  # Step 2: 全パイプライン実行（Docker必要）
  python simulations/verification_prisms/run_verification.py --skip-md \
    --md-result simulations/single_crystal_tensile/Cu100_tension_300K_gpu_test/stress_strain.txt

  # Step 3: MD実行から全自動
  python simulations/verification_prisms/run_verification.py
"""

import argparse
import os
import sys
import shutil
import subprocess
from string import Template

sys.path.append(os.getcwd())

from simulations.verification_prisms.scripts.extract_md_params import (
    load_stress_strain,
    smooth_data,
    fit_elastic_modulus,
    find_yield_stress_offset,
    compute_tau0,
    rodrigues_from_miller,
)

# プロジェクトディレクトリ
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# Cu文献値の弾性定数 (GPa)
CU_ELASTIC_CONSTANTS = {
    'C11': 168.4,
    'C12': 121.4,
    'C44': 75.4,
}

# Docker設定
PRISMS_DOCKER_IMAGE = 'prisms-plasticity:latest'
PRISMS_BINARY = '/opt/prisms/main'


def run_md_simulation(config):
    """
    Step 1: MDシミュレーションを実行
    既存の single_crystal_tensile.py を subprocess で呼び出す。

    Parameters
    ----------
    config : dict
        シミュレーション設定
        - element: str ('Cu')
        - miller: tuple (h, k, l)
        - temperature: float (K)
        - max_strain: float
        - runpod: bool
        - gpu: bool

    Returns
    -------
    result_path : str
        stress_strain.txt のパス
    """
    driver = 'simulations/single_crystal_tensile/single_crystal_tensile.py'
    miller = config.get('miller', (1, 0, 0))

    cmd = [
        sys.executable, driver,
        '--element', config.get('element', 'Cu'),
        '--miller', str(miller[0]), str(miller[1]), str(miller[2]),
        '--temperature', str(config.get('temperature', 300)),
        '--max-strain', str(config.get('max_strain', 0.15)),
    ]

    if config.get('runpod'):
        cmd.append('--runpod')
    if config.get('gpu'):
        cmd.append('--gpu')

    print(f"[MD] コマンド: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[MD] エラー:\n{result.stderr}")
        raise RuntimeError("MDシミュレーションが失敗しました")

    print(result.stdout)

    # 結果パスを推定（single_crystal_tensile.pyの命名規則に従う）
    element = config.get('element', 'Cu')
    m = ''.join(str(x) for x in miller)
    temp = int(config.get('temperature', 300))
    result_dir = f"simulations/single_crystal_tensile/{element}{m}_tension_{temp}K"

    # GPU付きの場合のサフィックス
    candidates = [
        os.path.join(result_dir + '_gpu_test', 'stress_strain.txt'),
        os.path.join(result_dir + '_gpu', 'stress_strain.txt'),
        os.path.join(result_dir, 'stress_strain.txt'),
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"[MD] 結果ファイル: {path}")
            return path

    raise FileNotFoundError(f"MD結果ファイルが見つかりません: {candidates}")


def extract_parameters(md_result_path, miller=(1, 0, 0)):
    """
    Step 2: MD結果からPRISMS入力パラメータを抽出

    Parameters
    ----------
    md_result_path : str
        stress_strain.txt のパス
    miller : tuple
        引張軸方位

    Returns
    -------
    params : dict
        PRISMS入力に必要な全パラメータ（単位: MPa）
    """
    # データ読み込み・スムージング
    strain, stress = load_stress_strain(md_result_path)
    strain_s, stress_s = smooth_data(strain, stress)

    # 弾性率フィット
    E, r2 = fit_elastic_modulus(strain_s, stress_s)
    print(f"[パラメータ] ヤング率 E = {E:.1f} GPa (R² = {r2:.4f})")

    # 降伏応力
    sigma_y, eps_y = find_yield_stress_offset(strain_s, stress_s, E)
    print(f"[パラメータ] 降伏応力 σ_y = {sigma_y:.3f} GPa (ε_y = {eps_y:.4f})")

    # 臨界分解せん断応力
    tau0, max_sf = compute_tau0(sigma_y, miller)
    print(f"[パラメータ] τ₀ = {tau0:.3f} GPa (Schmid因子 = {max_sf:.4f})")

    # Rodrigues-Frank ベクトル
    rodrigues = rodrigues_from_miller(miller)
    print(f"[パラメータ] Rodrigues = [{rodrigues[0]:.6f}, {rodrigues[1]:.6f}, {rodrigues[2]:.6f}]")

    # Cu文献値の弾性定数 (GPa → MPa)
    C11 = CU_ELASTIC_CONSTANTS['C11'] * 1000  # MPa
    C12 = CU_ELASTIC_CONSTANTS['C12'] * 1000
    C44 = CU_ELASTIC_CONSTANTS['C44'] * 1000

    # Voce硬化パラメータ（経験則）
    tau0_mpa = tau0 * 1000  # GPa → MPa
    tau_s = 1.5 * tau0_mpa   # 飽和すべり抵抗
    h0 = 10.0 * tau0_mpa     # 初期硬化係数
    n_exp = 20.0              # べき乗則指数

    # 12すべり系で同一値（カンマ区切り文字列）
    tau0_list = ', '.join([f'{tau0_mpa:.2f}'] * 12)
    taus_list = ', '.join([f'{tau_s:.2f}'] * 12)
    h0_list = ', '.join([f'{h0:.2f}'] * 12)
    n_list = ', '.join([f'{n_exp:.1f}'] * 12)

    # 最大ひずみからシミュレーション時間を設定
    max_strain = float(strain[-1])
    strain_rate = 1.0e-3  # /s（準静的）
    total_time = max_strain / strain_rate
    dt = total_time / 1000  # 1000ステップ

    # z方向変位（単位長さあたり）
    disp_z = max_strain  # 単位長さ = 1 mm として

    params = {
        # 弾性定数 (MPa)
        'C11': f'{C11:.1f}',
        'C12': f'{C12:.1f}',
        'C44': f'{C44:.1f}',
        # 硬化パラメータ
        'TAU0_LIST': tau0_list,
        'TAUS_LIST': taus_list,
        'H0_LIST': h0_list,
        'N_LIST': n_list,
        # 時間積分
        'DT': f'{dt:.6e}',
        'TOTAL_TIME': f'{total_time:.6e}',
        # 境界条件
        'DISP_Z': f'{disp_z:.6f}',
        # 方位
        'R1': f'{rodrigues[0]:.8f}',
        'R2': f'{rodrigues[1]:.8f}',
        'R3': f'{rodrigues[2]:.8f}',
        # 出力
        'OUTPUT_DIR': 'output',
        # MD解析結果（後の比較用に保持）
        '_E_gpa': E,
        '_sigma_y_gpa': sigma_y,
        '_tau0_gpa': tau0,
        '_max_strain': max_strain,
        '_miller': miller,
    }

    return params


def generate_prisms_input(params, config=None):
    """
    Step 3: テンプレートからPRISMS入力ファイルを生成

    Parameters
    ----------
    params : dict
        extract_parameters() の戻り値
    config : dict, optional
        追加設定

    Returns
    -------
    input_dir : str
        生成した入力ファイルのディレクトリ
    """
    input_dir = os.path.join(OUTPUT_DIR, 'prisms_input')
    os.makedirs(input_dir, exist_ok=True)

    # テンプレートファイル → 出力ファイルのマッピング
    template_files = {
        'prm.prm.in': 'prm.prm',
        'BCinfo.txt.in': 'BCinfo.txt',
        'orientations.txt.in': 'orientations.txt',
    }

    # テンプレート置換（$KEY 形式）
    for tmpl_name, out_name in template_files.items():
        tmpl_path = os.path.join(TEMPLATE_DIR, tmpl_name)
        out_path = os.path.join(input_dir, out_name)

        with open(tmpl_path, 'r') as f:
            content = f.read()

        # $KEY をパラメータで置換
        tmpl = Template(content)
        # 内部用キー（_で始まる）を除外
        subst = {k: v for k, v in params.items() if not k.startswith('_')}
        rendered = tmpl.safe_substitute(subst)

        with open(out_path, 'w') as f:
            f.write(rendered)

        print(f"[入力生成] {out_name} → {out_path}")

    # 静的ファイルをコピー
    static_files = ['grainID.txt', 'slipNormals.txt', 'slipDirections.txt']
    for fname in static_files:
        src = os.path.join(TEMPLATE_DIR, fname)
        dst = os.path.join(input_dir, fname)
        shutil.copy2(src, dst)
        print(f"[入力生成] {fname} → {dst}")

    # 出力ディレクトリを作成
    os.makedirs(os.path.join(input_dir, 'output'), exist_ok=True)

    print(f"\n[入力生成] 完了: {input_dir}")
    return input_dir


def run_prisms_simulation(input_dir, docker_image=None, np_procs=4):
    """
    Step 4: PRISMS-Plasticity をDocker経由で実行

    Parameters
    ----------
    input_dir : str
        入力ファイルのディレクトリ
    docker_image : str, optional
        Dockerイメージ名
    np_procs : int
        MPI並列数

    Returns
    -------
    output_dir : str
        結果出力ディレクトリ
    """
    if docker_image is None:
        docker_image = PRISMS_DOCKER_IMAGE

    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.join(input_dir, 'output')

    cmd = [
        'docker', 'run', '--rm',
        '-v', f'{input_dir}:/workspace',
        '-w', '/workspace',
        docker_image,
        'mpirun', '--allow-run-as-root',
        '-np', str(np_procs),
        PRISMS_BINARY, 'prm.prm',
    ]

    print(f"[PRISMS] コマンド: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)

    if result.returncode != 0:
        print(f"[PRISMS] stdout:\n{result.stdout}")
        print(f"[PRISMS] stderr:\n{result.stderr}")
        raise RuntimeError("PRISMS-Plasticity の実行に失敗しました")

    print(f"[PRISMS] 実行完了")
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)

    return output_dir


def compare_results(md_path, prisms_dir, output_path=None, orientations_path=None):
    """
    Step 5: MDとPRISMSの結果を比較・プロット

    Parameters
    ----------
    md_path : str
        MD の stress_strain.txt パス
    prisms_dir : str
        PRISMS 出力ディレクトリ
    output_path : str, optional
        比較プロットの保存先
    orientations_path : str, optional
        OVITO方位データファイルパス（極点図パネル追加用）
    """
    if output_path is None:
        output_path = os.path.join(OUTPUT_DIR, 'comparison.png')

    # plot_comparison.py を呼び出し
    script = os.path.join(BASE_DIR, 'scripts', 'plot_comparison.py')
    cmd = [
        sys.executable, script,
        '--md', md_path,
        '--prisms', prisms_dir,
        '--output', output_path,
    ]

    if orientations_path:
        cmd.extend(['--orientations', orientations_path])

    print(f"[比較] コマンド: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[比較] エラー:\n{result.stderr}")
        raise RuntimeError("結果比較に失敗しました")

    print(result.stdout)
    print(f"[比較] プロット保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="PRISMS-Plasticity 検証パイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ワークフロー制御
    parser.add_argument('--skip-md', action='store_true',
                        help='MDシミュレーションをスキップ（既存結果を使用）')
    parser.add_argument('--generate-only', action='store_true',
                        help='入力ファイル生成のみ（PRISMS実行なし）')
    parser.add_argument('--compare-only', action='store_true',
                        help='結果比較のみ')

    # 入出力パス
    parser.add_argument('--md-result', type=str, default=None,
                        help='MD結果ファイル（stress_strain.txt）のパス')
    parser.add_argument('--prisms-dir', type=str, default=None,
                        help='PRISMS出力ディレクトリ（compare-only時）')
    parser.add_argument('--output', type=str, default=None,
                        help='比較プロットの保存先')
    parser.add_argument('--orientations', type=str, default=None,
                        help='OVITO方位データファイル（極点図パネル追加用）')

    # MD設定
    parser.add_argument('--element', type=str, default='Cu')
    parser.add_argument('--miller', type=int, nargs=3, default=[1, 0, 0],
                        help='結晶方位 (h k l)')
    parser.add_argument('--temperature', type=float, default=300)
    parser.add_argument('--max-strain', type=float, default=0.15)
    parser.add_argument('--runpod', action='store_true')
    parser.add_argument('--gpu', action='store_true')

    # PRISMS設定
    parser.add_argument('--docker-image', type=str, default=None,
                        help='PRISMS Dockerイメージ名')
    parser.add_argument('--np', type=int, default=4,
                        help='MPI並列数')

    args = parser.parse_args()
    miller = tuple(args.miller)

    print("=" * 60)
    print("PRISMS-Plasticity 検証パイプライン")
    print(f"  方位: {miller}, 元素: {args.element}")
    print("=" * 60)

    # --- Step 1: MD実行 ---
    if args.compare_only:
        md_result_path = args.md_result
        if md_result_path is None:
            parser.error("--compare-only には --md-result が必要です")
    elif args.skip_md:
        md_result_path = args.md_result
        if md_result_path is None:
            parser.error("--skip-md には --md-result が必要です")
        print(f"\n[MD] スキップ（既存結果を使用: {md_result_path}）")
    else:
        print("\n--- Step 1: MDシミュレーション ---")
        config = {
            'element': args.element,
            'miller': miller,
            'temperature': args.temperature,
            'max_strain': args.max_strain,
            'runpod': args.runpod,
            'gpu': args.gpu,
        }
        md_result_path = run_md_simulation(config)

    if not args.compare_only:
        # --- Step 2: パラメータ抽出 ---
        print("\n--- Step 2: パラメータ抽出 ---")
        params = extract_parameters(md_result_path, miller)

        # --- Step 3: 入力ファイル生成 ---
        print("\n--- Step 3: PRISMS入力ファイル生成 ---")
        input_dir = generate_prisms_input(params)

        if args.generate_only:
            print("\n[完了] --generate-only: 入力ファイル生成まで完了")
            return

        # --- Step 4: PRISMS実行 ---
        print("\n--- Step 4: PRISMS-Plasticity 実行 ---")
        prisms_output = run_prisms_simulation(
            input_dir,
            docker_image=args.docker_image,
            np_procs=args.np,
        )
    else:
        prisms_output = args.prisms_dir
        if prisms_output is None:
            parser.error("--compare-only には --prisms-dir が必要です")

    # --- Step 5: 結果比較 ---
    print("\n--- Step 5: 結果比較 ---")

    # 方位データの自動検出（--orientations 未指定時）
    orientations_path = args.orientations
    if orientations_path is None and md_result_path:
        # MD結果と同じディレクトリの orientations.txt を探す
        candidate = os.path.join(os.path.dirname(md_result_path), 'orientations.txt')
        if os.path.exists(candidate):
            orientations_path = candidate
            print(f"[比較] 方位データを自動検出: {orientations_path}")

    compare_results(md_result_path, prisms_output, args.output,
                    orientations_path=orientations_path)

    print("\n" + "=" * 60)
    print("パイプライン完了")
    print("=" * 60)


if __name__ == "__main__":
    main()
