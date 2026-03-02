import os
import sys
import argparse
import subprocess
import numpy as np
from dataclasses import dataclass, fields
from ase.build import fcc100, fcc110, fcc111, stack, sort
from ase.data import atomic_numbers
from ase.io import write

sys.path.append(os.getcwd())


@dataclass
class SimConfig:
    """引張試験シミュレーションの全パラメータを管理するデータクラス"""

    # --- 材料 ---
    element1: str = 'Cu'           # 下部スラブ（z下側）
    element2: str = 'Al'           # 上部スラブ（z上側）
    lattice1: float = 3.61         # element1の格子定数 (Å)
    lattice2: float = 4.05         # element2の格子定数 (Å)
    miller1: tuple = (1, 0, 0)     # element1のMiller指数
    miller2: tuple = (1, 0, 0)     # element2のMiller指数
    potential: str = 'potentials/AlCu.eam.alloy'  # プロジェクトルートからの相対パス

    # --- ジオメトリ ---
    target_xy: float = 30.0        # 目標側面寸法 (Å)
    target_z: float = 30.0         # 各金属の目標厚さ (Å)
    interface_gap: float = 2.5     # 初期界面間隔 (Å)
    vacuum: float = 2.0            # z方向バッファ (Å)
    grip_thick: float = 5.0        # グリップ厚さ (Å)
    rattle: float = 0.05           # 原子位置の初期擾乱 (Å)

    # --- 時間パラメータ ---
    dt: float = 0.002              # タイムステップ (ps)
    anneal_temp: float = 600.0     # 焼きなまし温度 (K)
    anneal_time: float = 10.0      # 焼きなまし時間 (ps)
    cool_time: float = 10.0        # 冷却時間 (ps)
    eq_time: float = 10.0          # 平衡化時間 (ps)

    # --- 引張試験 ---
    temp: float = 300.0            # 試験温度 (K)
    erate: float = 0.001           # ひずみ速度 (1/ps)
    max_strain: float = 0.30       # 最大ひずみ（halt未発動時の上限）

    # --- 破断検知 ---
    halt_strain: float = 0.03      # halt判定開始ひずみ
    halt_stress: float = 0.1       # halt判定応力閾値 (GPa)

    # --- 出力 ---
    thermo_freq: int = 100         # thermo出力頻度 (timesteps)
    dump_freq: int = 100           # dump出力頻度 (timesteps)

    # --- 実行 ---
    np: int = 1                    # MPI並列数
    gpu: bool = False              # GPU使用フラグ（RunPod時）
    runpod: bool = False           # RunPodでリモート実行
    runpod_id: str = None          # 既存のRunPod IDを使用
    keep_pod: bool = False         # RunPod実行後にPodを維持

    # --- フォルダ名 ---
    label: str = ""                # 明示的なサフィックス（空なら自動生成）

    def pair_coeff_elements(self):
        """sort()後の元素順序（原子番号昇順）を返す"""
        elems = sorted([self.element1, self.element2],
                       key=lambda e: atomic_numbers[e])
        return ' '.join(elems)

    def job_name(self):
        """
        ジョブ名（=出力フォルダ名）を生成。

        基本形: {element1}{miller1}_{element2}{miller2}_{temp}K
        - label指定時:  基本形_label  （例: Cu100_Al100_300K_xy60_z30）
        - label未指定:  デフォルトと異なるパラメータを自動付与
        - 全てデフォルト: 基本形のみ
        """
        m1 = ''.join(map(str, self.miller1))
        m2 = ''.join(map(str, self.miller2))
        base = f"{self.element1}{m1}_{self.element2}{m2}_{self.temp:.0f}K"

        if self.label:
            return f"{base}_{self.label}"

        # 自動検出: デフォルトと異なるパラメータをサフィックスに追加
        defaults = SimConfig()
        skip = {'element1', 'element2', 'miller1', 'miller2', 'temp',
                'label', 'np', 'potential', 'gpu', 'runpod', 'keep_pod'}
        short = {
            'lattice1': 'a1', 'lattice2': 'a2',
            'target_xy': 'xy', 'target_z': 'z',
            'interface_gap': 'gap', 'vacuum': 'vac',
            'grip_thick': 'grip', 'rattle': 'rattle',
            'dt': 'dt',
            'anneal_temp': 'aT', 'anneal_time': 'at',
            'cool_time': 'ct', 'eq_time': 'eqt',
            'erate': 'erate', 'max_strain': 'maxe',
            'halt_strain': 'hse', 'halt_stress': 'hss',
            'thermo_freq': 'tf', 'dump_freq': 'df',
        }
        extras = []
        for f in fields(self):
            if f.name in skip:
                continue
            val = getattr(self, f.name)
            dval = getattr(defaults, f.name)
            if val != dval:
                key = short.get(f.name, f.name)
                if isinstance(val, float):
                    extras.append(f"{key}{val:g}")
                else:
                    extras.append(f"{key}{val}")

        if extras:
            return base + "_" + "_".join(extras)
        return base


def layer_spacing(miller, a):
    """FCC結晶の面間隔を返す"""
    if miller == (1, 0, 0):
        return a / 2
    elif miller == (1, 1, 0):
        return a / (2 * np.sqrt(2))
    elif miller == (1, 1, 1):
        return a / np.sqrt(3)
    else:
        raise ValueError(f"Unsupported Miller index: {miller}")


def calc_layers(miller, a, target_z):
    """目標厚さ(Å)に必要な層数を計算"""
    d = layer_spacing(miller, a)
    return max(int(np.ceil(target_z / d)), 6)


def create_slab(element, miller, layers, a):
    """FCC表面スラブを構築（単位セル）"""
    if miller == (1, 0, 0):
        return fcc100(element, size=(1, 1, layers), a=a, vacuum=0)
    elif miller == (1, 1, 0):
        return fcc110(element, size=(1, 1, layers), a=a, vacuum=0)
    elif miller == (1, 1, 1):
        return fcc111(element, size=(1, 2, layers), a=a, orthogonal=True, vacuum=0)
    else:
        raise ValueError(f"Unsupported Miller index: {miller}")


def find_best_repeats(elem1, miller1, a1, elem2, miller2, a2, target_xy):
    """
    目標側面寸法に対してミスマッチを最小化する繰り返し数を探索。
    elem1側のセル寸法を基準とし、elem2側はstack(maxstrain=None)でelem1に合わせる。
    """
    unit1 = create_slab(elem1, miller1, layers=1, a=a1)
    unit2 = create_slab(elem2, miller2, layers=1, a=a2)
    dx1, dy1 = unit1.get_cell()[0, 0], unit1.get_cell()[1, 1]
    dx2, dy2 = unit2.get_cell()[0, 0], unit2.get_cell()[1, 1]

    # elem1繰り返し数（目標寸法に最も近い整数）
    nx1 = max(1, round(target_xy / dx1))
    ny1 = max(1, round(target_xy / dy1))
    Lx, Ly = nx1 * dx1, ny1 * dy1

    # elem2繰り返し数（elem1寸法に最も近い整数）
    nx2 = max(1, round(Lx / dx2))
    ny2 = max(1, round(Ly / dy2))

    # ミスマッチ（stack後にelem2側が受ける歪み）
    mismatch_x = (Lx - nx2 * dx2) / (nx2 * dx2) * 100
    mismatch_y = (Ly - ny2 * dy2) / (ny2 * dy2) * 100

    print(f"--- Supercell Matching ---")
    print(f"  {elem1} unit: dx={dx1:.3f} dy={dy1:.3f} Å")
    print(f"  {elem2} unit: dx={dx2:.3f} dy={dy2:.3f} Å")
    print(f"  {elem1} ({nx1}×{ny1}) → {Lx:.1f} × {Ly:.1f} Å")
    print(f"  {elem2} ({nx2}×{ny2}) → {nx2*dx2:.1f} × {ny2*dy2:.1f} Å (natural)")
    print(f"  Mismatch: x={mismatch_x:+.1f}%, y={mismatch_y:+.1f}%")

    return (nx1, ny1, 1), (nx2, ny2, 1)


def run_simulation(cfg: SimConfig):
    """
    異種金属界面引張試験（剛体グリップ方式）。

    フロー:
      1. 0K構造緩和（x,y箱寸法 + 原子位置）
      2. 焼きなまし（界面ミスフィット緩和）→ 冷却
      3. グリップ定義 + 平衡化
      4. 引張試験（下端固定、上端を一定速度で引張）
    """
    project_root = os.getcwd()
    sim_dir = os.path.join(project_root, "simulations", "CuAl_interface_strength")

    job_name = cfg.job_name()
    seed = hash(job_name) % 100000 + 1
    job_dir = os.path.join(sim_dir, job_name)
    os.makedirs(job_dir, exist_ok=True)

    # === 1. 構造構築 ===
    print(f"\n=== Building: {cfg.element1}{cfg.miller1} | {cfg.element2}{cfg.miller2}, T={cfg.temp}K ===")

    layers1 = calc_layers(cfg.miller1, cfg.lattice1, cfg.target_z)
    layers2 = calc_layers(cfg.miller2, cfg.lattice2, cfg.target_z)
    thick1 = layers1 * layer_spacing(cfg.miller1, cfg.lattice1)
    thick2 = layers2 * layer_spacing(cfg.miller2, cfg.lattice2)
    print(f"  {cfg.element1}: {layers1} layers → {thick1:.1f} Å")
    print(f"  {cfg.element2}: {layers2} layers → {thick2:.1f} Å")

    rep1, rep2 = find_best_repeats(
        cfg.element1, cfg.miller1, cfg.lattice1,
        cfg.element2, cfg.miller2, cfg.lattice2,
        cfg.target_xy,
    )

    slab1 = create_slab(cfg.element1, cfg.miller1, layers1, cfg.lattice1).repeat(rep1)
    slab2 = create_slab(cfg.element2, cfg.miller2, layers2, cfg.lattice2).repeat(rep2)

    # elem1(下) | elem2(上)、maxstrain=Noneでelem2をelem1セルに合わせる
    # distance引数を使わない（scipy最適化がO(N²)で遅い）
    # 代わりに手動で界面間隔を設定し、LAMMPS minimizeで最適化
    interface = stack(slab1, slab2, axis=2, maxstrain=None)
    # 界面間隔を調整: slab1最上面とslab2最下面の間隔を設定
    zmax1 = slab1.positions[:, 2].max()
    cell_z1 = slab1.get_cell()[2, 2]
    zmin2_rel = slab2.positions[:, 2].min()
    current_gap = (cell_z1 - zmax1) + zmin2_rel
    shift = cfg.interface_gap - current_gap
    # slab2原子（slab1原子数より後のインデックス）をシフト
    n1 = len(slab1)
    interface.positions[n1:, 2] += shift
    interface.cell[2, 2] += shift
    # z方向にバッファを追加（boundary s での read_data エラー回避）
    interface.center(vacuum=cfg.vacuum, axis=2)
    interface.pbc = [True, True, True]

    # 原子位置をわずかに乱す（対称性を崩し、初期緩和を助ける）
    interface.rattle(stdev=cfg.rattle, seed=seed)

    # ソート（原子番号昇順）
    interface = sort(interface)
    n_atoms = len(interface)
    cell = interface.get_cell()
    print(f"  Total atoms: {n_atoms}")
    print(f"  Cell: {cell[0,0]:.1f} × {cell[1,1]:.1f} × {cell[2,2]:.1f} Å")

    # === 2. LAMMPS入力生成 ===
    pot_path = os.path.abspath(os.path.join(project_root, cfg.potential))
    pair_elems = cfg.pair_coeff_elements()

    anneal_steps = int(cfg.anneal_time / cfg.dt)
    cool_steps = int(cfg.cool_time / cfg.dt)
    eq_steps = int(cfg.eq_time / cfg.dt)
    tensile_steps = int(cfg.max_strain / cfg.erate / cfg.dt)

    cmds = [
        # ===== セットアップ =====
        "units metal",
        "boundary p p s",
        "atom_style atomic",
        f"timestep {cfg.dt}",
        "read_data data.interface",
        "",
        "pair_style eam/alloy",
        f"pair_coeff * * {pot_path} {pair_elems}",
        "",
        "neighbor 2.0 bin",
        "neigh_modify delay 10 check yes",
        "",
        f"thermo {cfg.thermo_freq}",
        "thermo_style custom step temp pe press pxx pyy lx ly lz",
        "",

        # ===== Phase 0: Shake (原子の重なり解消) =====
        "print '=== Phase 0: Shake (High-T limit) ==='",
        f"velocity all create 1000.0 {seed} dist gaussian",
        "fix shake all nve/limit 0.1",
        "run 1000",
        "unfix shake",
        "velocity all set 0 0 0",
        "",

        # ===== Phase 1: 0K構造緩和 =====
        "# x,y方向の箱寸法 + 原子位置を同時最適化（z方向はshrink-wrap）",
        "print '=== Phase 1: 0K Relaxation ==='",
        "fix relax all box/relax x 0.0 y 0.0 vmax 0.001",
        "minimize 1.0e-8 1.0e-10 10000 100000",
        "unfix relax",
        "",

        # ===== Phase 2: 焼きなまし =====
        "# 界面近傍の原子配置を緩和（ミスフィット軽減）",
        f"print '=== Phase 2: Annealing at {cfg.anneal_temp}K for {cfg.anneal_time}ps ==='",
        "reset_timestep 0",
        f"velocity all create {cfg.anneal_temp} {seed} dist gaussian",
        f"fix ann all npt temp {cfg.anneal_temp} {cfg.anneal_temp} 0.1 x 0.0 0.0 1.0 y 0.0 0.0 1.0",
        f"run {anneal_steps}",
        "unfix ann",
        "",

        # ===== Phase 2b: 冷却 =====
        f"print '=== Phase 2b: Cooling {cfg.anneal_temp}K -> {cfg.temp}K ==='",
        f"fix cool all npt temp {cfg.anneal_temp} {cfg.temp} 0.1 x 0.0 0.0 1.0 y 0.0 0.0 1.0",
        f"run {cool_steps}",
        "unfix cool",
        "",

        # ===== Phase 3: グリップ定義 =====
        "# 焼きなまし後の座標を基準にグリップ領域を定義",
        "print '=== Phase 3: Define Grips ==='",
        "variable zmin equal bound(all,zmin)",
        "variable zmax equal bound(all,zmax)",
        f"variable grip_thick equal {cfg.grip_thick}",
        "variable zlo_grip equal ${zmin}+${grip_thick}",
        "variable zhi_grip equal ${zmax}-${grip_thick}",
        "",
        "region rgn_bot block INF INF INF INF INF ${zlo_grip}",
        "region rgn_top block INF INF INF INF ${zhi_grip} INF",
        "group bottom region rgn_bot",
        "group top region rgn_top",
        "group mobile subtract all bottom top",
        "",
        "print 'Bottom grip: $(count(bottom)) atoms'",
        "print 'Top grip:    $(count(top)) atoms'",
        "print 'Mobile:      $(count(mobile)) atoms'",
        "",

        # ===== Phase 4: 平衡化（グリップ拘束あり） =====
        "print '=== Phase 4: Equilibration with Grips ==='",
        "reset_timestep 0",
        "velocity bottom set 0.0 0.0 0.0",
        "velocity top set 0.0 0.0 0.0",
        "# グリップ: nve（時間積分）+ setforce 0（力ゼロ）→ 等速運動",
        "fix nve_bot bottom nve",
        "fix nve_top top nve",
        "fix freeze_bot bottom setforce 0.0 0.0 0.0",
        "fix freeze_top top setforce 0.0 0.0 0.0",
        f"fix eq mobile npt temp {cfg.temp} {cfg.temp} 0.1 x 0.0 0.0 1.0 y 0.0 0.0 1.0",
        f"run {eq_steps}",
        "unfix eq",
        "",

        # ===== Phase 5: 引張試験 =====
        "print '=== Phase 5: Tensile Test ==='",
        "reset_timestep 0",
        "",
        "# グリップ重心間距離で初期材料長を定義",
        "variable zcm_bot equal xcm(bottom,z)",
        "variable zcm_top equal xcm(top,z)",
        "variable tmp_L equal v_zcm_top-v_zcm_bot",
        "variable L0 equal ${tmp_L}",
        "print 'Initial grip distance L0 = ${L0} Ang'",
        "",
        "# 引っ張り速度 = erate × L0",
        f"variable vpull equal {cfg.erate}*v_L0",
        "velocity top set 0.0 0.0 ${vpull}",
        f"print 'Pull velocity = ${{vpull}} Ang/ps (erate={cfg.erate}/ps)'",
        "",
        "# 可動原子: NVTで温度制御のみ (NPTはグリップ固定と競合するため不可)",
        f"fix pull mobile nvt temp {cfg.temp} {cfg.temp} 0.1",
        "",
        "# ひずみ: グリップ重心の相対変位から計算",
        "variable strain equal (xcm(top,z)-xcm(bottom,z)-v_L0)/v_L0",
        "",
        "# 応力: 原子応力の総和 / 材料体積 (bar·Å³ / ų → bar → GPa)",
        "compute peratom all stress/atom NULL",
        "compute szz_sum all reduce sum c_peratom[3]",
        "variable Lmat equal xcm(top,z)-xcm(bottom,z)",
        "variable stress_zz equal c_szz_sum/(lx*ly*v_Lmat)/10000",
        "",
        "# 出力設定",
        f"thermo {cfg.thermo_freq}",
        "thermo_style custom step temp v_strain v_stress_zz pxx pyy lx ly v_Lmat",
        "",
        f'fix out1 all print {cfg.dump_freq} "${{strain}} ${{stress_zz}}" file stress_strain.txt screen no',
        f"dump 1 all custom {cfg.dump_freq} dump.lammpstrj id type x y z"
        " c_peratom[1] c_peratom[2] c_peratom[3]"
        " c_peratom[4] c_peratom[5] c_peratom[6]",
        "",
        "# 破断検知: halt_strain以降、応力がhalt_stress未満になったら停止",
        f'variable should_halt equal "(v_strain > {cfg.halt_strain}) * (v_stress_zz < {cfg.halt_stress})"',
        f"fix halt_frac all halt {cfg.thermo_freq} v_should_halt > 0.5 error soft",
        "",
        f"run {tensile_steps}",
    ]

    # ファイル出力
    data_file = os.path.join(job_dir, "data.interface")
    write(data_file, interface, format='lammps-data', atom_style='atomic')

    inp_file = os.path.join(job_dir, "in.tensile")
    with open(inp_file, 'w') as f:
        for cmd in cmds:
            f.write(cmd + "\n")

    print(f"\nInput files → {job_dir}")

    if cfg.runpod:
        # RunPodでリモート実行
        from scripts.runpod_runner import run_on_runpod
        print("Running LAMMPS on RunPod...")
        try:
            # ポテンシャルファイルのパス解決
            # cfg.potential は "potentials/AlCu.eam.alloy" のような相対パスまたは絶対パス
            pot_path = os.path.abspath(cfg.potential)
            if not os.path.exists(pot_path):
                 print(f"Warning: Potential file not found at {pot_path}")

            run_on_runpod(
                job_dir=job_dir,
                input_file="in.tensile",
                pot_path=pot_path,
                np=cfg.np,
                gpu=cfg.gpu,
                keep_pod=cfg.keep_pod,
                pod_id=cfg.runpod_id,
            )
        except ImportError:
            print("Error: scripts.runpod_runner module not found.")
        
    else:
        # ローカル実行
        print("Running LAMMPS...")
        lammps_cmd = os.environ.get("ASE_LAMMPSRUN_COMMAND", "/home/iwash/lammps/build/lmp")
        if cfg.np > 1:
            cmd = f"mpirun -np {cfg.np} {lammps_cmd} -in in.tensile > log.lammps"
        else:
            cmd = f"{lammps_cmd} < in.tensile > log.lammps"
        print(f"  {cmd}")

        result = subprocess.run(cmd, shell=True, cwd=job_dir)
        if result.returncode == 0:
            print("Simulation finished successfully.")
        else:
            print(f"Simulation failed with return code {result.returncode}")
            log_path = os.path.join(job_dir, "log.lammps")
            if os.path.exists(log_path):
                subprocess.run(["tail", "-n", "20", log_path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="異種金属界面引張試験（剛体グリップ方式）")

    # 材料
    mat = parser.add_argument_group("材料")
    mat.add_argument("--e1", type=str, default='Cu',
                     help="下部スラブ元素 (default: Cu)")
    mat.add_argument("--e2", type=str, default='Al',
                     help="上部スラブ元素 (default: Al)")
    mat.add_argument("--a1", type=float, default=3.61,
                     help="element1の格子定数 [Å] (default: 3.61)")
    mat.add_argument("--a2", type=float, default=4.05,
                     help="element2の格子定数 [Å] (default: 4.05)")
    mat.add_argument("--m1", type=int, nargs=3, default=[1, 0, 0],
                     help="element1のMiller指数 (default: 1 0 0)")
    mat.add_argument("--m2", type=int, nargs=3, default=[1, 0, 0],
                     help="element2のMiller指数 (default: 1 0 0)")
    mat.add_argument("--potential", type=str, default='potentials/AlCu.eam.alloy',
                     help="ポテンシャルファイル（プロジェクトルートからの相対パス）")

    # 条件
    cond = parser.add_argument_group("条件")
    cond.add_argument("--temp", type=float, default=300.0,
                      help="試験温度 [K] (default: 300)")
    cond.add_argument("--erate", type=float, default=0.001,
                      help="ひずみ速度 [1/ps] (default: 0.001)")
    cond.add_argument("--max_strain", type=float, default=0.30,
                      help="最大ひずみ (default: 0.30)")

    # ジオメトリ
    geo = parser.add_argument_group("ジオメトリ")
    geo.add_argument("--target_xy", type=float, default=30.0,
                     help="目標側面寸法 [Å] (default: 30)")
    geo.add_argument("--target_z", type=float, default=30.0,
                     help="各金属の目標厚さ [Å] (default: 30)")
    geo.add_argument("--grip_thick", type=float, default=5.0,
                     help="グリップ厚さ [Å] (default: 5.0)")

    # フェーズ
    phase = parser.add_argument_group("フェーズ")
    phase.add_argument("--anneal_temp", type=float, default=600.0,
                       help="焼きなまし温度 [K] (default: 600)")
    phase.add_argument("--anneal_time", type=float, default=10.0,
                       help="焼きなまし時間 [ps] (default: 10)")
    phase.add_argument("--cool_time", type=float, default=10.0,
                       help="冷却時間 [ps] (default: 10)")
    phase.add_argument("--eq_time", type=float, default=10.0,
                       help="平衡化時間 [ps] (default: 10)")

    # 出力
    out = parser.add_argument_group("出力")
    out.add_argument("--thermo_freq", type=int, default=100,
                     help="thermo出力頻度 [timesteps] (default: 100)")
    out.add_argument("--dump_freq", type=int, default=100,
                     help="dump出力頻度 [timesteps] (default: 100)")

    # 実行
    exe = parser.add_argument_group("実行")
    exe.add_argument("--np", type=int, default=1,
                     help="MPI並列数 (default: 1)")
    exe.add_argument("--gpu", action='store_true',
                     help="GPU使用（RunPod時）")
    exe.add_argument("--runpod", action='store_true',
                     help="RunPodでリモート実行（RUNPOD_API_KEY環境変数が必要）")
    exe.add_argument("--runpod-id", type=str, default=None,
                     help="既存のRunPod IDを使用（起動待ち時間短縮）")
    exe.add_argument("--keep-pod", action='store_true',
                     help="RunPod実行後にPodを停止しない（連続実行時のコスト削減）")
    exe.add_argument("--label", type=str, default="",
                     help="出力フォルダ名のサフィックス（未指定時は変更パラメータから自動生成）")

    args = parser.parse_args()
    cfg = SimConfig(
        element1=args.e1,
        element2=args.e2,
        lattice1=args.a1,
        lattice2=args.a2,
        miller1=tuple(args.m1),
        miller2=tuple(args.m2),
        potential=args.potential,
        target_xy=args.target_xy,
        target_z=args.target_z,
        grip_thick=args.grip_thick,
        temp=args.temp,
        erate=args.erate,
        max_strain=args.max_strain,
        anneal_temp=args.anneal_temp,
        anneal_time=args.anneal_time,
        cool_time=args.cool_time,
        eq_time=args.eq_time,
        thermo_freq=args.thermo_freq,
        dump_freq=args.dump_freq,
        np=args.np,
        gpu=args.gpu,
        runpod=args.runpod,
        runpod_id=args.runpod_id,
        keep_pod=args.keep_pod,
        label=args.label,
    )
    run_simulation(cfg)
