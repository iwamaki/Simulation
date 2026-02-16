"""
FCC単結晶 引張/圧縮試験

ase.build.bulk() で完全な周期結晶を構築し、
fix deform (z軸) + fix npt (x,y側面ゼロ圧) で変形試験を行う。
結晶方位（Miller指数）でz軸＝負荷軸の結晶学的方向を指定する。

使用例:
  # Cu (100) 引張（デフォルト）
  python simulations/single_crystal_tensile/single_crystal_tensile.py

  # Cu (111) 圧縮
  python simulations/single_crystal_tensile/single_crystal_tensile.py \
      --miller 1 1 1 --mode compression --no-halt

  # Al (110) 引張
  python simulations/single_crystal_tensile/single_crystal_tensile.py \
      --element Al --lattice 4.05 --miller 1 1 0 \
      --potential potentials/Al_zhou.eam.alloy
"""

import os
import sys
import argparse
import subprocess
import hashlib
from dataclasses import dataclass, fields
from ase.build import bulk, cut
from ase.io import write

sys.path.append(os.getcwd())


# 結晶方位テーブル: 直交セルベクトル（conventional cubic cell基準）
# c がz軸（=負荷軸）方向
ORIENTATIONS = {
    (1, 0, 0): {'a': (1, 0, 0),  'b': (0, 1, 0),  'c': (0, 0, 1)},
    (1, 1, 0): {'a': (-1, 1, 0), 'b': (0, 0, 1),  'c': (1, 1, 0)},
    (1, 1, 1): {'a': (1, -1, 0), 'b': (1, 1, -2), 'c': (1, 1, 1)},
}


@dataclass
class SimConfig:
    """単結晶変形試験の全パラメータ"""

    # --- 材料 ---
    element: str = 'Cu'
    lattice: float = 3.615          # 格子定数 (Å)
    miller: tuple = (1, 0, 0)       # z軸の結晶方位
    potential: str = 'potentials/Cu_zhou.eam.alloy'
    pair_style: str = 'eam/alloy'

    # --- ジオメトリ ---
    target_size: float = 40.0       # 目標箱寸法 (Å)

    # --- 時間 ---
    dt: float = 0.002               # タイムステップ (ps)
    eq_time: float = 5.0            # 平衡化時間 (ps)  ※完全結晶は2-3psで十分

    # --- 変形 ---
    load_mode: str = 'tension'      # tension / compression
    temp: float = 300.0             # 試験温度 (K)
    erate: float = 0.001            # ひずみ速度 (1/ps)
    max_strain: float = 0.30        # 最大ひずみ

    # --- 破断検知 ---
    halt_enabled: bool = True
    halt_strain: float = 0.05       # halt判定開始ひずみ
    halt_stress: float = 0.1        # halt判定応力閾値 (GPa)

    # --- 出力 ---
    thermo_freq: int = 100
    dump_freq: int = 500

    # --- 実行 ---
    np: int = 1
    gpu: bool = False
    runpod: bool = False
    keep_pod: bool = False
    label: str = ""

    def signed_erate(self):
        return self.erate if self.load_mode == 'tension' else -self.erate

    def pair_coeff_line(self, pot_path):
        if self.pair_style in ('eam/alloy', 'eam/fs'):
            return f"pair_coeff * * {pot_path} {self.element}"
        elif self.pair_style == 'eam':
            return f"pair_coeff 1 1 {pot_path}"
        else:
            raise ValueError(f"未対応: {self.pair_style}")

    def job_name(self):
        m = ''.join(map(str, self.miller))
        base = f"{self.element}{m}_{self.load_mode}_{self.temp:.0f}K"
        if self.label:
            return f"{base}_{self.label}"
        defaults = SimConfig()
        skip = {'element', 'miller', 'load_mode', 'temp',
                'label', 'np', 'potential', 'pair_style', 'halt_enabled',
                'gpu', 'runpod', 'keep_pod'}
        short = {
            'lattice': 'a', 'target_size': 'L', 'dt': 'dt',
            'eq_time': 'eqt', 'erate': 'erate', 'max_strain': 'maxe',
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


def build_crystal(element, a, miller, target_size):
    """
    FCC単結晶のバルク周期構造を構築。

    bulk() で完全な従来型単位胞を作り、cut() で指定方位に回転。
    repeat() でスーパーセルに展開する。
    """
    if miller not in ORIENTATIONS:
        supported = ', '.join(str(m) for m in ORIENTATIONS)
        raise ValueError(f"未対応の Miller指数: {miller} (対応: {supported})")

    base = bulk(element, 'fcc', a=a, cubic=True)

    if miller == (1, 0, 0):
        unit = base
    else:
        orient = ORIENTATIONS[miller]
        unit = cut(base, a=orient['a'], b=orient['b'], c=orient['c'])

    unit.pbc = True

    # 各方向の繰り返し数
    dx, dy, dz = unit.cell.lengths()
    nx = max(1, round(target_size / dx))
    ny = max(1, round(target_size / dy))
    nz = max(1, round(target_size / dz))

    crystal = unit.repeat((nx, ny, nz))
    return crystal, (nx, ny, nz), len(unit)


def run_simulation(cfg: SimConfig):
    """
    単結晶変形試験を実行。

    Phase 1: 0K構造緩和（ポテンシャルの平衡格子定数に調整）
    Phase 2: NPT平衡化（目標温度・ゼロ圧力）
    Phase 3: z軸変形試験（fix deform z + fix npt x,y）
    """
    project_root = os.getcwd()
    sim_dir = os.path.join(project_root, "simulations", "single_crystal_tensile")

    job_name = cfg.job_name()
    seed = int(hashlib.md5(job_name.encode()).hexdigest(), 16) % 100000 + 1
    job_dir = os.path.join(sim_dir, job_name)
    os.makedirs(job_dir, exist_ok=True)

    # === 構造構築 ===
    m_str = ''.join(map(str, cfg.miller))
    print(f"\n=== {cfg.element}({m_str}) {cfg.load_mode}, T={cfg.temp}K ===")

    crystal, (nx, ny, nz), n_unit = build_crystal(
        cfg.element, cfg.lattice, cfg.miller, cfg.target_size)

    cell = crystal.get_cell()
    print(f"  Unit cell: {n_unit} atoms")
    print(f"  Supercell: ({nx} x {ny} x {nz})")
    print(f"  Box: {cell[0,0]:.1f} x {cell[1,1]:.1f} x {cell[2,2]:.1f} Ang")
    print(f"  Total: {len(crystal)} atoms")

    # === LAMMPS入力生成 ===
    pot_path = os.path.abspath(os.path.join(project_root, cfg.potential))
    signed_erate = cfg.signed_erate()
    eq_steps = int(cfg.eq_time / cfg.dt)
    deform_steps = int(cfg.max_strain / cfg.erate / cfg.dt)

    # (110)/(111)方位ではせん断変形を許容するため三斜晶系を使用
    needs_triclinic = cfg.miller != (1, 0, 0)
    eq_coupling = "tri" if needs_triclinic else "aniso"
    npt_lateral = ("x 0.0 0.0 1.0 y 0.0 0.0 1.0"
                   " xy 0.0 0.0 1.0 xz 0.0 0.0 1.0 yz 0.0 0.0 1.0"
                   if needs_triclinic else
                   "x 0.0 0.0 1.0 y 0.0 0.0 1.0")

    cmds = [
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        f"timestep {cfg.dt}",
        "read_data data.crystal",
        *(["change_box all triclinic"] if needs_triclinic else []),
        "",
        f"pair_style {cfg.pair_style}",
        cfg.pair_coeff_line(pot_path),
        "",
        "neighbor 2.0 bin",
        "neigh_modify delay 10 check yes",
        "",
        f"thermo {cfg.thermo_freq}",
        "thermo_style custom step temp pe press pxx pyy pzz lx ly lz",
        "",

        # Phase 1: 0K構造緩和
        "print '=== Phase 1: 0K Relaxation ==='",
        "fix relax all box/relax aniso 0.0 vmax 0.001",
        "minimize 1.0e-8 1.0e-10 10000 100000",
        "unfix relax",
        "",

        # Phase 2: NPT平衡化
        f"print '=== Phase 2: Equilibration at {cfg.temp}K ==='",
        "reset_timestep 0",
        f"velocity all create {cfg.temp} {seed} dist gaussian",
        f"fix eq all npt temp {cfg.temp} {cfg.temp} 0.1 {eq_coupling} 0.0 0.0 1.0",
        f"run {eq_steps}",
        "unfix eq",
        "",

        # Phase 3: 変形試験
        f"print '=== Phase 3: {cfg.load_mode.capitalize()} ==='",
        "reset_timestep 0",
        "",
        "# 初期z寸法の記録",
        "variable tmp equal lz",
        "variable lz0 equal ${tmp}",
        "variable tmp delete",
        "",
        "# z軸変形 + x,yゼロ圧力（Poisson収縮を許容）",
        f"fix deform_fix all deform 1 z erate {signed_erate} remap x",
        f"fix npt_fix all npt temp {cfg.temp} {cfg.temp} 0.1"
        f" {npt_lateral}",
        "",
        "# ひずみ・応力（sigma = -P [GPa]）",
        "variable strain equal (lz-v_lz0)/v_lz0",
        "variable stress equal -pzz/10000",
        "",
        "# 欠陥解析用compute",
        "# stress/atom: 単位は bar*Å³（実応力はボロノイ体積で割る必要あり）",
        "compute peratom all stress/atom NULL",
        f"compute cna all cna/atom {cfg.lattice * 0.854:.3f}",
        "compute csym all centro/atom fcc",
        "",
        f"thermo {cfg.thermo_freq}",
        "thermo_style custom step temp v_strain v_stress pxx pyy pzz lx ly lz",
        "",
        f'fix out1 all print {cfg.thermo_freq} "${{strain}} ${{stress}}"'
        f' file stress_strain.txt screen no',
        f"dump 1 all custom {cfg.dump_freq} dump.lammpstrj id type x y z"
        " c_cna c_csym"
        " c_peratom[1] c_peratom[2] c_peratom[3]"
        " c_peratom[4] c_peratom[5] c_peratom[6]",
        "",
    ]

    # 破断検知
    if cfg.halt_enabled:
        if cfg.load_mode == 'tension':
            cond = (f'"(v_strain > {cfg.halt_strain}) * '
                    f'(v_stress < {cfg.halt_stress})"')
        else:
            cond = (f'"(v_strain < -{cfg.halt_strain}) * '
                    f'(v_stress > -{cfg.halt_stress})"')
        cmds.extend([
            f"variable should_halt equal {cond}",
            f"fix halt_fix all halt {cfg.thermo_freq} v_should_halt > 0.5"
            " error soft",
            "",
        ])

    cmds.append(f"run {deform_steps}")

    # === ファイル出力 ===
    write(os.path.join(job_dir, "data.crystal"),
          crystal, format='lammps-data', atom_style='atomic')

    with open(os.path.join(job_dir, "in.deform"), 'w') as f:
        for cmd in cmds:
            f.write(cmd + "\n")

    # === 実行 ===
    print(f"\nOutput -> {job_dir}")

    if cfg.runpod:
        # RunPodでリモート実行
        from scripts.runpod_runner import run_on_runpod
        print("Running LAMMPS on RunPod...")
        run_on_runpod(
            job_dir=job_dir,
            input_file="in.deform",
            pot_path=pot_path,
            np=cfg.np,
            gpu=cfg.gpu,
            keep_pod=cfg.keep_pod,
        )
    else:
        # ローカル実行
        print("Running LAMMPS...")
        lammps = os.environ.get("ASE_LAMMPSRUN_COMMAND",
                                "/home/iwash/lammps/build/lmp")
        if cfg.np > 1:
            run_cmd = f"mpirun -np {cfg.np} {lammps} -in in.deform > log.lammps"
        else:
            run_cmd = f"{lammps} < in.deform > log.lammps"
        print(f"  {run_cmd}")

        result = subprocess.run(run_cmd, shell=True, cwd=job_dir)
        if result.returncode == 0:
            print("Done.")
        else:
            print(f"FAILED (rc={result.returncode})")
            log_path = os.path.join(job_dir, "log.lammps")
            if os.path.exists(log_path):
                subprocess.run(["tail", "-n", "20", log_path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FCC単結晶 引張/圧縮試験",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  python simulations/single_crystal_tensile/single_crystal_tensile.py
  python simulations/single_crystal_tensile/single_crystal_tensile.py --miller 1 1 1 --mode compression --no-halt
  python simulations/single_crystal_tensile/single_crystal_tensile.py --element Al --lattice 4.05 --potential potentials/Al_zhou.eam.alloy
""")

    mat = parser.add_argument_group("材料")
    mat.add_argument("--element", default='Cu', help="元素記号 (default: Cu)")
    mat.add_argument("--lattice", type=float, default=3.615,
                     help="格子定数 [Ang] (default: 3.615)")
    mat.add_argument("--miller", type=int, nargs=3, default=[1, 0, 0],
                     help="z軸の結晶方位 (default: 1 0 0)")
    mat.add_argument("--potential", default='potentials/Cu_zhou.eam.alloy',
                     help="ポテンシャルファイル")
    mat.add_argument("--pair-style", default='eam/alloy',
                     choices=['eam', 'eam/alloy', 'eam/fs'])

    load = parser.add_argument_group("変形条件")
    load.add_argument("--mode", default='tension',
                      choices=['tension', 'compression'],
                      help="引張 or 圧縮 (default: tension)")
    load.add_argument("--temp", type=float, default=300.0,
                      help="試験温度 [K] (default: 300)")
    load.add_argument("--erate", type=float, default=0.001,
                      help="ひずみ速度 [1/ps] (default: 0.001)")
    load.add_argument("--max-strain", type=float, default=0.30,
                      help="最大ひずみ (default: 0.30)")

    geo = parser.add_argument_group("ジオメトリ")
    geo.add_argument("--target-size", type=float, default=40.0,
                     help="目標箱寸法 [Ang] (default: 40)")

    phase = parser.add_argument_group("フェーズ")
    phase.add_argument("--eq-time", type=float, default=5.0,
                       help="平衡化時間 [ps] (default: 5)")

    halt_grp = parser.add_argument_group("破断検知")
    halt_grp.add_argument("--no-halt", action='store_true',
                          help="破断検知を無効化")
    halt_grp.add_argument("--halt-strain", type=float, default=0.05)
    halt_grp.add_argument("--halt-stress", type=float, default=0.1)

    out = parser.add_argument_group("出力")
    out.add_argument("--thermo-freq", type=int, default=100)
    out.add_argument("--dump-freq", type=int, default=500)

    exe = parser.add_argument_group("実行")
    exe.add_argument("--np", type=int, default=1, help="MPI並列数")
    exe.add_argument("--gpu", action='store_true', help="GPU使用（RunPod時）")
    exe.add_argument("--runpod", action='store_true',
                     help="RunPodでリモート実行（RUNPOD_API_KEY環境変数が必要）")
    exe.add_argument("--keep-pod", action='store_true',
                     help="RunPod実行後にPodを停止しない（連続実行時のコスト削減）")
    exe.add_argument("--label", default="", help="フォルダ名サフィックス")

    args = parser.parse_args()
    cfg = SimConfig(
        element=args.element, lattice=args.lattice,
        miller=tuple(args.miller), potential=args.potential,
        pair_style=args.pair_style, target_size=args.target_size,
        eq_time=args.eq_time, load_mode=args.mode,
        temp=args.temp, erate=args.erate, max_strain=args.max_strain,
        halt_enabled=not args.no_halt,
        halt_strain=args.halt_strain, halt_stress=args.halt_stress,
        thermo_freq=args.thermo_freq, dump_freq=args.dump_freq,
        np=args.np, gpu=args.gpu,
        runpod=args.runpod, keep_pod=args.keep_pod,
        label=args.label,
    )
    run_simulation(cfg)
