"""
バッチ実行スクリプト: 複数条件の引張試験を順次実行する。

使い方:
    プロジェクトルートから実行:
        python simulations/CuAl_interface_strength/batch_run.py

カスタマイズ方法:
    1. base = SimConfig(...) で共通のベース設定を定義する
       - デフォルトから変えたいパラメータだけ指定すればよい
       - 例: SimConfig(erate=0.005, target_xy=50.0)

    2. experiments リストに (説明文, SimConfig) のタプルを追加する
       - replace(base, ...) でbaseから一部だけ変更したコピーを作る
       - 例: replace(base, temp=600)          → 温度だけ変更
             replace(base, miller1=(1,1,1))   → 方位だけ変更
             replace(base, temp=10, np=4)     → 複数パラメータ同時変更

    3. 不要な条件はコメントアウトまたは削除する

SimConfigの主要パラメータ:
    材料:       element1/2 (元素), lattice1/2 (格子定数), miller1/2 (面方位),
                potential (ポテンシャルファイルパス)
    ジオメトリ: target_xy/target_z (セル寸法), grip_thick, interface_gap
    時間:       anneal_temp/anneal_time, cool_time, eq_time, dt
    引張:       temp (試験温度), erate (ひずみ速度), max_strain
    破断検知:   halt_strain, halt_stress
    出力:       thermo_freq, dump_freq
    実行:       np (MPI並列数)

出力先:
    simulations/CuAl_interface_strength/{基本名}[_{サフィックス}]/
    基本名: {element1}{miller1}_{element2}{miller2}_{temp}K

    フォルダ名の決定ルール:
      1. label指定時: 基本名_label     例: Cu100_Al100_300K_xy60_z30
      2. label未指定: デフォルトと異なるパラメータを自動付与
         例: Cu100_Al100_300K_erate0.002_xy60
      3. 全てデフォルト: 基本名のみ   例: Cu100_Al100_300K
"""

import sys
import os
import time
from dataclasses import replace

# プロジェクトルートからの実行を前提
sys.path.insert(0, os.path.join(os.getcwd(), "simulations", "CuAl_interface_strength"))
from CuAl_interface_strength import SimConfig, run_simulation

# === 共通設定（全実験のベース。変えたいパラメータだけ指定） ===
# 基準: Cu(100)-Al(100), 300K, erate=0.002
base = SimConfig(erate=0.002, np=4,
                 miller1=(1, 0, 0), miller2=(1, 0, 0))

# === 実験条件リスト: (説明文, replace(base, ...)) のタプル ===
experiments = [
    # --- システムサイズ依存性の検証 (基準: Cu(100)-Al(100)) ---

    # 1. Base (現状のサイズ感: xy=30, z=30)
    ("Size_Base / 30x30_Z30",
     replace(base, target_xy=30.0, target_z=30.0, label="xy30_z30")),

    # 2. 界面面積を拡大 (xy=60, 4倍の断面積)
    ("Size_LargeXY / 60x60_Z30",
     replace(base, target_xy=60.0, target_z=30.0, label="xy60_z30")),

    # 3. 厚みを拡大 (z=60, バルク領域を増やす)
    ("Size_LargeZ / 30x30_Z60",
     replace(base, target_xy=30.0, target_z=60.0, label="xy30_z60")),

    # 4. 全体を拡大 (ユーザー提案に近い、xy=40 -> 60で統一)
    ("Size_LargeAll / 60x60_Z60",
     replace(base, target_xy=60.0, target_z=60.0, label="xy60_z60")),
]

print(f"Starting batch simulation with {len(experiments)} conditions...")
print(f"Base config: {base}\n")

for i, (desc, cfg) in enumerate(experiments):
    print(f"[{i+1}/{len(experiments)}] Running: {desc}")

    start_time = time.time()
    try:
        run_simulation(cfg)
        elapsed = time.time() - start_time
        print(f"-> Finished in {elapsed:.1f} sec ({elapsed/60:.1f} min)\n")
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"-> Failed after {elapsed:.1f} sec: {e}\n")

print("All simulations completed.")
