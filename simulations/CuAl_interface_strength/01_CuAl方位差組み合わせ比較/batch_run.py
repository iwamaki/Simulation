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
    simulations/CuAl_interface_strength/{element1}{miller1}_{element2}{miller2}_{temp}K/
    例: Cu100_Al100_300K/, Cu111_Al111_10K/
"""

import sys
import os
import time
from dataclasses import replace

# プロジェクトルートからの実行を前提
sys.path.insert(0, os.path.join(os.getcwd(), "simulations", "CuAl_interface_strength"))
from CuAl_interface_strength import SimConfig, run_simulation

# === 共通設定（全実験のベース。変えたいパラメータだけ指定） ===
base = SimConfig(erate=0.002, np=4)

# === 実験条件リスト: (説明文, replace(base, ...)) のタプル ===
experiments = [
    # --- 結晶方位の組み合わせ比較 (300K, erate=0.002) ---
    ("Ori_100_100 / Cu(100)-Al(100)",
     replace(base, miller1=(1, 0, 0), miller2=(1, 0, 0))),

    ("Ori_110_110 / Cu(110)-Al(110)",
     replace(base, miller1=(1, 1, 0), miller2=(1, 1, 0))),

    ("Ori_111_111 / Cu(111)-Al(111)",
     replace(base, miller1=(1, 1, 1), miller2=(1, 1, 1))),

    ("Ori_100_110 / Cu(100)-Al(110)",
     replace(base, miller1=(1, 0, 0), miller2=(1, 1, 0))),

    ("Ori_100_111 / Cu(100)-Al(111)",
     replace(base, miller1=(1, 0, 0), miller2=(1, 1, 1))),

    ("Ori_110_100 / Cu(110)-Al(100)",
     replace(base, miller1=(1, 1, 0), miller2=(1, 0, 0))),

    ("Ori_110_111 / Cu(110)-Al(111)",
     replace(base, miller1=(1, 1, 0), miller2=(1, 1, 1))),

    ("Ori_111_100 / Cu(111)-Al(100)",
     replace(base, miller1=(1, 1, 1), miller2=(1, 0, 0))),

    ("Ori_111_110 / Cu(111)-Al(110)",
     replace(base, miller1=(1, 1, 1), miller2=(1, 1, 0))),
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
