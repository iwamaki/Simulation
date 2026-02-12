# GEMINI.md

This file provides guidance to Gemini when working with code in this repository.

## Project Overview

原子・分子スケールのシミュレーション統合環境。MD（分子動力学）とDFT（第一原理計算）をPythonで管理し、構造構築・計算実行・結果解析を行う。

## Environment

- **Python**: 3.12 (`.venv/` に仮想環境)
- **LAMMPS**: `/home/iwash/lammps/build/lmp` (ソースビルド版)
- **Quantum ESPRESSO**: `/home/iwash/qe/build/bin/` (v7.4, ソースビルド版)
- **Platform**: WSL2 (Ubuntu 24.04) on Windows

### Setup & Run

```bash
source .venv/bin/activate

# 汎用ツール
python scripts/<script_name>.py

# テーマ固有のシミュレーション（プロジェクトルートから実行）
python simulations/<テーマ>/<driver>.py [options]
```

## Architecture

```
scripts/               # 汎用ユーティリティ（テーマに依存しない）
potentials/            # ポテンシャルファイル（MD用）
notebooks/             # Jupyter Notebook（解析・可視化用）
simulations/
  └── <テーマ>/         # テーマ単位で管理
      ├── <driver>.py   #   実行ドライバ（構造構築→入力生成→実行）
      ├── PROGRESS.md   #   テーマ固有の進捗・知見メモ
      └── <条件名>/      #   個別条件の結果フォルダ（自動生成）
```

### scripts/ と simulations/ の役割分担

- **`scripts/`**: どのテーマでも使える汎用ツール（プロット、デバッグ、テスト等）
- **`simulations/<テーマ>/`**: テーマ固有のドライバスクリプト＋出力データを同一フォルダに格納。テーマ固有の知見は `PROGRESS.md` に記録する。

### Key Libraries

- **ASE (`ase`)**: 結晶構造構築、各種計算コードへの入出力
- **NumPy/SciPy**: 数値計算・データ処理
- **Matplotlib**: グラフ作成

## Conventions

- コメント・ドキュメントは日本語で記述
- ドライバスクリプトはプロジェクトルートから実行する前提（`os.getcwd()` でルートを参照）
- テーマ固有の設定・Tips・進捗は各 `simulations/<テーマ>/PROGRESS.md` に記録する
