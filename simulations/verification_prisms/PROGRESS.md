# PRISMS-Plasticity 検証プロジェクト

## 目的
MDシミュレーション (`single_crystal_tensile.py`) で得られた単結晶金属の応力ひずみ応答を、結晶塑性有限要素法 (CP-FEM) コードである **PRISMS-Plasticity** を用いて検証・比較する。

主な着眼点:
1.  弾性領域の立ち上がり (ヤング率) の整合性
2.  降伏点 (Yield Stress) の整合性
3.  加工硬化挙動 (Hardening) の定性的な一致

## 実装状態

### 完了
- [x] `scripts/extract_md_params.py` — MDデータからパラメータ抽出（E, σ_y, τ₀, Schmid因子, Rodrigues）
- [x] `scripts/plot_comparison.py` — MD vs CP-FEM 比較プロット（VTU対応）
- [x] `run_verification.py` — 5ステップパイプライン実装済み
- [x] テンプレート群（`prm.prm.in`, `BCinfo.txt.in`, `orientations.txt.in`, 静的ファイル）
- [x] `docker/Dockerfile.prisms` — deal.II v9.5.2ベース、PRISMS-Plasticityビルド
- [x] パラメータ抽出の動作確認（Cu(100)既存データ）
- [x] 入力ファイル生成の動作確認（`--generate-only`）

### 未完了
- [ ] Dockerイメージのビルド・動作確認
- [ ] PRISMS-Plasticity実行テスト
- [ ] MD vs CP-FEM 比較プロット生成

## パイプライン使い方

```bash
# Step 1: パラメータ抽出 + 入力生成のみ（Docker不要）
python simulations/verification_prisms/run_verification.py --skip-md \
  --md-result simulations/single_crystal_tensile/Cu100_tension_300K_gpu_test/stress_strain.txt \
  --generate-only

# Step 2: 全パイプライン実行（Docker必要）
python simulations/verification_prisms/run_verification.py --skip-md \
  --md-result simulations/single_crystal_tensile/Cu100_tension_300K_gpu_test/stress_strain.txt

# Step 3: MD実行から全自動
python simulations/verification_prisms/run_verification.py
```

## テスト結果 (Cu(100) 300K)

```
ヤング率 E = 78.0 GPa (R² = 0.9999)
降伏応力 σ_y = 7.677 GPa (ε_y = 0.1005)
臨界分解せん断応力 τ₀ = 3.134 GPa (Schmid因子 = 0.4082)
Rodrigues-Frank = [0, 0, 0]  ← (100)は恒等回転
```

**注意**: MDの降伏応力（7.7 GPa）はバルク実験値（〜数十MPa）より遥かに高い。これはMDの高ひずみ速度（〜10⁸/s）と完全結晶（転位フリー）に起因する。CP-FEMとの比較では、MDから抽出したτ₀を直接使用するため、弾性域の整合性が主な検証対象となる。

## システム構成

```
simulations/verification_prisms/
├── run_verification.py         # パイプライン統括スクリプト
├── scripts/
│   ├── extract_md_params.py    # MD結果からパラメータ抽出
│   └── plot_comparison.py      # 結果比較プロット
├── templates/
│   ├── prm.prm.in              # PRISMSメインパラメータ（フラットset形式）
│   ├── BCinfo.txt.in           # 境界条件（z軸引張）
│   ├── orientations.txt.in     # 結晶方位（Rodrigues-Frank）
│   ├── grainID.txt             # 8×8×8ボクセル単結晶
│   ├── slipNormals.txt         # FCC {111} 法線 12系
│   └── slipDirections.txt      # FCC <110> 方向 12系
├── docker/
│   └── Dockerfile.prisms       # PRISMS-Plasticity実行環境
└── output/
    └── prisms_input/           # 生成された入力ファイル
```

## 技術メモ

### 単位変換
- MD出力: GPa → PRISMS入力: MPa（×1000）
- 弾性定数は文献値（Cu: C11=168.4, C12=121.4, C44=75.4 GPa）

### Voce硬化パラメータ（経験則）
- τ_s = 1.5 × τ₀（飽和すべり抵抗）
- h₀ = 10 × τ₀（初期硬化係数）
- n = 20（べき乗則指数）
- 潜在硬化比 = 1.4

### PRISMS-Plasticity テンプレート形式
旧ファイル（`simulation.prm.in`）はdeal.IIの`subsection`形式で書かれていたが、PRISMSはフラットな`set key = value`形式を使用する。`prm.prm.in`に置き換え済み。
