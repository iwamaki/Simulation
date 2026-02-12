# 単結晶変形試験 (single_crystal_tensile)

## 概要
FCC単結晶の一軸引張/圧縮試験。
`ase.build.bulk()` + `ase.build.cut()` で完全な周期結晶を構築し、
`fix deform z` + `fix npt x,y` で変形試験を行う。

## シミュレーションフロー
1. **Phase 1**: 0K構造緩和（`box/relax aniso` で全方向最適化）
2. **Phase 2**: NPT平衡化（5ps, 完全結晶は2-3psで熱平衡に達する）
   - (100): `aniso`（直方体）
   - (110)/(111): `tri`（三斜晶系 — せん断自由度を解放）
3. **Phase 3**: 変形試験（`fix deform z erate` + `fix npt x,y` ゼロ圧力）
   - (110)/(111): `change_box all triclinic` で三斜晶系に変換し、xy/xz/yz tilt も解放

## 欠陥解析用の出力
dumpファイルに以下を含む（Ovitoで直接可視化可能）:
- **CNA** (`c_cna`): 1=FCC, 2=HCP(積層欠陥), 3=BCC, 4=ICO, 5=OTHER(転位コア等)
- **Centro-symmetry** (`c_csym`): 0=完全結晶, ~10-15=積層欠陥, ~25+=表面/空孔
- **Per-atom stress** (`c_peratom[1-6]`): 応力場の可視化（単位: bar*Å³、実応力にはボロノイ体積で割る必要あり）

### Ovitoでの可視化Tips
- CNA値でカラーリング → FCC(青)/HCP(赤)/OTHER(白) で転位・積層欠陥が明瞭
- `Select Type` → CNA=1(FCC) を選択 → `Delete Selected` で欠陥原子のみ表示
- centro-symmetry > 閾値 で欠陥抽出も有効

## 応力・ひずみの定義
- ひずみ: ε = (Lz - Lz₀) / Lz₀（工学ひずみ）
- 応力: σ = -Pzz / 10000 [GPa]（Cauchy応力, σ = -P）

## 使い方
```bash
# デフォルト: Cu(100) 引張 300K
python simulations/single_crystal_tensile/single_crystal_tensile.py

# 方位依存性
python simulations/single_crystal_tensile/single_crystal_tensile.py --miller 1 1 1
python simulations/single_crystal_tensile/single_crystal_tensile.py --miller 1 1 0

# 圧縮
python simulations/single_crystal_tensile/single_crystal_tensile.py --mode compression --no-halt

# 大きめの系（転位観察には100Å以上推奨、理想は150-200Å）
python simulations/single_crystal_tensile/single_crystal_tensile.py --target-size 100
```

## 設計メモ
- 結晶構築: `ase.build.bulk(cubic=True)` + `ase.build.cut()` で完全周期結晶
  - slab builder (fcc100等) はz方向の周期性が壊れるため使用不可
- **三斜晶系対応**: (110)/(111) では `change_box all triclinic` + tilt 解放で、すべり系活動に伴うせん断変形を許容。(100) は高対称性のため直方体で十分
- CNA cutoff = 0.854 × 格子定数（1NNと2NNの中間）
  - 大ひずみ域ではカットオフが初期値固定のため誤分類の可能性あり。高精度解析には OVITO の Adaptive CNA を推奨
- 平衡化5ps: 完全結晶は熱化が速い。欠陥入り構造には `--eq-time` で延長可
- 乱数シード: `hashlib.md5` ベース（PYTHONHASHSEED に依存せず再現性あり）

## 既知の制限事項
- **ひずみ速度**: 0.001 /ps = 10⁹ /s（実験より12桁高い）。速度感度テスト（`--erate 0.0005` 等）を推奨
- **システムサイズ**: デフォルト 40 Å は動作テスト用。転位核生成・すべりの観察には 100 Å 以上が必要（周期像相互作用の回避）
- **stress/atom**: LAMMPS出力は bar*Å³（応力×体積）。実応力への変換にはボロノイ体積割りが必要

## テスト結果 (2026-02-12)
小モデル（20 Å, 10%ひずみ）で動作検証:
- **Cu(100)**: 864原子, ~7.9 GPa@10% — 完全結晶のため降伏なし（正常）
- **Cu(111)**: 576原子, triclinic, ~11.6 GPa@10% — E₁₁₁ > E₁₀₀ の弾性異方性を正しく再現
