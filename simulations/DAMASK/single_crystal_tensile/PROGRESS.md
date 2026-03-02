# CPFEM (結晶塑性有限要素法) - DAMASK

## 環境情報
- **DAMASK**: 3.0.2 (PPA版: `ppa:damask-multiphysics/ppa`)
- **ソルバー**: `DAMASK_grid`（スペクトル法）、`DAMASK_mesh`（FEM）
- **Python**: `damask` パッケージ 3.0.2
- **セットアップ日**: 2026-02-24

## セットアップ手順（再現用）
```bash
sudo add-apt-repository ppa:damask-multiphysics/ppa
sudo apt-get update
sudo apt-get install -y damask
pip install damask  # venv内で使う場合
```

## 進捗
- [x] DAMASK インストール完了
- [x] DAMASK_grid / DAMASK_mesh 動作確認
- [x] Python damask モジュール動作確認
- [x] 単結晶引張テスト問題のドライバ完成（`single_crystal_tensile.py`）
- [x] Cu [100], [111] でテスト実行・応力ひずみCSV出力確認
- [ ] 実際の解析テーマの開始

## 知見・Tips
- PPA版はPETSc, HDF5, FFTW等の依存関係を自動解決してくれる
- システムPythonに `python3-damask` が入るが、venv内では `pip install damask` が必要
- DAMASK_grid は `--geom`, `--load`, `--material` の3フラグが必須
- FCC {111}<110> 滑り系の `h_sl-sl` 相互作用パラメータは7個必要（cF lattice）
- `damask.Result.get()` は `flatten=True`（デフォルト）で直接 ndarray を返す（dict ではない）
- `add_stress_Cauchy()` / `add_strain()` はHDF5に書き込むため、2回呼ぶとエラー
- 弾性率の検証: インクリメント数を十分に取らないと第1ステップで降伏してしまう（CRSSが低い場合）
