"""
GSF曲線のプロット

gsf_calculate.py が出力した gsf_results.csv を読み込み、
GSF曲線を描画して gsf_curves.png に保存する。

使い方:
  python simulations/GSF_Calculate/plot_gsf.py
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = os.getcwd()
DATA_DIR = os.path.join("simulations", "GSF_Calculate", "data")

COLORS = {'Cu': '#E74C3C', 'Ag': '#95A5A6', 'Au': '#F1C40F'}


def load_csv(csv_path):
    """CSVファイルを読み込み、{element: (disp_frac, gsf)} の辞書を返す。"""
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        elements = header[1:]
        data = {elem: [] for elem in elements}
        disp_frac = []
        for row in reader:
            disp_frac.append(float(row[0]))
            for i, elem in enumerate(elements):
                data[elem].append(float(row[i + 1]))

    disp_frac = np.array(disp_frac)
    results = {}
    for elem in elements:
        gsf = np.array(data[elem])
        results[elem] = (disp_frac, gsf)
    return results


def plot_gsf(results, out_path):
    """GSF曲線をプロットして保存。"""
    fig, ax = plt.subplots(figsize=(8, 5))

    for element, (disp_frac, gsf) in results.items():
        gamma_us = np.max(gsf)
        gamma_isf = gsf[-1]
        color = COLORS.get(element, None)
        label = (f"{element}  "
                 f"$\\gamma_{{us}}$={gamma_us:.0f}, "
                 f"$\\gamma_{{isf}}$={gamma_isf:.0f} mJ/m$^2$")
        ax.plot(disp_frac, gsf, '-o', color=color,
                markersize=3, linewidth=2, label=label)

    ax.set_xlabel("Displacement / partial Burgers vector", fontsize=12)
    ax.set_ylabel("GSF Energy (mJ/m$^2$)", fontsize=12)
    ax.set_title("{111}<112> Generalized Stacking Fault Energy", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.axhline(y=0, color='black', linewidth=0.5)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved: {out_path}")
    plt.close()


def main():
    csv_path = os.path.join(PROJECT_ROOT, DATA_DIR, "gsf_results.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} が見つかりません。")
        print("先に gsf_calculate.py を実行してください。")
        return

    results = load_csv(csv_path)
    out_path = os.path.join(PROJECT_ROOT, DATA_DIR, "gsf_curves.png")
    plot_gsf(results, out_path)


if __name__ == '__main__':
    main()
