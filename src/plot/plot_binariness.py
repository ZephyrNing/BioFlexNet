import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib import Path
import re
import sys

# Import shared plotting params from project
sys.path.append("/rds/general/user/zn324/home/Flexible-Neurons-main")
from src.plot.params import FIGURE_SIZE, FONTSIZE_TITLE, FONTSIZE_LABEL, FONTSIZE_TICK, PAD_TITLE, DPI_FIGURE

FIGURE_SIZE_TRIPLE = (FIGURE_SIZE[0] * 2.4, FIGURE_SIZE[1] * 0.85)
FIGURE_SIZE_MEAN   = (FIGURE_SIZE[0] * 1.2, FIGURE_SIZE[1] * 0.9)

def smooth_curve(data, weight=0.9):
    s = pd.Series(data).astype(float).ffill().bfill()
    arr = s.values
    if len(arr) == 0:
        return []
    out = []
    last = 0.0
    for x in arr:
        last = last * weight + (1 - weight) * x
        out.append(last)
    return out

def _find_binariness_cols(df):
    pat = re.compile(r"^Binariness\s+\d+$")
    return [c for c in df.columns if pat.match(c)]

def _prepare_df(df, num_steps_per_epoch=39, keep_first_n_epoch=180, exclude_the_first_n_epoch=0):
    df = df.iloc[exclude_the_first_n_epoch * num_steps_per_epoch:].copy()
    df.reset_index(drop=True, inplace=True)
    df.fillna(0, inplace=True)
    keep_first_n_row = num_steps_per_epoch * keep_first_n_epoch
    return df.iloc[:keep_first_n_row]

def plot_binariness_three_in_one(
    file_paths,                
    smooth=True,
    rolling_std_window=10,
    linewidth_mean=3.0,
    linewidth_layer=1.0,
    alpha_layer=0.85,
    alpha_fill=0.18,
    num_steps_per_epoch=39,
    keep_first_n_epoch=180,
    exclude_the_first_n_epoch=0,
    save_path=Path("figures") / "binariness" / "img_binariness_three_in_one.png",
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    models = list(file_paths.keys())
    n = len(models)
    fig, axes = plt.subplots(1, n, figsize=FIGURE_SIZE_TRIPLE, sharey=True)
    plt.suptitle('ImageNet-100 Binariness', fontsize=24, fontweight='bold')
    if n == 1:
        axes = [axes]

    handles_global, labels_global = [], []
    mean_color = sns.color_palette("tab10")[0]

    for idx, (model, path) in enumerate(file_paths.items()):
        ax = axes[idx]
        df = pd.read_csv(path)
        df = _prepare_df(df, num_steps_per_epoch, keep_first_n_epoch, exclude_the_first_n_epoch)

        bin_cols = _find_binariness_cols(df)
        if not bin_cols:
            print(f"[!] {model}: No 'Binariness i' columns found, skip.")
            continue

        for c in bin_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").ffill().bfill()

        epochs = pd.to_numeric(df["Epoch"], errors="coerce").ffill().bfill()
        mean_bin = df[bin_cols].mean(axis=1).astype(float)
        std_bin  = df[bin_cols].std(axis=1).astype(float)

        if smooth:
            mean_bin = pd.Series(smooth_curve(mean_bin.values))
            std_band = pd.Series(
                smooth_curve(std_bin.rolling(rolling_std_window).mean().ffill().bfill().values, weight=0.9)
            )
        else:
            std_band = std_bin.rolling(rolling_std_window).mean().ffill().bfill()

        n_layers = len(bin_cols)
        layer_palette = sns.color_palette("tab20", min(n_layers, 20))
        colors = [layer_palette[i % len(layer_palette)] for i in range(n_layers)]

        for i, c in enumerate(bin_cols):
            y = df[c].values
            if smooth:
                y = smooth_curve(y)
            line, = ax.plot(epochs, y, lw=linewidth_layer, color=colors[i], alpha=alpha_layer, label=c)
            if idx == 0:
                handles_global.append(line)
                labels_global.append(c)

        mean_line, = ax.plot(epochs, mean_bin, lw=linewidth_mean, color=mean_color, label="Mean Binariness", zorder=5)
        ax.fill_between(
            epochs,
            np.clip(mean_bin - std_band, 0, 1),
            np.clip(mean_bin + std_band, 0, 1),
            color=mean_color, alpha=alpha_fill, label="Mean ± SD"
        )
        if idx == 0:
            handles_global.append(mean_line)
            labels_global.append("Mean Binariness")

        ax.set_title(model, fontsize=FONTSIZE_TITLE, pad=PAD_TITLE)
        ax.set_xlabel("Epochs", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)
        if idx == 0:
            ax.set_ylabel("Binariness", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)
        ax.set_ylim(0.0, 0.5)
        ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)

    fig.legend(
        handles_global, labels_global,
        loc="center right",
        bbox_to_anchor=(0.97, 0.5),
        fontsize=FONTSIZE_LABEL * 0.7,
        ncol=1,
        frameon=True,
        framealpha=0.95,
        title="Layers",
        title_fontsize=FONTSIZE_LABEL * 0.75,
    )

    plt.tight_layout()
    plt.subplots_adjust(right=0.86)
    plt.savefig(save_path, dpi=DPI_FIGURE, bbox_inches="tight")
    plt.savefig(str(save_path).replace(".png", ".pdf"), dpi=DPI_FIGURE, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"[✓] Saved: {save_path}")

def plot_mean_binariness_multi_models(
    file_paths,                
    smooth=True,
    linewidth=3.0,
    num_steps_per_epoch=39,
    keep_first_n_epoch=180,
    exclude_the_first_n_epoch=0,
    save_path=Path("figures") / "binariness" / "mean_binariness_multi_models.png",
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=FIGURE_SIZE_MEAN)
    palette = sns.color_palette("tab10", len(file_paths))

    for idx, (model, path) in enumerate(file_paths.items()):
        df = pd.read_csv(path)
        df = _prepare_df(df, num_steps_per_epoch, keep_first_n_epoch, exclude_the_first_n_epoch)

        bin_cols = _find_binariness_cols(df)
        if not bin_cols:
            print(f"[!] {model}: No 'Binariness i' columns found, skip.")
            continue

        for c in bin_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").ffill().bfill()

        epochs = pd.to_numeric(df["Epoch"], errors="coerce").ffill().bfill()
        mean_bin = df[bin_cols].mean(axis=1).astype(float)
        if smooth:
            mean_bin = pd.Series(smooth_curve(mean_bin.values))

        ax.plot(epochs, mean_bin, lw=linewidth, color=palette[idx], label=model)

    ax.set_title("Models • Mean Binariness", fontsize=FONTSIZE_TITLE, pad=PAD_TITLE)
    ax.set_xlabel("Epochs", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)
    ax.set_ylabel("Mean Binariness", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)
    ax.set_ylim(0.0, 0.5)
    ax.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    ax.legend(loc="best", fontsize=FONTSIZE_LABEL * 0.75)

    plt.tight_layout()
    plt.savefig(save_path, dpi=DPI_FIGURE, bbox_inches="tight")
    plt.savefig(str(save_path).replace(".png", ".pdf"), dpi=DPI_FIGURE, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"[✓] Saved: {save_path}")

if __name__ == "__main__":
    data_dir = Path("data_csv/cifar10_training_curve_adjusted")
    with open(data_dir / "flex_registry.json", "r") as f:
        registry = json.load(f)
    file_paths_all = {registry[k]: str(data_dir / f"{k}.csv") for k in registry}
    file_paths_3 = dict(list(file_paths_all.items())[:3])

    plot_binariness_three_in_one(
        file_paths=file_paths_3,
        smooth=True,
        rolling_std_window=12,
        num_steps_per_epoch=39,
        keep_first_n_epoch=180,
        exclude_the_first_n_epoch=0,
        save_path=Path("figures") / "binariness" / "img_binariness_3in1.pdf",
    )

    plot_mean_binariness_multi_models(
        file_paths=file_paths_3,
        smooth=True,
        num_steps_per_epoch=39,
        keep_first_n_epoch=180,
        exclude_the_first_n_epoch=0,
        save_path=Path("figures") / "binariness" / "img_binariness.pdf",
    )
