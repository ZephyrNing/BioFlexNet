import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from pathlib import Path
import re
import sys
from scipy.stats import pearsonr
import matplotlib.gridspec as gridspec

# Import plotting params from project
sys.path.append("/rds/general/user/zn324/home/Flexible-Neurons-main")
from src.plot.params import FIGURE_SIZE, FONTSIZE_TITLE, FONTSIZE_LABEL, FONTSIZE_TICK, PAD_TITLE, DPI_FIGURE

FIGURE_SIZE_TRIPLE = (FIGURE_SIZE[0] * 2.6, FIGURE_SIZE[1] * 0.9)

def smooth_curve(data, weight=0.98):
    s = pd.Series(data).astype(float).ffill().bfill()
    arr = s.values
    if len(arr) == 0:
        return []
    out = []
    last = arr[0]
    for x in arr:
        last = last * weight + (1 - weight) * x
        out.append(last)
    return out

def _find_binariness_cols(df):
    pat = re.compile(r"^Binariness\s+\d+$")
    return [c for c in df.columns if pat.match(c)]

def _find_conv_ratio_cols(df):
    pat = re.compile(r"^Conv\s*Ratio\s+\d+$")
    return [c for c in df.columns if pat.match(c)]

def _prepare_df(df, num_steps_per_epoch=39, keep_first_n_epoch=180, exclude_the_first_n_epoch=0):
    df = df.iloc[exclude_the_first_n_epoch * num_steps_per_epoch:].copy()
    df.reset_index(drop=True, inplace=True)
    df.fillna(0, inplace=True)
    keep_first_n_row = num_steps_per_epoch * keep_first_n_epoch
    return df.iloc[:keep_first_n_row]

def plot_conv_binariness_dashboard_one_model(
    model_name: str,
    csv_path: str,
    smooth=True,
    layer_roll_window=15,
    rolling_std_window=16,
    linewidth_mean=3.0,
    linewidth_layer=1.4,
    alpha_layer=0.9,
    alpha_fill_layer=0.15,
    alpha_fill=0.18,
    degree_fit=2,
    show_fit=False,
    show_mean_binariness=True,
    num_steps_per_epoch=39,
    keep_first_n_epoch=180,
    exclude_the_first_n_epoch=0,
    y_limit=(0.0, 0.5),
    legend_bboxt_anchor=(0.96, 0.5),
    save_path=Path("figures") / "binariness" / "cifar_conv_ratio_dashboard.png",
    gap12_cols=0.8,
    gap23_cols=0.8
):
    save_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = _prepare_df(df, num_steps_per_epoch, keep_first_n_epoch, exclude_the_first_n_epoch)

    conv_cols = _find_conv_ratio_cols(df)
    bin_cols  = _find_binariness_cols(df)
    if not conv_cols:
        print(f"[!] {model_name}: No 'Conv Ratio i' columns found.")
        return
    if not bin_cols:
        print(f"[!] {model_name}: No 'Binariness i' columns found.")
        return

    for c in conv_cols + bin_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").ffill().bfill()

    epochs    = pd.to_numeric(df["Epoch"], errors="coerce").ffill().bfill()
    mean_conv = df[conv_cols].mean(axis=1).astype(float)
    std_conv  = df[conv_cols].std(axis=1).astype(float)
    mean_bin  = df[bin_cols].mean(axis=1).astype(float)

    if smooth:
        mean_conv_plot = pd.Series(smooth_curve(mean_conv.values, weight=0.98))
        mean_bin_plot  = pd.Series(smooth_curve(mean_bin.values,  weight=0.98)) if show_mean_binariness else mean_bin
        std_band = std_conv.rolling(rolling_std_window).mean().ffill().bfill()
        std_band = pd.Series(smooth_curve(std_band.values, weight=0.9))
    else:
        mean_conv_plot = mean_conv
        mean_bin_plot  = mean_bin
        std_band = std_conv.rolling(rolling_std_window).mean().ffill().bfill()

    fig = plt.figure(figsize=FIGURE_SIZE_TRIPLE)
    width_ratios = [1, 1, gap12_cols, 1, 1, gap23_cols, 1, 1]
    gs = gridspec.GridSpec(
        1, 8, figure=fig, width_ratios=width_ratios, wspace=0.05
    )
    ax_l = fig.add_subplot(gs[0, 0:2])
    ax_c = fig.add_subplot(gs[0, 3:5])
    ax_r = fig.add_subplot(gs[0, 6:8])

    n_layers = len(conv_cols)
    layer_palette = sns.color_palette("tab20", min(n_layers, 20))
    colors = [layer_palette[i % len(layer_palette)] for i in range(n_layers)]
    layer_handles, layer_labels = [], []

    for i, c in enumerate(conv_cols):
        y_raw = df[c].astype(float)
        roll_mean = y_raw.rolling(layer_roll_window).mean().ffill().bfill()
        roll_std  = y_raw.rolling(layer_roll_window).std().ffill().bfill().fillna(0)

        if smooth:
            y_line = pd.Series(smooth_curve(roll_mean.values, weight=0.98))
        else:
            y_line = roll_mean

        ax_l.fill_between(
            epochs,
            np.clip(roll_mean - roll_std, y_limit[0], y_limit[1]),
            np.clip(roll_mean + roll_std, y_limit[0], y_limit[1]),
            color=colors[i], alpha=alpha_fill_layer
        )
        line, = ax_l.plot(epochs, y_line, lw=linewidth_layer, color=colors[i], alpha=alpha_layer, label=c)

        if i < len(layer_palette):
            layer_handles.append(line)
            layer_labels.append(c)

    ax_l.set_title("Conv Ratio per Layer", fontsize=FONTSIZE_TITLE, pad=PAD_TITLE)
    ax_l.set_xlabel("Epochs", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)
    ax_l.set_ylabel("Conv Ratio", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)
    ax_l.set_ylim(*y_limit)
    ax_l.tick_params(axis="both", labelsize=FONTSIZE_TICK)

    mean_color = sns.color_palette("tab10")[0]
    mline, = ax_c.plot(epochs, mean_conv_plot, lw=linewidth_mean, color=mean_color, label="Mean Conv Ratio", zorder=5)
    ax_c.fill_between(
        epochs,
        np.clip(mean_conv_plot - std_band, y_limit[0], y_limit[1]),
        np.clip(mean_conv_plot + std_band, y_limit[0], y_limit[1]),
        color=mean_color, alpha=alpha_fill, label="Mean ± SD"
    )

    center_handles, center_labels = ax_c.get_legend_handles_labels()

    if show_fit:
        x = np.array(epochs, dtype=float)
        y = np.array(mean_conv_plot, dtype=float)
        coeffs = np.polyfit(x, y, deg=degree_fit)
        poly   = np.poly1d(coeffs)
        x_fit  = np.linspace(x.min(), x.max(), 300)
        y_fit  = poly(x_fit)
        fit_line, = ax_c.plot(x_fit, y_fit, lw=2.0, ls="--", color=mean_color, alpha=0.9, label=f"Poly Fit (deg={degree_fit})")
        center_handles.append(fit_line)
        center_labels.append(f"Poly Fit (deg={degree_fit})")
        eq = " + ".join([f"{coeffs[i]:.3g}·x^{degree_fit - i}" for i in range(degree_fit)]) + f" + {coeffs[-1]:.3g}"
        ax_c.text(0.02, 0.95, f"Fit: y = {eq}", transform=ax_c.transAxes,
                  fontsize=FONTSIZE_TICK*0.9, va="top")

    if show_mean_binariness:
        ax_c2 = ax_c.twinx()
        bin_color = sns.color_palette("tab10")[2]
        bline, = ax_c2.plot(epochs, mean_bin_plot, lw=2.2, color=bin_color, label="Mean Binariness")
        ax_c2.set_ylim(*y_limit)
        ax_c2.tick_params(axis="y", labelsize=FONTSIZE_TICK)

        h2, l2 = ax_c2.get_legend_handles_labels()
        center_handles += h2
        center_labels  += l2

    ax_c.set_title("Global Trend", fontsize=FONTSIZE_TITLE, pad=PAD_TITLE)
    ax_c.set_xlabel("Epochs", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)
    ax_c.set_ylabel("Conv Ratio", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)
    ax_c.set_ylim(*y_limit)
    ax_c.tick_params(axis="both", labelsize=FONTSIZE_TICK)

    conv_stack = df[conv_cols].values.flatten()
    bin_stack  = df[bin_cols].values.flatten()
    mask = np.isfinite(conv_stack) & np.isfinite(bin_stack)
    conv_stack = conv_stack[mask]
    bin_stack  = bin_stack[mask]

    if len(conv_stack) > 3:
        r, p = pearsonr(conv_stack, bin_stack)
        r_text = f"Pearson r = {r:.3f}"
    else:
        r_text = "Pearson r = N/A"

    qx1, qx2 = np.quantile(conv_stack, [0.01, 0.99])
    qy1, qy2 = np.quantile(bin_stack,  [0.01, 0.99])
    pad_x = max(0.01, 0.05 * (qx2 - qx1))
    pad_y = max(0.01, 0.05 * (qy2 - qy1))
    xlim = (max(0.0, qx1 - pad_x), min(1.0, qx2 + pad_x))
    ylim = (max(0.0, qy1 - pad_y), min(1.0, qy2 + pad_y))

    ax_r.set_title("Binariness vs Conv Ratio", fontsize=FONTSIZE_TITLE, pad=PAD_TITLE)
    ax_r.set_xlabel("Conv Ratio", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)
    ax_r.set_ylabel("Binariness", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)

    if len(conv_stack) < 5000:
        ax_r.scatter(conv_stack, bin_stack, s=6, alpha=0.25)
    else:
        hb = ax_r.hexbin(conv_stack, bin_stack, gridsize=70, bins='log', cmap="mako")
        cb = fig.colorbar(hb, ax=ax_r, fraction=0.046, pad=0.04)
        cb.set_label("log(count)", fontsize=FONTSIZE_LABEL*0.8)

    ax_r.set_xlim(*xlim)
    ax_r.set_ylim(*ylim)
    ax_r.tick_params(axis="both", labelsize=FONTSIZE_TICK)
    ax_r.text(0.98, 0.02, r_text, transform=ax_r.transAxes,
              fontsize=FONTSIZE_TICK*0.9, ha="right", va="bottom")
    ax_r.axvline(np.median(conv_stack), ls=":", lw=1, color="gray", alpha=0.6)
    ax_r.axhline(np.median(bin_stack),  ls=":", lw=1, color="gray", alpha=0.6)

    legend_handles = layer_handles + center_handles
    legend_labels  = layer_labels  + center_labels

    fig.legend(
        legend_handles, legend_labels,
        loc="center right",
        bbox_to_anchor=legend_bboxt_anchor,
        fontsize=FONTSIZE_LABEL * 0.7,
        ncol=1,
        frameon=True,
        framealpha=0.95,
        title="Layers & Trends",
        title_fontsize=FONTSIZE_LABEL * 0.75,
    )

    fig.suptitle(f"{model_name} Conv Ratio–Binariness Evolution and Correlation", fontsize=24, y=0.99, fontweight='bold')
    fig.tight_layout(rect=[0.0, 0.0, 0.86, 0.99])

    plt.savefig(save_path, dpi=DPI_FIGURE, bbox_inches="tight")
    plt.savefig(str(save_path).replace(".png", ".pdf"), dpi=DPI_FIGURE, bbox_inches="tight", format="pdf")
    plt.close()
    print(f"[✓] Saved: {save_path}")

if __name__ == "__main__":
    data_dir = Path("data_csv/cifar10_training_curve_adjusted")
    with open(data_dir / "flex_registry.json", "r") as f:
        registry = json.load(f)

    key0 = list(registry.keys())[2]
    model_name = registry[key0]
    csv_path   = str(data_dir / f"{key0}.csv")

    plot_conv_binariness_dashboard_one_model(
        model_name=model_name,
        csv_path=csv_path,
        smooth=True,
        layer_roll_window=15,
        rolling_std_window=15,
        linewidth_mean=3.0,
        linewidth_layer=1.4,
        alpha_layer=0.9,
        alpha_fill_layer=0.15,
        alpha_fill=0.18,
        degree_fit=2,
        show_fit=False,
        show_mean_binariness=True,
        y_limit=(0.0, 1.0),
        legend_bboxt_anchor=(1.02, 0.5),
        num_steps_per_epoch=39,
        keep_first_n_epoch=180,
        exclude_the_first_n_epoch=0,
        save_path=Path("figures") / "binariness" / f"{model_name}_conv_ratio_dashboard.png",
        gap12_cols=0.45,
        gap23_cols=0.6,
    )
