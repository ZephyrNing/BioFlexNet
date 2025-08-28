import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import sys
sys.path.append("/rds/general/user/zn324/home/Flexible-Neurons-main")
from src.plot.params import FIGURE_SIZE, FONTSIZE_TITLE, FONTSIZE_LABEL, FONTSIZE_TICK, PAD_TITLE, DPI_FIGURE

FIGURE_SIZE = (FIGURE_SIZE[0] * 0.8, FIGURE_SIZE[1])

def smooth_curve(data, weight=0.9, start_from=3, baseline=0.1):
    smoothed = []
    for i, point in enumerate(data):
        if i < start_from:
            smoothed.append(baseline * weight + point * (1 - weight))
        else:
            if i == start_from:
                last = point
            smoothed_val = last * weight + (1 - weight) * point
            smoothed.append(smoothed_val)
            last = smoothed_val
    return smoothed

def plot_training_curve_combined(
    file_paths,
    smooth=True,
    rolling_std_window=10,
    save_as=None,
    label_every_n_epochs=20,
    linewidth=4,
    total_epochs=180,
    keep_first_n_epoch=180,
    exclude_the_first_n_epoch=0,
    best_point_start_epoch=20,
):
    legend_size_adjustment_factor = 0.75
    num_steps_per_epoch = 39
    keep_first_n_row = num_steps_per_epoch * keep_first_n_epoch

    fig, axs = plt.subplots(1, 2, figsize=(FIGURE_SIZE[0]*2, FIGURE_SIZE[1]), sharex=False)
    plt.suptitle('MNIST Top-1 Performance', fontsize=24, fontweight='bold')

    color_map = {}
    best_acc_points = []
    best_loss_points = []

    for model, path in file_paths.items():
        data = pd.read_csv(path)

        data = data.iloc[exclude_the_first_n_epoch * num_steps_per_epoch:]
        data.reset_index(drop=True, inplace=True)
        data.fillna(0, inplace=True)

        def handle_outliers(data, lower_percentile=3, upper_percentile=97):
            float_columns = data.select_dtypes(include=[np.float64, np.float32]).columns
            for column in float_columns:
                lower_bound = np.percentile(data[column], lower_percentile)
                upper_bound = np.percentile(data[column], upper_percentile)
                data.loc[data[column] < lower_bound, column] = lower_bound
                data.loc[data[column] > upper_bound, column] = upper_bound
            return data

        data = handle_outliers(data)
        data = data.iloc[:keep_first_n_row]

        train_acc_raw = data["Train Accuracy"]
        valid_acc_raw = data["Valid Accuracy"]
        train_loss_raw = data["Train Loss"]
        valid_loss_raw = data["Valid Loss"]

        train_acc = pd.Series(smooth_curve(train_acc_raw, weight=0.9) if smooth else train_acc_raw)
        valid_acc = pd.Series(smooth_curve(valid_acc_raw, weight=0.998) if smooth else valid_acc_raw)
        train_loss = pd.Series(smooth_curve(train_loss_raw, weight=0.9) if smooth else train_loss_raw)
        valid_loss = pd.Series(smooth_curve(valid_loss_raw, weight=0.998) if smooth else valid_loss_raw)

        train_acc_std = train_acc_raw.rolling(window=rolling_std_window).std().fillna(0)
        valid_acc_std = valid_acc_raw.rolling(window=rolling_std_window).std().fillna(0)
        train_loss_std = train_loss_raw.rolling(window=rolling_std_window).std().fillna(0)
        valid_loss_std = valid_loss_raw.rolling(window=rolling_std_window).std().fillna(0)

        color = sns.color_palette("tab10")[list(file_paths.keys()).index(model)]
        color_map[model] = color

        epochs = data["Epoch"]

        sns.lineplot(ax=axs[0], x=epochs, y=train_acc, linestyle="-", linewidth=1.5, color=color)
        axs[0].fill_between(epochs, train_acc - train_acc_std, train_acc + train_acc_std, color=color, alpha=0.2)

        sns.lineplot(ax=axs[1], x=epochs, y=train_loss, linestyle="-", linewidth=1.5, color=color)
        axs[1].fill_between(epochs, train_loss - train_loss_std, train_loss + train_loss_std, color=color, alpha=0.2)

        axs[0].plot(epochs, valid_acc, linestyle="--", linewidth=1.5, color=color)
        axs[0].fill_between(epochs, valid_acc - valid_acc_std, valid_acc + valid_acc_std, color=color, alpha=0.2)

        axs[1].plot(epochs, valid_loss, linestyle="--", linewidth=1.5, color=color)
        axs[1].fill_between(epochs, valid_loss - valid_loss_std, valid_loss + valid_loss_std, color=color, alpha=0.2)

        mask = epochs >= best_point_start_epoch
        epochs_valid = epochs[mask].reset_index(drop=True)
        valid_acc_sub = valid_acc[mask]
        valid_loss_sub = valid_loss[mask]

        best_acc_idx = np.argmax(valid_acc_sub)
        best_acc_epoch = epochs_valid[best_acc_idx]
        best_acc_value = valid_acc_sub.iloc[best_acc_idx]
        best_acc_points.append((best_acc_epoch, best_acc_value, color))

        best_loss_idx = np.argmin(valid_loss_sub)
        best_loss_epoch = epochs_valid[best_loss_idx]
        best_loss_value = valid_loss_sub.iloc[best_loss_idx]
        best_loss_points.append((best_loss_epoch, best_loss_value, color))

    offset_unit_acc = (axs[0].get_ylim()[1] - axs[0].get_ylim()[0]) * 0.04
    for i, (epoch, value, color) in enumerate(best_acc_points):
        axs[0].axvline(x=epoch, linestyle=":", linewidth=1, color=color, alpha=0.7)
        axs[0].scatter(epoch, value, color=color, s=40, marker='o')

    offset_unit_loss = (axs[1].get_ylim()[1] - axs[1].get_ylim()[0]) * 0.07
    for i, (epoch, value, color) in enumerate(best_loss_points):
        axs[1].axvline(x=epoch, linestyle=":", linewidth=1, color=color, alpha=0.7)
        axs[1].scatter(epoch, value, color=color, s=40, marker='o')

    axs[0].set_title("Accuracy Curves", fontsize=FONTSIZE_TITLE, pad=PAD_TITLE)
    axs[0].set_xlabel("Epochs", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)
    axs[0].set_ylabel("Accuracy", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)
    axs[0].set_ylim(0, 1.0)

    axs[1].set_title("Loss Curves", fontsize=FONTSIZE_TITLE, pad=PAD_TITLE)
    axs[1].set_xlabel("Epochs", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)
    axs[1].set_ylabel("Loss", fontsize=FONTSIZE_LABEL, labelpad=PAD_TITLE)

    for ax in axs:
        legend_lines = []
        legend_labels = []
        legend_lines.append(plt.Line2D([0], [0], color="black", linestyle="-", linewidth=4))
        legend_labels.append("Train")
        legend_lines.append(plt.Line2D([0], [0], color="black", linestyle="--", linewidth=4))
        legend_labels.append("Validation")
        ax.legend(
            legend_lines,
            legend_labels,
            loc="center right",
            fontsize=FONTSIZE_LABEL * legend_size_adjustment_factor,
        )

    model_lines = [plt.Line2D([0], [0], color=color_map[model], linestyle="-", linewidth=4) for model in file_paths.keys()]
    model_legend = fig.legend(
        model_lines,
        file_paths.keys(),
        loc="center right",
        bbox_to_anchor=(1.25, 0.5),
        fontsize=FONTSIZE_LABEL * legend_size_adjustment_factor,
    )

    plt.tight_layout()
    plt.savefig(save_as, dpi=DPI_FIGURE, bbox_inches="tight")

    save_as_pdf = str(save_as).replace('.png', '.pdf')
    plt.savefig(save_as_pdf, dpi=DPI_FIGURE, bbox_inches="tight", format='pdf')

    plt.show()

if __name__ == "__main__":
    save_dir = Path("figures") / "training_curve"
    save_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data_csv/mnist_training_curve")

    with open(data_dir / "registry.json", "r") as f:
        registry = json.load(f)

    file_paths = {registry[key]: str(data_dir / f"{key}.csv") for key in registry}

    plot_training_curve_combined(
        file_paths=file_paths,
        smooth=True,
        rolling_std_window=12,
        save_as=save_dir / "mnist_top_accuracy.png",
    )
