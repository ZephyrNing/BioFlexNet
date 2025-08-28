import os
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Config
checkpoints_dir = './checkpoints/mnist'
attack_types = ['FGSM', 'PGD', 'SPGD']
model_prefix = 'mnist_VGG6_'
output_path = './attack_plots/mnist_attack_plot.png'

model_display_names = [
    'ANN_FLEX', 
    'ANN', 
    'SNN_FLEX', 
    'SNN', 
    'SNN_BP_FLEX', 
    'SNN_BP'
]

os.makedirs(os.path.dirname(output_path), exist_ok=True)

sns.set(style="whitegrid", context="talk", palette="tab10")
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['font.family'] = 'sans-serif'

def load_attack_data(model_dir, attack_type):
    """Load accuracy and AUC data for one model and one attack"""
    attack_dir = os.path.join(model_dir, 'results', attack_type)
    accuracy_file = os.path.join(attack_dir, f'{attack_type}_accuracies.json')
    auc_file = os.path.join(attack_dir, f'{attack_type}_auc.json')

    if not os.path.exists(accuracy_file) or not os.path.exists(auc_file):
        print(f"Warning: Missing files for {attack_type} in {model_dir}")
        return None, None

    with open(accuracy_file, 'r') as f:
        accuracy_data = json.load(f)
    with open(auc_file, 'r') as f:
        auc_data = json.load(f)

    epsilons = accuracy_data['epsilons']
    accuracies = accuracy_data['accuracies']
    auc_key = f"{attack_type} Attack Area Under Curve Top 1"
    auc_score = auc_data[auc_key]

    return epsilons, accuracies, auc_score

def plot_combined_attack(models_data, model_names, display_names):
    """Plot combined accuracy and AUC figures"""
    n_attacks = len(attack_types)
    fig, axes = plt.subplots(2, n_attacks, figsize=(7 * n_attacks, 10), gridspec_kw={'height_ratios': [3, 1]})
    plt.suptitle('MNIST Adversarial Attack Performance', fontsize=24, fontweight='bold')

    colors = sns.color_palette("tab10", n_colors=len(model_names))
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']

    for idx, attack_type in enumerate(attack_types):
        # Accuracy vs Epsilon
        ax1 = axes[0, idx]
        for model_idx, model_name in enumerate(model_names):
            data = models_data[model_name][attack_type]
            epsilons, accuracies, auc_score = data
            ax1.plot(
                epsilons, accuracies,
                label=display_names[model_idx],
                color=colors[model_idx],
                marker=markers[model_idx % len(markers)],
                linewidth=2,
                markersize=6
            )
        ax1.set_title(f'{attack_type} Attack', fontsize=18)
        ax1.set_xlabel('Epsilon', fontsize=16)
        if idx == 0:
            ax1.set_ylabel('Top-1 Accuracy', fontsize=16)
        ax1.set_ylim(0, 1)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # AUC bar plot
        ax2 = axes[1, idx]
        auc_values = [models_data[m][attack_type][2] for m in model_names]
        bars = ax2.bar(
            np.arange(len(display_names)), auc_values, 
            color=colors,
            alpha=0.8
        )
        ax2.set_xlabel('Models', fontsize=16)
        if idx == 0:
            ax2.set_ylabel('AUC', fontsize=16)
        ax2.set_ylim(0, max(auc_values) * 1.2)
        ax2.set_xticks([])

        ax2.grid(True, linestyle='--', alpha=0.7)

        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.4f}',
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords='offset points',
                         ha='center', va='bottom', fontsize=10)

    # Legend
    lines = []
    labels = []
    for model_idx, short_name in enumerate(display_names):
        line = plt.Line2D(
            [], [], color=colors[model_idx],
            marker=markers[model_idx % len(markers)],
            linestyle='-', linewidth=2, markersize=8,
            label=short_name
        )
        lines.append(line)
        labels.append(short_name)

    fig.legend(
        handles=lines,
        labels=labels,
        loc='lower center',
        ncol=len(model_names),
        fontsize=20,
        frameon=False,
        bbox_to_anchor=(0.5, -0.05)
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {output_path}")
    plt.close()

def main():
    all_models = [d for d in os.listdir(checkpoints_dir) if d.startswith(model_prefix)]
    if not all_models:
        print("No models found with prefix", model_prefix)
        return

    print(f"Found models: {all_models}")

    models_data = {}
    for model in all_models:
        model_dir = os.path.join(checkpoints_dir, model)
        models_data[model] = {}
        for attack_type in attack_types:
            epsilons, accuracies, auc_score = load_attack_data(model_dir, attack_type)
            if epsilons is not None and accuracies is not None:
                models_data[model][attack_type] = (epsilons, accuracies, auc_score)

    if models_data:
        plot_combined_attack(models_data, list(models_data.keys()), model_display_names)

if __name__ == "__main__":
    main()
