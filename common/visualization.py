import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def _set_dynamic_ylim(ax, values, pad_ratio: float = 0.12):
    low = min(values)
    high = max(values)
    span = max(1e-6, high - low)
    lower = max(0.0, low - span * pad_ratio)
    upper = min(1.05, high + span * pad_ratio)
    if upper - lower < 0.05:
        mid = (upper + lower) / 2
        lower = max(0.0, mid - 0.03)
        upper = min(1.05, mid + 0.03)
    ax.set_ylim([lower, upper])


def _plot_grouped_bars(ax, x, left_vals, right_vals, width, left_label, right_label, left_color, right_color, value_fmt: str = "{:.4f}"):
    left_bars = ax.bar(x - width / 2, left_vals, width, label=left_label, alpha=0.85, color=left_color)
    right_bars = ax.bar(x + width / 2, right_vals, width, label=right_label, alpha=0.85, color=right_color)

    all_vals = list(left_vals) + list(right_vals)
    _set_dynamic_ylim(ax, all_vals)
    ymin, ymax = ax.get_ylim()
    ypad = (ymax - ymin) * 0.02

    for bar, value in zip(left_bars, left_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ypad, value_fmt.format(value), ha="center", va="bottom", fontsize=8)
    for bar, value in zip(right_bars, right_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + ypad, value_fmt.format(value), ha="center", va="bottom", fontsize=8)


def plot_metric_summary(metrics_by_model: Dict[str, Dict[str, float]], title: str, save_path: str):
    metrics = ["accuracy", "precision", "recall", "false_negative_rate", "misclassification_rate"]
    models = list(metrics_by_model.keys())
    values = [[metrics_by_model[model].get(metric, 0.0) for metric in metrics] for model in models]

    x = np.arange(len(metrics))
    width = 0.35 if len(models) > 1 else 0.5

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, model in enumerate(models):
        ax.bar(x + (idx - (len(models) - 1) / 2) * width, values[idx], width, label=model, alpha=0.85)

    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in metrics])
    ax.set_ylim([0, 1.05])
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confidence_summary(metrics_by_model: Dict[str, Dict[str, float]], title: str, save_path: str):
    models = list(metrics_by_model.keys())
    values = [metrics_by_model[model].get("avg_confidence", 0.0) for model in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(models, values, color=["steelblue", "darkorange"][:len(models)], alpha=0.85)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel("Average Confidence")
    ax.set_title(title)

    for i, value in enumerate(values):
        ax.text(i, value + 0.01, f"{value:.3f}", ha="center", va="bottom")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_degradation_comparison(tabnet_degradation: Dict[str, float], node_degradation: Dict[str, float], title: str, save_path: str):
    metrics = ["accuracy_drop", "recall_drop", "fnr_increase", "confidence_degradation"]
    labels = [m.replace("_", " ").title() for m in metrics]
    tabnet_values = [abs(tabnet_degradation.get(m, 0.0)) for m in metrics]
    node_values = [abs(node_degradation.get(m, 0.0)) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, tabnet_values, width, label="TabNet", alpha=0.85)
    ax.bar(x + width / 2, node_values, width, label="NODE", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Degradation Magnitude")
    ax.set_title(title)
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_comprehensive_comparison(
    tabnet_baseline_metrics: Dict[str, float],
    tabnet_adversarial_metrics: Dict[str, float],
    node_baseline_metrics: Dict[str, float],
    node_adversarial_metrics: Dict[str, float],
    tabnet_degradation: Dict[str, float],
    node_degradation: Dict[str, float],
    dataset_name: str,
    save_path: str,
):
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    models = ["TabNet", "NODE"]
    x = np.arange(len(models))
    width = 0.34

    ax1 = fig.add_subplot(gs[0, 0])
    baseline_acc = [tabnet_baseline_metrics["accuracy"], node_baseline_metrics["accuracy"]]
    adversarial_acc = [tabnet_adversarial_metrics["accuracy"], node_adversarial_metrics["accuracy"]]
    _plot_grouped_bars(ax1, x, baseline_acc, adversarial_acc, width, "Baseline", "Adversarial", "steelblue", "coral")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 1])
    baseline_fnr = [tabnet_baseline_metrics["false_negative_rate"], node_baseline_metrics["false_negative_rate"]]
    adversarial_fnr = [tabnet_adversarial_metrics["false_negative_rate"], node_adversarial_metrics["false_negative_rate"]]
    _plot_grouped_bars(ax2, x, baseline_fnr, adversarial_fnr, width, "Baseline", "Adversarial", "seagreen", "firebrick")
    ax2.set_ylabel("False Negative Rate")
    ax2.set_title("False Negative Rate")
    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    baseline_conf = [tabnet_baseline_metrics["avg_confidence"], node_baseline_metrics["avg_confidence"]]
    adversarial_conf = [tabnet_adversarial_metrics["avg_confidence"], node_adversarial_metrics["avg_confidence"]]
    _plot_grouped_bars(ax3, x, baseline_conf, adversarial_conf, width, "Baseline", "Adversarial", "mediumpurple", "darkorange")
    ax3.set_ylabel("Average Confidence")
    ax3.set_title("Confidence")
    ax3.set_xticks(x)
    ax3.set_xticklabels(models)
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    deg_metrics = ["accuracy_drop", "recall_drop", "fnr_increase", "confidence_degradation"]
    x_deg = np.arange(len(deg_metrics))
    tabnet_deg = [abs(tabnet_degradation.get(m, 0.0)) for m in deg_metrics]
    node_deg = [abs(node_degradation.get(m, 0.0)) for m in deg_metrics]
    _plot_grouped_bars(ax4, x_deg, tabnet_deg, node_deg, width, "TabNet", "NODE", "steelblue", "darkorange", value_fmt="{:.5f}")
    ax4.set_ylabel("Degradation")
    ax4.set_title("Performance Degradation")
    ax4.set_xticks(x_deg)
    ax4.set_xticklabels([m.replace("_", " ").title() for m in deg_metrics], rotation=15, ha="right")
    ax4.legend()

    fig.suptitle(f"Comprehensive Baseline vs Adversarial Analysis - {dataset_name}", fontsize=14)
    plt.savefig(save_path, dpi=320, bbox_inches='tight')
    plt.close()


def plot_attack_comparison(
    attack_results_by_model: Dict[str, Dict[str, Dict[str, float]]],
    dataset_name: str,
    save_path: str,
):
    metric_specs = [
        ("accuracy", "Accuracy"),
        ("false_negative_rate", "False Negative Rate"),
        ("avg_confidence", "Average Confidence"),
    ]

    preferred_order = ["Baseline", "FGSM", "PGD", "BRPA"]
    all_conditions = set()
    for model_results in attack_results_by_model.values():
        all_conditions.update(model_results.keys())

    conditions = [c for c in preferred_order if c in all_conditions]
    for c in sorted(all_conditions):
        if c not in conditions:
            conditions.append(c)

    models = list(attack_results_by_model.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    x = np.arange(len(conditions))
    width = 0.35 if len(models) > 1 else 0.6

    for axis_idx, (metric_key, metric_title) in enumerate(metric_specs):
        ax = axes[axis_idx]
        for model_idx, model_name in enumerate(models):
            values = [
                attack_results_by_model[model_name].get(condition, {}).get(metric_key, 0.0)
                for condition in conditions
            ]
            bar_pos = x + (model_idx - (len(models) - 1) / 2) * width
            ax.bar(bar_pos, values, width=width, label=model_name, alpha=0.85)

        ax.set_title(metric_title)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        if metric_key == "accuracy" or metric_key == "avg_confidence":
            ax.set_ylim([0.0, 1.05])
        if metric_key == "false_negative_rate":
            max_v = 0.0
            for model_name in models:
                for condition in conditions:
                    max_v = max(max_v, attack_results_by_model[model_name].get(condition, {}).get(metric_key, 0.0))
            ax.set_ylim([0.0, max(0.05, max_v * 1.25)])
        if axis_idx == 0:
            ax.set_ylabel("Score")
            ax.legend()

    fig.suptitle(f"Baseline vs FGSM vs PGD vs BRPA - {dataset_name}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_adversarial_common_metrics_combined(
    attack_results_by_model: Dict[str, Dict[str, Dict[str, float]]],
    dataset_name: str,
    save_path: str,
):
    metric_specs = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1 Score"),
        ("false_negative_rate", "False Negative Rate"),
        ("misclassification_rate", "Misclassification Rate"),
    ]

    attack_order = ["FGSM", "PGD", "BRPA"]
    models = list(attack_results_by_model.keys())

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    x = np.arange(len(attack_order))
    width = 0.36 if len(models) > 1 else 0.6

    for axis_idx, (metric_key, metric_title) in enumerate(metric_specs):
        ax = axes[axis_idx]
        all_vals = []

        for model_idx, model_name in enumerate(models):
            values = [
                attack_results_by_model[model_name].get(attack, {}).get(metric_key, 0.0)
                for attack in attack_order
            ]
            all_vals.extend(values)
            bar_pos = x + (model_idx - (len(models) - 1) / 2) * width
            ax.bar(bar_pos, values, width=width, label=model_name, alpha=0.85)

        ax.set_title(metric_title)
        ax.set_xticks(x)
        ax.set_xticklabels(attack_order)

        if metric_key in {"accuracy", "precision", "recall", "f1_score"}:
            ax.set_ylim([0.0, 1.05])
        else:
            max_v = max(all_vals) if all_vals else 0.0
            ax.set_ylim([0.0, max(0.05, max_v * 1.2)])

        if axis_idx % 3 == 0:
            ax.set_ylabel("Score")
        if axis_idx == 0:
            ax.legend()

    fig.suptitle(f"Adversarial Common Metrics (FGSM/PGD/BRPA) - {dataset_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=320, bbox_inches='tight')
    plt.close()


def plot_adversarial_metrics_confidence_combined(
    attack_results_by_model: Dict[str, Dict[str, Dict[str, float]]],
    dataset_name: str,
    save_path: str,
):
    attacks = ["FGSM", "PGD", "BRPA"]
    metrics = ["accuracy", "precision", "recall", "f1_score", "avg_confidence"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1", "Avg Confidence"]
    models = list(attack_results_by_model.keys())

    fig, axes = plt.subplots(len(models), 1, figsize=(12, 4 * len(models)))
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        matrix = np.array(
            [[attack_results_by_model[model].get(attack, {}).get(metric, 0.0) for metric in metrics] for attack in attacks]
        )
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            vmin=0.0,
            vmax=1.0,
            xticklabels=metric_labels,
            yticklabels=attacks,
            cbar=True,
            ax=ax,
        )
        ax.set_title(f"{model}: Adversarial Metrics + Confidence")
        ax.set_xlabel("Metrics")
        ax.set_ylabel("Attacks")

    fig.suptitle(f"Combined Adversarial Metrics and Confidence - {dataset_name}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=320, bbox_inches="tight")
    plt.close()


def plot_adversarial_confidence_combined(
    attack_results_by_model: Dict[str, Dict[str, Dict[str, float]]],
    dataset_name: str,
    save_path: str,
):
    attacks = ["FGSM", "PGD", "BRPA"]
    models = list(attack_results_by_model.keys())

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(attacks))
    width = 0.36 if len(models) > 1 else 0.6
    all_vals = []

    for model_idx, model_name in enumerate(models):
        values = [
            attack_results_by_model[model_name].get(attack, {}).get("avg_confidence", 0.0)
            for attack in attacks
        ]
        all_vals.extend(values)
        bar_pos = x + (model_idx - (len(models) - 1) / 2) * width
        bars = ax.bar(bar_pos, values, width=width, label=model_name, alpha=0.85)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    max_v = max(all_vals) if all_vals else 0.0
    ax.set_ylim([0.0, max(0.1, min(1.05, max_v * 1.2 if max_v > 0 else 1.0))])
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylabel("Average Confidence")
    ax.set_title(f"Adversarial Confidence (FGSM/PGD/BRPA) - {dataset_name}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=320, bbox_inches="tight")
    plt.close()