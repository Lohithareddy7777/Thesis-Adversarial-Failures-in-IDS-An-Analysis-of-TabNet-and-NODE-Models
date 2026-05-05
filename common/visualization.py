import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

def _save_plot(save_path: str, *, dpi: int = 300, tight_rect=None):
    save_path = Path(save_path)
    if not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)
    if tight_rect is None:
        plt.tight_layout()
        plt.savefig(save_path.as_posix(), dpi=dpi, bbox_inches='tight')
    else:
        plt.tight_layout(rect=tight_rect)
        plt.savefig(save_path.as_posix(), dpi=dpi, bbox_inches='tight')
    plt.close('all')

def _set_dynamic_ylim(ax, values, pad_ratio: float = 0.12):
    if not values or all(v == 0 for v in values):
        ax.set_ylim([0, 0.1])
        return
    high = max(values)
    span = max(1e-6, high)
    upper = high + span * pad_ratio
    upper = min(1.05, upper)
    if upper < 0.05:
        upper = 0.1
    ax.set_ylim([0, upper])

def plot_summary_dataframe_bars(df: pd.DataFrame, title: str, save_path: str):
    """Plot numeric summary dataframe columns as bar charts."""
    if df is None or df.empty:
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return

    id_cols = [col for col in df.columns if col not in numeric_cols]
    labels = df[id_cols].astype(str).agg(" | ".join, axis=1).tolist() if id_cols else [f"Row {i+1}" for i in range(len(df))]

    n_metrics = len(numeric_cols)
    ncols = 2 if n_metrics > 1 else 1
    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7*ncols, 4.5*nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)

    x = np.arange(len(df))
    palette = sns.color_palette("tab10", max(3, len(df)))

    for idx, metric in enumerate(numeric_cols):
        ax = axes.flat[idx]
        values = df[metric].astype(float).tolist()
        bars = ax.bar(x, values, color=palette[:len(df)], alpha=0.88)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.4f}",
                   ha="center", va="bottom", fontsize=8)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        _set_dynamic_ylim(ax, values)
        ax.set_ylabel("Score")

    for idx in range(n_metrics, len(axes.flat)):
        axes.flat[idx].axis("off")

    fig.suptitle(title, fontsize=14)
    _save_plot(save_path, dpi=320, tight_rect=[0, 0, 1, 0.96])

def plot_feature_importance_shift(baseline_importance: pd.DataFrame, adversarial_importance: pd.DataFrame, title: str, save_path: str, top_n: int = 15):
    """RQ2: Plot TabNet feature importance shift."""
    if baseline_importance.empty or adversarial_importance.empty:
        return
    
    merged = baseline_importance.head(top_n).set_index('feature').join(
        adversarial_importance.set_index('feature'), lsuffix='_baseline', rsuffix='_adversarial')
    merged = merged.dropna()
    merged['shift'] = np.abs(merged['importance_baseline'] - merged['importance_adversarial'])
    merged = merged.sort_values('shift', ascending=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    y_pos = np.arange(len(merged))
    ax1.barh(y_pos, merged['shift'].values, color='steelblue', alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(merged.index)
    ax1.set_xlabel('Absolute Importance Shift')
    ax1.set_title('Feature Importance Shift Magnitude')
    ax1.set_xlim([0, merged['shift'].max() * 1.15])
    
    for i, v in enumerate(merged['shift'].values):
        ax1.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=8)
    
    merged_sorted = merged.sort_values('importance_baseline', ascending=True)
    y_pos2 = np.arange(len(merged_sorted))
    width = 0.35
    
    ax2.barh(y_pos2 - width/2, merged_sorted['importance_baseline'].values, width, label='Baseline', alpha=0.85, color='green')
    ax2.barh(y_pos2 + width/2, merged_sorted['importance_adversarial'].values, width, label='Adversarial', alpha=0.85, color='red')
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(merged_sorted.index)
    ax2.set_xlabel('Importance Score')
    ax2.set_title('Baseline vs Adversarial Feature Importance')
    ax2.set_xlim([0, max(merged_sorted['importance_baseline'].max(), merged_sorted['importance_adversarial'].max()) * 1.15])
    ax2.legend()
    
    fig.suptitle(title, fontsize=14)
    _save_plot(save_path, dpi=320)

def plot_activation_variance_comparison(activation_variance_metrics, title: str, save_path: str):
    """RQ2: Plot NODE activation variance changes."""
    if not activation_variance_metrics:
        return
    
    layer_keys = sorted([k for k in activation_variance_metrics.keys() if 'layer_' in k and 'variance_change' in k],
                       key=lambda x: int(x.split('_')[1]))
    if not layer_keys:
        return
    
    layer_names = [k.replace('_', ' ').title() for k in layer_keys]
    values = [activation_variance_metrics[k] for k in layer_keys]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(values)), values, color='steelblue', alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel('Variance Change (Baseline - Adversarial)', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(range(len(layer_names)))
    ax.set_xticklabels(layer_names, rotation=15, ha='right')
    ax.set_ylim([0, max(values) * 1.15 if values else 0.1])
    ax.grid(axis='y', alpha=0.3)
    
    _save_plot(save_path, dpi=320)

def plot_confidence_variation(confidence_baseline: np.ndarray, confidence_adversarial: np.ndarray, title: str, save_path: str, bins: int = 20):
    """RQ2: Plot confidence variation and degradation patterns."""
    confidence_drop = confidence_baseline - confidence_adversarial
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Left: Distribution
    ax = axes[0]
    ax.hist(confidence_drop, bins=bins, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(confidence_drop), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidence_drop):.4f}')
    ax.set_xlabel('Confidence Drop')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Confidence Drops')
    ax.legend()
    
    # Middle: Scatter
    ax = axes[1]
    scatter = ax.scatter(confidence_baseline, confidence_adversarial, alpha=0.5, s=20, c=confidence_drop, cmap='RdYlGn_r')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1)
    ax.set_xlabel('Baseline Confidence')
    ax.set_ylabel('Adversarial Confidence')
    ax.set_title('Confidence Degradation Pattern')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    plt.colorbar(scatter, ax=ax, label='Confidence Drop')
    
    # Right: Quantiles
    ax = axes[2]
    quantiles = np.arange(0.1, 1.0, 0.1)
    quantile_drops = [np.percentile(confidence_drop, q*100) for q in quantiles]
    bars = ax.bar(range(len(quantiles)), quantile_drops, color='coral', alpha=0.85)
    ax.set_ylabel('Confidence Drop')
    ax.set_title('Confidence Drop by Percentile')
    ax.set_xticks(range(len(quantiles)))
    ax.set_xticklabels([f'{q:.1f}' for q in quantiles], rotation=45)
    ax.set_ylim([0, max(quantile_drops) * 1.15])
    for i, v in enumerate(quantile_drops):
        ax.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)
    
    fig.suptitle(title, fontsize=14)
    _save_plot(save_path, dpi=320, tight_rect=[0, 0, 1, 0.97])

def plot_internal_behavior_deviations(baseline_metrics, adversarial_metrics, metric_names, model_name: str, title: str, save_path: str):
    """RQ2: Plot internal behavior deviations."""
    if not metric_names:
        return
    
    baseline_vals = [baseline_metrics.get(m, 0.0) for m in metric_names]
    adversarial_vals = [adversarial_metrics.get(m, 0.0) for m in metric_names]
    deviations = np.abs(np.array(baseline_vals) - np.array(adversarial_vals))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Deviation magnitude
    ax = axes[0]
    x_pos = np.arange(len(metric_names))
    bars = ax.bar(x_pos, deviations, color='firebrick', alpha=0.85)
    ax.set_ylabel('Absolute Deviation')
    ax.set_title(f'{model_name}: Internal Behavior Deviations')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names], rotation=45, ha='right')
    ax.set_ylim([0, max(deviations) * 1.15 if deviations.max() > 0 else 0.1])
    for bar, val in zip(bars, deviations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Right: Baseline vs Adversarial
    ax = axes[1]
    x_pos = np.arange(len(metric_names))
    width = 0.35
    ax.bar(x_pos - width/2, baseline_vals, width, label='Baseline', alpha=0.85, color='green')
    ax.bar(x_pos + width/2, adversarial_vals, width, label='Adversarial', alpha=0.85, color='red')
    ax.set_ylabel('Metric Value')
    ax.set_title(f'{model_name}: Baseline vs Adversarial Metrics')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_names], rotation=45, ha='right')
    max_val = max(max(baseline_vals), max(adversarial_vals))
    ax.set_ylim([0, max_val * 1.15 if max_val > 0 else 0.1])
    ax.legend()
    
    fig.suptitle(title, fontsize=14)
    _save_plot(save_path, dpi=320, tight_rect=[0, 0, 1, 0.97])

def plot_dataframe_table(df, title: str, save_path: str, fontsize: int = 9):
    """Fallback table visualization."""
    if df is None or df.empty:
        return
    fig, ax = plt.subplots(figsize=(min(14, max(8, df.shape[1])), min(8, max(4, df.shape[0]))))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1, 1.2)
    ax.set_title(title)
    _save_plot(save_path)

def plot_rq2_summary(summary_df: pd.DataFrame, title: str, save_path: str):
    """Plot RQ2 summary metrics."""
    if summary_df is None or summary_df.empty:
        return
    
    df = summary_df.copy()
    if "dataset" in df.columns and "Dataset" not in df.columns:
        df = df.rename(columns={"dataset": "Dataset"})
    
    if "Dataset" in df.columns:
        df = df.set_index("Dataset")
    
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        plot_dataframe_table(summary_df, title, save_path)
        return
    
    n_metrics = len(numeric_df.columns)
    ncols = min(3, max(1, n_metrics))
    nrows = int(np.ceil(n_metrics / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.5*nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    
    for idx, metric in enumerate(numeric_df.columns):
        ax = axes.flat[idx]
        values = numeric_df[metric]
        bars = ax.bar(values.index.astype(str), values.values, color="steelblue", alpha=0.85)
        for bar, value in zip(bars, values.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim([0.0, max(0.05, float(values.max()) * 1.2)])
    
    for idx in range(n_metrics, len(axes.flat)):
        axes.flat[idx].axis("off")
    
    fig.suptitle(title, fontsize=14)
    _save_plot(save_path, dpi=320, tight_rect=[0, 0, 1, 0.96])

def plot_failure_characteristics(failure_stats, title: str, save_path: str):
    """Plot failure characteristics."""
    if not failure_stats:
        return
    
    keys = [k for k in ['confidence_drop', 'prediction_flips', 'high_confidence_baseline_failures',
                        'large_confidence_drops', 'medium_confidence_drops', 'small_confidence_drops']
            if k in failure_stats]
    if not keys:
        return
    
    values = [float(failure_stats[key]) for key in keys]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar([k.replace("_", " ").title() for k in keys], values, color="steelblue", alpha=0.85)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylim([0, max(values) * 1.15 if values else 0.1])
    _save_plot(save_path)

def plot_metric_summary(metrics_by_model, title: str, save_path: str):
    """Plot metric summary by model."""
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
    _save_plot(save_path)

def plot_confidence_summary(metrics_by_model, title: str, save_path: str):
    """Plot confidence summary."""
    models = list(metrics_by_model.keys())
    values = [metrics_by_model[model].get("avg_confidence", 0.0) for model in models]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, values, color="steelblue", alpha=0.85)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylim([0, 1.05])
    ax.set_ylabel("Average Confidence")
    ax.set_title(title)
    _save_plot(save_path)

def plot_comprehensive_comparison(tabnet_baseline_metrics, tabnet_adversarial_metrics, node_baseline_metrics, 
                                 node_adversarial_metrics, tabnet_degradation, node_degradation, dataset_name: str, save_path: str):
    """Plot comprehensive baseline vs adversarial comparison."""
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    
    models = ["TabNet", "NODE"]
    x = np.arange(len(models))
    width = 0.34
    
    # Accuracy
    ax = fig.add_subplot(gs[0, 0])
    baseline_acc = [tabnet_baseline_metrics["accuracy"], node_baseline_metrics["accuracy"]]
    adversarial_acc = [tabnet_adversarial_metrics["accuracy"], node_adversarial_metrics["accuracy"]]
    ax.bar(x - width/2, baseline_acc, width, label="Baseline", alpha=0.85)
    ax.bar(x + width/2, adversarial_acc, width, label="Adversarial", alpha=0.85)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim([0, 1.05])
    ax.legend()
    
    # FNR
    ax = fig.add_subplot(gs[0, 1])
    baseline_fnr = [tabnet_baseline_metrics["false_negative_rate"], node_baseline_metrics["false_negative_rate"]]
    adversarial_fnr = [tabnet_adversarial_metrics["false_negative_rate"], node_adversarial_metrics["false_negative_rate"]]
    ax.bar(x - width/2, baseline_fnr, width, label="Baseline", alpha=0.85)
    ax.bar(x + width/2, adversarial_fnr, width, label="Adversarial", alpha=0.85)
    ax.set_ylabel("False Negative Rate")
    ax.set_title("False Negative Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim([0, 1.05])
    ax.legend()
    
    # Confidence
    ax = fig.add_subplot(gs[1, 0])
    baseline_conf = [tabnet_baseline_metrics["avg_confidence"], node_baseline_metrics["avg_confidence"]]
    adversarial_conf = [tabnet_adversarial_metrics["avg_confidence"], node_adversarial_metrics["avg_confidence"]]
    ax.bar(x - width/2, baseline_conf, width, label="Baseline", alpha=0.85)
    ax.bar(x + width/2, adversarial_conf, width, label="Adversarial", alpha=0.85)
    ax.set_ylabel("Average Confidence")
    ax.set_title("Confidence")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim([0, 1.05])
    ax.legend()
    
    # Degradation
    ax = fig.add_subplot(gs[1, 1])
    deg_metrics = ["accuracy_drop", "recall_drop", "fnr_increase", "confidence_degradation"]
    x_deg = np.arange(len(deg_metrics))
    tabnet_deg = [abs(tabnet_degradation.get(m, 0.0)) for m in deg_metrics]
    node_deg = [abs(node_degradation.get(m, 0.0)) for m in deg_metrics]
    ax.bar(x_deg - width/2, tabnet_deg, width, label="TabNet", alpha=0.85)
    ax.bar(x_deg + width/2, node_deg, width, label="NODE", alpha=0.85)
    ax.set_ylabel("Degradation")
    ax.set_title("Performance Degradation")
    ax.set_xticks(x_deg)
    ax.set_xticklabels([m.replace("_", " ").title() for m in deg_metrics], rotation=15, ha="right")
    ax.set_ylim([0, 1.05])
    ax.legend()
    
    fig.suptitle(f"Comprehensive Baseline vs Adversarial Analysis - {dataset_name}", fontsize=14)
    _save_plot(save_path, dpi=320)

def plot_attack_comparison(attack_results_by_model, dataset_name: str, save_path: str):
    """Plot attack comparison (Baseline vs FGSM vs PGD)."""
    metric_specs = [("accuracy", "Accuracy"), ("false_negative_rate", "False Negative Rate"), ("avg_confidence", "Average Confidence")]
    
    conditions = ["Baseline", "FGSM", "PGD"]
    models = list(attack_results_by_model.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    x = np.arange(len(conditions))
    width = 0.35 if len(models) > 1 else 0.6
    
    for axis_idx, (metric_key, metric_title) in enumerate(metric_specs):
        ax = axes[axis_idx]
        for model_idx, model_name in enumerate(models):
            values = [attack_results_by_model[model_name].get(condition, {}).get(metric_key, 0.0) for condition in conditions]
            bar_pos = x + (model_idx - (len(models) - 1) / 2) * width
            ax.bar(bar_pos, values, width=width, label=model_name, alpha=0.85)
        
        ax.set_title(metric_title)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions)
        ax.set_ylim([0.0, 1.05])
        if axis_idx == 0:
            ax.set_ylabel("Score")
            ax.legend()
    
    fig.suptitle(f"Baseline vs FGSM vs PGD - {dataset_name}", fontsize=14)
    _save_plot(save_path)

def plot_adversarial_common_metrics_combined(attack_results_by_model, dataset_name: str, save_path: str):
    """Plot adversarial common metrics."""
    metric_specs = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"),
                   ("f1_score", "F1 Score"), ("false_negative_rate", "False Negative Rate"),
                   ("misclassification_rate", "Misclassification Rate")]
    
    attack_order = ["FGSM", "PGD"]
    models = list(attack_results_by_model.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    x = np.arange(len(attack_order))
    width = 0.36 if len(models) > 1 else 0.6
    
    for axis_idx, (metric_key, metric_title) in enumerate(metric_specs):
        ax = axes[axis_idx]
        for model_idx, model_name in enumerate(models):
            values = [attack_results_by_model[model_name].get(attack, {}).get(metric_key, 0.0) for attack in attack_order]
            bar_pos = x + (model_idx - (len(models) - 1) / 2) * width
            ax.bar(bar_pos, values, width=width, label=model_name, alpha=0.85)
        
        ax.set_title(metric_title)
        ax.set_xticks(x)
        ax.set_xticklabels(attack_order)
        ax.set_ylim([0.0, 1.05])
        
        if axis_idx % 3 == 0:
            ax.set_ylabel("Score")
        if axis_idx == 0:
            ax.legend()
    
    fig.suptitle(f"Adversarial Common Metrics (FGSM/PGD) - {dataset_name}", fontsize=14)
    _save_plot(save_path, dpi=320, tight_rect=[0, 0, 1, 0.97])

def plot_adversarial_confidence_combined(attack_results_by_model, dataset_name: str, save_path: str):
    """Plot adversarial confidence comparison."""
    attacks = ["FGSM", "PGD"]
    models = list(attack_results_by_model.keys())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(attacks))
    width = 0.36 if len(models) > 1 else 0.6
    
    for model_idx, model_name in enumerate(models):
        values = [attack_results_by_model[model_name].get(attack, {}).get("avg_confidence", 0.0) for attack in attacks]
        bar_pos = x + (model_idx - (len(models) - 1) / 2) * width
        bars = ax.bar(bar_pos, values, width=width, label=model_name, alpha=0.85)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    
    ax.set_ylim([0.0, 1.05])
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylabel("Average Confidence")
    ax.set_title(f"Adversarial Confidence (FGSM/PGD) - {dataset_name}")
    ax.legend()
    
    _save_plot(save_path, dpi=320)

def plot_feature_importance(importance_df: pd.DataFrame, title: str, save_path: str, top_n: int = 20):
    """Plot top-N features by importance."""
    df = importance_df.head(top_n).copy()
    df = df.sort_values('importance', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(df['feature'].astype(str), df['importance'], color='steelblue', alpha=0.85)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height()/2, f'{width:.4f}', ha='left', va='center', fontsize=9)
    
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_ylabel('Feature', fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.set_xlim([0, df['importance'].max() * 1.15 if len(df) > 0 else 0.1])
    
    _save_plot(save_path)

def plot_rq2_correlation_heatmap(correlation_df: pd.DataFrame, title: str, save_path: str):
    """RQ2: Plot correlation heatmap."""
    if correlation_df.empty:
        return
    
    pivot_data = {}
    for _, row in correlation_df.iterrows():
        model = row.get('Model', 'Unknown')
        metric = row.get('Internal_Metric', 'Unknown')
        outcome = row.get('Outcome', 'Unknown')
        corr_val = row.get('Pearson_R', 0.0)
        key = f"{metric} → {outcome}"
        if key not in pivot_data:
            pivot_data[key] = {}
        pivot_data[key][model] = corr_val
    
    if not pivot_data:
        return
    
    heatmap_df = pd.DataFrame(pivot_data).T.fillna(0)
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot_data) * 0.4)))
    sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
               cbar_kws={'label': 'Pearson Correlation'}, ax=ax, vmin=-1, vmax=1)
    ax.set_title(title, fontsize=12, pad=20)
    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Internal Metric → Failure Outcome', fontsize=11)
    _save_plot(save_path)

def create_graph_montage(image_paths, save_path: str, title: str = None, ncols: int = 2):
    """Create a montage of images."""
    valid_paths = [Path(path) for path in image_paths if path and Path(path).is_file()]
    if not valid_paths:
        return
    
    n_images = len(valid_paths)
    ncols = max(1, min(ncols, n_images))
    nrows = int(np.ceil(n_images / ncols))
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.5*nrows))
    axes = np.atleast_1d(axes).reshape(nrows, ncols)
    
    for index, ax in enumerate(axes.flat):
        if index >= n_images:
            ax.axis("off")
            continue
        
        image = plt.imread(valid_paths[index])
        ax.imshow(image)
        ax.set_title(valid_paths[index].name, fontsize=9)
        ax.axis("off")
    
    if title:
        fig.suptitle(title, fontsize=14)
    
    _save_plot(save_path, dpi=300, tight_rect=[0, 0, 1, 0.96] if title else None)

def plot_model_failure_comparison(df: pd.DataFrame, title: str, save_path: str):
    """Plot model failure comparison."""
    if df is None or df.empty:
        return
    
    metric_col = "Metric" if "Metric" in df.columns else df.columns[0]
    model_cols = [col for col in df.columns if col != metric_col and col != "Difference"]
    if not model_cols:
        return
    
    x = np.arange(len(df[metric_col]))
    width = 0.35 if len(model_cols) > 1 else 0.6
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, model_col in enumerate(model_cols):
        values = df[model_col].astype(float).tolist()
        offset = (idx - (len(model_cols) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width, label=model_col, alpha=0.85)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(df[metric_col].astype(str), rotation=20, ha="right")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.set_ylim([0, max(df[model_cols].max().max(), 0.1) * 1.15])
    _save_plot(save_path)

def plot_rq2_correlation_table(correlation_df: pd.DataFrame, title: str, save_path: str):
    """RQ2: Plot correlation analysis as table."""
    if correlation_df is None or correlation_df.empty:
        return
    
    display_cols = ['feature', 'importance_shift', 'correlation_with_failures', 'correlation_pvalue']
    if all(col in correlation_df.columns for col in display_cols):
        table_df = correlation_df[display_cols].head(10).copy()
        table_df['importance_shift'] = table_df['importance_shift'].apply(lambda x: f"{x:.4f}")
        table_df['correlation_with_failures'] = table_df['correlation_with_failures'].apply(lambda x: f"{x:.4f}")
        table_df['correlation_pvalue'] = table_df['correlation_pvalue'].apply(lambda x: f"{x:.4f}")
        table_df.columns = ['Feature', 'Importance Shift', 'Correlation', 'P-Value']
    else:
        table_df = correlation_df.head(10).copy()
        table_df = table_df.round(4)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(table_df.columns)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    fig.suptitle(title, fontsize=12, weight='bold')
    _save_plot(save_path)

def plot_internal_behavior_summary(summary_rows: pd.DataFrame, title: str, save_path: str):
    """Plot a single unified internal behavior analysis summary."""
    if summary_rows is None or summary_rows.empty:
        return

    df = summary_rows.copy()
    required_cols = [col for col in ['Metric', 'Baseline', 'Adversarial', 'Change', 'Failure_Link'] if col in df.columns]
    numeric_cols = [col for col in ['Baseline', 'Adversarial', 'Change', 'Failure_Link'] if col in df.columns and np.issubdtype(df[col].dtype, np.number)]
    if 'Metric' not in df.columns or not numeric_cols:
        plot_dataframe_table(df, title, save_path)
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(df.head(4).iterrows()):
        ax = axes[idx]
        values = [float(row[col]) for col in numeric_cols]
        bars = ax.bar(numeric_cols, values, color='steelblue', alpha=0.88)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:.4f}", ha='center', va='bottom', fontsize=8)
        ax.set_title(str(row['Metric']))
        ax.tick_params(axis='x', rotation=20)
        ax.set_ylim([0, max(values) * 1.2 if values else 0.1])

    for idx in range(min(4, len(df)), 4):
        axes[idx].axis('off')

    fig.suptitle(title, fontsize=14)
    _save_plot(save_path, dpi=320, tight_rect=[0, 0, 1, 0.95])
