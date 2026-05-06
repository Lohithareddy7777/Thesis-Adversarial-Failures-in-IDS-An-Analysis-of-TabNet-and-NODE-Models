from __future__ import annotations

import os
import shutil
import json
import gc
from datetime import datetime
from typing import Type

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split

from common.utils import set_random_seed, get_feature_bounds, ensure_dir
from common.visualization import (
    plot_metric_summary,
    plot_confidence_summary,
    plot_comprehensive_comparison,
    plot_attack_comparison,
    plot_adversarial_common_metrics_combined,
    plot_adversarial_confidence_combined,
    create_graph_montage,
    plot_rq2_summary,
    plot_failure_characteristics,
    plot_dataframe_table,
    plot_summary_dataframe_bars,
    plot_feature_importance_shift,
    plot_internal_behavior_summary,
)


def _print_step(title: str):
    print("\n" + "▶" * 40)
    print(title)
    print("▶" * 40)


def _split_train_val(X: np.ndarray, y: np.ndarray, random_state: int, val_ratio: float = 0.2):
    return train_test_split(
        X,
        y,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y,
    )


def _normalize_attack_configs(attack_configs, attack_type: str, epsilon: float, alpha: float, num_iter: int):
    if attack_configs:
        return [
            {
                "attack_type": cfg.get("attack_type", "PGD").upper(),
                "epsilon": float(cfg.get("epsilon", epsilon)),
                "alpha": float(cfg.get("alpha", alpha)),
                "num_iter": int(cfg.get("num_iter", num_iter)),
                "perturbation_ratio": float(cfg.get("perturbation_ratio", 0.1)),
            }
            for cfg in attack_configs
        ]
    return [
        {
            "attack_type": attack_type.upper(),
            "epsilon": float(epsilon),
            "alpha": float(alpha),
            "num_iter": int(num_iter),
            "perturbation_ratio": 0.1,
        }
    ]


def _select_primary_attack(normalized_attack_configs: list[dict]):
    for candidate in ("PGD", "FGSM"):
        selected = next((cfg for cfg in normalized_attack_configs if cfg["attack_type"] == candidate), None)
        if selected is not None:
            return selected
    return normalized_attack_configs[0]


def _attack_kwargs(attack_cfg: dict) -> dict:
    return {
        "epsilon": attack_cfg.get("epsilon", 0.01),
        "alpha": attack_cfg.get("alpha", 0.005),
        "num_iter": attack_cfg.get("num_iter", 3),
        "clip_bounds": attack_cfg.get("bounds"),
    }


def _generate_adv_inputs(
    attacker,
    attack_cfg: dict,
    model_tabnet,
    model_node,
    X_test: np.ndarray,
    y_test: np.ndarray,
    bounds,
    include_node: bool = True,
):
    attack_name = attack_cfg["attack_type"]
    attack_kwargs = _attack_kwargs({**attack_cfg, "bounds": bounds})
    tabnet_adv = attacker.generate_attack_tabnet(attack_name, model_tabnet, X_test, y_test, **attack_kwargs)
    node_adv = None
    if include_node:
        node_adv = attacker.generate_attack(attack_name, model_node, X_test, y_test, **attack_kwargs)
    return tabnet_adv, node_adv


def _run_baseline_phase(
    trainer,
    evaluator,
    X_train_use: np.ndarray,
    X_test_use: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
    tag: str,
    include_node: bool = True,
):
    X_tr, X_val, y_tr, y_val = _split_train_val(X_train_use, y_train, random_state=random_state)
    tabnet_model = trainer.train_tabnet(X_tr, y_tr, X_val, y_val, num_classes=2, max_epochs=40, patience=8)
    node_model = None
    if include_node:
        node_model = trainer.train_node(X_tr, y_tr, X_val, y_val, num_classes=2, max_epochs=40, patience=8)

    y_pred_tabnet_base, y_proba_tabnet_base = trainer.predict_tabnet(tabnet_model, X_test_use)
    tabnet_baseline_metrics = evaluator.evaluate_baseline(y_test, y_pred_tabnet_base, y_proba_tabnet_base, f"TabNet-{tag}")
    y_pred_tabnet_train, _ = trainer.predict_tabnet(tabnet_model, X_train_use)

    y_pred_node_base = None
    y_proba_node_base = None
    node_baseline_metrics = None
    y_pred_node_train = None
    if include_node:
        y_pred_node_base, y_proba_node_base = trainer.predict_node(node_model, X_test_use)
        node_baseline_metrics = evaluator.evaluate_baseline(y_test, y_pred_node_base, y_proba_node_base, f"NODE-{tag}")
        y_pred_node_train, _ = trainer.predict_node(node_model, X_train_use)

    tabnet_train_acc = float(np.mean(y_pred_tabnet_train == y_train))
    node_train_acc = float(np.mean(y_pred_node_train == y_train)) if include_node else None
    overfit_diagnostics = {
        "TabNet": {
            "train_accuracy": tabnet_train_acc,
            "test_accuracy": float(tabnet_baseline_metrics["accuracy"]),
            "generalization_gap": tabnet_train_acc - float(tabnet_baseline_metrics["accuracy"]),
        },
    }
    if include_node and node_baseline_metrics is not None and node_train_acc is not None:
        overfit_diagnostics["NODE"] = {
            "train_accuracy": node_train_acc,
            "test_accuracy": float(node_baseline_metrics["accuracy"]),
            "generalization_gap": node_train_acc - float(node_baseline_metrics["accuracy"]),
        }

    print("\nOverfitting diagnostics (train vs test):")
    for model_name, diag in overfit_diagnostics.items():
        print(
            f"  {model_name:6s} | train={diag['train_accuracy']:.4f} "
            f"test={diag['test_accuracy']:.4f} gap={diag['generalization_gap']:.4f}"
        )

    return {
        "tabnet_model": tabnet_model,
        "node_model": node_model,
        "y_pred_tabnet_base": y_pred_tabnet_base,
        "y_proba_tabnet_base": y_proba_tabnet_base,
        "y_pred_node_base": y_pred_node_base,
        "y_proba_node_base": y_proba_node_base,
        "tabnet_baseline_metrics": tabnet_baseline_metrics,
        "node_baseline_metrics": node_baseline_metrics,
        "overfit_diagnostics": overfit_diagnostics,
    }


def _evaluate_attack_results(
    trainer,
    evaluator,
    y_test: np.ndarray,
    tabnet_model,
    node_model,
    X_test_adv_tabnet: np.ndarray,
    X_test_adv_node: np.ndarray,
    tabnet_baseline_metrics: dict,
    node_baseline_metrics: dict | None,
    attack_name: str,
    tag: str,
    include_node: bool = True,
):
    y_pred_tabnet_adv, y_proba_tabnet_adv = trainer.predict_tabnet(tabnet_model, X_test_adv_tabnet)
    tabnet_adv_metrics = evaluator.evaluate_adversarial(y_test, y_pred_tabnet_adv, y_proba_tabnet_adv, f"TabNet-{tag}", attack_name)

    y_pred_node_adv = None
    y_proba_node_adv = None
    node_adv_metrics = None
    if include_node and X_test_adv_node is not None and node_model is not None:
        y_pred_node_adv, y_proba_node_adv = trainer.predict_node(node_model, X_test_adv_node)
        node_adv_metrics = evaluator.evaluate_adversarial(y_test, y_pred_node_adv, y_proba_node_adv, f"NODE-{tag}", attack_name)

    tabnet_adv_degradation = evaluator.compute_degradation(tabnet_baseline_metrics, tabnet_adv_metrics, f"TabNet-{tag}", attack_name)
    node_adv_degradation = None
    if include_node and node_baseline_metrics is not None and node_adv_metrics is not None:
        node_adv_degradation = evaluator.compute_degradation(node_baseline_metrics, node_adv_metrics, f"NODE-{tag}", attack_name)
    return {
        "X_test_adv_tabnet": X_test_adv_tabnet,
        "X_test_adv_node": X_test_adv_node,
        "y_pred_tabnet_adv": y_pred_tabnet_adv,
        "y_proba_tabnet_adv": y_proba_tabnet_adv,
        "y_pred_node_adv": y_pred_node_adv,
        "y_proba_node_adv": y_proba_node_adv,
        "tabnet_adv_metrics": tabnet_adv_metrics,
        "node_adv_metrics": node_adv_metrics,
        "tabnet_adv_degradation": tabnet_adv_degradation,
        "node_adv_degradation": node_adv_degradation,
    }


def _run_adversarial_phase(
    attacker,
    trainer,
    evaluator,
    attack_cfg: dict,
    tabnet_model,
    node_model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    bounds,
    tabnet_baseline_metrics: dict,
    node_baseline_metrics: dict | None,
    tag: str,
    include_node: bool = True,
):
    X_test_adv_tabnet, X_test_adv_node = _generate_adv_inputs(
        attacker,
        attack_cfg,
        tabnet_model,
        node_model,
        X_test,
        y_test,
        bounds,
        include_node,
    )
    return _evaluate_attack_results(
        trainer,
        evaluator,
        y_test,
        tabnet_model,
        node_model,
        X_test_adv_tabnet,
        X_test_adv_node,
        tabnet_baseline_metrics,
        node_baseline_metrics,
        attack_cfg["attack_type"],
        tag,
        include_node,
    )


def _build_baseline_summary_rows(tabnet_baseline_metrics: dict, node_baseline_metrics: dict | None):
    rows = [{"Model": "TabNet", **tabnet_baseline_metrics}]
    if node_baseline_metrics is not None:
        rows.append({"Model": "NODE", **node_baseline_metrics})
    return rows


def _build_adversarial_summary_rows(attack_results: dict, include_node: bool = True):
    rows = []
    for attack_key, attack_data in attack_results.items():
        rows.append(
            {
                "Attack": attack_key,
                "Model": "TabNet",
                "accuracy": attack_data["tabnet_adv_metrics"].get("accuracy"),
                "false_negative_rate": attack_data["tabnet_adv_metrics"].get("false_negative_rate"),
                "misclassification_rate": attack_data["tabnet_adv_metrics"].get("misclassification_rate"),
                "avg_confidence": attack_data["tabnet_adv_metrics"].get("avg_confidence"),
                "confidence_degradation": attack_data["tabnet_adv_degradation"].get("confidence_degradation"),
            }
        )
        if include_node and attack_data.get("node_adv_metrics") is not None:
            rows.append(
                {
                    "Attack": attack_key,
                    "Model": "NODE",
                    "accuracy": attack_data["node_adv_metrics"].get("accuracy"),
                    "false_negative_rate": attack_data["node_adv_metrics"].get("false_negative_rate"),
                    "misclassification_rate": attack_data["node_adv_metrics"].get("misclassification_rate"),
                    "avg_confidence": attack_data["node_adv_metrics"].get("avg_confidence"),
                    "confidence_degradation": attack_data["node_adv_degradation"].get("confidence_degradation") if attack_data.get("node_adv_degradation") else np.nan,
                }
            )
    return rows


def _format_results_dataframe(df: pd.DataFrame, ordered_columns: list[str], sort_cols: list[str] | None = None) -> pd.DataFrame:
    missing = [col for col in ordered_columns if col not in df.columns]
    if missing:
        for col in missing:
            df[col] = np.nan

    formatted = df[ordered_columns].copy()

    if "Attack" in formatted.columns:
        default_attack_order = ["Baseline", "FGSM", "PGD"]
        seen_attacks = [
            attack for attack in formatted["Attack"].dropna().astype(str).unique().tolist()
            if attack not in default_attack_order
        ]
        attack_order = pd.CategoricalDtype(default_attack_order + seen_attacks, ordered=True)
        formatted["Attack"] = formatted["Attack"].astype(str).astype(attack_order)
    if "Model" in formatted.columns:
        model_order = pd.CategoricalDtype(["TabNet", "NODE"], ordered=True)
        formatted["Model"] = formatted["Model"].astype(model_order)

    if sort_cols:
        formatted = formatted.sort_values(sort_cols).reset_index(drop=True)

    float_cols = formatted.select_dtypes(include=[np.floating]).columns
    formatted[float_cols] = formatted[float_cols].round(6)

    if "Attack" in formatted.columns:
        formatted["Attack"] = formatted["Attack"].astype(str)
    if "Model" in formatted.columns:
        formatted["Model"] = formatted["Model"].astype(str)

    return formatted


def _baseline_columns() -> list[str]:
    return [
        "Model",
        "accuracy",
        "false_negative_rate",
        "misclassification_rate",
        "avg_confidence",
    ]


def _adversarial_columns() -> list[str]:
    return [
        "Attack",
        "Model",
        "accuracy",
        "false_negative_rate",
        "misclassification_rate",
        "avg_confidence",
        "confidence_degradation",
    ]


def _unified_columns() -> list[str]:
    return [
        "Scenario",
        "Attack",
        "Model",
        "accuracy",
        "false_negative_rate",
        "misclassification_rate",
        "avg_confidence",
        "confidence_degradation",
    ]


def _build_unified_results_dataframe(baseline_df: pd.DataFrame, adversarial_df: pd.DataFrame) -> pd.DataFrame:
    baseline_unified = baseline_df.copy()
    baseline_unified.insert(0, "Scenario", "Baseline")
    baseline_unified.insert(1, "Attack", "Baseline")
    baseline_unified["confidence_degradation"] = np.nan

    adversarial_unified = adversarial_df.copy()
    adversarial_unified.insert(0, "Scenario", "Adversarial")

    unified = pd.concat([baseline_unified, adversarial_unified], ignore_index=True, sort=False)
    unified = _format_results_dataframe(unified, _unified_columns(), sort_cols=None)
    scenario_order = pd.CategoricalDtype(["Baseline", "Adversarial"], ordered=True)
    unified["Scenario"] = unified["Scenario"].astype(scenario_order)
    unified = unified.sort_values(["Scenario", "Attack", "Model"]).reset_index(drop=True)
    unified["Scenario"] = unified["Scenario"].astype(str)
    return unified


def _write_metric_analysis_report(
    baseline_df: pd.DataFrame,
    adversarial_df: pd.DataFrame,
    dataset_name: str,
    report_path: str,
):
    lines = [
        f"Dataset: {dataset_name}",
        "Analysis based on plotted summary metrics only:",
        "- accuracy",
        "- false_negative_rate",
        "- misclassification_rate",
        "- avg_confidence",
        "- confidence_degradation",
        "",
        "Baseline (per model):",
    ]

    for _, row in baseline_df.iterrows():
        lines.append(
            (
                f"- {row['Model']}: accuracy={row['accuracy']:.4f}, "
                f"fnr={row['false_negative_rate']:.4f}, "
                f"misclassification={row['misclassification_rate']:.4f}, "
                f"confidence={row['avg_confidence']:.4f}"
            )
        )

    lines.extend(["", "Adversarial impact (per attack/model):"])
    for _, row in adversarial_df.iterrows():
        lines.append(
            (
                f"- {row['Attack']} | {row['Model']}: accuracy={row['accuracy']:.4f}, "
                f"fnr={row['false_negative_rate']:.4f}, "
                f"misclassification={row['misclassification_rate']:.4f}, "
                f"confidence={row['avg_confidence']:.4f}, "
                f"confidence_degradation={row['confidence_degradation']:.4f}"
            )
        )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _build_core_services(*, trainer_cls, evaluator_cls, attacker_cls, random_state: int, results_dir: str):
    return (
        trainer_cls(random_state=random_state),
        evaluator_cls(output_dir=results_dir),
        attacker_cls(),
    )


def _cleanup_stale_files(directory: str, stale_files: list[str]):
    """Generic cleanup function to remove redundant output files."""
    for filename in stale_files:
        path = os.path.join(directory, filename)
        if os.path.exists(path):
            os.remove(path)


def _cleanup_redundant_adversarial_outputs(adversarial_dir: str):
    # Keep only combined plots to avoid duplicated per-attack artifacts.
    stale_files = [
        "adversarial_metrics_fgsm.png",
        "adversarial_metrics_pgd.png",
        "adversarial_confidence_fgsm.png",
        "adversarial_confidence_pgd.png",
        "degradation_comparison_fgsm.png",
        "degradation_comparison_pgd.png",
        "adversarial_metrics_confidence_combined.png",
    ]
    _cleanup_stale_files(adversarial_dir, stale_files)


def _cleanup_redundant_root_outputs(results_dir: str):
    _cleanup_stale_files(results_dir, ["attack_comparison.png"])


def _build_summary_image(results_dir: str, baseline_dir: str, adversarial_dir: str, dataset_name: str):
    montage_path = os.path.join(results_dir, "all_graphs_summary.png")
    image_paths = [
        os.path.join(baseline_dir, "baseline_metrics.png"),
        os.path.join(baseline_dir, "baseline_confidence.png"),
        os.path.join(results_dir, "comprehensive_comparison.png"),
        os.path.join(adversarial_dir, "adversarial_attacks_combined.png"),
        os.path.join(adversarial_dir, "adversarial_common_metrics_combined.png"),
        os.path.join(adversarial_dir, "adversarial_confidence_combined.png"),
    ]
    create_graph_montage(image_paths, montage_path, title=f"All Graphs Summary - {dataset_name}", ncols=2)


def _run_single_experiment(
    *,
    trainer,
    evaluator,
    attacker,
    X_train_use: np.ndarray,
    X_test_use: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    epsilon: float,
    attack_type: str,
    alpha: float,
    num_iter: int,
    perturbation_ratio: float,
    bounds,
    tag: str,
    random_state: int,
    include_node: bool = True,
):
    baseline_results = _run_baseline_phase(
        trainer,
        evaluator,
        X_train_use,
        X_test_use,
        y_train,
        y_test,
        random_state,
        tag,
        include_node,
    )
    tabnet_model = baseline_results["tabnet_model"]
    node_model = baseline_results["node_model"]
    tabnet_baseline_metrics = baseline_results["tabnet_baseline_metrics"]
    node_baseline_metrics = baseline_results["node_baseline_metrics"]

    attack_name = attack_type.upper()
    attack_eval = _run_adversarial_phase(
        attacker,
        trainer,
        evaluator,
        {
            "attack_type": attack_name,
            "epsilon": epsilon,
            "alpha": alpha,
            "num_iter": num_iter,
            "perturbation_ratio": perturbation_ratio,
        },
        tabnet_model,
        node_model,
        X_test_use,
        y_test,
        bounds,
        tabnet_baseline_metrics,
        node_baseline_metrics,
        tag,
        include_node,
    )

    return {
        "tabnet_model": tabnet_model,
        "node_model": node_model,
        "tabnet_baseline_metrics": tabnet_baseline_metrics,
        "node_baseline_metrics": node_baseline_metrics,
        "overfit_diagnostics": baseline_results["overfit_diagnostics"],
        "y_pred_tabnet_base": baseline_results["y_pred_tabnet_base"],
        "y_proba_tabnet_base": baseline_results["y_proba_tabnet_base"],
        "y_pred_node_base": baseline_results.get("y_pred_node_base"),
        "y_proba_node_base": baseline_results.get("y_proba_node_base"),
        "attack_eval": attack_eval,
    }


def run_full_pipeline(
    *,
    data_file: str,
    test_file: str | None = None,
    label_column: str,
    epsilon: float,
    attack_type: str,
    alpha: float,
    num_iter: int,
    random_state: int,
    dataset_name: str,
    preprocessor_cls: Type,
    selector_cls: Type,
    trainer_cls: Type,
    attacker_cls: Type,
    evaluator_cls: Type,
    internal_analyzer_cls: Type,
    failure_analyzer_cls: Type,
    use_lasso_prefilter: bool = False,
    lasso_max_features: int | None = None,
    max_rows: int | None = None,
    attack_configs: list[dict] | None = None,
    results_root: str = "results",
    results_subdir: str | None = None,
    overwrite_results: bool = False,
    timestamped_runs: bool = True,
    include_node: bool = True,
):
    set_random_seed(random_state)

    run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if results_subdir:
        results_dir = os.path.join(results_root, results_subdir)
    elif timestamped_runs:
        results_dir = os.path.join(results_root, run_id)
    else:
        results_dir = results_root

    if overwrite_results and os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    ensure_dir(results_root)
    baseline_dir = os.path.join(results_dir, "baseline")
    adversarial_dir = os.path.join(results_dir, "adversarial")
    comparison_dir = os.path.join(results_dir, "comparison")
    ensure_dir(baseline_dir)
    ensure_dir(adversarial_dir)
    ensure_dir(comparison_dir)

    print("\n" + "=" * 80)
    print(" " * max(0, (80 - len(dataset_name) - 18) // 2) + f"{dataset_name} EXPERIMENTAL PIPELINE")
    print("=" * 80)
    print(f"Dataset: {data_file}")
    print(f"Epsilon: {epsilon}")
    if attack_configs:
        attacks_label = ", ".join([cfg.get("attack_type", "PGD").upper() for cfg in attack_configs])
        print(f"Attacks: {attacks_label}")
    else:
        print(f"Attack: {attack_type.upper()} | Alpha: {alpha} | Iterations: {num_iter}")
    print(f"Random State: {random_state}")
    print("=" * 80 + "\n")
    if timestamped_runs and not results_subdir:
        print(f"Run ID: {run_id}")
    else:
        print("Run ID: latest")

    _print_step("STEP 1: DATA PREPROCESSING")
    data_dir = os.path.dirname(data_file) or "."
    data_name = os.path.basename(data_file)
    preprocessor = preprocessor_cls(data_dir=data_dir, random_state=random_state)
    # If an explicit test file path was passed, forward only the basename
    test_file_param = os.path.basename(test_file) if test_file else None
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline(
        data_name,
        label_column,
        use_lasso_prefilter=use_lasso_prefilter,
        lasso_max_features=lasso_max_features,
        max_rows=max_rows,
        test_file=test_file_param,
    )

    if use_lasso_prefilter:
        _print_step("STEP 2: LASSO PREFILTER + FEATURE SELECTION")
    else:
        _print_step("STEP 2: FEATURE SELECTION")
    selector = selector_cls(random_state=random_state)
    (X_train_full, X_test_full, _, _, _, _) = selector.feature_selection_pipeline(
        X_train, y_train, X_test, feature_names, top_k=10
    )
    _print_step("STEP 3-7: TRAIN/EVALUATE FULL FEATURES")
    trainer, evaluator, attacker = _build_core_services(
        trainer_cls=trainer_cls,
        evaluator_cls=evaluator_cls,
        attacker_cls=attacker_cls,
        random_state=random_state,
        results_dir=results_dir,
    )
    bounds_full = get_feature_bounds(X_train_full)

    normalized_attack_configs = _normalize_attack_configs(attack_configs, attack_type, epsilon, alpha, num_iter)
    preferred_primary = _select_primary_attack(normalized_attack_configs)

    full_results = _run_single_experiment(
        trainer=trainer,
        evaluator=evaluator,
        attacker=attacker,
        X_train_use=X_train_full,
        X_test_use=X_test_full,
        y_train=y_train,
        y_test=y_test,
        epsilon=preferred_primary["epsilon"],
        attack_type=preferred_primary["attack_type"],
        alpha=preferred_primary["alpha"],
        num_iter=preferred_primary["num_iter"],
        perturbation_ratio=preferred_primary["perturbation_ratio"],
        bounds=bounds_full,
        tag="Full",
        random_state=random_state,
        include_node=include_node,
    )
    tabnet_model = full_results["tabnet_model"]
    node_model = full_results["node_model"]
    tabnet_baseline_metrics = full_results["tabnet_baseline_metrics"]
    node_baseline_metrics = full_results["node_baseline_metrics"]
    overfit_diagnostics = full_results["overfit_diagnostics"]
    primary_attack_eval = full_results["attack_eval"]

    attack_results = {
        preferred_primary["attack_type"]: primary_attack_eval
    }

    for cfg in normalized_attack_configs:
        attack_name = cfg["attack_type"]
        if attack_name == preferred_primary["attack_type"]:
            continue

        attack_results[attack_name] = _run_adversarial_phase(
            attacker,
            trainer,
            evaluator,
            cfg,
            tabnet_model,
            node_model,
            X_test_full,
            y_test,
            bounds_full,
            tabnet_baseline_metrics,
            node_baseline_metrics,
            "Full",
            include_node,
        )

    primary_attack_name = preferred_primary["attack_type"]
    selected_primary = attack_results[primary_attack_name]
    # Attach baseline predictions/probabilities so analyzers have full context
    # (these were returned from the single experiment)
    selected_primary["y_pred_tabnet_base"] = full_results.get("y_pred_tabnet_base")
    selected_primary["y_proba_tabnet_base"] = full_results.get("y_proba_tabnet_base")
    selected_primary["y_pred_node_base"] = full_results.get("y_pred_node_base")
    selected_primary["y_proba_node_base"] = full_results.get("y_proba_node_base")
    tabnet_adv_metrics = selected_primary["tabnet_adv_metrics"]
    node_adv_metrics = selected_primary["node_adv_metrics"]
    tabnet_adv_degradation = selected_primary["tabnet_adv_degradation"]
    node_adv_degradation = selected_primary["node_adv_degradation"]

    _print_step("STEP 8: INTERNAL BEHAVIOR & FAILURE ANALYSIS (RQ2)")
    internal_analyzer = internal_analyzer_cls(output_dir=results_dir)
    failure_analyzer = failure_analyzer_cls(output_dir=results_dir)

    tabnet_feature_importance_base, tabnet_feature_importance_adv, tabnet_importance_shift = internal_analyzer.compute_tabnet_feature_importance(
        tabnet_model, X_test_full, selected_primary["X_test_adv_tabnet"], feature_names
    )
    tabnet_decision_shift = internal_analyzer.compute_decision_boundary_shift(
        selected_primary["y_proba_tabnet_base"][:, 1], selected_primary["y_proba_tabnet_adv"][:, 1]
    )

    node_analysis_results = {}
    if include_node and node_model is not None:
        node_embeddings_base = internal_analyzer.extract_node_embeddings(node_model, X_test_full)
        node_embeddings_adv = internal_analyzer.extract_node_embeddings(node_model, selected_primary["X_test_adv_node"])
        node_activation_variance = internal_analyzer.compute_node_activation_variance(node_embeddings_base, node_embeddings_adv)
        node_decision_shift = internal_analyzer.compute_decision_boundary_shift(
            selected_primary["y_proba_node_base"][:, 1], selected_primary["y_proba_node_adv"][:, 1]
        )
        node_analysis_results = {
            "activation_variance": node_activation_variance,
            "decision_shift": node_decision_shift
        }

    tabnet_failure_df, tabnet_failure_stats = failure_analyzer.analyze_failures(
        X_test_full, y_test, selected_primary["y_pred_tabnet_base"], selected_primary["y_pred_tabnet_adv"],
        selected_primary["y_proba_tabnet_base"][:, 1], selected_primary["y_proba_tabnet_adv"][:, 1],
        feature_names, "TabNet", primary_attack_name
    )

    tabnet_vulnerable_analysis = failure_analyzer.identify_vulnerable_samples(
        selected_primary["y_proba_tabnet_base"][:, 1], selected_primary["y_proba_tabnet_adv"][:, 1],
        selected_primary["y_pred_tabnet_base"], selected_primary["y_pred_tabnet_adv"], threshold=0.2
    )

    node_failure_stats = None
    node_vulnerable_analysis = None
    if include_node and node_model is not None:
        node_failure_df, node_failure_stats = failure_analyzer.analyze_failures(
            X_test_full, y_test, selected_primary["y_pred_node_base"], selected_primary["y_pred_node_adv"],
            selected_primary["y_proba_node_base"][:, 1], selected_primary["y_proba_node_adv"][:, 1],
            feature_names, "NODE", primary_attack_name
        )
        node_vulnerable_analysis = failure_analyzer.identify_vulnerable_samples(
            selected_primary["y_proba_node_base"][:, 1], selected_primary["y_proba_node_adv"][:, 1],
            selected_primary["y_pred_node_base"], selected_primary["y_pred_node_adv"], threshold=0.2
        )

    if node_failure_stats is not None:
        failure_analyzer.compare_model_failures(tabnet_failure_stats, node_failure_stats)

    # ========== COMPREHENSIVE RQ2 ANALYSIS ==========
    internal_dir = adversarial_dir
    
    tabnet_corr_df, tabnet_corr_summary = internal_analyzer.compute_rq2_correlation_analysis(
        X_test_full, selected_primary["X_test_adv_tabnet"], 
        selected_primary["y_pred_tabnet_base"], selected_primary["y_pred_tabnet_adv"],
        selected_primary["y_proba_tabnet_base"][:, 1], selected_primary["y_proba_tabnet_adv"][:, 1],
        y_test, feature_names, tabnet_model, "TabNet"
    )
    if include_node and node_analysis_results.get("activation_variance"):
        node_corr_summary = internal_analyzer.compute_rq2_node_correlation_analysis(
            X_test_full, selected_primary["X_test_adv_node"],
            selected_primary["y_pred_node_base"], selected_primary["y_pred_node_adv"],
            selected_primary["y_proba_node_base"][:, 1], selected_primary["y_proba_node_adv"][:, 1],
            y_test, node_model, "NODE"
        )
    tabnet_internal_metrics = ["accuracy", "false_negative_rate", "misclassification_rate", "avg_confidence"]
    tabnet_internal_baseline = tabnet_baseline_metrics
    tabnet_internal_adversarial = selected_primary["tabnet_adv_metrics"]
    tabnet_feature_importance = pd.DataFrame({
        'feature': feature_names[:len(tabnet_feature_importance_base)],
        'baseline': tabnet_feature_importance_base['importance'].values[:len(feature_names)],
        'adversarial': tabnet_feature_importance_adv['importance'].values[:len(feature_names)],
    })
    tabnet_feature_importance['shift'] = np.abs(tabnet_feature_importance['baseline'] - tabnet_feature_importance['adversarial'])

    confidence_drop_tabnet = selected_primary["y_proba_tabnet_base"][:, 1] - selected_primary["y_proba_tabnet_adv"][:, 1]
    confidence_drop_node = None
    if include_node and selected_primary.get("y_proba_node_base") is not None:
        confidence_drop_node = selected_primary["y_proba_node_base"][:, 1] - selected_primary["y_proba_node_adv"][:, 1]

    verdict_rows = [{
        'Metric': 'TabNet Feature Shift',
        'Baseline': float(tabnet_feature_importance['baseline'].mean()),
        'Adversarial': float(tabnet_feature_importance['adversarial'].mean()),
        'Change': float(tabnet_feature_importance['shift'].mean()),
        'Failure_Link': float(tabnet_corr_summary['importance_shift_failure_correlation']),
    }, {
        'Metric': 'TabNet Confidence',
        'Baseline': float(selected_primary["y_proba_tabnet_base"][:, 1].mean()),
        'Adversarial': float(selected_primary["y_proba_tabnet_adv"][:, 1].mean()),
        'Change': float(np.mean(confidence_drop_tabnet)),
        'Failure_Link': float(tabnet_corr_summary['confidence_drop_failure_correlation']),
    }]

    if include_node and node_analysis_results.get("activation_variance"):
        verdict_rows.append({
            'Metric': 'NODE Confidence',
            'Baseline': float(selected_primary["y_proba_node_base"][:, 1].mean()),
            'Adversarial': float(selected_primary["y_proba_node_adv"][:, 1].mean()),
            'Change': float(np.mean(confidence_drop_node)) if confidence_drop_node is not None else 0.0,
            'Failure_Link': float(node_corr_summary.get('confidence_drop_failure_correlation', 0.0)),
        })
        verdict_rows.append({
            'Metric': 'NODE Behavior Deviation',
            'Baseline': float(node_baseline_metrics['avg_confidence']) if node_baseline_metrics else 0.0,
            'Adversarial': float(selected_primary["node_adv_metrics"]['avg_confidence']) if selected_primary.get("node_adv_metrics") else 0.0,
            'Change': float(node_baseline_metrics['avg_confidence'] - selected_primary["node_adv_metrics"]['avg_confidence']) if node_baseline_metrics and selected_primary.get("node_adv_metrics") else 0.0,
            'Failure_Link': float(node_corr_summary.get('confidence_drop_failure_correlation', 0.0)),
        })

    verdict_df = pd.DataFrame(verdict_rows)
    summary_plot_path = os.path.join(internal_dir, "internal_behavior_summary.png")
    plot_internal_behavior_summary(verdict_df, f"Internal Behavior Analysis - {dataset_name}", summary_plot_path)

    summary_report_path = os.path.join(internal_dir, "internal_behavior_report.txt")
    with open(summary_report_path, 'w') as f:
        f.write("INTERNAL BEHAVIOR ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"Attack: {primary_attack_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("Measurements used:\n")
        f.write("1. Feature importance shifts\n")
        f.write("2. Decision pattern changes\n")
        f.write("3. Confidence variation\n")
        f.write("4. Internal behavior deviations\n\n")
        f.write("Verdict:\n")
        for _, row in verdict_df.iterrows():
            f.write(
                f"- {row['Metric']}: baseline={row['Baseline']:.6f}, adversarial={row['Adversarial']:.6f}, "
                f"change={row['Change']:.6f}, failure_link={row['Failure_Link']:.6f}\n"
            )
        f.write("\nConclusion:\n")
        f.write("These internal behavior changes show how adversarial inputs alter attention, confidence, and model stability.\n")
        f.write("The failure link values indicate whether the behavior change is associated with detection errors.\n")

    baseline_plot_metrics = {"TabNet": tabnet_baseline_metrics}
    if include_node and node_baseline_metrics is not None:
        baseline_plot_metrics["NODE"] = node_baseline_metrics

    plot_metric_summary(
        baseline_plot_metrics,
        f"Baseline Metrics - {dataset_name}",
        os.path.join(baseline_dir, "baseline_metrics.png"),
    )
    plot_confidence_summary(
        baseline_plot_metrics,
        f"Baseline Confidence - {dataset_name}",
        os.path.join(baseline_dir, "baseline_confidence.png"),
    )

    _cleanup_redundant_adversarial_outputs(adversarial_dir)

    if include_node and node_baseline_metrics is not None and node_adv_metrics is not None and node_adv_degradation is not None:
        plot_comprehensive_comparison(
            tabnet_baseline_metrics,
            tabnet_adv_metrics,
            node_baseline_metrics,
            node_adv_metrics,
            tabnet_adv_degradation,
            node_adv_degradation,
            dataset_name,
            os.path.join(comparison_dir, "unified_rq1.png"),
        )

    attack_results_by_model = {
        "TabNet": {"Baseline": tabnet_baseline_metrics},
    }
    if include_node and node_baseline_metrics is not None:
        attack_results_by_model["NODE"] = {"Baseline": node_baseline_metrics}
    for attack_key, attack_data in attack_results.items():
        attack_results_by_model["TabNet"][attack_key] = attack_data["tabnet_adv_metrics"]
        if include_node and attack_data.get("node_adv_metrics") is not None:
            attack_results_by_model["NODE"][attack_key] = attack_data["node_adv_metrics"]
    _cleanup_redundant_root_outputs(results_dir)

    attacks_only_by_model = {
        model_name: {
            attack_key: metrics
            for attack_key, metrics in model_metrics.items()
            if attack_key != "Baseline"
        }
        for model_name, model_metrics in attack_results_by_model.items()
    }
    plot_attack_comparison(
        attacks_only_by_model,
        dataset_name,
        os.path.join(adversarial_dir, "adversarial_attacks_combined.png"),
    )
    plot_adversarial_common_metrics_combined(
        attacks_only_by_model,
        dataset_name,
        os.path.join(adversarial_dir, "adversarial_common_metrics_combined.png"),
    )
    plot_adversarial_confidence_combined(
        attacks_only_by_model,
        dataset_name,
        os.path.join(adversarial_dir, "adversarial_confidence_combined.png"),
    )
    _build_summary_image(results_dir, baseline_dir, adversarial_dir, dataset_name)

    # Save baseline/adversarial summaries as bar-chart plots with thesis RQ1 metrics only
    baseline_df = pd.DataFrame(_build_baseline_summary_rows(tabnet_baseline_metrics, node_baseline_metrics))
    baseline_df = _format_results_dataframe(baseline_df, _baseline_columns(), sort_cols=["Model"])
    baseline_plot_path = os.path.join(baseline_dir, "baseline_single.png")
    plot_summary_dataframe_bars(baseline_df, f"Baseline Summary - {dataset_name}", baseline_plot_path)

    adversarial_df = pd.DataFrame(_build_adversarial_summary_rows(attack_results, include_node=include_node))
    adversarial_df = _format_results_dataframe(adversarial_df, _adversarial_columns(), sort_cols=["Attack", "Model"])
    adversarial_plot_path = os.path.join(adversarial_dir, "adversarial_single.png")
    plot_summary_dataframe_bars(adversarial_df, f"Adversarial Summary - {dataset_name}", adversarial_plot_path)

    unified_df = _build_unified_results_dataframe(baseline_df, adversarial_df)
    unified_plot_path = os.path.join(comparison_dir, "unified_results.png")
    plot_summary_dataframe_bars(unified_df, f"Unified Results - {dataset_name}", unified_plot_path)
    create_graph_montage(
        [baseline_plot_path, adversarial_plot_path, unified_plot_path],
        os.path.join(comparison_dir, "unified_clear.png"),
        title=f"Clear Summary Plots - {dataset_name}",
        ncols=1,
    )
    analysis_report_path = os.path.join(results_dir, "analysis_summary.txt")
    _write_metric_analysis_report(baseline_df, adversarial_df, dataset_name, analysis_report_path)

    _print_step("STEP 10: FINALIZING RESULTS")
    print("Saved simplified outputs:")
    print(f"  {baseline_plot_path}")
    print(f"  {adversarial_plot_path}")
    print(f"  {unified_plot_path}")
    print(f"  {os.path.join(comparison_dir, 'unified_clear.png')}")
    print(f"  {analysis_report_path}")
    print("Unified plots regenerated in baseline/adversarial/comparison folders.")

    _print_step("STEP 9: SAVING ANALYSIS OUTPUTS")
    internal_dir = adversarial_dir

    # Save analysis outputs as plots/images instead of CSVs
    plot_feature_importance_shift(
        tabnet_feature_importance_base,
        tabnet_feature_importance_adv,
        f"TabNet Feature Importance Shift - {dataset_name}",
        os.path.join(internal_dir, "tabnet_feature_importance_comparison.png"),
    )
    if include_node and node_analysis_results is not None:
        node_activation_variance = node_analysis_results.get("activation_variance")
        node_decision_shift = node_analysis_results.get("decision_shift")
        if node_activation_variance:
            plot_rq2_summary(
                pd.DataFrame([node_activation_variance]),
                "NODE Activation Variance - Summary",
                os.path.join(internal_dir, "node_activation_variance.png"),
            )
        if node_decision_shift:
            plot_rq2_summary(
                pd.DataFrame([node_decision_shift]),
                "NODE Decision Boundary Shift - Summary",
                os.path.join(internal_dir, "node_decision_shift.png"),
            )
        if node_failure_stats:
            plot_failure_characteristics(
                node_failure_stats,
                "NODE Failure Characteristics",
                os.path.join(internal_dir, "node_failure_characteristics.png"),
            )

    print(f"\nAnalysis outputs saved to: {internal_dir}/")
    print(f"  - Internal behavior summary plot")
    print(f"  - Internal behavior verdict report")

    # Cleanup memory before final summary
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print("\n" + "=" * 80)
    print(" " * 25 + "PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {results_dir}/")
    print(f"Baseline plots: {baseline_dir}/")
    print(f"Adversarial plots: {adversarial_dir}/")
    print(f"Comparison outputs: {comparison_dir}/")
    print("\n" + "=" * 80 + "\n")