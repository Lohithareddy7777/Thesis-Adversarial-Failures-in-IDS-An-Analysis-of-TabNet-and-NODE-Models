from __future__ import annotations

import os
import shutil
from datetime import datetime
from typing import Type

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common.utils import set_random_seed, get_feature_bounds, ensure_dir
from common.visualization import (
    plot_metric_summary,
    plot_confidence_summary,
    plot_comprehensive_comparison,
    plot_attack_comparison,
    plot_adversarial_common_metrics_combined,
    plot_adversarial_metrics_confidence_combined,
    plot_adversarial_confidence_combined,
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
    for candidate in ["PGD", "FGSM", "BRPA"]:
        for cfg in normalized_attack_configs:
            if cfg["attack_type"] == candidate:
                return cfg
    return normalized_attack_configs[0]


def _generate_adv_inputs(attacker, attack_cfg: dict, model_tabnet, model_node, X_test: np.ndarray, y_test: np.ndarray, bounds):
    attack_name = attack_cfg["attack_type"]
    if attack_name == "BRPA":
        X_test_adv_tabnet = attacker.bounded_perturbation_attack(
            X_test,
            epsilon=attack_cfg["epsilon"],
            perturbation_ratio=attack_cfg["perturbation_ratio"],
            clip_bounds=bounds,
        )
        return X_test_adv_tabnet, X_test_adv_tabnet.copy()

    return (
        attacker.generate_attack_tabnet(
            attack_name,
            model_tabnet,
            X_test,
            y_test,
            epsilon=attack_cfg["epsilon"],
            alpha=attack_cfg["alpha"],
            num_iter=attack_cfg["num_iter"],
            clip_bounds=bounds,
        ),
        attacker.generate_attack(
            attack_name,
            model_node,
            X_test,
            y_test,
            epsilon=attack_cfg["epsilon"],
            alpha=attack_cfg["alpha"],
            num_iter=attack_cfg["num_iter"],
            clip_bounds=bounds,
        ),
    )


def _run_baseline_phase(trainer, evaluator, X_train_use: np.ndarray, X_test_use: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, random_state: int, tag: str):
    X_tr, X_val, y_tr, y_val = _split_train_val(X_train_use, y_train, random_state=random_state)
    tabnet_model = trainer.train_tabnet(X_tr, y_tr, X_val, y_val, num_classes=2, max_epochs=40, patience=8)
    node_model = trainer.train_node(X_tr, y_tr, X_val, y_val, num_classes=2, max_epochs=40, patience=8)

    y_pred_tabnet_base, y_proba_tabnet_base = trainer.predict_tabnet(tabnet_model, X_test_use)
    tabnet_baseline_metrics = evaluator.evaluate_baseline(y_test, y_pred_tabnet_base, y_proba_tabnet_base, f"TabNet-{tag}")
    y_pred_tabnet_train, _ = trainer.predict_tabnet(tabnet_model, X_train_use)

    y_pred_node_base, y_proba_node_base = trainer.predict_node(node_model, X_test_use)
    node_baseline_metrics = evaluator.evaluate_baseline(y_test, y_pred_node_base, y_proba_node_base, f"NODE-{tag}")
    y_pred_node_train, _ = trainer.predict_node(node_model, X_train_use)

    tabnet_train_acc = float(np.mean(y_pred_tabnet_train == y_train))
    node_train_acc = float(np.mean(y_pred_node_train == y_train))
    overfit_diagnostics = {
        "TabNet": {
            "train_accuracy": tabnet_train_acc,
            "test_accuracy": float(tabnet_baseline_metrics["accuracy"]),
            "generalization_gap": tabnet_train_acc - float(tabnet_baseline_metrics["accuracy"]),
        },
        "NODE": {
            "train_accuracy": node_train_acc,
            "test_accuracy": float(node_baseline_metrics["accuracy"]),
            "generalization_gap": node_train_acc - float(node_baseline_metrics["accuracy"]),
        },
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
    node_baseline_metrics: dict,
    attack_name: str,
    tag: str,
):
    y_pred_tabnet_adv, y_proba_tabnet_adv = trainer.predict_tabnet(tabnet_model, X_test_adv_tabnet)
    tabnet_adv_metrics = evaluator.evaluate_adversarial(y_test, y_pred_tabnet_adv, y_proba_tabnet_adv, f"TabNet-{tag}", attack_name)
    y_pred_node_adv, y_proba_node_adv = trainer.predict_node(node_model, X_test_adv_node)
    node_adv_metrics = evaluator.evaluate_adversarial(y_test, y_pred_node_adv, y_proba_node_adv, f"NODE-{tag}", attack_name)
    tabnet_adv_degradation = evaluator.compute_degradation(tabnet_baseline_metrics, tabnet_adv_metrics, f"TabNet-{tag}", attack_name)
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
    node_baseline_metrics: dict,
    tag: str,
):
    X_test_adv_tabnet, X_test_adv_node = _generate_adv_inputs(
        attacker,
        attack_cfg,
        tabnet_model,
        node_model,
        X_test,
        y_test,
        bounds,
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
    )


def _build_baseline_summary_rows(tabnet_baseline_metrics: dict, node_baseline_metrics: dict):
    return [
        {"Model": "TabNet", **tabnet_baseline_metrics},
        {"Model": "NODE", **node_baseline_metrics},
    ]


def _build_adversarial_summary_rows(attack_results: dict):
    rows = []
    for attack_key, attack_data in attack_results.items():
        rows.append({"Attack": attack_key, "Model": "TabNet", **attack_data["tabnet_adv_metrics"]})
        rows.append({"Attack": attack_key, "Model": "NODE", **attack_data["node_adv_metrics"]})
    return rows


def _format_results_dataframe(df: pd.DataFrame, ordered_columns: list[str], sort_cols: list[str] | None = None) -> pd.DataFrame:
    missing = [col for col in ordered_columns if col not in df.columns]
    if missing:
        for col in missing:
            df[col] = np.nan

    formatted = df[ordered_columns].copy()

    if "Attack" in formatted.columns:
        attack_order = pd.CategoricalDtype(["FGSM", "PGD", "BRPA"], ordered=True)
        formatted["Attack"] = formatted["Attack"].astype(attack_order)
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
        "precision",
        "recall",
        "f1_score",
        "false_negative_rate",
        "false_positive_rate",
        "misclassification_rate",
        "avg_confidence",
        "std_confidence",
        "train_accuracy",
        "generalization_gap",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
    ]


def _adversarial_columns() -> list[str]:
    return [
        "Attack",
        "Model",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "false_negative_rate",
        "false_positive_rate",
        "misclassification_rate",
        "avg_confidence",
        "std_confidence",
        "true_positives",
        "true_negatives",
        "false_positives",
        "false_negatives",
    ]


def _build_core_services(*, trainer_cls, evaluator_cls, attacker_cls, random_state: int, results_dir: str):
    return (
        trainer_cls(random_state=random_state),
        evaluator_cls(output_dir=results_dir),
        attacker_cls(),
    )


def _cleanup_redundant_adversarial_outputs(adversarial_dir: str):
    # Keep only combined plots to avoid duplicated per-attack artifacts.
    stale_files = [
        "adversarial_metrics_fgsm.png",
        "adversarial_metrics_pgd.png",
        "adversarial_metrics_brpa.png",
        "adversarial_confidence_fgsm.png",
        "adversarial_confidence_pgd.png",
        "adversarial_confidence_brpa.png",
        "degradation_comparison_fgsm.png",
        "degradation_comparison_pgd.png",
        "degradation_comparison_brpa.png",
    ]
    for filename in stale_files:
        path = os.path.join(adversarial_dir, filename)
        if os.path.exists(path):
            os.remove(path)


def _cleanup_redundant_root_outputs(results_dir: str):
    stale_files = [
        "attack_comparison.png",
    ]
    for filename in stale_files:
        path = os.path.join(results_dir, filename)
        if os.path.exists(path):
            os.remove(path)


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
    )

    return {
        "tabnet_model": tabnet_model,
        "node_model": node_model,
        "tabnet_baseline_metrics": tabnet_baseline_metrics,
        "node_baseline_metrics": node_baseline_metrics,
        "overfit_diagnostics": baseline_results["overfit_diagnostics"],
        "attack_eval": attack_eval,
    }


def run_full_pipeline(
    *,
    data_file: str,
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
    attack_configs: list[dict] | None = None,
    results_root: str = "results",
    results_subdir: str | None = None,
    overwrite_results: bool = False,
    timestamped_runs: bool = True,
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
    ensure_dir(baseline_dir)
    ensure_dir(adversarial_dir)

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
    X_train, X_test, y_train, y_test, feature_names = preprocessor.preprocess_pipeline(
        data_name,
        label_column,
        use_lasso_prefilter=use_lasso_prefilter,
        lasso_max_features=lasso_max_features,
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
        )

    primary_attack_name = preferred_primary["attack_type"]
    selected_primary = attack_results[primary_attack_name]
    tabnet_adv_metrics = selected_primary["tabnet_adv_metrics"]
    node_adv_metrics = selected_primary["node_adv_metrics"]
    tabnet_adv_degradation = selected_primary["tabnet_adv_degradation"]
    node_adv_degradation = selected_primary["node_adv_degradation"]

    plot_metric_summary(
        {"TabNet": tabnet_baseline_metrics, "NODE": node_baseline_metrics},
        f"Baseline Metrics - {dataset_name}",
        os.path.join(baseline_dir, "baseline_metrics.png"),
    )
    plot_confidence_summary(
        {"TabNet": tabnet_baseline_metrics, "NODE": node_baseline_metrics},
        f"Baseline Confidence - {dataset_name}",
        os.path.join(baseline_dir, "baseline_confidence.png"),
    )

    _cleanup_redundant_adversarial_outputs(adversarial_dir)

    plot_comprehensive_comparison(
        tabnet_baseline_metrics,
        tabnet_adv_metrics,
        node_baseline_metrics,
        node_adv_metrics,
        tabnet_adv_degradation,
        node_adv_degradation,
        dataset_name,
        os.path.join(results_dir, "comprehensive_comparison.png"),
    )

    attack_results_by_model = {
        "TabNet": {"Baseline": tabnet_baseline_metrics},
        "NODE": {"Baseline": node_baseline_metrics},
    }
    for attack_key, attack_data in attack_results.items():
        attack_results_by_model["TabNet"][attack_key] = attack_data["tabnet_adv_metrics"]
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
    plot_adversarial_metrics_confidence_combined(
        attacks_only_by_model,
        dataset_name,
        os.path.join(adversarial_dir, "adversarial_metrics_confidence_combined.png"),
    )
    plot_adversarial_confidence_combined(
        attacks_only_by_model,
        dataset_name,
        os.path.join(adversarial_dir, "adversarial_confidence_combined.png"),
    )

    # Keep final outputs intentionally simple: one baseline CSV and one adversarial CSV.
    baseline_summary_path = os.path.join(baseline_dir, "baseline_single.csv")
    baseline_df = pd.DataFrame(_build_baseline_summary_rows(tabnet_baseline_metrics, node_baseline_metrics))
    baseline_df["train_accuracy"] = baseline_df["Model"].map(lambda m: overfit_diagnostics[m]["train_accuracy"])
    baseline_df["generalization_gap"] = baseline_df["Model"].map(lambda m: overfit_diagnostics[m]["generalization_gap"])
    baseline_df = _format_results_dataframe(baseline_df, _baseline_columns(), sort_cols=["Model"])
    baseline_df.to_csv(baseline_summary_path, index=False)

    adversarial_summary_path = os.path.join(adversarial_dir, "adversarial_single.csv")
    adversarial_df = pd.DataFrame(_build_adversarial_summary_rows(attack_results))
    adversarial_df = _format_results_dataframe(adversarial_df, _adversarial_columns(), sort_cols=["Attack", "Model"])
    adversarial_df.to_csv(adversarial_summary_path, index=False)

    _print_step("STEP 10: FINALIZING RESULTS")
    print("Saved simplified outputs:")
    print(f"  {os.path.join(baseline_dir, 'baseline_single.csv')}")
    print(f"  {os.path.join(adversarial_dir, 'adversarial_single.csv')}")
    print("Plots regenerated in baseline/adversarial and dataset root folders.")

    print("\n" + "=" * 80)
    print(" " * 25 + "PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"\nAll results saved to: {results_dir}/")
    print(f"Baseline plots: {baseline_dir}/")
    print(f"Adversarial plots: {adversarial_dir}/")
    print("\n" + "=" * 80 + "\n")