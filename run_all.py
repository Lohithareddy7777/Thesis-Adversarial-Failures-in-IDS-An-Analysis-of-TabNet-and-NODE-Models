from __future__ import annotations

import os
import gc
import torch

from common.preprocessing import UNSW_NB15_Preprocessor
from common.feature_selection import UNSW_FeatureSelector
from common.model_training import ModelTrainer
from common.adversarial_core import AdversarialAttacker
from common.evaluation import ModelEvaluator
from common.internal_analysis import InternalBehaviorAnalyzer
from common.failure_analysis import FailureAnalyzer
from common.pipeline_runner import run_full_pipeline

DATASET_BASE = {
    "label_column": "label",
    "random_state": 42,
    "lasso_max_features": 20,
    "attack_type": "PGD",
    "epsilon": 0.01,
    "alpha": 0.005,
    "num_iter": 3,
    "attack_configs": [
        {"attack_type": "FGSM", "epsilon": 0.1},
        {"attack_type": "PGD", "epsilon": 0.1, "alpha": 0.01, "num_iter": 5},
    ],
    "use_lasso_prefilter": True,
}

DATASET1 = {
    **DATASET_BASE,
    "name": "UNSW-NB15",
    "data_file": "dataset1/data/UNSW_NB15_training-set.csv",
    "test_file": "dataset1/data/UNSW_NB15_testing-set.csv",
    "results_subdir": "dataset1",
}

DATASET2 = {
    **DATASET_BASE,
    "name": "WUSTL_HDRL_2024",
    "data_file": "dataset2/data/wustl_hdrl_2024.csv",
    "results_subdir": "dataset2",
}

DATASET3 = {
    **DATASET_BASE,
    "name": "MachineLearningCVE",
    "label_column": "Label",
    "data_file": "dataset3/data/MachineLearningCVE",
    "results_subdir": "dataset3",
    "max_rows": 200000,
}

DATASET4 = {
    **DATASET_BASE,
    "name": "Real_network_data_attacks",
    "label_column": "Label",
    "data_file": "dataset4/data/Real_network_data_attacks_labeled.csv",
    "results_subdir": "dataset4",
}

PIPELINE_COMPONENTS = {
    "preprocessor_cls": UNSW_NB15_Preprocessor,
    "selector_cls": UNSW_FeatureSelector,
    "trainer_cls": ModelTrainer,
    "attacker_cls": AdversarialAttacker,
    "evaluator_cls": ModelEvaluator,
    "internal_analyzer_cls": InternalBehaviorAnalyzer,
    "failure_analyzer_cls": FailureAnalyzer,
}


def _keep_only_plots(results_dir: str) -> None:
    keep_relpaths = {
        os.path.join("baseline", "baseline_single.png"),
        os.path.join("adversarial", "adversarial_single.png"),
        os.path.join("adversarial", "internal_behavior_report.txt"),
        os.path.join("adversarial", "internal_behavior_summary.png"),
        os.path.join("adversarial", "tabnet_feature_importance_comparison.png"),
        os.path.join("adversarial", "node_activation_variance.png"),
        os.path.join("adversarial", "node_decision_shift.png"),
        os.path.join("adversarial", "node_failure_characteristics.png"),
        os.path.join("comparison", "unified_results.png"),
        os.path.join("comparison", "unified_clear.png"),
        os.path.join("comparison", "unified_rq1.png"),
        "analysis_summary.txt",
    }

    for root, _, files in os.walk(results_dir):
        for name in files:
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, results_dir)
            if rel_path not in keep_relpaths:
                os.remove(abs_path)

    for root, dirs, _ in os.walk(results_dir, topdown=False):
        for d in dirs:
            path = os.path.join(root, d)
            if not os.listdir(path):
                os.rmdir(path)


def _run_dataset(
    config: dict,
    use_lasso_prefilter: bool,
    include_node: bool,
) -> None:
    results_subdir = config["results_subdir"]
    primary_attack = config["attack_configs"][0]

    run_full_pipeline(
        dataset_name=config["name"],
        data_file=config["data_file"],
        test_file=config.get("test_file"),
        label_column=config["label_column"],
        epsilon=primary_attack.get("epsilon", config["epsilon"]),
        attack_type=primary_attack.get("attack_type", config["attack_type"]),
        alpha=primary_attack.get("alpha", config["alpha"]),
        num_iter=primary_attack.get("num_iter", config["num_iter"]),
        random_state=config["random_state"],
        use_lasso_prefilter=use_lasso_prefilter,
        lasso_max_features=config.get("lasso_max_features"),
        max_rows=config.get("max_rows"),
        attack_configs=config.get("attack_configs"),
        results_root="results",
        results_subdir=results_subdir,
        overwrite_results=True,
        timestamped_runs=False,
        include_node=include_node,
        **PIPELINE_COMPONENTS,
    )
    _keep_only_plots(os.path.join("results", results_subdir))
    
    # Cleanup memory after each dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_all(include_node: bool = True) -> None:
    print("\n" + "=" * 80)
    print(" " * 22 + "THESIS EVALUATION - SHARED PIPELINE")
    print("=" * 80 + "\n")
    print(f"Model mode: {'TabNet + NODE' if include_node else 'TabNet only (thesis default)'}\n")

    for config in [DATASET1, DATASET2, DATASET3]:
        attacks_label = ", ".join(cfg["attack_type"] for cfg in config.get("attack_configs", []))
        model_label = "TabNet+NODE" if include_node else "TabNet"
        print(f"▶ {config['name']} | {model_label} | LASSO + {attacks_label}")
        _run_dataset(
            config,
            use_lasso_prefilter=config["use_lasso_prefilter"],
            include_node=include_node,
        )

    print("\n" + "=" * 80)
    print(" " * 30 + "ALL RUNS COMPLETE")
    print("Results: results/dataset1, results/dataset2, results/dataset3 (baseline/adversarial/comparison only)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run thesis pipelines")
    parser.add_argument("--include-node", dest="include_node", action="store_true", help="Include NODE alongside TabNet")
    parser.add_argument("--no-node", dest="include_node", action="store_false", help="Run TabNet only (no NODE)")
    parser.set_defaults(include_node=True)
    args = parser.parse_args()

    # Env var overrides CLI if present (useful for automated runs)
    env_val = os.getenv("THESIS_INCLUDE_NODE")
    if env_val is not None:
        include_node = env_val.lower() not in ("0", "false", "no")
    else:
        include_node = args.include_node

    run_all(include_node=include_node)