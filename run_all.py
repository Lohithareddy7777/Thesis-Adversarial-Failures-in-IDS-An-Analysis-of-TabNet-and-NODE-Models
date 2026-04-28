from __future__ import annotations

import os

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
        {"attack_type": "BRPA", "epsilon": 0.1, "perturbation_ratio": 0.1},
    ],
    "use_lasso_prefilter": True,
}

DATASET1 = {
    **DATASET_BASE,
    "name": "UNSW-NB15",
    "data_file": "dataset1/data/UNSW_NB15_training-set.csv",
    "results_subdir": "dataset1",
}

DATASET2 = {
    **DATASET_BASE,
    "name": "WUSTL_HDRL_2024",
    "data_file": "dataset2/data/wustl_hdrl_2024.csv",
    "results_subdir": "dataset2",
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
    keep_ext = {".png", ".jpg", ".jpeg", ".svg", ".webp", ".csv"}
    for root, _, files in os.walk(results_dir):
        for name in files:
            ext = os.path.splitext(name)[1].lower()
            if ext not in keep_ext:
                os.remove(os.path.join(root, name))

    for root, dirs, _ in os.walk(results_dir, topdown=False):
        for d in dirs:
            path = os.path.join(root, d)
            if not os.listdir(path):
                os.rmdir(path)


def _run_dataset(
    config: dict,
    use_lasso_prefilter: bool,
) -> None:
    results_subdir = config["results_subdir"]
    primary_attack = config["attack_configs"][0]

    run_full_pipeline(
        dataset_name=config["name"],
        data_file=config["data_file"],
        label_column=config["label_column"],
        epsilon=primary_attack.get("epsilon", config["epsilon"]),
        attack_type=primary_attack.get("attack_type", config["attack_type"]),
        alpha=primary_attack.get("alpha", config["alpha"]),
        num_iter=primary_attack.get("num_iter", config["num_iter"]),
        random_state=config["random_state"],
        use_lasso_prefilter=use_lasso_prefilter,
        lasso_max_features=config.get("lasso_max_features"),
        attack_configs=config.get("attack_configs"),
        results_root="results",
        results_subdir=results_subdir,
        overwrite_results=True,
        timestamped_runs=False,
        **PIPELINE_COMPONENTS,
    )
    _keep_only_plots(os.path.join("results", results_subdir))


def run_all() -> None:
    print("\n" + "=" * 80)
    print(" " * 22 + "THESIS EVALUATION - SHARED PIPELINE")
    print("=" * 80 + "\n")

    for config in [DATASET1, DATASET2]:
        attacks_label = ", ".join(cfg["attack_type"] for cfg in config.get("attack_configs", []))
        print(f"▶ {config['name']} | LASSO + {attacks_label}")
        _run_dataset(
            config,
            use_lasso_prefilter=config["use_lasso_prefilter"],
        )

    print("\n" + "=" * 80)
    print(" " * 30 + "ALL RUNS COMPLETE")
    print("Results: results/dataset1 and results/dataset2 (baseline/adversarial plots only)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all()