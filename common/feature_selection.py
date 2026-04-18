import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import VarianceThreshold
from typing import Tuple, List
import os

from common.utils import set_random_seed


class UNSW_FeatureSelector:
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.selected_features = None
        self.feature_importance = None
        self.top_k_indices = None
        
        set_random_seed(random_state)
    
    def remove_low_variance(
        self, 
        X: np.ndarray, 
        feature_names: List[str],
        threshold: float = 0.01
    ) -> Tuple[np.ndarray, List[str]]:
        print(f"Removing low variance features (threshold={threshold})...")
        
        selector = VarianceThreshold(threshold=threshold)
        X_filtered = selector.fit_transform(X)
        
        mask = selector.get_support()
        filtered_features = [f for f, m in zip(feature_names, mask) if m]
        
        removed = len(feature_names) - len(filtered_features)
        print(f"Removed {removed} low variance features")
        print(f"Remaining features: {len(filtered_features)}")
        
        return X_filtered, filtered_features
    
    def remove_correlated_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
        threshold: float = 0.90
    ) -> Tuple[np.ndarray, List[str]]:
        print(f"Removing highly correlated features (threshold={threshold})...")
        
        df = pd.DataFrame(X, columns=feature_names)
        corr_matrix = df.corr().abs()
        
        upper_triangle = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        high_corr = (corr_matrix.where(upper_triangle) > threshold)
        
        to_drop = set()
        for column in high_corr.columns:
            if high_corr[column].any():
                to_drop.add(column)
        
        keep_features = [f for f in feature_names if f not in to_drop]
        keep_indices = [i for i, f in enumerate(feature_names) if f not in to_drop]
        
        X_filtered = X[:, keep_indices]
        
        removed = len(to_drop)
        print(f"Removed {removed} highly correlated features")
        print(f"Remaining features: {len(keep_features)}")
        
        return X_filtered, keep_features
    
    def compute_feature_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        method: str = "lasso"
    ) -> pd.DataFrame:
        if method not in {"lasso", "lasso_logistic"}:
            raise ValueError(f"Unsupported importance method: {method}")

        print("Computing feature importance using L1-regularized logistic regression...")

        lasso_model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            random_state=self.random_state,
            max_iter=5000,
            C=1.0,
            class_weight="balanced"
        )

        lasso_model.fit(X, y)

        if lasso_model.coef_.ndim == 2:
            coefficients = np.abs(lasso_model.coef_[0])
        else:
            coefficients = np.abs(lasso_model.coef_)

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': coefficients,
            'coefficient': lasso_model.coef_[0] if lasso_model.coef_.ndim == 2 else lasso_model.coef_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance_df
        
        print("Top 10 most important features:")
        print(importance_df.head(10).to_string(index=False))
        
        return importance_df
    
    def select_top_k_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
        importance_df: pd.DataFrame,
        k: int = 10
    ) -> Tuple[np.ndarray, List[str], List[int]]:
        print(f"\nSelecting top {k} features...")
        
        top_features = importance_df.head(k)['feature'].tolist()
        top_indices = [feature_names.index(f) for f in top_features]
        
        X_topk = X[:, top_indices]
        
        self.selected_features = top_features
        self.top_k_indices = top_indices
        
        print(f"Selected features: {top_features}")
        
        return X_topk, top_features, top_indices
    
    def feature_selection_pipeline(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        top_k: int = 10,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.90,
        importance_method: str = "lasso"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], pd.DataFrame]:
        print("="*80)
        print("UNSW-NB15 FEATURE SELECTION PIPELINE")
        print("="*80)
        print(f"Initial features: {len(feature_names)}")
        
        X_train_filt, features_filt = self.remove_low_variance(
            X_train, feature_names, variance_threshold
        )
        X_test_filt = X_test[:, [feature_names.index(f) for f in features_filt]]
        
        X_train_filt, features_filt = self.remove_correlated_features(
            X_train_filt, features_filt, correlation_threshold
        )
        X_test_filt = X_test[:, [feature_names.index(f) for f in features_filt]]
        
        importance_df = self.compute_feature_importance(
            X_train_filt, y_train, features_filt, method=importance_method
        )
        
        X_train_topk, topk_features, topk_indices_filt = self.select_top_k_features(
            X_train_filt, features_filt, importance_df, top_k
        )
        X_test_topk = X_test_filt[:, topk_indices_filt]
        
        print("\n" + "="*80)
        print("FEATURE SELECTION COMPLETE")
        print("="*80)
        print(f"Full feature set: {X_train_filt.shape[1]} features")
        print(f"Top-{top_k} feature set: {X_train_topk.shape[1]} features")
        
        return (X_train_filt, X_test_filt, X_train_topk, X_test_topk, 
                topk_features, importance_df)
    
    def save_feature_selection_results(self, output_dir: str, importance_df: pd.DataFrame):
        os.makedirs(output_dir, exist_ok=True)
        
        importance_path = os.path.join(output_dir, 'feature_importance.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"Saved feature importance to: {importance_path}")