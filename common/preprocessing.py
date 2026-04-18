import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from typing import Tuple
import os

from common.utils import set_random_seed


class UNSW_NB15_Preprocessor:
    
    def __init__(self, data_dir: str = "data", random_state: int = 42):
        self.data_dir = data_dir
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.lasso_selector = None
        
        set_random_seed(random_state)
    
    def load_data(self, file_pattern: str = "UNSW_NB15_training-set.csv") -> pd.DataFrame:
        possible_paths = [
            os.path.join(self.data_dir, file_pattern),
            os.path.join("../datasets", file_pattern),
            os.path.join("../datasets/archive", file_pattern)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Loading data from: {path}")
                df = pd.read_csv(path)
                print(f"Loaded {len(df)} samples with {len(df.columns)} features")
                return df
        
        raise FileNotFoundError(f"Could not find dataset file: {file_pattern}")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        print(f"Initial shape: {df.shape}")
        
        id_cols = [col for col in df.columns if 'id' in col.lower()]
        if id_cols:
            print(f"Removing identifier columns: {id_cols}")
            df = df.drop(columns=id_cols)
        
        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            print(f"Handling {missing_before} missing values")
            
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            
            cat_cols = df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        duplicates_before = df.duplicated().sum()
        if duplicates_before > 0:
            print(f"Removing {duplicates_before} duplicate rows")
            df = df.drop_duplicates()
        
        print(f"Final shape after cleaning: {df.shape}")
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True, log: bool = True) -> pd.DataFrame:
        cat_cols = df.select_dtypes(include=['object']).columns
        
        if len(cat_cols) > 0:
            if log:
                print(f"Encoding categorical columns: {list(cat_cols)}")
            
            for col in cat_cols:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        df[col] = df[col].astype(str)
                        known_categories = set(self.label_encoders[col].classes_)
                        fallback = self.label_encoders[col].classes_[0]
                        df[col] = df[col].apply(lambda x: x if x in known_categories else fallback)
                        df[col] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def normalize_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled

    def apply_lasso_prefilter(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: list,
        max_features: int | None = None,
        C: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, list]:
        print("Applying Lasso prefilter...")
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            random_state=self.random_state,
            max_iter=5000,
            C=C,
            class_weight="balanced",
        )
        selector = SelectFromModel(model, prefit=False, max_features=max_features)
        selector.fit(X_train, y_train)

        X_train_sel = selector.transform(X_train)
        X_test_sel = selector.transform(X_test)
        mask = selector.get_support()
        selected_features = [f for f, keep in zip(feature_names, mask) if keep]
        self.lasso_selector = selector

        print(f"Lasso prefilter kept {len(selected_features)} / {len(feature_names)} features")
        return X_train_sel, X_test_sel, selected_features
    
    def prepare_train_test_split(
        self, 
        df: pd.DataFrame, 
        label_column: str = 'label',
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        possible_labels = [label_column, 'Label', 'attack_cat', 'Label_binary']
        label_col = None
        
        for col in possible_labels:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            raise ValueError(f"Could not find label column. Available columns: {df.columns.tolist()}")
        
        print(f"Using label column: {label_col}")
        
        X = df.drop(columns=[label_col]).copy()
        y = df[label_col].copy()

        # Remove target-leakage proxy columns when a different label is selected.
        leakage_candidates = {'label', 'Label', 'attack_cat', 'Label_binary'}
        leakage_candidates.discard(label_col)
        leakage_cols = [col for col in X.columns if col in leakage_candidates]
        if leakage_cols:
            print(f"Removing leakage-prone label proxy columns: {leakage_cols}")
            X = X.drop(columns=leakage_cols)
        
        if y.dtype == 'object':
            y = (y != 'normal').astype(int)
        elif len(np.unique(y)) > 2:
            y = (y != 0).astype(int)
        
        X_train_df, X_test_df, y_train, y_test = train_test_split(
            X, y.values,
            test_size=test_size, 
            random_state=self.random_state,
            stratify=y
        )

        X_train_df = self.encode_categorical(X_train_df.copy(), fit=True, log=True)
        X_test_df = self.encode_categorical(X_test_df.copy(), fit=False, log=False)

        self.feature_names = X_train_df.columns.tolist()
        X_train = X_train_df.values
        X_test = X_test_df.values
        
        print(f"Train set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Attack rate in train: {np.mean(y_train):.2%}")
        print(f"Attack rate in test: {np.mean(y_test):.2%}")
        
        return X_train, X_test, y_train, y_test, self.feature_names
    
    def preprocess_pipeline(
        self,
        file_pattern: str = "UNSW_NB15_training-set.csv",
        label_column: str = 'label',
        use_lasso_prefilter: bool = False,
        lasso_max_features: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        print("="*80)
        print("UNSW-NB15 PREPROCESSING PIPELINE")
        print("="*80)
        
        df = self.load_data(file_pattern)
        
        df = self.clean_data(df)
        
        X_train, X_test, y_train, y_test, feature_names = self.prepare_train_test_split(
            df, label_column
        )
        
        X_train = self.normalize_features(X_train, fit=True)
        X_test = self.normalize_features(X_test, fit=False)

        if use_lasso_prefilter:
            X_train, X_test, feature_names = self.apply_lasso_prefilter(
                X_train,
                y_train,
                X_test,
                feature_names,
                max_features=lasso_max_features,
            )
        
        print("\nPreprocessing complete!")
        print("="*80)
        
        return X_train, X_test, y_train, y_test, feature_names