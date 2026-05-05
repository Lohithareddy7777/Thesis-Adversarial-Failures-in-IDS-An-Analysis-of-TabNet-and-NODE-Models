import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from typing import Tuple
from pathlib import Path
import os
import logging

from common.utils import set_random_seed
logger = logging.getLogger(__name__)


class UNSW_NB15_Preprocessor:
    
    def __init__(self, data_dir: str = "data", random_state: int = 42):
        self.data_dir = data_dir
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.lasso_selector = None
        
        set_random_seed(random_state)

    def _read_csv(self, path: str) -> pd.DataFrame:
        for encoding in ("utf-8", "latin-1", "cp1252"):
            try:
                return pd.read_csv(path, skipinitialspace=True, low_memory=False, encoding=encoding)
            except UnicodeDecodeError:
                continue
        return pd.read_csv(path, skipinitialspace=True, low_memory=False, encoding_errors="replace")
    
    def load_data(self, file_pattern: str = "UNSW_NB15_training-set.csv") -> pd.DataFrame:
        possible_paths = [
            os.path.join(self.data_dir, file_pattern),
            os.path.join("../datasets", file_pattern),
            os.path.join("../datasets/archive", file_pattern)
        ]
        
        for path in possible_paths:
            if os.path.isfile(path):
                logger.info(f"Loading data from: {path}")
                df = self._read_csv(path)
                df.columns = df.columns.str.strip()
                logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features")
                return df

        directory_candidates = [
            os.path.join(self.data_dir, file_pattern),
            self.data_dir,
        ]

        for directory in directory_candidates:
            if not os.path.isdir(directory):
                continue

            csv_files = sorted(Path(directory).glob("*.csv"))
            if not csv_files:
                csv_files = sorted(Path(directory).rglob("*.csv"))

            if csv_files:
                logger.info(f"Loading data from directory: {directory}")
                frames = [self._read_csv(str(csv_file)) for csv_file in csv_files]
                for frame in frames:
                    frame.columns = frame.columns.str.strip()
                df = pd.concat(frames, ignore_index=True, sort=False)
                logger.info(f"Loaded {len(df)} samples with {len(df.columns)} features from {len(csv_files)} CSV files")
                return df
        
        raise FileNotFoundError(f"Could not find dataset file: {file_pattern}")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info(f"Initial shape: {df.shape}")
        
        id_cols = [col for col in df.columns if 'id' in col.lower()]
        if id_cols:
            logger.info(f"Removing identifier columns: {id_cols}")
            df = df.drop(columns=id_cols)
        # Replace infinite values with NaN so they are handled below
        df = df.replace([np.inf, -np.inf], np.nan)

        missing_before = df.isnull().sum().sum()
        if missing_before > 0:
            logger.info(f"Handling {missing_before} missing values")
            
            num_cols = df.select_dtypes(include=[np.number]).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            
            cat_cols = df.select_dtypes(include=['object']).columns
            for col in cat_cols:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        duplicates_before = df.duplicated().sum()
        if duplicates_before > 0:
            logger.info(f"Removing {duplicates_before} duplicate rows")
            df = df.drop_duplicates()
        
        # After filling missing values, ensure numeric columns contain finite values
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            col_vals = df[col]
            if not np.all(np.isfinite(col_vals)) or np.nanmax(np.abs(col_vals)) > 1e12:
                median = np.nanmedian(col_vals)
                if np.isnan(median):
                    median = 0.0
                mask = ~np.isfinite(col_vals) | (np.abs(col_vals) > 1e12)
                df.loc[mask, col] = median

        logger.info(f"Final shape after cleaning: {df.shape}")
        return df
    
    def encode_categorical(self, df: pd.DataFrame, fit: bool = True, log: bool = True) -> pd.DataFrame:
        cat_cols = df.select_dtypes(include=['object']).columns
        
        if len(cat_cols) > 0:
            if log:
                logger.info(f"Encoding categorical columns: {list(cat_cols)}")
            
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
        logger.info("Applying Lasso prefilter...")
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

        logger.info(f"Lasso prefilter kept {len(selected_features)} / {len(feature_names)} features")
        return X_train_sel, X_test_sel, selected_features
    
    def prepare_train_test_split(
        self, 
        df: pd.DataFrame, 
        label_column: str = 'label',
        test_size: float = 0.2,
        max_rows: int | None = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        possible_labels = [label_column, 'Label', 'attack_cat', 'Label_binary']
        label_col = None
        
        for col in possible_labels:
            if col in df.columns:
                label_col = col
                break
        
        if label_col is None:
            raise ValueError(f"Could not find label column. Available columns: {df.columns.tolist()}")
        
        logger.info(f"Using label column: {label_col}")

        if max_rows is not None and len(df) > max_rows:
            logger.info(f"Sampling down to {max_rows} rows for faster training while preserving class balance")
            df, _ = train_test_split(
                df,
                train_size=max_rows,
                random_state=self.random_state,
                stratify=df[label_col],
            )
        
        X = df.drop(columns=[label_col]).copy()
        y = df[label_col].copy()

        # Remove target-leakage proxy columns when a different label is selected.
        leakage_candidates = {'label', 'Label', 'attack_cat', 'Label_binary'}
        leakage_candidates.discard(label_col)
        leakage_cols = [col for col in X.columns if col in leakage_candidates]
        if leakage_cols:
            logger.info(f"Removing leakage-prone label proxy columns: {leakage_cols}")
            X = X.drop(columns=leakage_cols)
        
        if not is_numeric_dtype(y):
            # Treat common normal labels as background/0 for string/categorical labels.
            normal_tokens = {"normal", "benign"}
            y_lower = y.astype(str).str.strip().str.lower()
            y = (~y_lower.isin(normal_tokens)).astype(int)
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
        
        logger.info(f"Train set: {X_train.shape[0]} samples")
        logger.info(f"Test set: {X_test.shape[0]} samples")
        logger.info(f"Attack rate in train: {np.mean(y_train):.2%}")
        logger.info(f"Attack rate in test: {np.mean(y_test):.2%}")
        
        return X_train, X_test, y_train, y_test, self.feature_names
    
    def preprocess_pipeline(
        self,
        file_pattern: str = "UNSW_NB15_training-set.csv",
        label_column: str = 'label',
        use_lasso_prefilter: bool = False,
        lasso_max_features: int | None = None,
        max_rows: int | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
        logger.info("="*80)
        logger.info("UNSW-NB15 PREPROCESSING PIPELINE")
        logger.info("="*80)
        
        df = self.load_data(file_pattern)
        
        df = self.clean_data(df)
        
        X_train, X_test, y_train, y_test, feature_names = self.prepare_train_test_split(
            df, label_column, max_rows=max_rows
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
        
        logger.info("Preprocessing complete!")
        logger.info("="*80)
        
        return X_train, X_test, y_train, y_test, feature_names
