import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

def preprocess_and_split(
    df: pd.DataFrame,
    target_col: str = "Diabetes_binary",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    processed_dir: str = "data/processed"
):
    """
    Performs leak-free preprocessing and stratified splitting:
    1. Separate features (X) and target (y).
    2. Perform stratified split: Train (70%), Validation (15%), Test (15%).
    3. Fit StandardScaler & SimpleImputer strictly on training split.
    4. Transform Train, Validation, and Test splits.
    5. Save processed numpy arrays to data/processed/.
    """
    os.makedirs(processed_dir, exist_ok=True)
    
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataframe.")
        
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    
    feature_names = list(X.columns)
    
    continuous_cols = [col for col in ["BMI", "MentHlth", "PhysHlth"] if col in feature_names]
    passthrough_cols = [col for col in feature_names if col not in continuous_cols]
    
    print(f"\n[PREPROCESSING] Feature Separation: {len(feature_names)} features total", flush=True)
    print(f" -> Continuous features (StandardScaler): {continuous_cols}", flush=True)
    print(f" -> Binary/Ordinal features (Passthrough): {len(passthrough_cols)} features", flush=True)
    
    # 1. First Split: Train+Val (85%) and Test (15%)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    # 2. Second Split: Train (70%) and Validation (15% overall)
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=relative_val_size,
        stratify=y_train_val,
        random_state=random_state
    )
    
    print("\n[PREPROCESSING] Dataset Splitting Summary (Stratified):", flush=True)
    print(f" -> Train Set:      {X_train.shape[0]:,} samples ({len(X_train)/len(df):.1%}) | Positive Class Ratio: {y_train.mean():.2%}", flush=True)
    print(f" -> Validation Set: {X_val.shape[0]:,} samples ({len(X_val)/len(df):.1%}) | Positive Class Ratio: {y_val.mean():.2%}", flush=True)
    print(f" -> Test Set:       {X_test.shape[0]:,} samples ({len(X_test)/len(df):.1%}) | Positive Class Ratio: {y_test.mean():.2%}", flush=True)
    
    # 3. Create Preprocessing Pipeline
    continuous_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    passthrough_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", continuous_transformer, continuous_cols),
            ("cat", passthrough_transformer, passthrough_cols)
        ],
        remainder="passthrough"
    )
    
    # 4. Strict Leak-Free Fitting: FIT ONLY ON X_TRAIN
    print("[PREPROCESSING] Fitting feature scalers/imputers STRICTLY on Training Set...", flush=True)
    X_train_proc = preprocessor.fit_transform(X_train)
    
    # TRANSFORM Validation and Test sets using fitted preprocessor
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    all_processed_features = continuous_cols + passthrough_cols
    
    # Save processed splits
    np.savez_compressed(
        os.path.join(processed_dir, "train_data.npz"),
        X=X_train_proc, y=y_train.values
    )
    np.savez_compressed(
        os.path.join(processed_dir, "val_data.npz"),
        X=X_val_proc, y=y_val.values
    )
    np.savez_compressed(
        os.path.join(processed_dir, "test_data.npz"),
        X=X_test_proc, y=y_test.values
    )
    
    print(f"[PREPROCESSING] Processed splits saved to: {processed_dir}/", flush=True)
    
    return {
        "X_train": X_train_proc,
        "y_train": y_train.values,
        "X_val": X_val_proc,
        "y_val": y_val.values,
        "X_test": X_test_proc,
        "y_test": y_test.values,
        "preprocessor": preprocessor,
        "feature_names": all_processed_features,
        "raw_splits": (X_train, X_val, X_test, y_train, y_val, y_test)
    }
