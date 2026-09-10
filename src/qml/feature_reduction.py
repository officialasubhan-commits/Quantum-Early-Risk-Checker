import os
from typing import Any
import joblib
import numpy as np
from sklearn.decomposition import PCA

def reduce_features_for_qml(
    X_train: Any,
    X_val: Any,
    X_test: Any,
    n_components: int = 6,
    models_dir: str = "models"
) -> tuple:
    """
    Applies PCA feature reduction from 21 classical features down to n_components (6) quantum features.
    Strictly fits the PCA reducer ONLY on X_train to avoid data leakage.
    """
    os.makedirs(models_dir, exist_ok=True)
    
    print(f"\n[QML FEATURE REDUCTION] Fitting PCA (21 -> {n_components} quantum features) strictly on Train set...", flush=True)
    pca = PCA(n_components=n_components, random_state=42)
    
    X_train_q = pca.fit_transform(X_train)
    X_val_q = pca.transform(X_val)
    X_test_q = pca.transform(X_test)
    
    explained_variance = float(sum(pca.explained_variance_ratio_))
    print(f" -> PCA Cumulative Explained Variance ({n_components} components): {explained_variance:.2%}", flush=True)
    
    reducer_path = os.path.join(models_dir, "qml_pca_reducer.joblib")
    joblib.dump(pca, reducer_path)
    print(f" -> Saved QML Feature Reducer artifact: {reducer_path}", flush=True)
    
    return X_train_q, X_val_q, X_test_q, pca
