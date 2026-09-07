import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_val_score

def get_classical_models(random_state: int = 42) -> dict:
    """
    Instantiates the classical ML model classifiers configured with class_weight='balanced'
    to handle class imbalance and fixed random seed for reproducibility.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=random_state
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state
        ),
        "Support Vector Machine": CalibratedClassifierCV(
            estimator=LinearSVC(
                class_weight="balanced",
                max_iter=2000,
                random_state=random_state,
                dual=False
            ),
            cv=3
        )
    }
    return models

def train_and_cross_validate(models: dict, X_train: np.ndarray, y_train: np.ndarray, cv_splits: int = 5):
    """
    Performs 5-fold stratified cross-validation on the training set to evaluate model stability.
    """
    print(f"\n--- Running {cv_splits}-Fold Stratified Cross-Validation on Training Data ---", flush=True)
    cv_results = {}
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"[CROSS-VALIDATION] Evaluating {name}...", flush=True)
        start_time = time.time()
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring="f1_macro", n_jobs=-1)
        duration = time.time() - start_time
        
        cv_results[name] = {
            "mean_f1_macro": float(np.mean(scores)),
            "std_f1_macro": float(np.std(scores)),
            "fold_scores": [float(s) for s in scores],
            "cv_duration_sec": float(duration)
        }
        print(f" -> {name} Macro F1: {np.mean(scores):.4f} (±{np.std(scores):.4f}) | CV Time: {duration:.2f}s", flush=True)
        
    return cv_results
