import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "heart_disease")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "heart_disease")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "heart_disease")

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

def train_heart_classical_models():
    print("=" * 70, flush=True)
    print(" [CLASSICAL ML] Training & Benchmarking Heart Disease Models... ", flush=True)
    print("=" * 70, flush=True)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))

    X_train_raw = train_df[FEATURE_COLS]
    y_train = train_df["target"].to_numpy()
    X_val_raw = val_df[FEATURE_COLS]
    y_val = val_df["target"].to_numpy()

    # Fit Imputer + StandardScaler preprocessor pipeline
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_val_proc = preprocessor.transform(X_val_raw)

    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    joblib.dump(preprocessor, preprocessor_path)
    print(f" -> Fitted & Saved Preprocessor (Imputer + StandardScaler) to: {preprocessor_path}", flush=True)

    # Benchmark candidates
    candidates = {
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
        "SupportVectorMachine": SVC(probability=True, random_state=42, C=1.0, kernel="rbf"),  # type: ignore[arg-type]
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
    }

    benchmark_results = {}
    best_model_name = None
    best_model_obj = None
    best_val_auc = -1.0

    for name, clf in candidates.items():
        clf.fit(X_train_proc, y_train)
        val_probs = clf.predict_proba(X_val_proc)[:, 1]
        val_preds = (val_probs >= 0.5).astype(int)

        acc = float(accuracy_score(y_val, val_preds))
        prec = float(precision_score(y_val, val_preds, zero_division="warn"))
        rec = float(recall_score(y_val, val_preds, zero_division="warn"))
        f1 = float(f1_score(y_val, val_preds, zero_division="warn"))
        auc = float(roc_auc_score(y_val, val_probs))
        cm = confusion_matrix(y_val, val_preds)
        tn, fp, fn, tp = map(int, cm.ravel())
        spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        benchmark_results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall_sensitivity": rec,
            "specificity": spec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
        }

        print(f" -> Candidate {name:28s} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Rec: {rec:.4f} | Spec: {spec:.4f}", flush=True)

        if auc > best_val_auc:
            best_val_auc = auc
            best_model_name = name
            best_model_obj = clf

    print("-" * 70, flush=True)
    print(f" -> Best Performing Model on Validation Set: {best_model_name} (ROC-AUC: {best_val_auc:.4f})", flush=True)

    best_model_path = os.path.join(MODELS_DIR, "best_classical_model.joblib")
    joblib.dump(best_model_obj, best_model_path)
    print(f" -> Saved Best Classical Model to: {best_model_path}", flush=True)

    report = {
        "disease_id": "heart_disease",
        "selected_best_model": best_model_name,
        "best_model_path": best_model_path,
        "validation_benchmark_comparison": benchmark_results
    }

    report_path = os.path.join(REPORTS_DIR, "classical_ml_experiment_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f" -> Saved Classical ML Experiment Report to: {report_path}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    train_heart_classical_models()
