import os
import sys
from typing import Any
import json
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.qml.vqc_model import VariationalQuantumClassifier

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "heart_disease")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "heart_disease")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "heart_disease")

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

def train_heart_qml_model(n_qubits: int = 6, n_layers: int = 2, epochs: int = 40, lr: float = 0.08, seed: int = 42):
    print("=" * 70, flush=True)
    print(" [QML TRAINING] Training 6-Qubit VQC for Heart Disease Detection... ", flush=True)
    print("=" * 70, flush=True)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    train_df = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))

    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))

    X_train_proc = preprocessor.transform(train_df[FEATURE_COLS])
    y_train = train_df["target"].values

    X_val_proc = preprocessor.transform(val_df[FEATURE_COLS])
    y_val: Any = val_df["target"].values

    # Fit PCA Reducer (13 -> 6 features)
    pca = PCA(n_components=n_qubits, random_state=seed)
    X_train_q = pca.fit_transform(X_train_proc)
    X_val_q = pca.transform(X_val_proc)

    explained_var_ratio = float(sum(pca.explained_variance_ratio_))
    print(f" -> Fitted PCA Reducer (13 -> {n_qubits} Qubits). Total Variance Retained: {explained_var_ratio*100:.2f}%", flush=True)

    pca_path = os.path.join(MODELS_DIR, "qml_pca_reducer.joblib")
    joblib.dump(pca, pca_path)
    print(f" -> Saved QML PCA Reducer to: {pca_path}", flush=True)

    # Initialize 6-Qubit VQC
    vqc = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers, seed=seed)
    
    start_time = time.time()
    vqc.fit(X_train_q, np.asarray(y_train), epochs=epochs, lr=lr, batch_size=32)
    training_time = time.time() - start_time

    val_probs: Any = vqc.predict_proba(X_val_q)
    val_preds: Any = (val_probs >= 0.5).astype(int)

    acc = float(accuracy_score(y_val, val_preds))
    prec = float(precision_score(y_val, val_preds, zero_division="warn"))
    rec = float(recall_score(y_val, val_preds, zero_division="warn"))
    f1 = float(f1_score(y_val, val_preds, zero_division="warn"))
    auc = float(roc_auc_score(y_val, val_probs))
    cm = confusion_matrix(y_val, val_preds)
    tn, fp, fn, tp = map(int, cm.ravel())
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    print(f" -> 6-Qubit VQC Validation Metrics | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Rec: {rec:.4f} | Spec: {spec:.4f}", flush=True)

    # Save QML Parameters
    params_path = os.path.join(MODELS_DIR, "qml_vqc_params.npz")
    np.savez_compressed(
        params_path,
        weights_ry=vqc.weights_ry,
        weights_rz=vqc.weights_rz,
        bias=vqc.bias,
        scale=vqc.scale,
        n_qubits=n_qubits,
        n_layers=n_layers
    )
    print(f" -> Saved Trained QML Parameters to: {params_path}", flush=True)

    report = {
        "disease_id": "heart_disease",
        "model_name": "6-Qubit Variational Quantum Classifier (QML)",
        "n_qubits": n_qubits,
        "n_layers": n_layers,
        "pca_variance_retained": explained_var_ratio,
        "training_time_sec": training_time,
        "validation_metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall_sensitivity": rec,
            "specificity": spec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
        }
    }

    report_path = os.path.join(REPORTS_DIR, "qml_experiment_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f" -> Saved QML Experiment Report to: {report_path}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    train_heart_qml_model()
