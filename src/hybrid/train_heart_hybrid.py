import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.qml.vqc_model import VariationalQuantumClassifier
from src.hybrid.fusion_model import HybridEnsembleClassifier

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "heart_disease")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "heart_disease")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports", "heart_disease")

FEATURE_COLS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", 
    "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

def train_heart_hybrid_model():
    print("=" * 70, flush=True)
    print(" [HYBRID FUSION] Training Classical-Quantum Ensemble for Heart Disease... ", flush=True)
    print("=" * 70, flush=True)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    val_df = pd.read_csv(os.path.join(PROCESSED_DIR, "val.csv"))

    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    classical_model = joblib.load(os.path.join(MODELS_DIR, "best_classical_model.joblib"))
    pca_reducer = joblib.load(os.path.join(MODELS_DIR, "qml_pca_reducer.joblib"))

    qml_params = np.load(os.path.join(MODELS_DIR, "qml_vqc_params.npz"))
    vqc = VariationalQuantumClassifier(
        n_qubits=int(qml_params["n_qubits"]),
        n_layers=int(qml_params["n_layers"]),
        seed=42
    )
    vqc.weights_ry = qml_params["weights_ry"]
    vqc.weights_rz = qml_params["weights_rz"]
    vqc.bias = float(qml_params["bias"])
    vqc.scale = float(qml_params["scale"])

    X_val_proc = preprocessor.transform(val_df[FEATURE_COLS])
    y_val = val_df["target"].values

    start_time = time.time()
    hybrid_ensemble = HybridEnsembleClassifier(classical_model=classical_model, qml_model=vqc, pca_reducer=pca_reducer)
    hybrid_ensemble.fit_fusion(X_val_proc, y_val)
    training_time = time.time() - start_time

    val_probs = hybrid_ensemble.predict_proba(X_val_proc)
    val_preds = hybrid_ensemble.predict(X_val_proc)

    acc = float(accuracy_score(y_val, val_preds))
    prec = float(precision_score(y_val, val_preds, zero_division=0))
    rec = float(recall_score(y_val, val_preds, zero_division=0))
    f1 = float(f1_score(y_val, val_preds, zero_division=0))
    auc = float(roc_auc_score(y_val, val_probs))
    cm = confusion_matrix(y_val, val_preds)
    tn, fp, fn, tp = map(int, cm.ravel())
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    print(f" -> Hybrid Validation Metrics | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Rec: {rec:.4f} | Spec: {spec:.4f}", flush=True)
    w_class = hybrid_ensemble.optimal_weight
    w_quant = 1.0 - hybrid_ensemble.optimal_weight
    print(f" -> Optimal Blend Weights: {w_class:.3f} Classical + {w_quant:.3f} Quantum", flush=True)
    print(f" -> Optimal Decision Threshold: {hybrid_ensemble.optimal_threshold:.3f}", flush=True)

    hybrid_path = os.path.join(MODELS_DIR, "hybrid_fusion_model.joblib")
    joblib.dump(hybrid_ensemble, hybrid_path)
    print(f" -> Saved Trained Hybrid Fusion Model to: {hybrid_path}", flush=True)

    report = {
        "disease_id": "heart_disease",
        "model_name": "Hybrid Classical-Quantum Ensemble Classifier",
        "optimal_classical_weight": w_class,
        "optimal_quantum_weight": w_quant,
        "optimal_threshold": hybrid_ensemble.optimal_threshold,
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

    report_path = os.path.join(REPORTS_DIR, "hybrid_experiment_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f" -> Saved Hybrid Experiment Report to: {report_path}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    train_heart_hybrid_model()
