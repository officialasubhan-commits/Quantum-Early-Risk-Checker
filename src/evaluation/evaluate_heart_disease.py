import os
import sys
import json
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

def evaluate_heart_models():
    print("=" * 70, flush=True)
    print(" [HELD-OUT TEST EVALUATION] Evaluating Heart Disease Models on Test Set (46 Samples)... ", flush=True)
    print("=" * 70, flush=True)

    test_df = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))

    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.joblib"))
    classical_model = joblib.load(os.path.join(MODELS_DIR, "best_classical_model.joblib"))
    pca_reducer = joblib.load(os.path.join(MODELS_DIR, "qml_pca_reducer.joblib"))
    hybrid_model = joblib.load(os.path.join(MODELS_DIR, "hybrid_fusion_model.joblib"))

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

    X_test_proc = preprocessor.transform(test_df[FEATURE_COLS])
    X_test_q = pca_reducer.transform(X_test_proc)
    y_test = test_df["target"].values

    def compute_metrics(y_true, y_probs, y_preds):
        acc = float(accuracy_score(y_true, y_preds))
        prec = float(precision_score(y_true, y_preds, zero_division="warn"))
        rec = float(recall_score(y_true, y_preds, zero_division="warn"))
        f1 = float(f1_score(y_true, y_preds, zero_division="warn"))
        auc = float(roc_auc_score(y_true, y_probs))
        cm = confusion_matrix(y_true, y_preds)
        tn, fp, fn, tp = map(int, cm.ravel())
        spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        return {
            "accuracy": acc,
            "precision": prec,
            "recall_sensitivity": rec,
            "specificity": spec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
        }

    # 1. Classical RF
    c_probs = classical_model.predict_proba(X_test_proc)[:, 1]
    c_preds = (c_probs >= 0.5).astype(int)
    c_metrics = compute_metrics(y_test, c_probs, c_preds)

    # 2. QML VQC
    q_probs = vqc.predict_proba(X_test_q)
    q_preds = (q_probs >= 0.5).astype(int)
    q_metrics = compute_metrics(y_test, q_probs, q_preds)

    # 3. Hybrid Ensemble
    h_probs = hybrid_model.predict_proba(X_test_proc)
    h_preds = hybrid_model.predict(X_test_proc)
    h_metrics = compute_metrics(y_test, h_probs, h_preds)

    print(f" -> Classical RF  | Acc: {c_metrics['accuracy']:.4f} | F1: {c_metrics['f1_score']:.4f} | AUC: {c_metrics['roc_auc']:.4f} | Rec: {c_metrics['recall_sensitivity']:.4f} | Spec: {c_metrics['specificity']:.4f}", flush=True)
    print(f" -> 6-Qubit VQC   | Acc: {q_metrics['accuracy']:.4f} | F1: {q_metrics['f1_score']:.4f} | AUC: {q_metrics['roc_auc']:.4f} | Rec: {q_metrics['recall_sensitivity']:.4f} | Spec: {q_metrics['specificity']:.4f}", flush=True)
    print(f" -> Hybrid Model  | Acc: {h_metrics['accuracy']:.4f} | F1: {h_metrics['f1_score']:.4f} | AUC: {h_metrics['roc_auc']:.4f} | Rec: {h_metrics['recall_sensitivity']:.4f} | Spec: {h_metrics['specificity']:.4f}", flush=True)

    summary_report = {
        "disease_id": "heart_disease",
        "disease_name": "Cardiovascular / Heart Disease",
        "test_sample_count": len(y_test),
        "classical_rf": c_metrics,
        "quantum_vqc": q_metrics,
        "hybrid_ensemble": h_metrics
    }

    report_path = os.path.join(REPORTS_DIR, "heart_vs_all_comparison.json")
    with open(report_path, "w") as f:
        json.dump(summary_report, f, indent=4)
    print(f" -> Saved Heart Disease Comparison Report to: {report_path}", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    evaluate_heart_models()
