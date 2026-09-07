import os
import time
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from src.qml.vqc_model import VariationalQuantumClassifier
from src.hybrid.fusion_model import HybridEnsembleClassifier

def run_hybrid_experiment():
    print("=" * 70, flush=True)
    print(" Phase 5: Hybrid Classical-Quantum Model Experiment ", flush=True)
    print("=" * 70, flush=True)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    
    # 1. Load Preprocessed Datasets (Same splits)
    train_data = np.load("data/processed/train_data.npz")
    val_data   = np.load("data/processed/val_data.npz")
    test_data  = np.load("data/processed/test_data.npz")
    
    X_train, y_train = train_data["X"], train_data["y"]
    X_val, y_val     = val_data["X"], val_data["y"]
    X_test, y_test   = test_data["X"], test_data["y"]
    
    print(f"\n[DATASET VERIFICATION] Dataset Splits Loaded:", flush=True)
    print(f" -> Train Set:      {len(X_train):,} samples", flush=True)
    print(f" -> Validation Set: {len(X_val):,} samples (Used for Hybrid Fusion Tuning)", flush=True)
    print(f" -> Test Set:       {len(X_test):,} samples [UNTOUCHED HELD-OUT EVALUATION]", flush=True)
    
    # 2. Load Pre-trained Artifacts
    print("\n[ARTIFACT LOADING] Loading Classical Baseline & QML Model Artifacts...", flush=True)
    classical_model = joblib.load("models/best_classical_model.joblib")
    pca_reducer = joblib.load("models/qml_pca_reducer.joblib")
    
    qml_params = np.load("models/qml_vqc_params.npz")
    vqc = VariationalQuantumClassifier(n_qubits=6, n_layers=2, seed=42)
    vqc.weights_ry = qml_params["weights_ry"]
    vqc.weights_rz = qml_params["weights_rz"]
    vqc.bias = float(qml_params["bias"])
    vqc.scale = float(qml_params["scale"])
    
    print(" -> Classical Model (Random Forest): Loaded", flush=True)
    print(" -> QML Feature Reducer (PCA 21->6): Loaded", flush=True)
    print(" -> QML Model (6-Qubit VQC):         Loaded", flush=True)
    
    # 3. Instantiate & Train Hybrid Fusion Classifier (Fitted strictly on Validation Set)
    hybrid_model = HybridEnsembleClassifier(classical_model, vqc, pca_reducer)
    
    start_fusion_train = time.time()
    hybrid_model.fit_fusion(X_val, y_val)
    fusion_train_duration = time.time() - start_fusion_train
    
    # Save trained hybrid fusion model
    joblib.dump(hybrid_model, "models/hybrid_fusion_model.joblib")
    print(f" -> Saved Trained Hybrid Fusion Model: models/hybrid_fusion_model.joblib", flush=True)
    
    # 4. Evaluate ALL THREE Models on the EXACT SAME Held-Out Test Set (38,052 samples)
    print(f"\n[HYBRID EVALUATION] Evaluating Classical, Quantum & Hybrid on 38,052 Test Set...", flush=True)
    
    # A. Classical Baseline Evaluation
    start_rf_inf = time.time()
    P_rf_test = classical_model.predict_proba(X_test)[:, 1]
    y_pred_rf = (P_rf_test >= 0.5).astype(int)
    rf_inf_time = time.time() - start_rf_inf
    
    # B. Standalone QML Evaluation
    X_test_q = pca_reducer.transform(X_test)
    start_qml_inf = time.time()
    P_qml_test = vqc.predict_proba(X_test_q)
    y_pred_qml = (P_qml_test >= 0.5).astype(int)
    qml_inf_time = time.time() - start_qml_inf
    
    # C. Hybrid Model Evaluation
    start_hybrid_inf = time.time()
    P_hybrid_test = hybrid_model.predict_proba(X_test)
    y_pred_hybrid = hybrid_model.predict(X_test)
    hybrid_inf_time = time.time() - start_hybrid_inf
    
    # Helper metric calculator
    def compute_all_metrics(name, y_true, y_pred, y_prob, train_t, inf_t):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_m = f1_score(y_true, y_pred, average="macro", zero_division=0)
        auc = roc_auc_score(y_true, y_prob)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        return {
            "model_name": name,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall_sensitivity": float(rec),
            "specificity": float(spec),
            "f1_score": float(f1),
            "f1_macro": float(f1_m),
            "roc_auc": float(auc),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)},
            "training_time_sec": float(train_t),
            "inference_time_sec": float(inf_t),
            "inference_ms_per_sample": float((inf_t / len(y_true)) * 1000.0)
        }

    rf_eval = compute_all_metrics("Random Forest (Classical)", y_test, y_pred_rf, P_rf_test, 3.197, rf_inf_time)
    qml_eval = compute_all_metrics("Variational Quantum Classifier (QML)", y_test, y_pred_qml, P_qml_test, 62.15, qml_inf_time)
    hybrid_eval = compute_all_metrics("Hybrid Classical-Quantum Ensemble", y_test, y_pred_hybrid, P_hybrid_test, fusion_train_duration, hybrid_inf_time)
    
    print("\n" + "=" * 70, flush=True)
    print(" EMPIRICAL TEST SET EVALUATION COMPARISON (38,052 SAMPLES) ", flush=True)
    print("=" * 70, flush=True)
    print(f"Classical RF  -> Accuracy: {rf_eval['accuracy']:.4f} | F1: {rf_eval['f1_score']:.4f} | AUC: {rf_eval['roc_auc']:.4f} | Rec: {rf_eval['recall_sensitivity']:.4f} | Spec: {rf_eval['specificity']:.4f}", flush=True)
    print(f"Quantum VQC   -> Accuracy: {qml_eval['accuracy']:.4f} | F1: {qml_eval['f1_score']:.4f} | AUC: {qml_eval['roc_auc']:.4f} | Rec: {qml_eval['recall_sensitivity']:.4f} | Spec: {qml_eval['specificity']:.4f}", flush=True)
    print(f"Hybrid Model  -> Accuracy: {hybrid_eval['accuracy']:.4f} | F1: {hybrid_eval['f1_score']:.4f} | AUC: {hybrid_eval['roc_auc']:.4f} | Rec: {hybrid_eval['recall_sensitivity']:.4f} | Spec: {hybrid_eval['specificity']:.4f}", flush=True)
    print("=" * 70, flush=True)
    
    # 5. Export JSON Reports
    full_comparison = {
        "classical_rf": rf_eval,
        "quantum_vqc": qml_eval,
        "hybrid_ensemble": hybrid_eval,
        "experiment_metadata": {
            "dataset_version": "CDC BRFSS 2015",
            "total_records": 253680,
            "classical_features": 21,
            "pca_quantum_features": 6,
            "num_qubits": 6,
            "circuit_layers": 2,
            "circuit_params": 26,
            "fusion_method": f"Optimal Blended Probability ({hybrid_model.optimal_weight:.3f} Class + {1.0-hybrid_model.optimal_weight:.3f} Quant) + Logistic Meta-Classifier",
            "optimal_threshold": hybrid_model.optimal_threshold,
            "test_samples": len(y_test),
            "random_seed": 42
        }
    }
    
    with open("reports/hybrid_experiment_report.json", "w") as f:
        json.dump(hybrid_eval, f, indent=4)
        
    with open("reports/hybrid_vs_all_comparison.json", "w") as f:
        json.dump(full_comparison, f, indent=4)
        
    with open("experiments/hybrid_experiment_report.json", "w") as f:
        json.dump(hybrid_eval, f, indent=4)
        
    with open("experiments/hybrid_vs_all_comparison.json", "w") as f:
        json.dump(full_comparison, f, indent=4)

    print(f" -> Saved Hybrid Experiment Reports to reports/ & experiments/", flush=True)
    
    # 6. Generate Visualizations
    generate_hybrid_visualizations(rf_eval, qml_eval, hybrid_eval, y_test, P_rf_test, P_qml_test, P_hybrid_test)
    
    return hybrid_eval, full_comparison

def generate_hybrid_visualizations(rf, qml, hybrid, y_test, P_rf, P_qml, P_hybrid):
    print("[VISUALIZATION] Rendering Hybrid evaluation plots...", flush=True)
    
    # 1. Confusion Matrix for Hybrid Model
    plt.figure(figsize=(5, 4))
    cm = np.array([
        [hybrid["confusion_matrix"]["TN"], hybrid["confusion_matrix"]["FP"]],
        [hybrid["confusion_matrix"]["FN"], hybrid["confusion_matrix"]["TP"]]
    ])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", cbar=False,
                xticklabels=["No Diabetes (0)", "Diabetes (1)"],
                yticklabels=["No Diabetes (0)", "Diabetes (1)"])
    plt.title("Hybrid Ensemble Confusion Matrix (Test Set)", fontsize=11, fontweight="bold")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig("reports/hybrid_confusion_matrix.png", dpi=300)
    plt.savefig("experiments/hybrid_confusion_matrix.png", dpi=300)
    plt.close()
    
    # 2. Metric Comparison Bar Chart: Classical vs QML vs Hybrid
    plt.figure(figsize=(10, 5))
    metrics_list = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]
    
    rf_scores = [rf["accuracy"], rf["precision"], rf["recall_sensitivity"], rf["specificity"], rf["f1_score"], rf["roc_auc"]]
    qml_scores = [qml["accuracy"], qml["precision"], qml["recall_sensitivity"], qml["specificity"], qml["f1_score"], qml["roc_auc"]]
    hyb_scores = [hybrid["accuracy"], hybrid["precision"], hybrid["recall_sensitivity"], hybrid["specificity"], hybrid["f1_score"], hybrid["roc_auc"]]
    
    x = np.arange(len(metrics_list))
    width = 0.25
    
    plt.bar(x - width, rf_scores, width, label="Classical RF Baseline", color="#3b82f6")
    plt.bar(x, qml_scores, width, label="Quantum VQC Baseline", color="#06b6d4")
    plt.bar(x + width, hyb_scores, width, label="Hybrid Ensemble Model", color="#8b5cf6")
    
    plt.title("Comprehensive Metric Comparison: Classical vs Quantum vs Hybrid Ensemble", fontsize=12, fontweight="bold")
    plt.ylabel("Score")
    plt.xticks(x, metrics_list)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("reports/hybrid_vs_all_comparison.png", dpi=300)
    plt.savefig("experiments/hybrid_vs_all_comparison.png", dpi=300)
    plt.close()
    
    # 3. ROC Curves Comparison
    plt.figure(figsize=(7, 5))
    fpr_rf, tpr_rf, _ = roc_curve(y_test, P_rf)
    fpr_qml, tpr_qml, _ = roc_curve(y_test, P_qml)
    fpr_hyb, tpr_hyb, _ = roc_curve(y_test, P_hybrid)
    
    plt.plot(fpr_rf, tpr_rf, color="#3b82f6", label=f"Classical RF (AUC = {rf['roc_auc']:.4f})", linewidth=2)
    plt.plot(fpr_qml, tpr_qml, color="#06b6d4", label=f"Quantum VQC (AUC = {qml['roc_auc']:.4f})", linewidth=1.8, linestyle="--")
    plt.plot(fpr_hyb, tpr_hyb, color="#8b5cf6", label=f"Hybrid Ensemble (AUC = {hybrid['roc_auc']:.4f})", linewidth=2.5)
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    
    plt.title("ROC Curve Comparison: Classical vs Quantum vs Hybrid Ensemble", fontsize=12, fontweight="bold")
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("reports/hybrid_roc_curves.png", dpi=300)
    plt.savefig("experiments/hybrid_roc_curves.png", dpi=300)
    plt.close()
    
    print(" -> Generated & saved 3 technical plots (hybrid_confusion_matrix.png, hybrid_vs_all_comparison.png, hybrid_roc_curves.png)", flush=True)

if __name__ == "__main__":
    run_hybrid_experiment()
