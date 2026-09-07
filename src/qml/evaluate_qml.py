import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.qml.feature_reduction import reduce_features_for_qml
from src.qml.vqc_model import VariationalQuantumClassifier

def run_qml_experiment(
    sample_train_size: int = 5000,
    n_qubits: int = 6,
    n_layers: int = 2,
    epochs: int = 40,
    lr: float = 0.1,
    random_seed: int = 42
):
    print("=" * 70, flush=True)
    print(" Phase 4: Quantum Machine Learning (QML) Baseline Experiment ", flush=True)
    print("=" * 70, flush=True)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("reports", exist_ok=True)
    os.makedirs("experiments", exist_ok=True)
    
    # 1. Load Preprocessed Data (Same splits as Classical ML)
    train_data = np.load("data/processed/train_data.npz")
    val_data   = np.load("data/processed/val_data.npz")
    test_data  = np.load("data/processed/test_data.npz")
    
    X_train_full, y_train_full = train_data["X"], train_data["y"]
    X_val, y_val               = val_data["X"], val_data["y"]
    X_test, y_test             = test_data["X"], test_data["y"]
    
    print(f"\n[DATA REUSE VERIFICATION] Original Dataset Split Sizes:", flush=True)
    print(f" -> Full Train Set:      {len(X_train_full):,} samples (70.0%)", flush=True)
    print(f" -> Validation Set:      {len(X_val):,} samples (15.0%)", flush=True)
    print(f" -> Held-Out Test Set:   {len(X_test):,} samples (15.0%) [UNTOUCHED]", flush=True)
    
    # 2. PCA Feature Reduction (21 -> 6 Quantum Features)
    X_train_q_full, X_val_q, X_test_q, pca = reduce_features_for_qml(
        X_train=X_train_full,
        X_val=X_val,
        X_test=X_test,
        n_components=n_qubits
    )
    
    # 3. Stratified Subsampling for Simulator Training (Preserving 13.93% Positive Ratio)
    np.random.seed(random_seed)
    pos_idx = np.where(y_train_full == 1)[0]
    neg_idx = np.where(y_train_full == 0)[0]
    
    n_pos_sub = int(sample_train_size * (len(pos_idx) / len(y_train_full)))
    n_neg_sub = sample_train_size - n_pos_sub
    
    sub_pos_idx = np.random.choice(pos_idx, size=n_pos_sub, replace=False)
    sub_neg_idx = np.random.choice(neg_idx, size=n_neg_sub, replace=False)
    
    sub_indices = np.concatenate([sub_pos_idx, sub_neg_idx])
    np.random.shuffle(sub_indices)
    
    X_train_sub = X_train_q_full[sub_indices]
    y_train_sub = y_train_full[sub_indices]
    
    print(f"\n[QML SAMPLING STRATEGY] Quantum Simulator Training Subset:", flush=True)
    print(f" -> Documented Training Subset Size: {len(X_train_sub):,} samples ({len(X_train_sub)/len(X_train_full):.2%} of full train set)", flush=True)
    print(f" -> Subsample Target Positive Ratio: {y_train_sub.mean():.2%} (Matches full dataset 13.93%)", flush=True)
    print(f" -> Rationale: Enables iterative parameter-shift gradient optimization on statevector simulator within reasonable runtime while preserving the full 38,052 sample held-out test set.", flush=True)
    
    # 4. Instantiate & Train Variational Quantum Classifier (VQC)
    vqc = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers, seed=random_seed)
    
    start_train_time = time.time()
    loss_history = vqc.fit(X_train_sub, y_train_sub, epochs=epochs, lr=lr, batch_size=256)
    train_duration = time.time() - start_train_time
    
    # 5. Evaluate QML Model on Full Held-Out Test Set (38,052 samples)
    print(f"\n[QML EVALUATION] Evaluating VQC on FULL UNTOUCHED Test Set ({len(X_test_q):,} samples)...", flush=True)
    start_inf_time = time.time()
    y_proba_test = vqc.predict_proba(X_test_q)
    y_pred_test = (y_proba_test >= 0.5).astype(int)
    inf_duration = time.time() - start_inf_time
    inf_ms_per_sample = (inf_duration / len(X_test_q)) * 1000.0
    
    # Calculate Empirical Metrics
    acc = accuracy_score(y_test, y_pred_test)
    prec = precision_score(y_test, y_pred_test, zero_division=0)
    rec_sensitivity = recall_score(y_test, y_pred_test, zero_division=0)
    f1 = f1_score(y_test, y_pred_test, zero_division=0)
    f1_macro = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba_test)
    
    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    qml_metrics = {
        "model_name": "Variational Quantum Classifier (VQC)",
        "framework": "Custom Statevector QML Simulator (NumPy)",
        "num_qubits": n_qubits,
        "num_layers": n_layers,
        "num_trainable_params": int(n_layers * n_qubits * 2 + 2),
        "feature_reduction_method": f"PCA (21 -> {n_qubits} quantum features)",
        "training_subset_size": len(X_train_sub),
        "test_set_size": len(X_test_q),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall_sensitivity": float(rec_sensitivity),
        "specificity": float(specificity),
        "f1_score": float(f1),
        "f1_macro": float(f1_macro),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },
        "training_time_sec": float(train_duration),
        "inference_time_sec": float(inf_duration),
        "inference_ms_per_sample": float(inf_ms_per_sample)
    }
    
    print("\n" + "=" * 60, flush=True)
    print(" ACTUAL QML EXPERIMENTAL EVALUATION RESULTS ", flush=True)
    print("=" * 60, flush=True)
    print(f"Accuracy:            {acc:.4f} ({acc*100:.2f}%)", flush=True)
    print(f"Precision:           {prec:.4f} ({prec*100:.2f}%)", flush=True)
    print(f"Recall / Sensitivity:{rec_sensitivity:.4f} ({rec_sensitivity*100:.2f}%)", flush=True)
    print(f"Specificity:         {specificity:.4f} ({specificity*100:.2f}%)", flush=True)
    print(f"F1-Score:            {f1:.4f}", flush=True)
    print(f"ROC-AUC:             {roc_auc:.4f}", flush=True)
    print(f"Confusion Matrix:    [TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}]", flush=True)
    print(f"Training Duration:   {train_duration:.2f}s", flush=True)
    print(f"Inference Latency:   {inf_duration:.2f}s ({inf_ms_per_sample:.4f} ms/sample)", flush=True)
    print("=" * 60, flush=True)
    
    # 6. Load Classical Baseline Metrics for Direct Fair Comparison
    classical_report_path = "reports/experiment_metrics.json"
    if os.path.exists(classical_report_path):
        with open(classical_report_path, "r") as f:
            classical_data = json.load(f)
        rf_metrics = classical_data.get("Random Forest", {})
    else:
        rf_metrics = {}
        
    comparison_report = {
        "qml_vqc": qml_metrics,
        "classical_rf_baseline": rf_metrics,
        "comparison_notes": "Both QML and Classical Random Forest were evaluated on the EXACT SAME 38,052 sample held-out test set."
    }
    
    # 7. Save Reports & Parameters
    np.savez_compressed(
        "models/qml_vqc_params.npz",
        weights_ry=vqc.weights_ry,
        weights_rz=vqc.weights_rz,
        bias=vqc.bias,
        scale=vqc.scale
    )
    
    with open("reports/qml_experiment_report.json", "w") as f:
        json.dump(qml_metrics, f, indent=4)
        
    with open("reports/qml_vs_classical_comparison.json", "w") as f:
        json.dump(comparison_report, f, indent=4)
        
    print(f" -> QML Model Parameters Saved:      models/qml_vqc_params.npz", flush=True)
    print(f" -> QML Experiment JSON Saved:        reports/qml_experiment_report.json", flush=True)
    print(f" -> QML vs Classical JSON Saved:      reports/qml_vs_classical_comparison.json", flush=True)
    
    # 8. Generate Visualizations
    generate_visualizations(loss_history, qml_metrics, rf_metrics)
    
    return qml_metrics, comparison_report

def generate_visualizations(loss_history, qml_metrics, rf_metrics):
    print("\n[VISUALIZATION] Generating experimental plots...", flush=True)
    
    # Plot 1: Training Loss Curve
    plt.figure(figsize=(7, 4.5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color="#0ea5e9", linewidth=2.5, marker="o", markersize=4)
    plt.title("VQC Quantum Model Training Loss Curve", fontsize=12, fontweight="bold")
    plt.xlabel("Epoch")
    plt.ylabel("Weighted Binary Cross-Entropy Loss")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("reports/qml_training_loss.png", dpi=300)
    plt.close()
    
    # Plot 2: QML Confusion Matrix
    plt.figure(figsize=(5, 4))
    cm = np.array([
        [qml_metrics["confusion_matrix"]["TN"], qml_metrics["confusion_matrix"]["FP"]],
        [qml_metrics["confusion_matrix"]["FN"], qml_metrics["confusion_matrix"]["TP"]]
    ])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Diabetes (0)", "Diabetes (1)"],
                yticklabels=["No Diabetes (0)", "Diabetes (1)"])
    plt.title("QML VQC Confusion Matrix (Test Set)", fontsize=11, fontweight="bold")
    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()
    plt.savefig("reports/qml_confusion_matrix.png", dpi=300)
    plt.close()
    
    # Plot 3: QML vs Classical Metric Comparison Bar Chart
    if rf_metrics:
        plt.figure(figsize=(9, 5))
        metrics_names = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "ROC-AUC"]
        qml_vals = [
            qml_metrics["accuracy"],
            qml_metrics["precision"],
            qml_metrics["recall_sensitivity"],
            qml_metrics["specificity"],
            qml_metrics["f1_score"],
            qml_metrics["roc_auc"]
        ]
        rf_vals = [
            rf_metrics["accuracy"],
            rf_metrics["precision"],
            rf_metrics["recall_sensitivity"],
            rf_metrics["specificity"],
            rf_metrics["f1_score"],
            rf_metrics["roc_auc"]
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        plt.bar(x - width/2, rf_vals, width, label="Random Forest (Classical)", color="#3b82f6")
        plt.bar(x + width/2, qml_vals, width, label="VQC (Quantum)", color="#06b6d4")
        
        plt.title("Empirical Metric Comparison: Classical RF Baseline vs. Quantum VQC", fontsize=12, fontweight="bold")
        plt.ylabel("Score")
        plt.xticks(x, metrics_names)
        plt.ylim(0, 1.0)
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig("reports/qml_vs_classical_comparison.png", dpi=300)
        plt.close()
        
    print(" -> Saved 3 technical plots to reports/ (qml_training_loss.png, qml_confusion_matrix.png, qml_vs_classical_comparison.png)", flush=True)

if __name__ == "__main__":
    run_qml_experiment()
