import time
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

def evaluate_model_performance(model, model_name: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Fits the model on the full training set, records training and inference duration,
    and computes empirical metrics on the test set.
    """
    print(f"\n[EVALUATION] Training and Evaluating: {model_name}...", flush=True)
    
    # 1. Measure Training Time
    start_train = time.time()
    model.fit(X_train, y_train)
    train_duration_sec = time.time() - start_train
    
    # 2. Measure Inference Time
    start_inf = time.time()
    y_pred = model.predict(X_test)
    inf_duration_sec = time.time() - start_inf
    inf_time_per_sample_ms = (inf_duration_sec / len(X_test)) * 1000.0
    
    # Get class probabilities for ROC-AUC
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        decision_scores = model.decision_function(X_test)
        y_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min() + 1e-12)
    else:
        y_proba = None

    # 3. Calculate Empirical Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec_sensitivity = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
        specificity = 0.0

    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = 0.0

    metrics = {
        "model_name": model_name,
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
        "training_time_sec": float(train_duration_sec),
        "inference_time_sec": float(inf_duration_sec),
        "inference_ms_per_sample": float(inf_time_per_sample_ms)
    }
    
    print(f" -> Accuracy:    {acc:.4f}", flush=True)
    print(f" -> Precision:   {prec:.4f}", flush=True)
    print(f" -> Sensitivity: {rec_sensitivity:.4f}", flush=True)
    print(f" -> Specificity: {specificity:.4f}", flush=True)
    print(f" -> F1 Score:    {f1:.4f}", flush=True)
    print(f" -> ROC-AUC:     {roc_auc:.4f}", flush=True)
    print(f" -> Confusion Matrix: [TN={tn:,}, FP={fp:,}, FN={fn:,}, TP={tp:,}]", flush=True)
    print(f" -> Training Time: {train_duration_sec:.3f}s | Inference Time: {inf_duration_sec:.3f}s ({inf_time_per_sample_ms:.4f} ms/sample)", flush=True)
    
    return metrics, model
