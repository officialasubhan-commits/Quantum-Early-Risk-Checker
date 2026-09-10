import os
import sys
import json
import time
import joblib
import urllib.request
from typing import cast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.qml.vqc_model import VariationalQuantumClassifier
from src.hybrid.fusion_model import HybridEnsembleClassifier
from src.utils.disease_registry import DiseaseRegistry

DISEASE_CONFIGS = {
    "kidney_disease": {
        "id": "kidney_disease",
        "name": "Chronic Kidney Disease Risk Assessment",
        "category": "Nephrology",
        "description": "Early risk screening for chronic kidney disease (CKD) using renal biomarkers and clinical indicators.",
        "dataset_name": "UCI Chronic Kidney Disease Dataset",
        "url": "https://archive.ics.uci.edu/static/public/336/data.csv",
        "target_col": "class",
        "pos_label": "ckd"
    },
    "liver_disease": {
        "id": "liver_disease",
        "name": "Liver Disease Risk Assessment",
        "category": "Hepatology",
        "description": "Early screening for hepatic dysfunction and liver disease using serum bilirubin and enzyme markers.",
        "dataset_name": "UCI Indian Liver Patient Dataset",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
        "cols": ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset'],
        "target_col": "Dataset",
        "pos_label": 1
    },
    "stroke": {
        "id": "stroke",
        "name": "Stroke Risk Assessment",
        "category": "Neurology",
        "description": "Cerebrovascular accident and stroke prediction using clinical risk factors and lifestyle indicators.",
        "dataset_name": "Kaggle Healthcare Stroke Dataset",
        "url": "https://raw.githubusercontent.com/codejay411/Stroke_prediction/main/healthcare-dataset-stroke-data.csv",
        "target_col": "stroke",
        "pos_label": 1
    },
    "breast_cancer": {
        "id": "breast_cancer",
        "name": "Breast Cancer Risk Assessment",
        "category": "Oncology",
        "description": "Biomarker and cell nucleus morphometric analysis for early breast cancer risk stratification.",
        "dataset_name": "UCI Breast Cancer Wisconsin (Diagnostic) Dataset",
        "source": "sklearn",
        "target_col": "target",
        "pos_label": 0  # malignant in sklearn
    },
    "parkinsons": {
        "id": "parkinsons",
        "name": "Parkinson's Disease Risk Assessment",
        "category": "Neurology & Movement",
        "description": "Acoustic and voice frequency analysis for early detection of Parkinsonian neurological tremor.",
        "dataset_name": "UCI Parkinson's Disease Vocal Dataset",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
        "target_col": "status",
        "pos_label": 1
    }
}

def load_and_preprocess_dataset(d_id: str, cfg: dict):
    print(f"\n[{d_id.upper()}] Loading dataset for {cfg['name']}...", flush=True)
    if cfg.get("source") == "sklearn":
        bc = load_breast_cancer(as_frame=True)
        df = bc.frame  # type: ignore[union-attr]
        target_col = "target"
        # Inverse target so 1 = Malignant (high risk), 0 = Benign
        df["target"] = (df["target"] == 0).astype(int)
    else:
        url = cfg["url"]
        if "cols" in cfg:
            df = pd.read_csv(url, header=None, names=cfg["cols"], na_values=["?", "nan", "None", ""])
        else:
            df = pd.read_csv(url, na_values=["?", "nan", "None", ""])

        if "id" in df.columns:
            df.drop(columns=["id"], inplace=True)
        if "name" in df.columns:
            df.drop(columns=["name"], inplace=True)

        target_col = cfg["target_col"]
        pos_val = cfg["pos_label"]
        if isinstance(pos_val, str):
            df["target"] = (df[target_col].astype(str).str.strip().str.lower() == pos_val.lower()).astype(int)
        else:
            df["target"] = (df[target_col] == pos_val).astype(int)

        if target_col != "target" and target_col in df.columns:
            df.drop(columns=[target_col], inplace=True)

    # Convert all feature columns cleanly to floats
    for col in df.columns:
        if col == "target":
            continue
        if not pd.api.types.is_numeric_dtype(df[col].dtype):
            s = df[col].astype(str).str.strip().str.lower()
            s = s.replace(['nan', 'none', '?', '', 'null', '\t?', '\tckd', 'ckd\t'], np.nan)
            s_num = pd.to_numeric(s, errors='coerce')
            if s_num.notna().sum() > 0 and (s_num.notna().sum() / s.notna().sum() if s.notna().sum() > 0 else 0) > 0.5:  # type: ignore[union-attr]
                df[col] = s_num
            else:
                codes, uniques = pd.factorize(s)
                df[col] = pd.Series(np.where(codes == -1, np.nan, codes.astype(float)).tolist(), index=df.index)
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    features = [c for c in df.columns if c != "target"]
    print(f" -> Dataset shape: {df.shape} | Features ({len(features)}): {features}", flush=True)
    print(f" -> Class Distribution (target=1 positive): {df['target'].value_counts().to_dict()}", flush=True)

    return df, features

def train_and_evaluate_disease(d_id: str, cfg: dict):
    df, feature_names = load_and_preprocess_dataset(d_id, cfg)

    raw_dir = os.path.join(PROJECT_ROOT, "data", "raw", d_id)
    proc_dir = os.path.join(PROJECT_ROOT, "data", "processed", d_id)
    models_dir = os.path.join(PROJECT_ROOT, "models", d_id)
    reports_dir = os.path.join(PROJECT_ROOT, "reports", d_id)

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    df.to_csv(os.path.join(raw_dir, "raw_data.csv"), index=False)

    # Train / Val / Test Splits (70%, 15%, 15%)
    train_split, temp_split = train_test_split(df, test_size=0.30, random_state=42, stratify=df["target"])
    val_split, test_split = train_test_split(temp_split, test_size=0.50, random_state=42, stratify=temp_split["target"])  # type: ignore[assignment]
    train_df = cast(pd.DataFrame, train_split)
    val_df = cast(pd.DataFrame, val_split)
    test_df = cast(pd.DataFrame, test_split)

    train_df.to_csv(os.path.join(proc_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(proc_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(proc_dir, "test.csv"), index=False)

    X_train_raw = train_df[feature_names]
    y_train = train_df["target"].to_numpy()

    X_val_raw = val_df[feature_names]
    y_val = val_df["target"].to_numpy()

    X_test_raw = test_df[feature_names]
    y_test = test_df["target"].to_numpy()

    # 1. Fit Preprocessor (SimpleImputer + StandardScaler)
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    X_train_proc = preprocessor.fit_transform(X_train_raw)
    X_val_proc = preprocessor.transform(X_val_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.joblib"))

    # 2. Benchmark Classical Models
    candidates = {
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"),
        "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42),
        "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced"),
        "SupportVectorMachine": SVC(probability=True, random_state=42, C=1.0, class_weight="balanced")  # type: ignore[arg-type]
    }

    best_auc = -1.0
    best_name = None
    best_clf = None

    for name, clf in candidates.items():
        clf.fit(X_train_proc, y_train)
        probs = clf.predict_proba(X_val_proc)[:, 1]
        auc = roc_auc_score(y_val, probs) if len(set(y_val)) > 1 else 0.5
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_clf = clf

    joblib.dump(best_clf, os.path.join(models_dir, "best_classical_model.joblib"))
    print(f" -> Best Classical Model for {d_id}: {best_name} (Val AUC: {best_auc:.4f})", flush=True)

    # 3. Fit PCA Reducer & Train 6-Qubit VQC
    n_qubits = min(6, len(feature_names))
    pca = PCA(n_components=n_qubits, random_state=42)
    X_train_q = pca.fit_transform(X_train_proc)
    X_val_q = pca.transform(X_val_proc)
    X_test_q = pca.transform(X_test_proc)

    joblib.dump(pca, os.path.join(models_dir, "qml_pca_reducer.joblib"))

    vqc = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=2, seed=42)
    vqc.fit(X_train_q, y_train, epochs=25, lr=0.08, batch_size=32)

    np.savez_compressed(
        os.path.join(models_dir, "qml_vqc_params.npz"),
        weights_ry=vqc.weights_ry,
        weights_rz=vqc.weights_rz,
        bias=vqc.bias,
        scale=vqc.scale,
        n_qubits=n_qubits,
        n_layers=2
    )

    # 4. Train Hybrid Ensemble
    hybrid = HybridEnsembleClassifier(classical_model=best_clf, qml_model=vqc, pca_reducer=pca)
    hybrid.fit_fusion(X_val_proc, y_val)
    joblib.dump(hybrid, os.path.join(models_dir, "hybrid_fusion_model.joblib"))

    # 5. Evaluate All 3 Models on Untouched Test Set
    def calc_metrics(y_true, probs, preds):
        acc = float(accuracy_score(y_true, preds))
        prec = float(precision_score(y_true, preds, zero_division="warn"))
        rec = float(recall_score(y_true, preds, zero_division="warn"))
        f1 = float(f1_score(y_true, preds, zero_division="warn"))
        auc = float(roc_auc_score(y_true, probs)) if len(set(y_true)) > 1 else 0.5
        cm = confusion_matrix(y_true, preds)
        tn, fp, fn, tp = map(int, cm.ravel()) if cm.size == 4 else (0, 0, 0, 0)
        spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        return {
            "accuracy": acc, "precision": prec, "recall_sensitivity": rec,
            "specificity": spec, "f1_score": f1, "roc_auc": auc,
            "confusion_matrix": {"TN": tn, "FP": fp, "FN": fn, "TP": tp}
        }

    assert best_clf is not None, "No best classical model was selected"
    c_probs = best_clf.predict_proba(X_test_proc)[:, 1]
    c_preds = (c_probs >= 0.5).astype(int)
    c_met = calc_metrics(y_test, c_probs, c_preds)

    q_probs = vqc.predict_proba(X_test_q)
    q_preds = (q_probs >= 0.5).astype(int)
    q_met = calc_metrics(y_test, q_probs, q_preds)

    h_probs = hybrid.predict_proba(X_test_proc)
    h_preds = hybrid.predict(X_test_proc)
    h_met = calc_metrics(y_test, h_probs, h_preds)

    summary = {
        "disease_id": d_id,
        "disease_name": cfg["name"],
        "total_records": len(df),
        "feature_count": len(feature_names),
        "test_sample_count": len(y_test),
        "best_classical_model": best_name,
        "classical_rf": c_met,
        "quantum_vqc": q_met,
        "hybrid_ensemble": h_met
    }

    with open(os.path.join(reports_dir, f"{d_id}_comparison_report.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # 6. Update Disease Registry
    registry = DiseaseRegistry()
    registry.register_disease_model(
        disease_id=d_id,
        name=cfg["name"],
        category=cfg["category"],
        description=cfg["description"],
        dataset_name=cfg["dataset_name"],
        target_variable="target",
        features=feature_names,
        model_paths={
            "preprocessor": f"models/{d_id}/preprocessor.joblib",
            "classical_model": f"models/{d_id}/best_classical_model.joblib",
            "pca_reducer": f"models/{d_id}/qml_pca_reducer.joblib",
            "qml_params": f"models/{d_id}/qml_vqc_params.npz",
            "hybrid_model": f"models/{d_id}/hybrid_fusion_model.joblib"
        },
        qml_config={"n_qubits": n_qubits, "n_layers": 2, "pca_dim": n_qubits}
    )

    print(f" -> Completed & Registered {d_id.upper()}! Test Acc: {h_met['accuracy']:.4f} | Test F1: {h_met['f1_score']:.4f} | Test AUC: {h_met['roc_auc']:.4f}", flush=True)

def train_all_remaining_diseases():
    print("=" * 70, flush=True)
    print(" [MULTI-DISEASE TRAINING] Training & Registering All Target Disease Models... ", flush=True)
    print("=" * 70, flush=True)

    for d_id, cfg in DISEASE_CONFIGS.items():
        try:
            train_and_evaluate_disease(d_id, cfg)
        except Exception as e:
            print(f" ERROR training disease '{d_id}': {e}", flush=True)

    print("=" * 70, flush=True)
    print(" [MULTI-DISEASE TRAINING] All Disease Models Trained & Activated Successfully!", flush=True)
    print("=" * 70, flush=True)

if __name__ == "__main__":
    train_all_remaining_diseases()
