import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, brier_score_loss
)

from src.qml.vqc_model import VariationalQuantumClassifier
from src.hybrid.fusion_model import HybridEnsembleClassifier
from src.utils.disease_registry import DiseaseRegistry
from backend.database.connection import SessionLocal
from backend.database.repository import register_disease_record, register_dataset_record, register_model

class AutoDiseaseMLPipeline:
    """
    Automated Disease-Specific ML Pipeline for SIH26139 Platform.
    Executes end-to-end training and evaluation for classical, quantum, and hybrid models.
    """
    def __init__(self, disease_id: str, disease_name: str, category: str, description: str,
                 target_col: str, dataset_name: str, source: str, raw_csv_path: str,
                 symptoms: str = "", risk_factors: str = "", n_qubits: int = 6):
        self.disease_id = disease_id.lower()
        self.disease_name = disease_name
        self.category = category
        self.description = description
        self.target_col = target_col
        self.dataset_name = dataset_name
        self.source = source
        self.raw_csv_path = raw_csv_path
        self.symptoms = symptoms
        self.risk_factors = risk_factors
        self.n_qubits = n_qubits

        self.disease_model_dir = os.path.join(BASE_DIR, "models", self.disease_id)
        self.processed_data_dir = os.path.join(BASE_DIR, "data", "processed", self.disease_id)
        os.makedirs(self.disease_model_dir, exist_ok=True)
        os.makedirs(self.processed_data_dir, exist_ok=True)

    def run_pipeline(self) -> Dict[str, Any]:
        print("=" * 80)
        print(f" [AUTO ML PIPELINE] Training Models for Target Disease: '{self.disease_name}' ({self.disease_id})")
        print("=" * 80)

        # 1. Load Data
        if not os.path.exists(self.raw_csv_path):
            raise FileNotFoundError(f"Raw dataset file not found at '{self.raw_csv_path}'")

        df = pd.read_csv(self.raw_csv_path)
        print(f" 1. Loaded raw dataset: {df.shape[0]} samples, {df.shape[1]} total columns.")

        if self.target_col not in df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found in dataset columns: {list(df.columns)}")

        X_df = df.drop(columns=[self.target_col])
        y = df[self.target_col].values.astype(int)
        feature_names = list(X_df.columns)

        # 2. Stratified Train / Val / Test Split (80% Train, 10% Val, 10% Test)
        X_train_df, X_temp_df, y_train, y_temp = train_test_split(X_df, y, test_size=0.20, random_state=42, stratify=y)
        X_val_df, X_test_df, y_val, y_test = train_test_split(X_temp_df, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

        print(f" 2. Data Split complete: Train={len(y_train)}, Val={len(y_val)}, Test={len(y_test)}")

        # 3. Fit Preprocessing Pipeline
        preprocessor = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        X_train_proc = preprocessor.fit_transform(X_train_df)
        X_val_proc = preprocessor.transform(X_val_df)
        X_test_proc = preprocessor.transform(X_test_df)

        preprocessor_path = os.path.join(self.disease_model_dir, "preprocessor.joblib")
        joblib.dump(preprocessor, preprocessor_path)
        print(f" 3. Preprocessor fitted and saved to '{preprocessor_path}'.")

        # 4. Train Best Classical Model
        print(" 4. Training Classical ML models...")
        candidate_models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=80, max_depth=5, random_state=42),
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
            "SVC": SVC(probability=True, random_state=42)
        }

        best_classical_clf = None
        best_classical_name = ""
        best_val_auc = -1.0

        for name, clf in candidate_models.items():
            clf.fit(X_train_proc, y_train)
            val_probs = clf.predict_proba(X_val_proc)[:, 1]
            auc = roc_auc_score(y_val, val_probs) if len(np.unique(y_val)) > 1 else accuracy_score(y_val, (val_probs >= 0.5).astype(int))
            print(f"    - Candidate '{name}' Val ROC-AUC: {auc:.4f}")
            if auc > best_val_auc:
                best_val_auc = auc
                best_classical_clf = clf
                best_classical_name = name

        classical_path = os.path.join(self.disease_model_dir, "best_classical_model.joblib")
        joblib.dump(best_classical_clf, classical_path)
        print(f"    -> Selected '{best_classical_name}' as primary classical baseline. Saved to '{classical_path}'.")

        # 5. Train 6-Qubit VQC QML Simulator
        print(" 5. Fitting PCA and training 6-Qubit Variational Quantum Classifier (QML)...")
        pca_reducer = PCA(n_components=self.n_qubits, random_state=42)
        X_train_q = pca_reducer.fit_transform(X_train_proc)
        X_val_q = pca_reducer.transform(X_val_proc)
        X_test_q = pca_reducer.transform(X_test_proc)

        pca_path = os.path.join(self.disease_model_dir, "qml_pca_reducer.joblib")
        joblib.dump(pca_reducer, pca_path)

        vqc = VariationalQuantumClassifier(n_qubits=self.n_qubits, n_layers=2, seed=42)
        vqc.fit(X_train_q, y_train, epochs=15, lr=0.10, batch_size=32)

        qml_params_path = os.path.join(self.disease_model_dir, "qml_vqc_params.npz")
        np.savez(
            qml_params_path,
            weights_ry=vqc.weights_ry,
            weights_rz=vqc.weights_rz,
            bias=vqc.bias,
            scale=vqc.scale,
            n_qubits=self.n_qubits,
            n_layers=2
        )
        print(f"    -> Quantum parameters saved to '{qml_params_path}'.")

        # 6. Train Hybrid Ensemble Classifier
        print(" 6. Training Hybrid Ensemble Model...")
        hybrid_model = HybridEnsembleClassifier(
            classical_model=best_classical_clf,
            qml_model=vqc,
            pca_reducer=pca_reducer
        )
        hybrid_model.fit_fusion(X_val_proc, y_val)
        hybrid_path = os.path.join(self.disease_model_dir, "hybrid_fusion_model.joblib")
        joblib.dump(hybrid_model, hybrid_path)
        print(f"    -> Hybrid model saved to '{hybrid_path}'.")


        # 7. Evaluate All Architectures on Test Set
        print(" 7. Evaluating test metrics across Classical, QML, and Hybrid models...")
        metrics_dict = {
            "classical": self._eval_model(best_classical_clf, X_test_proc, y_test, is_vqc=False, name=f"{best_classical_name} (Classical)"),
            "qml": self._eval_model(vqc, X_test_q, y_test, is_vqc=True, name="6-Qubit VQC (Quantum)"),
            "hybrid": self._eval_model(hybrid_model, X_test_proc, y_test, is_vqc=False, name="Hybrid Ensemble")
        }

        # 8. Register Records in Database & DiseaseRegistry
        self._register_in_database(feature_names, metrics_dict, len(y_test))

        print(f" [AUTO ML PIPELINE] Successfully completed pipeline for '{self.disease_name}'.")
        print("=" * 80)
        return metrics_dict

    def _eval_model(self, model: Any, X_test: np.ndarray, y_test: np.ndarray, is_vqc: bool = False, name: str = "") -> Dict[str, Any]:
        raw_probs = model.predict_proba(X_test)
        if isinstance(raw_probs, np.ndarray) and raw_probs.ndim == 2:
            probs = raw_probs[:, 1]
        else:
            probs = np.array(raw_probs)

        preds = model.predict(X_test) if hasattr(model, "predict") else (probs >= 0.5).astype(int)


        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        
        try:
            auc = float(roc_auc_score(y_test, probs))
        except Exception:
            auc = acc

        tn, fp, fn, tp = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
        spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        brier = float(brier_score_loss(y_test, probs))

        print(f"    - [{name}] Acc: {acc:.4f} | F1: {f1:.4f} | ROC-AUC: {auc:.4f} | Rec: {rec:.4f} | Spec: {spec:.4f}")

        return {
            "model_name": name,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "brier_score": round(brier, 4),
            "confusion_matrix": {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}
        }

    def _register_in_database(self, feature_names: List[str], metrics_dict: Dict[str, Any], test_samples: int):
        db = SessionLocal()
        try:
            rel_model_dir = f"models/{self.disease_id}"
            model_paths = {
                "preprocessor": f"{rel_model_dir}/preprocessor.joblib",
                "classical_model": f"{rel_model_dir}/best_classical_model.joblib",
                "pca_reducer": f"{rel_model_dir}/qml_pca_reducer.joblib",
                "qml_params": f"{rel_model_dir}/qml_vqc_params.npz",
                "hybrid_model": f"{rel_model_dir}/hybrid_fusion_model.joblib"
            }

            # 1. Disease Registry
            register_disease_record(
                db,
                disease_id=self.disease_id,
                name=self.disease_name,
                category=self.category,
                description=self.description,
                symptoms=self.symptoms,
                risk_factors=self.risk_factors,
                dataset_name=self.dataset_name,
                target_variable=self.target_col,
                data_modality="Tabular Clinical Biomarkers",
                features=feature_names,
                qml_config={"n_qubits": self.n_qubits, "n_layers": 2, "pca_dim": self.n_qubits},
                status="ACTIVE",
                model_version="1.0.0"
            )

            # 2. Dataset Registry
            register_dataset_record(
                db,
                dataset_id=f"ds_{self.disease_id}",
                disease_id=self.disease_id,
                name=self.dataset_name,
                source=self.source,
                modality="Tabular Clinical Features",
                target_name=self.target_col,
                sample_count=pd.read_csv(self.raw_csv_path).shape[0],
                feature_count=len(feature_names),
                access_info="Open Biomedical Benchmark Access",
                raw_path=self.raw_csv_path,
                processed_path=os.path.join(self.processed_data_dir, "test.csv"),
                is_acquired=True
            )

            # 3. Model Registry Records (Classical, QML, Hybrid)
            for m_type, m_metrics in metrics_dict.items():
                register_model(
                    db,
                    disease_id=self.disease_id,
                    model_name=m_metrics["model_name"],
                    model_type=m_type.capitalize(),
                    version="1.0.0",
                    artifact_path=model_paths.get(f"{m_type}_model", model_paths["hybrid_model"]),
                    accuracy=m_metrics["accuracy"],
                    f1_score=m_metrics["f1_score"],
                    roc_auc=m_metrics["roc_auc"],
                    recall=m_metrics["recall"],
                    specificity=m_metrics["specificity"],
                    test_samples=test_samples
                )

            # 4. Sync in-memory DiseaseRegistry instance & registry.json
            reg = DiseaseRegistry()
            reg.register_disease_model(
                disease_id=self.disease_id,
                name=self.disease_name,
                category=self.category,
                description=self.description,
                dataset_name=self.dataset_name,
                target_variable=self.target_col,
                features=feature_names,
                model_paths=model_paths,
                qml_config={"n_qubits": self.n_qubits, "n_layers": 2, "pca_dim": self.n_qubits},
                symptoms=self.symptoms,
                risk_factors=self.risk_factors,
                data_modality="Tabular Clinical Biomarkers"
            )

            print(" 8. Registered records in SQLite database and updated DiseaseRegistry JSON.")
        except Exception as e:
            print(f" Warning: DB registration failed for '{self.disease_id}': {e}")
        finally:
            db.close()
