import os
import uuid
import datetime
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional

from backend.utils.config import MODEL_PATHS, FEATURE_NAMES_IN, FEATURE_NAMES_OUT, API_VERSION
from backend.schemas import (
    PatientFeatures,
    PredictionResponse,
    ExplainabilityResponse,
    FeatureContributionItem,
    HealthResponse,
    ModelInfoResponse,
    SingleDiseasePredictionResult,
    MultiDiseasePredictionResponse,
    MEDICAL_DISCLAIMER_TEXT
)

from src.qml.vqc_model import VariationalQuantumClassifier
from src.hybrid.fusion_model import HybridEnsembleClassifier
from src.explainability.patient_explainer import explain_patient_prediction
from src.utils.disease_registry import DiseaseRegistry
from backend.database.connection import SessionLocal
from backend.database.repository import save_assessment_record

class ModelService:
    _instance: Optional["ModelService"] = None
    _loaded: bool = False
    registry: DiseaseRegistry
    disease_artifacts: Dict[str, Dict[str, Any]]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelService, cls).__new__(cls)
            cls._instance._loaded = False
            cls._instance.registry = DiseaseRegistry()
            cls._instance.disease_artifacts = {}
            cls._instance._load_all_active_models()
        return cls._instance

    def _load_all_active_models(self):
        print("\n[FASTAPI SERVICE] Initializing Multi-Disease Model Service...", flush=True)
        active_diseases = self.registry.get_active_diseases()
        for d_id in active_diseases.keys():
            self._load_disease_artifacts(d_id)
        self._loaded = len(self.disease_artifacts) > 0

    def _load_disease_artifacts(self, disease_id: str):
        paths = self.registry.get_model_paths(disease_id)
        if not paths:
            return

        print(f" -> Loading artifacts for disease: '{disease_id}'", flush=True)
        artifacts: Dict[str, Any] = {
            "preprocessor": None,
            "classical_model": None,
            "pca_reducer": None,
            "qml_vqc": None,
            "hybrid_model": None
        }

        try:
            if "preprocessor" in paths and os.path.exists(paths["preprocessor"]):
                artifacts["preprocessor"] = joblib.load(paths["preprocessor"])
            if "classical_model" in paths and os.path.exists(paths["classical_model"]):
                artifacts["classical_model"] = joblib.load(paths["classical_model"])
            if "pca_reducer" in paths and os.path.exists(paths["pca_reducer"]):
                artifacts["pca_reducer"] = joblib.load(paths["pca_reducer"])
            if "qml_params" in paths and os.path.exists(paths["qml_params"]):
                qml_params = np.load(paths["qml_params"])
                n_qubits = int(qml_params.get("n_qubits", 6))
                n_layers = int(qml_params.get("n_layers", 2))
                vqc = VariationalQuantumClassifier(n_qubits=n_qubits, n_layers=n_layers, seed=42)
                vqc.weights_ry = qml_params["weights_ry"]
                vqc.weights_rz = qml_params["weights_rz"]
                vqc.bias = float(qml_params["bias"])
                vqc.scale = float(qml_params["scale"])
                artifacts["qml_vqc"] = vqc
            if "hybrid_model" in paths and os.path.exists(paths["hybrid_model"]):
                artifacts["hybrid_model"] = joblib.load(paths["hybrid_model"])

            self.disease_artifacts[disease_id.lower()] = artifacts
            print(f"    - Loaded model artifacts for '{disease_id}' successfully.", flush=True)
        except Exception as e:
            print(f" ERROR loading artifacts for '{disease_id}': {e}", flush=True)

    def _convert_to_dataframe(self, features_dict: Dict[str, Any], feature_names_in: List[str]) -> pd.DataFrame:
        ordered_data = {col: [features_dict[col]] for col in feature_names_in}
        return pd.DataFrame(ordered_data)

    def preprocess_disease(self, disease_id: str, features_dict: Dict[str, Any]) -> np.ndarray:
        d_id = disease_id.lower()
        if d_id not in self.disease_artifacts or self.disease_artifacts[d_id]["preprocessor"] is None:
            self._load_disease_artifacts(d_id)

        artifacts = self.disease_artifacts[d_id]
        required_features = self.registry.get_required_features(d_id)
        df = self._convert_to_dataframe(features_dict, required_features)
        X_proc = artifacts["preprocessor"].transform(df)
        return X_proc

    def predict_disease(self, disease_id: str, model_type: str, features_dict: Dict[str, Any], patient_id: Optional[str] = "PATIENT_001") -> PredictionResponse:
        p_id = patient_id or "PATIENT_001"
        d_id = disease_id.lower()
        disease_info = self.registry.get_disease_info(d_id)
        if not disease_info:
            raise ValueError(f"Unknown disease identifier '{disease_id}'. Registered diseases: {list(self.registry.get_all_diseases().keys())}")

        if disease_info.get("status") != "ACTIVE":
            raise ValueError(f"Disease model for '{disease_info.get('name')}' is currently PLANNED and not active yet.")

        # Validate input features
        is_valid, missing = self.registry.validate_features_for_disease(d_id, features_dict)
        if not is_valid:
            raise ValueError(f"Missing required clinical features for {disease_info.get('name')}: {missing}")

        if d_id not in self.disease_artifacts:
            self._load_disease_artifacts(d_id)

        artifacts = self.disease_artifacts[d_id]
        X_proc = self.preprocess_disease(d_id, features_dict)
        model_type_clean = model_type.lower()

        if model_type_clean in ["classical", "rf", "random_forest"]:
            prob = float(artifacts["classical_model"].predict_proba(X_proc)[0, 1])
            model_name = f"{type(artifacts['classical_model']).__name__} (Classical Baseline)"
            pred_class = int(prob >= 0.5)
        elif model_type_clean in ["qml", "quantum", "vqc"]:
            X_q = artifacts["pca_reducer"].transform(X_proc)
            prob = float(artifacts["qml_vqc"].predict_proba(X_q)[0])
            model_name = f"{artifacts['qml_vqc'].n_qubits}-Qubit Variational Quantum Classifier (QML)"
            pred_class = int(prob >= 0.5)
        elif model_type_clean in ["hybrid", "ensemble"]:
            prob = float(artifacts["hybrid_model"].predict_proba(X_proc)[0])
            pred_class = int(artifacts["hybrid_model"].predict(X_proc)[0])
            model_name = "Hybrid Classical-Quantum Ensemble Classifier"
        else:
            raise ValueError(f"Unsupported model_type '{model_type}'. Choose from 'classical', 'qml', 'hybrid'.")

        disease_name = disease_info.get("name", d_id.capitalize())
        label_pos = f"Elevated {disease_name} Risk"
        label_neg = f"Low {disease_name} Risk"
        label = label_pos if pred_class == 1 else label_neg
        req_id = str(uuid.uuid4())

        # Save to database
        try:
            db = SessionLocal()
            save_assessment_record(
                db,
                request_id=req_id,
                patient_id=p_id,
                disease_id=d_id,
                disease_name=disease_name,
                model_used=model_name,
                predicted_class=pred_class,
                predicted_label=label,
                risk_probability=round(prob, 4),
                is_high_risk=(pred_class == 1),
                feature_inputs=features_dict,
                disclaimer=MEDICAL_DISCLAIMER_TEXT
            )
            db.close()
        except Exception as e:
            print(f" Warning: DB save error in predict_disease: {e}", flush=True)

        return PredictionResponse(
            request_id=req_id,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            patient_id=p_id,
            disease_id=d_id,
            disease_name=disease_name,
            model_used=model_name,
            predicted_class=pred_class,
            predicted_label=label,
            risk_probability=round(prob, 4),
            is_high_risk=(pred_class == 1),
            disclaimer=MEDICAL_DISCLAIMER_TEXT
        )

    def predict_all_diseases(self, features_dict: Dict[str, Any], patient_id: Optional[str] = "PATIENT_001") -> MultiDiseasePredictionResponse:
        p_id = patient_id or "PATIENT_001"
        results: List[SingleDiseasePredictionResult] = []
        unsupported: List[Dict[str, Any]] = []

        all_diseases = self.registry.get_all_diseases()
        for d_id, info in all_diseases.items():
            if info.get("status") == "ACTIVE":
                is_valid, missing = self.registry.validate_features_for_disease(d_id, features_dict)
                if is_valid:
                    res = self.predict_disease(d_id, "hybrid", features_dict, patient_id=p_id)
                    results.append(SingleDiseasePredictionResult(
                        disease_id=d_id,
                        disease_name=info.get("name", d_id),
                        model_used=res.model_used,
                        predicted_class=res.predicted_class,
                        predicted_label=res.predicted_label,
                        risk_probability=res.risk_probability,
                        is_high_risk=res.is_high_risk,
                        status="SUCCESS",
                        missing_features=[]
                    ))
                else:
                    results.append(SingleDiseasePredictionResult(
                        disease_id=d_id,
                        disease_name=info.get("name", d_id),
                        model_used="N/A",
                        predicted_class=0,
                        predicted_label="Evaluation Skipped (Incomplete Feature Payload)",
                        risk_probability=0.0,
                        is_high_risk=False,
                        status="SKIPPED_MISSING_FEATURES",
                        missing_features=missing
                    ))
            else:
                unsupported.append({
                    "disease_id": d_id,
                    "disease_name": info.get("name", d_id),
                    "status": "PLANNED",
                    "reason": "Model architecture established; dataset training planned in subsequent phase."
                })

        return MultiDiseasePredictionResponse(
            request_id=str(uuid.uuid4()),
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            patient_id=p_id,
            disease_results=results,
            unsupported_diseases=unsupported,
            disclaimer=MEDICAL_DISCLAIMER_TEXT
        )

    # Legacy helper wrappers for backward compatibility
    def predict_classical(self, features: PatientFeatures, patient_id: Optional[str] = "PATIENT_001") -> PredictionResponse:
        f_dict = features.model_dump()
        return self.predict_disease("diabetes", "classical", f_dict, patient_id=patient_id or "PATIENT_001")

    def predict_qml(self, features: PatientFeatures, patient_id: Optional[str] = "PATIENT_001") -> PredictionResponse:
        f_dict = features.model_dump()
        return self.predict_disease("diabetes", "qml", f_dict, patient_id=patient_id or "PATIENT_001")

    def predict_hybrid(self, features: PatientFeatures, patient_id: Optional[str] = "PATIENT_001") -> PredictionResponse:
        f_dict = features.model_dump()
        return self.predict_disease("diabetes", "hybrid", f_dict, patient_id=patient_id or "PATIENT_001")

    def explain_disease(self, disease_id: str, features_dict: Dict[str, Any], patient_id: Optional[str] = "PATIENT_001") -> ExplainabilityResponse:
        p_id = patient_id or "PATIENT_001"
        d_id = disease_id.lower()
        disease_info = self.registry.get_disease_info(d_id)
        if not disease_info:
            raise ValueError(f"Unknown disease identifier '{disease_id}'")

        if d_id not in self.disease_artifacts:
            self._load_disease_artifacts(d_id)

        artifacts = self.disease_artifacts[d_id]
        required_features = self.registry.get_required_features(d_id)
        X_proc = self.preprocess_disease(d_id, features_dict)

        clf = artifacts["classical_model"]
        feature_importances = []
        if hasattr(clf, "feature_importances_"):
            imps = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imps = np.abs(clf.coef_[0])
        else:
            imps = np.ones(len(required_features)) / len(required_features)

        raw_vals = [float(features_dict.get(col, 0.0)) for col in required_features]
        sum_imp = sum(imps) if sum(imps) > 0 else 1.0
        norm_imps = [float(i / sum_imp) for i in imps]

        full_contribs = []
        for feat, val, imp in zip(required_features, raw_vals, norm_imps):
            full_contribs.append(FeatureContributionItem(
                feature=feat,
                display_name=feat.replace("_", " ").title(),
                feature_value=round(val, 2),
                contribution_score=round(imp * (1.0 if val != 0 else 0.1), 4)
            ))

        full_contribs.sort(key=lambda x: abs(x.contribution_score), reverse=True)
        top_contribs = full_contribs[:5]

        res = self.predict_disease(d_id, "hybrid", features_dict, patient_id=p_id)
        
        disease_name = disease_info.get("name", d_id)
        if res.is_high_risk:
            narrative = f"Patient presents elevated risk indicators for {disease_name}. Primary contributing biomarkers include {top_contribs[0].display_name} ({top_contribs[0].feature_value}) and {top_contribs[1].display_name} ({top_contribs[1].feature_value})."
        else:
            narrative = f"Patient presents low overall risk profile for {disease_name}. Key protective biomarkers include {top_contribs[0].display_name} ({top_contribs[0].feature_value})."

        return ExplainabilityResponse(
            request_id=res.request_id,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            patient_id=p_id,
            disease_id=d_id,
            disease_name=disease_name,
            model_used=res.model_used,
            predicted_class=res.predicted_class,
            predicted_label=res.predicted_label,
            risk_probability=res.risk_probability,
            is_high_risk=res.is_high_risk,
            top_contributing_features=top_contribs,
            full_feature_contributions=full_contribs,
            clinical_narrative=narrative,
            disclaimer=MEDICAL_DISCLAIMER_TEXT
        )

    def explain(self, features: PatientFeatures, patient_id: Optional[str] = "PATIENT_001") -> ExplainabilityResponse:
        p_id = patient_id or "PATIENT_001"
        f_dict = features.model_dump()
        X_proc = self.preprocess_disease("diabetes", f_dict)
        diabetes_artifacts = self.disease_artifacts["diabetes"]
        raw_explanation = explain_patient_prediction(
            diabetes_artifacts["hybrid_model"],
            X_proc[0],
            FEATURE_NAMES_OUT,
            patient_id=p_id
        )

        top_contribs = [FeatureContributionItem(**item) for item in raw_explanation["top_contributing_features"]]
        full_contribs = [FeatureContributionItem(**item) for item in raw_explanation["full_feature_contributions"]]
        pred_class = 1 if raw_explanation["is_high_risk"] else 0
        req_id = str(uuid.uuid4())
        top_contribs_dicts = [item.model_dump() for item in top_contribs]

        try:
            db = SessionLocal()
            save_assessment_record(
                db,
                request_id=req_id,
                patient_id=p_id,
                disease_id="diabetes",
                disease_name="Diabetes / Prediabetes",
                model_used="Hybrid Classical-Quantum Ensemble + Explainability Layer",
                predicted_class=pred_class,
                predicted_label=raw_explanation["predicted_label"],
                risk_probability=raw_explanation["risk_probability"],
                is_high_risk=raw_explanation["is_high_risk"],
                feature_inputs=f_dict,
                top_contributing_features=top_contribs_dicts,
                clinical_narrative=raw_explanation["clinical_narrative"],
                disclaimer=MEDICAL_DISCLAIMER_TEXT
            )
            db.close()
        except Exception as e:
            print(f" Warning: DB save error in explain: {e}", flush=True)

        return ExplainabilityResponse(
            request_id=req_id,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            patient_id=p_id,
            disease_id="diabetes",
            disease_name="Diabetes / Prediabetes",
            model_used="Hybrid Classical-Quantum Ensemble + Explainability Layer",
            predicted_class=pred_class,
            predicted_label=raw_explanation["predicted_label"],
            risk_probability=raw_explanation["risk_probability"],
            is_high_risk=raw_explanation["is_high_risk"],
            top_contributing_features=top_contribs,
            full_feature_contributions=full_contribs,
            clinical_narrative=raw_explanation["clinical_narrative"],
            disclaimer=MEDICAL_DISCLAIMER_TEXT
        )

    def get_health(self) -> HealthResponse:
        all_diseases = self.registry.get_all_diseases()
        active_diseases = self.registry.get_active_diseases()
        loaded_status = {}
        for d_id in active_diseases.keys():
            is_loaded = d_id in self.disease_artifacts and all(
                self.disease_artifacts[d_id][k] is not None for k in ["preprocessor", "classical_model", "pca_reducer", "qml_vqc", "hybrid_model"]
            )
            loaded_status[f"{d_id}_all_models_ready"] = is_loaded

        status_str = "OK" if any(loaded_status.values()) else "PARTIAL"
        return HealthResponse(
            status=status_str,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            models_loaded=loaded_status,
            registered_diseases_count=len(all_diseases),
            active_diseases_count=len(active_diseases),
            version=API_VERSION
        )

    def get_model_info(self) -> ModelInfoResponse:
        comparison_path = os.path.join(MODEL_PATHS["hybrid_model"], "..", "..", "reports", "hybrid_vs_all_comparison.json")
        comp_data = {}
        if os.path.exists(comparison_path):
            with open(comparison_path, "r") as f:
                comp_data = json.load(f)

        classical_info = comp_data.get("classical_rf", {
            "model_name": "Random Forest Baseline",
            "accuracy": 0.7348,
            "f1_score": 0.4444,
            "roc_auc": 0.8240
        })
        quantum_info = comp_data.get("quantum_vqc", {
            "model_name": "6-Qubit VQC Simulator",
            "accuracy": 0.8607,
            "f1_score": 0.0,
            "roc_auc": 0.4944
        })
        hybrid_info = comp_data.get("hybrid_ensemble", {
            "model_name": "Hybrid Classical-Quantum Ensemble",
            "accuracy": 0.7852,
            "f1_score": 0.4612,
            "roc_auc": 0.8240
        })
        dataset_info = comp_data.get("experiment_metadata", {
            "dataset_name": "CDC BRFSS 2015 Diabetes Health Indicators",
            "total_records": 253680,
            "classical_features": 21,
            "pca_quantum_features": 6,
            "test_samples": 38052
        })

        return ModelInfoResponse(
            baseline_model=classical_info,
            qml_model=quantum_info,
            hybrid_model=hybrid_info,
            dataset_info=dataset_info,
            medical_disclaimer=MEDICAL_DISCLAIMER_TEXT
        )
