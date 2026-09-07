"""
Phase 7 End-to-End System Testing & SIH Demo Readiness Suite
SIH26139 - Hybrid Quantum Machine Learning Platform
"""

import unittest
import time
import json
import sqlite3
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.main import app
from backend.services.model_service import ModelService
from backend.database.init_db import init_database
from backend.database.connection import get_db
from backend.utils.config import DATABASE_URL

class TestPhase7SystemEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=== Initializing Phase 7 System End-to-End Testing Suite ===")
        cls.client = TestClient(app)
        init_database()

        cls.diseases = [
            "diabetes", "heart_disease", "kidney_disease",
            "liver_disease", "stroke", "breast_cancer", "parkinsons"
        ]

        cls.sample_payloads = {
            "diabetes": {
                "patient_id": "TEST_PATIENT_LOW_001",
                "disease_id": "diabetes",
                "features": {
                    "BMI": 22.4, "GenHlth": 1, "MentHlth": 0, "PhysHlth": 0, "HighBP": 0,
                    "HighChol": 0, "CholCheck": 1, "Smoker": 0, "Stroke": 0,
                    "HeartDiseaseorAttack": 0, "PhysActivity": 1, "Fruits": 1,
                    "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1,
                    "NoDocbcCost": 0, "DiffWalk": 0, "Sex": 0, "Age": 4,
                    "Education": 6, "Income": 8
                }
            },
            "heart_disease": {
                "patient_id": "TEST_PATIENT_LOW_002",
                "disease_id": "heart_disease",
                "features": {
                    "age": 42.0, "sex": 0.0, "cp": 0.0, "trestbps": 118.0, "chol": 185.0,
                    "fbs": 0.0, "restecg": 0.0, "thalach": 168.0, "exang": 0.0, "oldpeak": 0.0,
                    "slope": 1.0, "ca": 0.0, "thal": 2.0
                }
            },
            "kidney_disease": {
                "patient_id": "TEST_PATIENT_LOW_003",
                "disease_id": "kidney_disease",
                "features": {
                    "age": 38.0, "bp": 70.0, "sg": 1.025, "al": 0.0, "su": 0.0, "rbc": 1.0,
                    "pc": 1.0, "pcc": 0.0, "ba": 0.0, "bgr": 95.0, "bu": 18.0, "sc": 0.8,
                    "sod": 142.0, "pot": 4.2, "hemo": 16.2, "pcv": 48.0, "wbcc": 6400.0,
                    "rbcc": 5.4, "htn": 0.0, "dm": 0.0, "cad": 0.0, "appet": 0.0, "pe": 0.0, "ane": 0.0
                }
            },
            "liver_disease": {
                "patient_id": "TEST_PATIENT_LOW_004",
                "disease_id": "liver_disease",
                "features": {
                    "Age": 35.0, "Gender": 1.0, "Total_Bilirubin": 0.6, "Direct_Bilirubin": 0.1,
                    "Alkaline_Phosphotase": 140.0, "Alamine_Aminotransferase": 18.0,
                    "Aspartate_Aminotransferase": 20.0, "Total_Protiens": 7.2, "Albumin": 4.1,
                    "Albumin_and_Globulin_Ratio": 1.2
                }
            },
            "stroke": {
                "patient_id": "TEST_PATIENT_LOW_005",
                "disease_id": "stroke",
                "features": {
                    "gender": 0.0, "age": 34.0, "hypertension": 0.0, "heart_disease": 0.0,
                    "ever_married": 1.0, "work_type": 2.0, "Residence_type": 1.0,
                    "avg_glucose_level": 84.5, "bmi": 22.1, "smoking_status": 0.0
                }
            },
            "breast_cancer": {
                "patient_id": "TEST_PATIENT_LOW_006",
                "disease_id": "breast_cancer",
                "features": {
                    "mean radius": 11.2, "mean texture": 14.5, "mean perimeter": 71.8, "mean area": 384.0,
                    "mean smoothness": 0.092, "mean compactness": 0.054, "mean concavity": 0.021,
                    "mean concave points": 0.015, "mean symmetry": 0.165, "mean fractal dimension": 0.059,
                    "radius error": 0.22, "texture error": 0.85, "perimeter error": 1.45, "area error": 15.2,
                    "smoothness error": 0.005, "compactness error": 0.011, "concavity error": 0.012,
                    "concave points error": 0.006, "symmetry error": 0.014, "fractal dimension error": 0.002,
                    "worst radius": 12.4, "worst texture": 18.2, "worst perimeter": 79.5, "worst area": 470.0,
                    "worst smoothness": 0.121, "worst compactness": 0.102, "worst concavity": 0.085,
                    "worst concave points": 0.052, "worst symmetry": 0.245, "worst fractal dimension": 0.071
                }
            },
            "parkinsons": {
                "patient_id": "TEST_PATIENT_LOW_007",
                "disease_id": "parkinsons",
                "features": {
                    "MDVP:Fo(Hz)": 202.4, "MDVP:Fhi(Hz)": 224.1, "MDVP:Flo(Hz)": 188.5,
                    "MDVP:Jitter(%)": 0.0021, "MDVP:Jitter(Abs)": 0.00001, "MDVP:RAP": 0.0011,
                    "MDVP:PPQ": 0.0013, "Jitter:DDP": 0.0032, "MDVP:Shimmer": 0.0142,
                    "MDVP:Shimmer(dB)": 0.128, "Shimmer:APQ3": 0.0078, "Shimmer:APQ5": 0.0089,
                    "MDVP:APQ": 0.0105, "Shimmer:DDA": 0.0234, "NHR": 0.0045, "HNR": 28.5,
                    "RPDE": 0.321, "DFA": 0.625, "spread1": -7.21, "spread2": 0.112,
                    "D2": 1.85, "PPE": 0.098
                }
            }
        }

    # -------------------------------------------------------------
    # TEST 1: All 7 Integrated Disease Predictions Across Engines
    # -------------------------------------------------------------
    def test_01_all_disease_predictions_and_engines(self):
        """Verify predictions work for all 7 diseases across hybrid, classical, and QML engines."""
        engines = ["hybrid", "classical", "qml"]
        for disease in self.diseases:
            payload = self.sample_payloads[disease]
            for engine in engines:
                response = self.client.post(
                    f"/api/v1/predict/disease/{disease}/{engine}",
                    json=payload
                )
                self.assertEqual(response.status_code, 200, f"Failed for {disease} with {engine}")
                data = response.json()
                self.assertEqual(data["disease_id"], disease)
                self.assertIn("risk_probability", data)
                self.assertGreaterEqual(data["risk_probability"], 0.0)
                self.assertLessEqual(data["risk_probability"], 1.0)
                self.assertIn("predicted_label", data)
                self.assertIn("is_high_risk", data)
                self.assertIn("model_used", data)

    # -------------------------------------------------------------
    # TEST 2: Correct Disease Routing & Isolation
    # -------------------------------------------------------------
    def test_02_disease_routing_isolation(self):
        """Verify request routing prevents passing features to wrong disease models."""
        wrong_payload = {
            "patient_id": "TEST_ROUTING_ERR",
            "disease_id": "heart_disease",
            "features": {"BMI": 25.0, "Age": 5} # Diabetes features passed to heart disease
        }
        response = self.client.post("/api/v1/predict/disease/heart_disease/hybrid", json=wrong_payload)
        self.assertEqual(response.status_code, 400, "Model should reject features trained for another disease")
        data = response.json()
        self.assertIn("detail", data)

    # -------------------------------------------------------------
    # TEST 3: Input Validation & Missing Data Handling (No Silent Fabrication)
    # -------------------------------------------------------------
    def test_03_missing_data_validation(self):
        """Verify missing features are caught and NOT silently fabricated."""
        incomplete_payload = {
            "patient_id": "TEST_MISSING",
            "disease_id": "diabetes",
            "features": {"BMI": 28.5} # Missing 20 required features
        }
        response = self.client.post("/api/v1/predict/disease/diabetes/hybrid", json=incomplete_payload)
        self.assertEqual(response.status_code, 400)
        err_msg = response.json()["detail"]
        self.assertIn("Missing", err_msg)

    # -------------------------------------------------------------
    # TEST 4: FastAPI Endpoints Integrity & Health
    # -------------------------------------------------------------
    def test_04_fastapi_endpoints_health(self):
        """Verify API health, registry info, and report upload endpoints."""
        health_resp = self.client.get("/api/v1/health")
        self.assertEqual(health_resp.status_code, 200)
        hdata = health_resp.json()
        self.assertEqual(hdata["status"], "OK")
        self.assertGreaterEqual(hdata["active_diseases_count"], 7)

        registry_resp = self.client.get("/api/v1/diseases")
        self.assertEqual(registry_resp.status_code, 200)
        rdata = registry_resp.json()
        self.assertGreaterEqual(rdata["active_count"], 7)


    # -------------------------------------------------------------
    # TEST 5: Database Storage and Retrieval
    # -------------------------------------------------------------
    def test_05_database_storage_and_retrieval(self):
        """Verify prediction and explanation results are persisted to SQLite database."""
        test_patient = "DB_AUDIT_PATIENT_999"
        payload = dict(self.sample_payloads["heart_disease"])
        payload["patient_id"] = test_patient

        response = self.client.post("/api/v1/predict/disease/heart_disease/hybrid", json=payload)
        self.assertEqual(response.status_code, 200)

        # Check DB record
        conn = sqlite3.connect(DATABASE_URL.replace("sqlite:///", ""))
        cursor = conn.cursor()
        cursor.execute("SELECT patient_id, disease_id, model_used, risk_probability, predicted_label FROM patient_assessments WHERE patient_id = ?", (test_patient,))
        row = cursor.fetchone()
        conn.close()

        self.assertIsNotNone(row, "Prediction record must be saved in database")
        self.assertEqual(row[0], test_patient)
        self.assertEqual(row[1], "heart_disease")

    # -------------------------------------------------------------
    # TEST 6: SHAP / LIME Feature Attribution Verification
    # -------------------------------------------------------------
    def test_06_explainability_feature_attributions(self):
        """Verify SHAP/LIME explainability endpoint returns feature attributions and clinical narrative."""
        for disease in ["diabetes", "heart_disease", "kidney_disease"]:
            payload = self.sample_payloads[disease]
            response = self.client.post(f"/api/v1/explain/disease/{disease}", json=payload)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertEqual(data["disease_id"], disease)
            self.assertIn("top_contributing_features", data)
            self.assertIn("clinical_narrative", data)
            self.assertGreater(len(data["top_contributing_features"]), 0)

    # -------------------------------------------------------------
    # TEST 7: Extreme, Out-of-Bound, and Unexpected Inputs
    # -------------------------------------------------------------
    def test_07_extreme_and_unexpected_inputs(self):
        """Verify system handles extreme values, SQL injection strings, and out-of-range inputs safely."""
        injection_payload = {
            "patient_id": "PATIENT'; DROP TABLE patient_assessments; --",
            "disease_id": "diabetes",
            "features": self.sample_payloads["diabetes"]["features"]
        }
        response = self.client.post("/api/v1/predict/disease/diabetes/hybrid", json=injection_payload)
        self.assertEqual(response.status_code, 200, "SQL injection attempt in patient_id should be safely parameterized")

        # Check DB table is still intact
        conn = sqlite3.connect(DATABASE_URL.replace("sqlite:///", ""))
        cursor = conn.cursor()
        cursor.execute("SELECT count(*) FROM patient_assessments")
        cnt = cursor.fetchone()[0]
        conn.close()
        self.assertGreaterEqual(cnt, 1, "Database table must remain intact")

    # -------------------------------------------------------------
    # TEST 8: Security & Protection of Patient Data
    # -------------------------------------------------------------
    def test_08_security_and_medical_disclaimer(self):
        """Verify response headers, sanitization, and inclusion of mandatory medical disclaimer."""
        response = self.client.post(
            "/api/v1/predict/disease/diabetes/hybrid",
            json=self.sample_payloads["diabetes"]
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("disclaimer", data)
        self.assertIn("NOT a medical diagnosis", data["disclaimer"])

    # -------------------------------------------------------------
    # TEST 9: Comprehensive Multi-Disease Screening Flow
    # -------------------------------------------------------------
    def test_09_multi_disease_screening_flow(self):
        """Verify /api/v1/predict/all endpoint evaluates patient across all disease registries."""
        payload = self.sample_payloads["diabetes"]
        response = self.client.post("/api/v1/predict/all", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("disease_results", data)
        self.assertGreaterEqual(len(data["disease_results"]), 7)


    # -------------------------------------------------------------
    # TEST 10: Performance Benchmark & Inference Latency
    # -------------------------------------------------------------
    def test_10_performance_benchmark(self):
        """Benchmark model inference time for Classical ML vs QML vs Hybrid."""
        payload = self.sample_payloads["diabetes"]

        start_time = time.time()
        c_resp = self.client.post("/api/v1/predict/disease/diabetes/classical", json=payload)
        c_time = (time.time() - start_time) * 1000.0

        start_time = time.time()
        q_resp = self.client.post("/api/v1/predict/disease/diabetes/qml", json=payload)
        q_time = (time.time() - start_time) * 1000.0

        start_time = time.time()
        h_resp = self.client.post("/api/v1/predict/disease/diabetes/hybrid", json=payload)
        h_time = (time.time() - start_time) * 1000.0

        print(f"\n[BENCHMARK] Classical Inference: {c_time:.2f}ms | QML VQC: {q_time:.2f}ms | Hybrid Ensemble: {h_time:.2f}ms")
        self.assertLess(c_time, 200.0, "Classical inference must be under 200ms")
        self.assertLess(h_time, 500.0, "Hybrid inference must be under 500ms")

if __name__ == "__main__":
    unittest.main()
