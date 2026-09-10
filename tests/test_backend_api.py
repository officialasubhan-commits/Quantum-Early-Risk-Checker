import os
import sys
import unittest
from fastapi.testclient import TestClient

# Ensure root directory is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app

class TestFastAPIBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.valid_payload = {
            "patient_id": "PATIENT_TEST_UNIT_01",
            "features": {
                "BMI": 32.0,
                "GenHlth": 4,
                "MentHlth": 5.0,
                "PhysHlth": 10.0,
                "HighBP": 1,
                "HighChol": 1,
                "CholCheck": 1,
                "Smoker": 1,
                "Stroke": 0,
                "HeartDiseaseorAttack": 0,
                "PhysActivity": 0,
                "Fruits": 0,
                "Veggies": 1,
                "HvyAlcoholConsump": 0,
                "AnyHealthcare": 1,
                "NoDocbcCost": 0,
                "DiffWalk": 1,
                "Sex": 1,
                "Age": 11,
                "Education": 4,
                "Income": 5
            }
        }

    def test_01_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "OK")
        self.assertTrue(data["models_loaded"]["diabetes_all_models_ready"])
        self.assertTrue(data["models_loaded"]["heart_disease_all_models_ready"])

    def test_02_api_v1_health(self):
        response = self.client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "OK")

    def test_03_model_info(self):
        response = self.client.get("/api/v1/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("baseline_model", data)
        self.assertIn("qml_model", data)
        self.assertIn("hybrid_model", data)
        self.assertIn("medical_disclaimer", data)

    def test_04_predict_classical(self):
        response = self.client.post("/api/v1/predict/classical", json=self.valid_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["patient_id"], "PATIENT_TEST_UNIT_01")
        self.assertIn("RandomForestClassifier", data["model_used"])
        self.assertGreaterEqual(data["risk_probability"], 0.0)
        self.assertLessEqual(data["risk_probability"], 1.0)
        self.assertIn("disclaimer", data)

    def test_05_predict_qml(self):
        response = self.client.post("/api/v1/predict/qml", json=self.valid_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["patient_id"], "PATIENT_TEST_UNIT_01")
        self.assertIn("6-Qubit Variational Quantum Classifier", data["model_used"])
        self.assertGreaterEqual(data["risk_probability"], 0.0)
        self.assertLessEqual(data["risk_probability"], 1.0)

    def test_06_predict_hybrid(self):
        response = self.client.post("/api/v1/predict/hybrid", json=self.valid_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["patient_id"], "PATIENT_TEST_UNIT_01")
        self.assertIn("Hybrid Classical-Quantum Ensemble Classifier", data["model_used"])
        self.assertGreaterEqual(data["risk_probability"], 0.0)
        self.assertLessEqual(data["risk_probability"], 1.0)

    def test_07_explainability(self):
        response = self.client.post("/api/v1/explain", json=self.valid_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["patient_id"], "PATIENT_TEST_UNIT_01")
        self.assertGreater(len(data["top_contributing_features"]), 0)
        self.assertIn("clinical_narrative", data)
        self.assertIn("disclaimer", data)

    def test_09_get_registered_diseases(self):
        response = self.client.get("/api/v1/diseases")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("registered_diseases", data)
        self.assertGreaterEqual(data["active_count"], 7)
        self.assertGreaterEqual(data["total_count"], 7)
        self.assertIn("diabetes", data["registered_diseases"])
        self.assertIn("heart_disease", data["registered_diseases"])
        self.assertIn("kidney_disease", data["registered_diseases"])
        self.assertIn("liver_disease", data["registered_diseases"])
        self.assertIn("stroke", data["registered_diseases"])
        self.assertIn("breast_cancer", data["registered_diseases"])
        self.assertIn("parkinsons", data["registered_diseases"])

    def test_10_predict_disease_specific_endpoint(self):
        payload = {
            "patient_id": "PATIENT_MULTI_01",
            "disease_id": "diabetes",
            "features": self.valid_payload["features"]
        }
        response = self.client.post("/api/v1/predict/disease/diabetes/hybrid", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["disease_id"], "diabetes")
        self.assertIn("Diabetes", data["disease_name"])
        self.assertGreaterEqual(data["risk_probability"], 0.0)

    def test_11_predict_all_diseases_endpoint(self):
        payload = {
            "patient_id": "PATIENT_ALL_DISEASES_01",
            "features": self.valid_payload["features"]
        }
        response = self.client.post("/api/v1/predict/all", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["patient_id"], "PATIENT_ALL_DISEASES_01")
        self.assertIn("disease_results", data)
        self.assertGreaterEqual(len(data["disease_results"]), 7)

        
        # Check Diabetes Result
        diabetes_res = next((r for r in data["disease_results"] if r["disease_id"] == "diabetes"), None)
        self.assertIsNotNone(diabetes_res)
        assert diabetes_res is not None
        self.assertEqual(diabetes_res["status"], "SUCCESS")
        self.assertGreaterEqual(diabetes_res["risk_probability"], 0.0)

    def test_12_predict_heart_disease_endpoint(self):
        heart_payload = {
            "patient_id": "PATIENT_HEART_TEST_01",
            "disease_id": "heart_disease",
            "features": {
                "age": 63.0, "sex": 1.0, "cp": 1.0, "trestbps": 145.0, "chol": 233.0,
                "fbs": 1.0, "restecg": 2.0, "thalach": 150.0, "exang": 0.0, "oldpeak": 2.3,
                "slope": 3.0, "ca": 0.0, "thal": 6.0
            }
        }
        response = self.client.post("/api/v1/predict/disease/heart_disease/hybrid", json=heart_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["disease_id"], "heart_disease")
        self.assertIn("Heart Disease", data["disease_name"])
        self.assertGreaterEqual(data["risk_probability"], 0.0)
        self.assertLessEqual(data["risk_probability"], 1.0)

    def test_13_predict_kidney_disease_endpoint(self):
        kidney_payload = {
            "patient_id": "PATIENT_CKD_TEST_01",
            "disease_id": "kidney_disease",
            "features": {
                "age": 55.0, "bp": 80.0, "sg": 1.02, "al": 1.0, "su": 0.0, "rbc": 0.0,
                "pc": 0.0, "pcc": 0.0, "ba": 0.0, "bgr": 120.0, "bu": 36.0, "sc": 1.2,
                "sod": 137.0, "pot": 4.4, "hemo": 15.4, "pcv": 44.0, "wbcc": 7800.0,
                "rbcc": 5.2, "htn": 1.0, "dm": 0.0, "cad": 0.0, "appet": 0.0, "pe": 0.0, "ane": 0.0
            }
        }
        response = self.client.post("/api/v1/predict/disease/kidney_disease/hybrid", json=kidney_payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["disease_id"], "kidney_disease")
        self.assertGreaterEqual(data["risk_probability"], 0.0)

if __name__ == "__main__":
    unittest.main()
