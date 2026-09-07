import os
import sys
import unittest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app
from backend.database.connection import SessionLocal
from backend.database.repository import get_assessment_records

class TestMultiDiseaseIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.test_payloads = {
            "diabetes": {
                "patient_id": "PATIENT_INTEG_DIABETES",
                "disease_id": "diabetes",
                "features": {
                    "BMI": 32.0, "GenHlth": 4, "MentHlth": 5.0, "PhysHlth": 10.0,
                    "HighBP": 1, "HighChol": 1, "CholCheck": 1, "Smoker": 1,
                    "Stroke": 0, "HeartDiseaseorAttack": 0, "PhysActivity": 0,
                    "Fruits": 0, "Veggies": 1, "HvyAlcoholConsump": 0, "AnyHealthcare": 1,
                    "NoDocbcCost": 0, "DiffWalk": 1, "Sex": 1, "Age": 11,
                    "Education": 4, "Income": 5
                }
            },
            "heart_disease": {
                "patient_id": "PATIENT_INTEG_HEART",
                "disease_id": "heart_disease",
                "features": {
                    "age": 63.0, "sex": 1.0, "cp": 1.0, "trestbps": 145.0, "chol": 233.0,
                    "fbs": 1.0, "restecg": 2.0, "thalach": 150.0, "exang": 0.0, "oldpeak": 2.3,
                    "slope": 3.0, "ca": 0.0, "thal": 6.0
                }
            },
            "kidney_disease": {
                "patient_id": "PATIENT_INTEG_CKD",
                "disease_id": "kidney_disease",
                "features": {
                    "age": 55.0, "bp": 80.0, "sg": 1.02, "al": 1.0, "su": 0.0, "rbc": 0.0,
                    "pc": 0.0, "pcc": 0.0, "ba": 0.0, "bgr": 120.0, "bu": 36.0, "sc": 1.2,
                    "sod": 137.0, "pot": 4.4, "hemo": 15.4, "pcv": 44.0, "wbcc": 7800.0,
                    "rbcc": 5.2, "htn": 1.0, "dm": 0.0, "cad": 0.0, "appet": 0.0, "pe": 0.0, "ane": 0.0
                }
            },
            "liver_disease": {
                "patient_id": "PATIENT_INTEG_LIVER",
                "disease_id": "liver_disease",
                "features": {
                    "Age": 65.0, "Gender": 1.0, "Total_Bilirubin": 0.7, "Direct_Bilirubin": 0.1,
                    "Alkaline_Phosphotase": 187.0, "Alamine_Aminotransferase": 16.0,
                    "Aspartate_Aminotransferase": 18.0, "Total_Protiens": 6.8, "Albumin": 3.3,
                    "Albumin_and_Globulin_Ratio": 0.9
                }
            },
            "stroke": {
                "patient_id": "PATIENT_INTEG_STROKE",
                "disease_id": "stroke",
                "features": {
                    "gender": 1.0, "age": 67.0, "hypertension": 0.0, "heart_disease": 1.0,
                    "ever_married": 1.0, "work_type": 2.0, "Residence_type": 1.0,
                    "avg_glucose_level": 228.69, "bmi": 36.6, "smoking_status": 1.0
                }
            },
            "breast_cancer": {
                "patient_id": "PATIENT_INTEG_BREAST",
                "disease_id": "breast_cancer",
                "features": {
                    "mean radius": 17.99, "mean texture": 10.38, "mean perimeter": 122.8, "mean area": 1001.0,
                    "mean smoothness": 0.1184, "mean compactness": 0.2776, "mean concavity": 0.3001,
                    "mean concave points": 0.1471, "mean symmetry": 0.2419, "mean fractal dimension": 0.07871,
                    "radius error": 1.095, "texture error": 0.9053, "perimeter error": 8.589, "area error": 153.4,
                    "smoothness error": 0.006399, "compactness error": 0.04904, "concavity error": 0.05373,
                    "concave points error": 0.01587, "symmetry error": 0.03003, "fractal dimension error": 0.006193,
                    "worst radius": 25.38, "worst texture": 17.33, "worst perimeter": 184.6, "worst area": 2019.0,
                    "worst smoothness": 0.1622, "worst compactness": 0.6656, "worst concavity": 0.7119,
                    "worst concave points": 0.2654, "worst symmetry": 0.4601, "worst fractal dimension": 0.1189
                }
            },
            "parkinsons": {
                "patient_id": "PATIENT_INTEG_PARKINSONS",
                "disease_id": "parkinsons",
                "features": {
                    "MDVP:Fo(Hz)": 119.992, "MDVP:Fhi(Hz)": 157.302, "MDVP:Flo(Hz)": 74.997,
                    "MDVP:Jitter(%)": 0.00784, "MDVP:Jitter(Abs)": 0.00007, "MDVP:RAP": 0.0037,
                    "MDVP:PPQ": 0.00554, "Jitter:DDP": 0.01109, "MDVP:Shimmer": 0.04374,
                    "MDVP:Shimmer(dB)": 0.426, "Shimmer:APQ3": 0.02182, "Shimmer:APQ5": 0.0313,
                    "MDVP:APQ": 0.02971, "Shimmer:DDA": 0.06545, "NHR": 0.02211, "HNR": 21.033,
                    "RPDE": 0.414783, "DFA": 0.815285, "spread1": -4.813031, "spread2": 0.266482,
                    "D2": 2.301442, "PPE": 0.284654
                }
            }
        }

    def test_01_registry_contains_all_diseases(self):
        response = self.client.get("/api/v1/diseases")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(data["active_count"], 7)
        self.assertGreaterEqual(data["total_count"], 7)

    def test_02_predict_individual_diseases(self):
        for d_id, payload in self.test_payloads.items():
            url = f"/api/v1/predict/disease/{d_id}/hybrid"
            response = self.client.post(url, json=payload)
            self.assertEqual(response.status_code, 200, f"Failed prediction for '{d_id}': {response.text}")
            data = response.json()
            self.assertEqual(data["disease_id"], d_id)
            self.assertGreaterEqual(data["risk_probability"], 0.0)
            self.assertLessEqual(data["risk_probability"], 1.0)
            self.assertIn("disclaimer", data)

    def test_03_explain_individual_diseases(self):
        for d_id, payload in self.test_payloads.items():
            url = f"/api/v1/explain/disease/{d_id}"
            response = self.client.post(url, json=payload)
            self.assertEqual(response.status_code, 200, f"Failed explainability for '{d_id}': {response.text}")
            data = response.json()
            self.assertEqual(data["disease_id"], d_id)
            self.assertGreater(len(data["top_contributing_features"]), 0)
            self.assertIn("clinical_narrative", data)

    def test_04_input_validation_catches_missing_features(self):
        invalid_payload = {
            "patient_id": "PATIENT_INVALID_01",
            "disease_id": "heart_disease",
            "features": {"age": 63.0}  # Missing required features!
        }
        url = "/api/v1/predict/disease/heart_disease/hybrid"
        response = self.client.post(url, json=invalid_payload)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Missing required clinical features", response.json()["detail"])

    def test_05_predict_all_diseases_flow(self):
        payload = self.test_payloads["diabetes"]
        url = "/api/v1/predict/all"
        response = self.client.post(url, json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreaterEqual(len(data["disease_results"]), 7)


    def test_06_database_persistence_records(self):
        db = SessionLocal()
        records = get_assessment_records(db, limit=10)
        db.close()
        self.assertGreater(len(records), 0)
        rec = records[0]
        self.assertIsNotNone(rec.disease_id)
        self.assertIsNotNone(rec.disease_name)
        self.assertIsNotNone(rec.model_used)
        self.assertIsNotNone(rec.timestamp)

if __name__ == "__main__":
    unittest.main()
