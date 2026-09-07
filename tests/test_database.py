import os
import sys
import unittest
import uuid
import datetime

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.database.connection import engine, Base, SessionLocal
from backend.database.models import AssessmentRecord, ModelRegistryRecord
from backend.database.repository import (
    save_assessment_record,
    get_assessment_records,
    get_assessment_by_request_id,
    register_model,
    get_registered_models
)
from backend.database.init_db import init_database

class TestDatabaseLayer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize database tables & seed model registry
        init_database()
        cls.db = SessionLocal()
        cls.test_req_id = str(uuid.uuid4())
        cls.test_patient_id = "PATIENT_DB_TEST_999"

    @classmethod
    def tearDownClass(cls):
        cls.db.close()

    def test_01_database_tables_exist(self):
        tables = engine.table_names() if hasattr(engine, "table_names") else engine.dialect.get_table_names(engine.connect())
        self.assertIn("patient_assessments", tables)
        self.assertIn("model_registry", tables)

    def test_02_insert_and_retrieve_assessment(self):
        sample_features = {
            "BMI": 30.5,
            "HighBP": 1,
            "HighChol": 1,
            "Age": 9
        }
        
        record = save_assessment_record(
            db=self.db,
            request_id=self.test_req_id,
            patient_id=self.test_patient_id,
            model_used="Hybrid Classical-Quantum Ensemble Classifier",
            predicted_class=1,
            predicted_label="Elevated Diabetes/Prediabetes Risk",
            risk_probability=0.7850,
            is_high_risk=True,
            feature_inputs=sample_features,
            top_contributing_features=[{"feature": "HighBP", "score": 0.4}],
            clinical_narrative="Test assessment clinical narrative.",
            disclaimer="Test Disclaimer"
        )

        self.assertIsNotNone(record.id)
        self.assertEqual(record.request_id, self.test_req_id)
        self.assertEqual(record.patient_id, self.test_patient_id)
        self.assertEqual(record.risk_probability, 0.7850)

    def test_03_retrieve_by_request_id(self):
        retrieved = get_assessment_by_request_id(self.db, self.test_req_id)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.patient_id, self.test_patient_id)
        self.assertEqual(retrieved.predicted_class, 1)

    def test_04_query_patient_records(self):
        records = get_assessment_records(self.db, patient_id=self.test_patient_id)
        self.assertGreater(len(records), 0)
        self.assertEqual(records[0].patient_id, self.test_patient_id)

    def test_05_model_registry_seeded(self):
        models = get_registered_models(self.db)
        self.assertGreaterEqual(len(models), 3)
        model_names = [m.model_name for m in models]
        self.assertTrue(any("Hybrid" in name for name in model_names))

if __name__ == "__main__":
    unittest.main()
