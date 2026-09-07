import unittest
from voice_ai.app.extraction.medical_extractor import MedicalEntityExtractor

class TestMedicalEntityExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = MedicalEntityExtractor()

    def test_01_extract_age_blood_pressure_and_smoking(self):
        transcript = "I am 45 years old. I have high blood pressure and I smoke."
        params, updated = self.extractor.extract_clinical_entities(transcript)

        self.assertEqual(params.get("age"), 45.0)
        self.assertEqual(params.get("Age"), 6) # 45 yo is BRFSS age category 6 (45-49)
        self.assertEqual(params.get("HighBP"), 1)
        self.assertEqual(params.get("Smoker"), 1)
        self.assertIn("age", updated)
        self.assertIn("HighBP", updated)
        self.assertIn("Smoker", updated)

    def test_02_extract_height_weight_and_bmi(self):
        transcript = "My weight is 88 kg and my height is 175 cm. Cholesterol is high."
        params, updated = self.extractor.extract_clinical_entities(transcript)

        bmi = params.get("BMI")
        self.assertIsNotNone(bmi)
        self.assertAlmostEqual(float(bmi or 0.0), 28.7, places=1)
        self.assertEqual(params.get("HighChol"), 1)
        self.assertEqual(params.get("CholCheck"), 1)

    def test_03_extract_user_corrections(self):
        initial_params = {"age": 45.0, "Age": 6, "HighBP": 1}
        correction_transcript = "Actually I am 52 years old."
        updated_params, updated = self.extractor.extract_clinical_entities(correction_transcript, initial_params)

        self.assertEqual(updated_params.get("age"), 52.0)
        self.assertEqual(updated_params.get("Age"), 7) # 52 yo is BRFSS age category 7 (50-54)
        self.assertEqual(updated_params.get("HighBP"), 1) # Retains HighBP

if __name__ == "__main__":
    unittest.main()
