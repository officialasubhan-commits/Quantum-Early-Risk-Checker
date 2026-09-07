import unittest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="starlette")
from starlette.testclient import TestClient
from voice_ai.app.main import app

class TestVoiceAIService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_endpoint(self):
        response = self.client.get("/api/v1/voice/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "OK")
        self.assertIn("Voice AI", data["service"])

    def test_02_multi_turn_voice_assessment_flow(self):
        session_id = "test_flow_session_101"

        # Turn 1: Partial information
        payload_1 = {
            "session_id": session_id,
            "user_input": "I am 45 years old. I have high blood pressure and I smoke.",
            "disease_id": "diabetes"
        }
        resp1 = self.client.post("/api/v1/voice/chat", json=payload_1)
        self.assertEqual(resp1.status_code, 200)
        data1 = resp1.json()

        self.assertFalse(data1["is_ready_for_prediction"])
        self.assertIn("BMI", data1["missing_features"])
        self.assertIn("audio_base64", data1)

        # Turn 2: Provide height, weight, and cholesterol
        payload_2 = {
            "session_id": session_id,
            "user_input": "My weight is 88 kg, height is 175 cm, cholesterol is high.",
            "disease_id": "diabetes"
        }
        resp2 = self.client.post("/api/v1/voice/chat", json=payload_2)
        self.assertEqual(resp2.status_code, 200)
        data2 = resp2.json()

        self.assertTrue(data2["is_ready_for_prediction"])
        self.assertIsNotNone(data2["prediction_result"])
        self.assertIn("disclaimer", data2)
        self.assertIn("NOT a medical diagnosis", data2["disclaimer"])

if __name__ == "__main__":
    unittest.main()
