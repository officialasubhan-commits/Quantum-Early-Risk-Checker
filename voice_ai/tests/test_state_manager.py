import unittest
from voice_ai.app.conversation.state_manager import ConversationStateManager, ConversationSession

class TestConversationStateManager(unittest.TestCase):
    def setUp(self):
        self.manager = ConversationStateManager()
        self.session_id = "test_session_001"

    def test_01_session_lifecycle_and_feature_tracking(self):
        session = self.manager.get_or_create_session(self.session_id, "diabetes")
        self.assertEqual(session.disease_id, "diabetes")

        session.update_features({"Age": 6, "HighBP": 1, "Smoker": 1})
        self.assertEqual(session.collected_features["Age"], 6)
        self.assertEqual(session.collected_features["HighBP"], 1)

    def test_02_detect_missing_features_and_generate_prompt(self):
        session = self.manager.get_or_create_session(self.session_id, "diabetes")
        session.update_features({"HighBP": 1, "Smoker": 1})

        required = ["Age", "BMI", "HighBP", "Smoker"]
        missing = session.get_missing_features(required)

        self.assertIn("Age", missing)
        self.assertIn("BMI", missing)

        prompt = self.manager.generate_followup_prompt(session, missing)
        self.assertIn("how old you are", prompt)

if __name__ == "__main__":
    unittest.main()
