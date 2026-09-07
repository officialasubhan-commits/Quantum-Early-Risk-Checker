import unittest
from voice_ai.app.services.voice_pipeline import VoiceAIPipelineService
from voice_ai.app.languages.language_manager import LanguageManager

class TestMultilingualVoicePipeline(unittest.TestCase):
    def setUp(self):
        self.pipeline = VoiceAIPipelineService()

    def test_1_english_conversation(self):
        """Test standard English conversation turn."""
        session_id = "test_session_en"
        resp1 = self.pipeline.process_text_turn(
            session_id=session_id,
            user_transcript="I am 45 years old, I have high blood pressure and I smoke.",
            disease_id="diabetes",
            language="en"
        )
        self.assertEqual(resp1.language, "en")
        self.assertEqual(resp1.extracted_data.get("age"), 45.0)
        self.assertEqual(resp1.extracted_data.get("HighBP"), 1)
        self.assertEqual(resp1.extracted_data.get("Smoker"), 1)
        self.assertFalse(resp1.is_ready_for_prediction)

        resp2 = self.pipeline.process_text_turn(
            session_id=session_id,
            user_transcript="My weight is 88 kg, height is 175 cm, cholesterol is high.",
            disease_id="diabetes",
            language="en"
        )
        self.assertTrue(resp2.is_ready_for_prediction)
        self.assertIsNotNone(resp2.prediction_result)

    def test_2_hindi_conversation(self):
        """Test Hindi conversation turn with Devanagari script."""
        session_id = "test_session_hi"
        resp = self.pipeline.process_text_turn(
            session_id=session_id,
            user_transcript="मेरी उम्र ४५ साल है और मुझे उच्च रक्तचाप (बीपी) की समस्या है।",
            disease_id="diabetes",
            language="hi"
        )
        self.assertEqual(resp.language, "hi")
        self.assertEqual(resp.extracted_data.get("age"), 45.0)
        self.assertEqual(resp.extracted_data.get("HighBP"), 1)

    def test_3_bengali_conversation(self):
        """Test Bengali conversation turn with Bengali script."""
        session_id = "test_session_bn"
        resp = self.pipeline.process_text_turn(
            session_id=session_id,
            user_transcript="আমার বয়স ৪০ বছর এবং আমার উচ্চ রক্তচাপ (High BP) আছে।",
            disease_id="diabetes",
            language="bn"
        )
        self.assertEqual(resp.language, "bn")
        self.assertEqual(resp.extracted_data.get("age"), 40.0)
        self.assertEqual(resp.extracted_data.get("HighBP"), 1)

    def test_4_hinglish_mixed_conversation(self):
        """Test Hindi-English code-switching conversation turn."""
        session_id = "test_session_hinglish"
        resp = self.pipeline.process_text_turn(
            session_id=session_id,
            user_transcript="Mera age 50 hai, blood pressure high hai and weight 85 kg hai.",
            disease_id="diabetes",
            language="auto"
        )
        self.assertEqual(resp.language, "hi")
        self.assertEqual(resp.extracted_data.get("age"), 50.0)
        self.assertEqual(resp.extracted_data.get("HighBP"), 1)

    def test_5_banglish_mixed_conversation(self):
        """Test Bengali-English code-switching conversation turn."""
        session_id = "test_session_banglish"
        resp = self.pipeline.process_text_turn(
            session_id=session_id,
            user_transcript="Amar age 40 years, cholesterol heavy high ache and I smoke cigarettes.",
            disease_id="diabetes",
            language="auto"
        )
        self.assertEqual(resp.language, "bn")
        self.assertEqual(resp.extracted_data.get("age"), 40.0)
        self.assertEqual(resp.extracted_data.get("HighChol"), 1)
        self.assertEqual(resp.extracted_data.get("Smoker"), 1)

    def test_6_language_switching_during_assessment(self):
        """Test language switching mid-assessment (Hindi -> English)."""
        session_id = "test_session_switch"
        # Turn 1 in Hindi
        resp1 = self.pipeline.process_text_turn(
            session_id=session_id,
            user_transcript="मेरी उम्र ५० साल है और मैं धूम्रपान करता हूं।",
            disease_id="diabetes",
            language="hi"
        )
        self.assertEqual(resp1.language, "hi")
        self.assertEqual(resp1.extracted_data.get("age"), 50.0)
        self.assertEqual(resp1.extracted_data.get("Smoker"), 1)

        # Turn 2 switch to English
        resp2 = self.pipeline.process_text_turn(
            session_id=session_id,
            user_transcript="My weight is 80 kg, height is 170 cm, BP is high, and cholesterol is high.",
            disease_id="diabetes",
            language="en"
        )
        self.assertEqual(resp2.language, "en")
        self.assertEqual(resp2.extracted_data.get("age"), 50.0)
        self.assertEqual(resp2.extracted_data.get("HighBP"), 1)
        self.assertEqual(resp2.extracted_data.get("HighChol"), 1)
        self.assertTrue(resp2.is_ready_for_prediction)



    def test_7_fallback_handling_for_unsupported_language(self):
        """Test clear fallback handling when unsupported language is passed."""
        session_id = "test_session_fallback"
        resp = self.pipeline.process_text_turn(
            session_id=session_id,
            user_transcript="Hello, I am 35 years old.",
            disease_id="diabetes",
            language="invalid_lang_code"
        )
        self.assertTrue(resp.fallback_used)
        self.assertIsNotNone(resp.fallback_reason)
        self.assertIn("unsupported", (resp.fallback_reason or "").lower())
        self.assertEqual(resp.language, "en")

if __name__ == "__main__":
    unittest.main()
