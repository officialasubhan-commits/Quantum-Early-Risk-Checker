import unittest
from voice_ai.app.conversation.question_flow import (
    get_questions_for_disease,
    get_question_by_id,
    CLINICAL_QUESTIONS_CATALOG
)
from voice_ai.app.conversation.intent_handler import ConversationalIntentHandler
from voice_ai.app.conversation.state_manager import ConversationSession, ConversationStateManager
from voice_ai.app.extraction.medical_extractor import MedicalEntityExtractor
from voice_ai.app.services.voice_pipeline import VoiceAIPipelineService

class TestConversationalVoiceAssessment(unittest.TestCase):
    def setUp(self):
        self.state_manager = ConversationStateManager()
        self.extractor = MedicalEntityExtractor()
        self.pipeline = VoiceAIPipelineService()

    def test_01_catalog_and_disease_tailoring(self):
        """Verifies all 29 questions exist and questions are tailored per disease schema."""
        self.assertEqual(len(CLINICAL_QUESTIONS_CATALOG), 29)

        # Diabetes questions
        diabetes_qs = get_questions_for_disease("diabetes")
        d_ids = [q["id"] for q in diabetes_qs]
        self.assertIn("q1_age", d_ids)
        self.assertIn("q2_sex", d_ids)
        self.assertIn("q3_height", d_ids)
        self.assertIn("q4_weight", d_ids)
        self.assertIn("q6_high_bp", d_ids)
        self.assertIn("q15_smoker", d_ids)
        self.assertIn("q19_fruits_veggies", d_ids)
        # Verify irrelevant questions like chest discomfort (cp - heart disease only) are pruned
        self.assertNotIn("q28_chest_discomfort", d_ids)

        # Heart disease questions
        heart_qs = get_questions_for_disease("heart_disease")
        h_ids = [q["id"] for q in heart_qs]
        self.assertIn("q28_chest_discomfort", h_ids)
        self.assertIn("q7_bp_reading", h_ids)
        self.assertIn("q8_heart_rate", h_ids)

    def test_02_single_question_flow_with_acknowledgments(self):
        """Verifies assistant asks ONE question at a time and provides warm confirmation receipts."""
        session = ConversationSession("test_session_flow", disease_id="diabetes", language="en")
        initial_speech, first_q = session.start_assessment("diabetes", "en")

        assert first_q is not None
        self.assertEqual(first_q["id"], "q1_age")
        self.assertEqual(initial_speech, "What is your age?")

        # Turn 1: User provides age
        res1 = session.handle_spoken_turn("I am 45 years old.")
        self.assertEqual(session.collected_features.get("age"), 45.0)
        self.assertEqual(session.collected_features.get("Age"), 6) # 45-49 is BRFSS category 6
        self.assertIn("Got it. I've recorded your age as 45.", res1["assistant_reply"])
        self.assertIn("What is your biological sex?", res1["assistant_reply"])

        # Turn 2: User provides sex
        res2 = session.handle_spoken_turn("Male")
        self.assertEqual(session.collected_features.get("Sex"), 1)
        self.assertIn("recorded as Male", res2["assistant_reply"])
        self.assertIn("What is your height?", res2["assistant_reply"])

    def test_03_multi_answer_extraction_and_question_skipping(self):
        """Verifies if user provides height and weight together, both are recorded, BMI calculated, and upcoming questions skipped."""
        session = ConversationSession("test_session_multi", disease_id="diabetes", language="en")
        session.start_assessment("diabetes", "en")

        # Answer age
        session.handle_spoken_turn("I am 45")
        # Answer sex
        session.handle_spoken_turn("Female")

        # Current question is height: User responds with both height and weight
        q3 = session.get_current_question()
        assert q3 is not None
        self.assertEqual(q3["id"], "q3_height")
        turn_res = session.handle_spoken_turn("My height is 165 centimeters and my weight is 70 kilograms.")

        self.assertEqual(session.collected_features.get("height"), 165.0)
        self.assertEqual(session.collected_features.get("weight"), 70.0)
        bmi_val = session.collected_features.get("BMI")
        assert isinstance(bmi_val, (int, float))
        self.assertAlmostEqual(float(bmi_val), 25.7, places=1)

        # Weight question should be skipped since weight was already extracted!
        next_q = turn_res.get("current_question")
        self.assertIsNotNone(next_q)
        assert next_q is not None
        self.assertNotEqual(next_q["id"], "q4_weight")
        self.assertEqual(next_q["id"], "q5_family_history")

    def test_04_user_correction_handling(self):
        """Verifies user can correct an existing answer."""
        session = ConversationSession("test_session_correction", disease_id="diabetes", language="en")
        session.start_assessment("diabetes", "en")

        session.handle_spoken_turn("I am 40 years old")
        self.assertEqual(session.collected_features["age"], 40.0)
        self.assertEqual(session.collected_features["Age"], 5)

        # User corrects
        session.handle_spoken_turn("Actually I am 52 years old")
        self.assertEqual(session.collected_features["age"], 52.0)
        self.assertEqual(session.collected_features["Age"], 7)

    def test_05_dont_know_and_skip_commands(self):
        """Verifies 'I don't know' stores value as unknown/null without guessing and 'skip' advances."""
        session = ConversationSession("test_session_commands", disease_id="diabetes", language="en")
        session.start_assessment("diabetes", "en")

        # User says "I don't know" to age
        res = session.handle_spoken_turn("I don't know")
        self.assertIn("Age", session.unknown_features)
        self.assertIsNone(session.collected_features.get("Age"))
        self.assertIn("mark that as unknown", res["assistant_reply"])

        # Current question is now sex: user says skip
        q_sex = session.get_current_question()
        assert q_sex is not None
        self.assertEqual(q_sex["id"], "q2_sex")
        res_skip = session.handle_spoken_turn("skip")
        self.assertIn("Sex", session.unknown_features)
        self.assertIn("Skipped", res_skip["assistant_reply"])
        q_h = session.get_current_question()
        assert q_h is not None
        self.assertEqual(q_h["id"], "q3_height")

    def test_06_repeat_and_back_commands(self):
        """Verifies repeat repeats the current question and back returns to previous question."""
        session = ConversationSession("test_session_nav", disease_id="diabetes", language="en")
        session.start_assessment("diabetes", "en")
        session.handle_spoken_turn("45")

        # Current is sex
        q_sex2 = session.get_current_question()
        assert q_sex2 is not None
        self.assertEqual(q_sex2["id"], "q2_sex")

        # Repeat
        rep_res = session.handle_spoken_turn("repeat")
        self.assertIn("What is your biological sex?", rep_res["assistant_reply"])
        q_sex3 = session.get_current_question()
        assert q_sex3 is not None
        self.assertEqual(q_sex3["id"], "q2_sex")

        # Go back
        back_res = session.handle_spoken_turn("go back")
        self.assertIn("What is your age?", back_res["assistant_reply"])
        q_age = session.get_current_question()
        assert q_age is not None
        self.assertEqual(q_age["id"], "q1_age")

    def test_07_multilingual_hindi_and_bengali(self):
        """Verifies Hindi and Bengali spoken assessments ask questions in the respective language."""
        # Hindi
        session_hi = ConversationSession("test_session_hi", disease_id="diabetes", language="hi")
        speech_hi, q_hi = session_hi.start_assessment("diabetes", "hi")
        self.assertIn("आपकी उम्र क्या है?", speech_hi)

        turn_hi = session_hi.handle_spoken_turn("मेरी उम्र 50 साल है")
        self.assertEqual(session_hi.collected_features.get("age"), 50.0)
        self.assertIn("दर्ज कर ली है", turn_hi["assistant_reply"])

        # Bengali
        session_bn = ConversationSession("test_session_bn", disease_id="diabetes", language="bn")
        speech_bn, q_bn = session_bn.start_assessment("diabetes", "bn")
        self.assertIn("আপনার বয়স কত?", speech_bn)

        turn_bn = session_bn.handle_spoken_turn("আমার বয়স ৪৮ বছর")
        self.assertEqual(session_bn.collected_features.get("age"), 48.0)
        self.assertIn("রেকর্ড করেছি", turn_bn["assistant_reply"])

    def test_08_pipeline_endpoints(self):
        """Verifies pipeline service conversational start and turn methods with TTS output."""
        start_res = self.pipeline.start_conversational_assessment("test_pipeline_session", "diabetes", "en")
        self.assertEqual(start_res.disease_id, "diabetes")
        assert start_res.current_question is not None
        self.assertEqual(start_res.current_question.id, "q1_age")
        self.assertIsNotNone(start_res.audio_base64)
        assert start_res.audio_base64 is not None
        self.assertTrue(len(start_res.audio_base64) > 100)

        turn_res = self.pipeline.process_conversational_turn("test_pipeline_session", "I am 50", "diabetes", "en")
        self.assertEqual(turn_res.extracted_data.get("age"), 50.0)
        self.assertIsNotNone(turn_res.audio_base64)
        assert turn_res.audio_base64 is not None
        self.assertTrue(len(turn_res.audio_base64) > 100)

if __name__ == "__main__":
    unittest.main()
