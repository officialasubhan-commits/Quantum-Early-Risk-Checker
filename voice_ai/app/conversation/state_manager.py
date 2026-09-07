from typing import Dict, Any, List, Optional, Tuple
from voice_ai.app.languages.language_manager import LanguageManager
from voice_ai.app.conversation.question_flow import (
    get_questions_for_disease,
    get_question_by_id,
    CLINICAL_QUESTIONS_CATALOG
)
from voice_ai.app.conversation.intent_handler import ConversationalIntentHandler
from voice_ai.app.extraction.medical_extractor import MedicalEntityExtractor

# Clinical default fallback values for non-critical demographic/lifestyle indicators
CLINICAL_DEFAULTS = {
    "MentHlth": 0.0, "PhysHlth": 0.0, "Fruits": 1, "Veggies": 1, "HvyAlcoholConsump": 0,
    "AnyHealthcare": 1, "NoDocbcCost": 0, "DiffWalk": 0, "Education": 5, "Income": 6,
    "cp": 0.0, "fbs": 0.0, "restecg": 0.0, "thalach": 150.0, "exang": 0.0, "oldpeak": 0.0,
    "slope": 1.0, "ca": 0.0, "thal": 2.0, "heart_rate": 72.0, "serum_sodium": 140.0,
    "serum_potassium": 4.3, "family_history": 0, "physical_activity_hours": 2.0,
    "CholCheck": 1, "GenHlth": 2, "PhysActivity": 1, "Stroke": 0, "HeartDiseaseorAttack": 0,
    "Sex": 1, "sex": 1.0
}

COMPLETION_MESSAGES = {
    "en": "I've collected the information needed for your assessment. I'll show you what I understood so you can review and correct anything before we continue.",
    "hi": "मैंने आपके मूल्यांकन के लिए आवश्यक जानकारी एकत्र कर ली है। मैं आपके सामने विवरण प्रस्तुत कर रहा हूँ ताकि आगे बढ़ने से पहले आप इसकी समीक्षा कर सकें।",
    "bn": "আমি আপনার মূল্যায়নের জন্য প্রয়োজনীয় তথ্য সংগ্রহ করেছি। আমি যা বুঝতে পেরেছি তা দেখাচ্ছি যাতে এগিয়ে যাওয়ার আগে আপনি পর্যালোচনা ও সংশোধন করতে পারেন।"
}

CLARIFICATION_PREFIXES = {
    "en": "I didn't quite catch that.",
    "hi": "माफ़ कीजिए, मैं समझ नहीं पाया।",
    "bn": "আমি ঠিক বুঝতে পারিনি।"
}

class ConversationSession:
    """
    Stateful multi-turn conversational session conducting single-question spoken assessments.
    Maintains question queue, current question, collected clinical features,
    unknown values, and review synchronization state.
    """
    def __init__(self, session_id: str, disease_id: str = "diabetes", language: str = "en"):
        self.session_id = session_id
        self.disease_id = disease_id.lower()
        self.active_language: str = language
        self.collected_features: Dict[str, Any] = {}
        self.unknown_features: List[str] = []
        self.skipped_features: List[str] = []
        self.history: List[Dict[str, str]] = []
        self.is_completed: bool = False
        self.status: str = "NOT_STARTED" # NOT_STARTED, IN_PROGRESS, PAUSED, AWAITING_REVIEW, COMPLETED

        self.question_queue: List[Dict[str, Any]] = []
        self.current_question_index: int = 0
        self.extractor = MedicalEntityExtractor()

        self._initialize_queue()

    def _initialize_queue(self):
        """Initializes tailored question queue for selected disease model."""
        self.question_queue = get_questions_for_disease(self.disease_id)
        self.current_question_index = 0

    def start_assessment(self, disease_id: Optional[str] = None, language: Optional[str] = None) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Starts or resets the spoken conversational assessment.
        Returns (initial_assistant_speech, current_question_dict).
        """
        if disease_id:
            self.disease_id = disease_id.lower()
        if language and language != "auto":
            self.active_language = language

        self._initialize_queue()
        self.collected_features.clear()
        self.unknown_features.clear()
        self.skipped_features.clear()
        self.history.clear()
        self.is_completed = False
        self.status = "IN_PROGRESS"

        first_q = self.get_current_question()
        if not first_q:
            return "No questions queued for this assessment.", None

        lang = self.active_language or "en"
        speech = first_q.get("questions", {}).get(lang) or first_q.get("questions", {}).get("en", "")
        self.history.append({"speaker": "assistant", "text": speech})
        return speech, first_q

    def get_current_question(self) -> Optional[Dict[str, Any]]:
        """Returns the currently active question dictionary."""
        if 0 <= self.current_question_index < len(self.question_queue):
            return self.question_queue[self.current_question_index]
        return None

    def is_question_answered(self, question: Dict[str, Any]) -> bool:
        """Checks if all primary feature keys for a question have been collected."""
        keys = question.get("feature_keys", [])
        if not keys:
            return False
        # If any key is in collected_features or unknown_features, it's considered answered
        for k in keys:
            if k in self.collected_features or k in self.unknown_features:
                return True
        return False

    def advance_to_next_unanswered_question(self) -> Optional[Dict[str, Any]]:
        """
        Advances the pointer to the next question in the queue that has not yet been answered.
        Skips questions already answered via multi-entity user responses.
        """
        self.current_question_index += 1
        while self.current_question_index < len(self.question_queue):
            q = self.question_queue[self.current_question_index]
            if not self.is_question_answered(q):
                return q
            self.current_question_index += 1
        return None

    def handle_spoken_turn(self, user_text: str) -> Dict[str, Any]:
        """
        Core conversational dialogue engine:
        1. Checks for control commands (skip, go back, repeat, stop, don't know, change).
        2. Extracts clinical entities using MedicalEntityExtractor.
        3. Formulates warm conversational acknowledgment.
        4. Transitions to next question or completion review.
        """
        lang = self.active_language or "en"
        self.history.append({"speaker": "user", "text": user_text})

        current_q = self.get_current_question()
        if not current_q or self.status == "AWAITING_REVIEW" or self.status == "COMPLETED":
            completion_msg = COMPLETION_MESSAGES.get(lang, COMPLETION_MESSAGES["en"])
            return {
                "assistant_reply": completion_msg,
                "current_question": None,
                "current_question_index": self.current_question_index,
                "total_questions": len(self.question_queue),
                "is_completed": True,
                "status": "AWAITING_REVIEW",
                "extracted_data": self.collected_features,
                "newly_extracted": {}
            }

        # 1. Detect Conversational Control Commands
        cmd = ConversationalIntentHandler.detect_command(user_text)

        if cmd == "stop":
            self.status = "PAUSED"
            msg = "Assessment paused. Say 'continue' or click start whenever you're ready to proceed." if lang == "en" else "मूल्यांकन रोक दिया गया है। जब भी आप तैयार हों, जारी रखें।"
            return {
                "assistant_reply": msg,
                "current_question": current_q,
                "current_question_index": self.current_question_index,
                "total_questions": len(self.question_queue),
                "is_completed": False,
                "status": "PAUSED",
                "extracted_data": self.collected_features,
                "newly_extracted": {}
            }

        if cmd == "repeat":
            repeat_q = current_q.get("questions", {}).get(lang) or current_q.get("questions", {}).get("en", "")
            return {
                "assistant_reply": repeat_q,
                "current_question": current_q,
                "current_question_index": self.current_question_index,
                "total_questions": len(self.question_queue),
                "is_completed": False,
                "status": self.status,
                "extracted_data": self.collected_features,
                "newly_extracted": {}
            }

        if cmd == "back":
            if self.current_question_index > 0:
                self.current_question_index -= 1
                prev_q = self.get_current_question()
                q_text = prev_q.get("questions", {}).get(lang) or prev_q.get("questions", {}).get("en", "") if prev_q else ""
                msg = f"Going back. {q_text}" if lang == "en" else f"पिछले प्रश्न पर लौट रहे हैं। {q_text}"
                return {
                    "assistant_reply": msg,
                    "current_question": prev_q,
                    "current_question_index": self.current_question_index,
                    "total_questions": len(self.question_queue),
                    "is_completed": False,
                    "status": self.status,
                    "extracted_data": self.collected_features,
                    "newly_extracted": {}
                }

        if cmd == "skip" or cmd == "dont_know":
            # Store value as unknown/null without guessing
            for k in current_q.get("feature_keys", []):
                self.unknown_features.append(k)
                if k not in self.collected_features:
                    self.collected_features[k] = None

            next_q = self.advance_to_next_unanswered_question()
            if next_q:
                next_text = next_q.get("questions", {}).get(lang) or next_q.get("questions", {}).get("en", "")
                ack = "Understood, we'll mark that as unknown. " if cmd == "dont_know" else "Skipped. "
                if lang == "hi":
                    ack = "ठीक है, इसे अज्ञात चिह्नित कर दिया गया है। " if cmd == "dont_know" else "प्रश्न छोड़ दिया गया। "
                elif lang == "bn":
                    ack = "বুঝেছি, এটি অজানা হিসেবে রাখা হলো। " if cmd == "dont_know" else "প্রশ্নটি বাদ দেওয়া হলো। "

                reply = ack + next_text
                return {
                    "assistant_reply": reply,
                    "current_question": next_q,
                    "current_question_index": self.current_question_index,
                    "total_questions": len(self.question_queue),
                    "is_completed": False,
                    "status": "IN_PROGRESS",
                    "extracted_data": self.collected_features,
                    "newly_extracted": {}
                }
            else:
                self.status = "AWAITING_REVIEW"
                self.is_completed = True
                completion_msg = COMPLETION_MESSAGES.get(lang, COMPLETION_MESSAGES["en"])
                return {
                    "assistant_reply": completion_msg,
                    "current_question": None,
                    "current_question_index": self.current_question_index,
                    "total_questions": len(self.question_queue),
                    "is_completed": True,
                    "status": "AWAITING_REVIEW",
                    "extracted_data": self.collected_features,
                    "newly_extracted": {}
                }

        # 2. Extract Clinical Entities for the Active Question
        updated_params, updated_keys = self.extractor.extract_clinical_entities(
            user_text,
            existing_params=self.collected_features,
            active_question_id=current_q.get("id")
        )

        newly_extracted = {k: updated_params[k] for k in updated_keys if k in updated_params}

        # 3. Handle Ambiguous / Unclear Response
        if not updated_keys:
            clarify = CLARIFICATION_PREFIXES.get(lang, CLARIFICATION_PREFIXES["en"])
            q_text = current_q.get("questions", {}).get(lang) or current_q.get("questions", {}).get("en", "")
            clarify_reply = f"{clarify} {q_text}"
            return {
                "assistant_reply": clarify_reply,
                "current_question": current_q,
                "current_question_index": self.current_question_index,
                "total_questions": len(self.question_queue),
                "is_completed": False,
                "status": self.status,
                "extracted_data": self.collected_features,
                "newly_extracted": {}
            }

        # Update session features
        self.collected_features.update(updated_params)

        # 4. Formulate Warm Conversational Acknowledgment
        primary_key = current_q.get("feature_keys", ["value"])[0]
        recorded_val = self.collected_features.get(primary_key, "")
        if isinstance(recorded_val, float) and recorded_val.is_integer():
            recorded_val = int(recorded_val)
        if primary_key == "Sex" or primary_key == "sex":
            recorded_val = "Male" if recorded_val == 1 else "Female"
        elif primary_key == "HighBP" or primary_key == "Smoker" or primary_key == "family_history":
            recorded_val = "Yes" if recorded_val == 1 else "No"

        ack_tmpl = current_q.get("acknowledgments", {}).get(lang) or current_q.get("acknowledgments", {}).get("en", "Got it. I've recorded {value}.")
        calc_bmi = self.collected_features.get("BMI") or self.collected_features.get("bmi") or ""
        try:
            ack_msg = ack_tmpl.format(value=recorded_val, bmi=calc_bmi)
        except Exception:
            ack_msg = f"Got it, recorded {recorded_val}."

        # 5. Advance to Next Unanswered Question
        next_q = self.advance_to_next_unanswered_question()
        if next_q:
            next_text = next_q.get("questions", {}).get(lang) or next_q.get("questions", {}).get("en", "")
            full_reply = f"{ack_msg} {next_text}"
            return {
                "assistant_reply": full_reply,
                "current_question": next_q,
                "current_question_index": self.current_question_index,
                "total_questions": len(self.question_queue),
                "is_completed": False,
                "status": "IN_PROGRESS",
                "extracted_data": self.collected_features,
                "newly_extracted": newly_extracted
            }
        else:
            # All required questions answered!
            self.status = "AWAITING_REVIEW"
            self.is_completed = True
            completion_msg = COMPLETION_MESSAGES.get(lang, COMPLETION_MESSAGES["en"])
            full_reply = f"{ack_msg} {completion_msg}"
            return {
                "assistant_reply": full_reply,
                "current_question": None,
                "current_question_index": self.current_question_index,
                "total_questions": len(self.question_queue),
                "is_completed": True,
                "status": "AWAITING_REVIEW",
                "extracted_data": self.collected_features,
                "newly_extracted": newly_extracted
            }

    def apply_clinical_defaults_for_missing(self, missing: List[str]):
        """Fills non-critical missing parameters with clinical defaults prior to prediction."""
        for feat in missing:
            if feat in CLINICAL_DEFAULTS and (feat not in self.collected_features or self.collected_features[feat] is None):
                self.collected_features[feat] = CLINICAL_DEFAULTS[feat]

    def set_language(self, language: str):
        """Sets active conversation language."""
        if language and language != "auto":
            self.active_language = language

    def update_features(self, new_params: Dict[str, Any]):
        """Updates collected features with new key-values."""
        self.collected_features.update(new_params)

    def get_missing_features(self, required_features: List[str]) -> List[str]:
        """Returns list of required feature keys that have not yet been collected."""
        missing = []
        for feat in required_features:
            if feat not in self.collected_features or self.collected_features[feat] is None:
                missing.append(feat)
        return missing


class ConversationStateManager:
    """
    Manages multi-turn conversation sessions across all active users.
    """
    DEFAULT_REQUIRED_FEATURES = {
        "diabetes": [
            "BMI", "GenHlth", "MentHlth", "PhysHlth", "HighBP", "HighChol",
            "CholCheck", "Smoker", "Stroke", "HeartDiseaseorAttack", "PhysActivity",
            "Fruits", "Veggies", "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost",
            "DiffWalk", "Sex", "Age", "Education", "Income"
        ],
        "heart_disease": [
            "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"
        ],
        "hypertension": [
            "age", "sex", "systolic_bp", "diastolic_bp", "heart_rate", "bmi", "fasting_glucose",
            "serum_sodium", "serum_potassium", "family_history", "smoking_status", "physical_activity_hours"
        ]
    }

    def __init__(self):
        self.sessions: Dict[str, ConversationSession] = {}

    def get_or_create_session(self, session_id: str, disease_id: str = "diabetes", language: str = "en") -> ConversationSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id, disease_id, language)
        else:
            session = self.sessions[session_id]
            if disease_id and session.disease_id != disease_id.lower():
                session.disease_id = disease_id.lower()
                session._initialize_queue()
            if language and language != "auto":
                session.active_language = language
        return self.sessions[session_id]

    def fetch_required_features(self, disease_id: str) -> List[str]:
        return self.DEFAULT_REQUIRED_FEATURES.get(disease_id.lower(), self.DEFAULT_REQUIRED_FEATURES["diabetes"])

    def generate_followup_prompt(self, session: ConversationSession, missing_critical: List[str], lang_override: Optional[str] = None) -> str:
        lang = lang_override or session.active_language or "en"
        disease_name = session.disease_id.replace("_", " ").title()

        if any(f in missing_critical for f in ["Age", "age"]):
            tmpl = LanguageManager.get_prompt_template(lang, "age_ask")
            return tmpl.format(disease=disease_name)

        if any(f in missing_critical for f in ["BMI", "bmi", "weight", "height"]):
            return LanguageManager.get_prompt_template(lang, "bmi_ask")

        if any(f in missing_critical for f in ["HighBP", "hypertension", "systolic_bp"]):
            return LanguageManager.get_prompt_template(lang, "bp_ask")

        if any(f in missing_critical for f in ["HighChol", "chol"]):
            return LanguageManager.get_prompt_template(lang, "chol_ask")

        if any(f in missing_critical for f in ["Smoker", "smoking_status"]):
            return LanguageManager.get_prompt_template(lang, "smoke_ask")

        return LanguageManager.get_prompt_template(lang, "age_ask").format(disease=disease_name)

