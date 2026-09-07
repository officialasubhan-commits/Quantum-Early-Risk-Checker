from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

MEDICAL_DISCLAIMER_TEXT = (
    "AI-Assisted Early Disease Risk Decision Support: "
    "This output is an automated probabilistic risk estimation designed solely for decision support. "
    "It is NOT a medical diagnosis and should not replace professional clinical evaluation by a licensed healthcare practitioner."
)

class TextChatRequest(BaseModel):
    session_id: str = Field("default_session", description="Session identifier for multi-turn dialogue")
    user_input: str = Field(..., description="User transcript or spoken text input")
    disease_id: Optional[str] = Field("diabetes", description="Target disease identifier (e.g. diabetes, heart_disease, hypertension)")
    language: Optional[str] = Field("auto", description="Preferred language code (auto, en, hi, bn, ta, te, mr, gu, kn, ml, pa, ur)")

class VoiceAudioRequest(BaseModel):
    session_id: str = Field("default_session", description="Session identifier")
    audio_base64: Optional[str] = Field(None, description="Base64 encoded audio string (WAV/MP3/PCM)")
    audio_format: str = Field("wav", description="Audio container format (wav, mp3, ogg, pcm)")
    disease_id: Optional[str] = Field("diabetes", description="Target disease identifier")
    language: Optional[str] = Field("auto", description="Target language code or auto-detect")

class ExtractedMedicalData(BaseModel):
    extracted_features: Dict[str, Any] = Field(default_factory=dict, description="Extracted clinical key-value parameters")
    updated_features: List[str] = Field(default_factory=list, description="List of feature names updated in this turn")
    target_disease: Optional[str] = Field(None, description="Inferred or explicitly set target disease")

class VoiceAssessmentResponse(BaseModel):
    session_id: str
    transcript: str
    language: str = Field("en", description="Active conversation language code")
    detected_language: str = Field("en", description="Automatically detected language code from speech or text")
    extracted_data: Dict[str, Any]
    missing_features: List[str]
    is_ready_for_prediction: bool
    assistant_reply: str
    prediction_result: Optional[Dict[str, Any]] = None
    audio_base64: Optional[str] = None
    audio_url: Optional[str] = None
    fallback_used: bool = Field(False, description="True if a fallback engine or language model was used")
    fallback_reason: Optional[str] = Field(None, description="Explanation if a fallback was triggered")
    disclaimer: str = MEDICAL_DISCLAIMER_TEXT

class ConversationStartRequest(BaseModel):
    session_id: str = Field("default_session", description="Session identifier")
    disease_id: Optional[str] = Field("diabetes", description="Target disease model")
    language: Optional[str] = Field("en", description="Preferred language (en, hi, bn)")

class QuestionInfo(BaseModel):
    id: str
    index: int
    category: str
    question_text: str
    feature_keys: List[str]

class ConversationStartResponse(BaseModel):
    session_id: str
    disease_id: str
    language: str
    assistant_reply: str
    current_question: Optional[QuestionInfo] = None
    current_question_index: int = 0
    total_questions: int = 0
    audio_base64: Optional[str] = None
    disclaimer: str = MEDICAL_DISCLAIMER_TEXT

class ConversationTurnRequest(BaseModel):
    session_id: str = Field("default_session", description="Session identifier")
    user_input: str = Field(..., description="User's spoken answer or text")
    disease_id: Optional[str] = Field("diabetes", description="Target disease model")
    language: Optional[str] = Field("en", description="Active language code")

class ConversationTurnResponse(BaseModel):
    session_id: str
    disease_id: str
    language: str
    user_transcript: str
    assistant_reply: str
    current_question: Optional[QuestionInfo] = None
    current_question_index: int = 0
    total_questions: int = 0
    is_completed: bool = False
    status: str = "IN_PROGRESS"
    extracted_data: Dict[str, Any] = Field(default_factory=dict)
    newly_extracted: Dict[str, Any] = Field(default_factory=dict)
    audio_base64: Optional[str] = None
    disclaimer: str = MEDICAL_DISCLAIMER_TEXT

