import base64
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, status
from typing import Optional

from voice_ai.app.schemas.voice_schemas import (
    TextChatRequest,
    VoiceAudioRequest,
    VoiceAssessmentResponse,
    ConversationStartRequest,
    ConversationStartResponse,
    ConversationTurnRequest,
    ConversationTurnResponse
)
from voice_ai.app.services.voice_pipeline import VoiceAIPipelineService

router = APIRouter(prefix="/api/v1/voice", tags=["Voice AI Assistant"])
pipeline_service = VoiceAIPipelineService()

@router.get("/health", tags=["System"])
def voice_health_check():
    """
    Returns Voice AI service health and active local model configurations.
    """
    return {
        "status": "OK",
        "service": "SIH26139 Standalone Local Voice AI Module",
        "stt_engine": "Local Whisper (faster-whisper)",
        "tts_engine": "Local Kokoro-82M / ONNX",
        "prediction_backend_target": pipeline_service.llm.ollama_url
    }

@router.post("/chat", response_model=VoiceAssessmentResponse)
def text_voice_chat(request: TextChatRequest):
    """
    Simulates spoken turn from text transcript input.
    Parses entities, tracks conversation state, queries prediction backend API, and returns speech audio in requested/detected language.
    """
    try:
        return pipeline_service.process_text_turn(
            session_id=request.session_id,
            user_transcript=request.user_input,
            disease_id=request.disease_id or "diabetes",
            language=request.language or "auto"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Voice processing failed: {str(e)}"
        )

@router.post("/process-audio", response_model=VoiceAssessmentResponse)
async def process_audio_file(
    session_id: str = Form("default_session"),
    disease_id: str = Form("diabetes"),
    language: str = Form("auto"),
    file: UploadFile = File(...)
):
    """
    Accepts uploaded WAV audio file, transcribes locally via STT, and returns full prediction response + TTS audio.
    """
    try:
        audio_bytes = await file.read()
        format_ext = file.filename.split(".")[-1] if (file.filename and "." in file.filename) else "wav"
        return pipeline_service.process_audio_turn(

            session_id=session_id,
            audio_bytes=audio_bytes,
            audio_format=format_ext,
            disease_id=disease_id,
            language=language
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Audio processing failed: {str(e)}"
        )

@router.post("/conversation/start", response_model=ConversationStartResponse)
def start_conversational_assessment(request: ConversationStartRequest):
    """
    Initializes a spoken health assessment dialogue asking one question at a time.
    Returns the first question formatted in the requested language along with TTS audio.
    """
    try:
        return pipeline_service.start_conversational_assessment(
            session_id=request.session_id,
            disease_id=request.disease_id or "diabetes",
            language=request.language or "en"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start conversational assessment: {str(e)}"
        )

@router.post("/conversation/turn", response_model=ConversationTurnResponse)
def process_conversational_turn(request: ConversationTurnRequest):
    """
    Processes user's spoken answer turn:
    - Extracts structured parameters
    - Formulates warm conversational acknowledgment
    - Returns next question or completion notice with TTS audio
    """
    try:
        return pipeline_service.process_conversational_turn(
            session_id=request.session_id,
            user_input=request.user_input,
            disease_id=request.disease_id or "diabetes",
            language=request.language or "en"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Conversational turn failed: {str(e)}"
        )


