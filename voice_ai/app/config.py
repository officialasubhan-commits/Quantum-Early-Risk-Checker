import os

class VoiceConfig:
    """
    Configuration settings for Local Voice AI Service.
    """
    VOICE_SERVICE_HOST: str = os.getenv("VOICE_SERVICE_HOST", "127.0.0.1")
    VOICE_SERVICE_PORT: int = int(os.getenv("VOICE_SERVICE_PORT", "8002"))
    
    # Target URL for existing FastAPI medical prediction backend
    PREDICTION_BACKEND_URL: str = os.getenv("PREDICTION_BACKEND_URL", "http://127.0.0.1:8001")
    
    # Local Speech-to-Text Model Settings
    WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL_SIZE", "tiny") # 'tiny', 'base', 'small', 'medium'
    DEVICE: str = os.getenv("DEVICE", "cpu")
    COMPUTE_TYPE: str = os.getenv("COMPUTE_TYPE", "int8")
    
    # Local LLM / Ollama Settings
    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "qwen2.5:latest")
    
    # Local Text-to-Speech (Kokoro) Settings
    KOKORO_MODEL_PATH: str = os.getenv("KOKORO_MODEL_PATH", "models/kokoro-v0_19.onnx")
    KOKORO_VOICES_PATH: str = os.getenv("KOKORO_VOICES_PATH", "models/voices.bin")
    DEFAULT_VOICE: str = os.getenv("DEFAULT_VOICE", "af_bella")

config = VoiceConfig()
