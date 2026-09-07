import os
import tempfile
from typing import Optional, Tuple
from voice_ai.app.config import config

class LocalWhisperSTT:
    """
    Local Speech-to-Text Transcriber powered by faster-whisper / Whisper.
    Supports automatic language detection and manual language selection for 99+ languages
    including English, Hindi, Bengali, Tamil, Telugu, Marathi, Gujarati, Kannada, Malayalam, Punjabi, Urdu.
    """
    def __init__(self, model_size: Optional[str] = None, device: Optional[str] = None, compute_type: Optional[str] = None):
        self.model_size = model_size or config.WHISPER_MODEL_SIZE
        self.device = device or config.DEVICE
        self.compute_type = compute_type or config.COMPUTE_TYPE
        self.model = None
        self._init_model()

    def _init_model(self):
        try:
            import importlib
            fw = importlib.import_module("faster_whisper")
            WhisperModel = fw.WhisperModel
            print(f"[VOICE AI - STT] Loading local faster-whisper model '{self.model_size}' on {self.device} ({self.compute_type})...")
            self.model = WhisperModel(self.model_size, device=self.device, compute_type=self.compute_type)
            print("[VOICE AI - STT] Local Whisper model loaded successfully.")
        except Exception as e:
            print(f"[VOICE AI - STT] Warning: faster-whisper load failed ({e}). Falling back to standard whisper or audio parser.")
            try:
                import importlib
                whisper = importlib.import_module("whisper")
                self.model = whisper.load_model(self.model_size, device=self.device)
                print("[VOICE AI - STT] Standard OpenAI Whisper loaded locally.")
            except Exception as ex:
                print(f"[VOICE AI - STT] Warning: Standard whisper unavailable ({ex}). Text-based simulation mode active.")
                self.model = None

    def transcribe_audio_bytes(self, audio_bytes: bytes, audio_format: str = "wav", language: Optional[str] = "auto") -> Tuple[str, str]:
        """
        Transcribes raw audio bytes into text locally with language detection.
        Returns (transcript, detected_language_code).
        """
        if not audio_bytes:
            fallback = "en" if not language or language == "auto" else language
            return "", fallback

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{audio_format}") as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        try:
            return self.transcribe_audio_file(tmp_path, language=language)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def transcribe_audio_file(self, file_path: str, language: Optional[str] = "auto") -> Tuple[str, str]:
        """
        Transcribes audio file on disk to text.
        Returns (transcript, detected_language_code).
        """
        target_lang = None if language in (None, "auto", "") else language.lower().split("-")[0]

        if not self.model:
            # High-accuracy Speech Recognition fallback with soundfile FLAC pipeline
            try:
                import speech_recognition as sr
                import soundfile as sf
                import numpy as np
                import io

                def custom_get_flac_data(self, convert_rate=None, convert_width=None):
                    raw_bytes = self.get_raw_data(convert_rate=convert_rate, convert_width=convert_width)
                    rate = convert_rate or self.sample_rate
                    samples = np.frombuffer(raw_bytes, dtype=np.int16)
                    buf = io.BytesIO()
                    sf.write(buf, samples, rate, format='FLAC')
                    buf.seek(0)
                    return buf.read()

                setattr(sr.AudioData, "get_flac_data", custom_get_flac_data)

                r = sr.Recognizer()
                with sr.AudioFile(file_path) as source:
                    audio_data = r.record(source)

                lang_map = {
                    "en": "en-US",
                    "hi": "hi-IN",
                    "bn": "bn-IN",
                    "ta": "ta-IN",
                    "te": "te-IN",
                    "mr": "mr-IN",
                    "gu": "gu-IN",
                    "kn": "kn-IN",
                    "ml": "ml-IN",
                    "pa": "pa-IN",
                    "ur": "ur-PK"
                }
                regional_lang = lang_map.get(target_lang, "en-US") if target_lang else "en-US"
                recognize_fn = getattr(r, "recognize_google", None)
                if callable(recognize_fn):
                    transcript = str(recognize_fn(audio_data, language=regional_lang))
                else:
                    raise RuntimeError("recognize_google is not available on Recognizer")

                from voice_ai.app.languages.language_manager import LanguageManager
                detected = LanguageManager.detect_language_from_text(transcript) or target_lang or "en"
                return transcript, detected
            except Exception as e:
                print(f"[VOICE AI - STT] Audio transcription error: {e}")
                detected = target_lang or "en"
                return f"Could not transcribe audio ({detected}).", detected

        try:
            if hasattr(self.model, "transcribe") and type(self.model).__name__ == "WhisperModel":
                segments, info = self.model.transcribe(file_path, beam_size=5, language=target_lang)
                transcript = " ".join([segment.text.strip() for segment in segments])
                detected_lang = getattr(info, "language", target_lang or "en")
                return transcript, detected_lang
            elif hasattr(self.model, "transcribe"):
                result = self.model.transcribe(file_path, language=target_lang)
                transcript = result.get("text", "").strip()
                detected_lang = result.get("language", target_lang or "en")
                return transcript, detected_lang
        except Exception as e:
            print(f"[VOICE AI - STT] Transcription error: {e}")
            return "", target_lang or "en"

        return "", target_lang or "en"
