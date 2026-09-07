import os
import io
import base64
import wave
import struct
import math
import numpy as np
from typing import Optional
from voice_ai.app.config import config

class LocalKokoroTTS:
    """
    Local Text-to-Speech Synthesizer powered by Kokoro-82M / ONNX.
    Converts text response to speech locally without paid APIs.
    """
    def __init__(self, model_path: Optional[str] = None, voices_path: Optional[str] = None):
        self.model_path = model_path or config.KOKORO_MODEL_PATH
        self.voices_path = voices_path or config.KOKORO_VOICES_PATH
        self.kokoro = None
        self._init_kokoro()

    def _init_kokoro(self):
        if os.path.exists(self.model_path) and os.path.exists(self.voices_path):
            try:
                from kokoro_onnx import Kokoro
                print(f"[VOICE AI - TTS] Loading local Kokoro-82M ONNX model from '{self.model_path}'...")
                self.kokoro = Kokoro(self.model_path, self.voices_path)
                print("[VOICE AI - TTS] Local Kokoro TTS model loaded successfully.")
            except Exception as e:
                print(f"[VOICE AI - TTS] Warning: Kokoro ONNX load failed ({e}). Synthetic WAV fallback engine active.")
                self.kokoro = None
        else:
            print(f"[VOICE AI - TTS] Kokoro model files not found at '{self.model_path}'. Synthetic local audio fallback active.")
            self.kokoro = None

    def synthesize_to_wav_bytes(self, text: str, voice: str = "af_bella", language: str = "en") -> bytes:
        """
        Synthesizes text into raw WAV audio bytes with multilingual support (en, hi, bn).
        """
        clean_lang = (language or "en").lower().split("-")[0]

        # 1. Native Indic / Multilingual Speech via gTTS
        if clean_lang in ["hi", "bn"]:
            try:
                from gtts import gTTS
                import soundfile as sf
                tts = gTTS(text=text, lang=clean_lang, slow=False)
                mp3_fp = io.BytesIO()
                tts.write_to_fp(mp3_fp)
                mp3_fp.seek(0)
                data, samplerate = sf.read(mp3_fp)
                wav_buf = io.BytesIO()
                sf.write(wav_buf, data, samplerate, format='WAV')
                return wav_buf.getvalue()
            except Exception as e:
                print(f"[VOICE AI - TTS] gTTS synthesis error for {clean_lang}: {e}")

        # 2. Kokoro ONNX for English
        if self.kokoro and clean_lang == "en":
            try:
                samples, sample_rate = self.kokoro.create(text, voice=voice, speed=1.0, lang="en-us")
                buf = io.BytesIO()
                import soundfile as sf
                sf.write(buf, samples, sample_rate, format='WAV')
                return buf.getvalue()
            except Exception as e:
                print(f"[VOICE AI - TTS] Kokoro synthesis error ({e}). Generating fallback audio.")

        # 3. Try pyttsx3 for offline English vocal speech
        try:
            import pyttsx3
            import tempfile
            engine = pyttsx3.init()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tmp_path = tmp.name
            engine.save_to_file(text, tmp_path)
            engine.runAndWait()
            if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
                with open(tmp_path, "rb") as f:
                    wav_data = f.read()
                os.remove(tmp_path)
                return wav_data
        except Exception:
            pass

        # 4. gTTS for English if available
        try:
            from gtts import gTTS
            import soundfile as sf
            tts = gTTS(text=text, lang="en", slow=False)
            mp3_fp = io.BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            data, samplerate = sf.read(mp3_fp)
            wav_buf = io.BytesIO()
            sf.write(wav_buf, data, samplerate, format='WAV')
            return wav_buf.getvalue()
        except Exception:
            pass

        # 5. Fallback pleasant tone envelope
        return self._generate_fallback_wav(text)

    def synthesize_to_base64(self, text: str, voice: str = "af_bella", language: str = "en") -> str:
        """
        Synthesizes text and returns base64 encoded WAV string.
        """
        wav_bytes = self.synthesize_to_wav_bytes(text, voice, language=language)
        return base64.b64encode(wav_bytes).decode("utf-8")

    def _generate_fallback_wav(self, text: str) -> bytes:
        """
        Generates a valid 16kHz 16-bit mono WAV file for local playback.
        """
        words = text.split()
        duration_sec = max(1.0, len(words) * 0.35)
        sample_rate = 16000
        n_samples = int(sample_rate * duration_sec)

        buf = io.BytesIO()
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)

            # Generate pleasant soft audio tone envelope
            t = np.linspace(0, duration_sec, n_samples, False)
            audio = 0.15 * np.sin(2 * np.pi * 440 * t) * np.exp(-1.5 * (t % 0.4))
            audio_int16 = (audio * 32767).astype(np.int16)

            for sample in audio_int16:
                wf.writeframes(struct.pack('<h', sample))

        return buf.getvalue()
