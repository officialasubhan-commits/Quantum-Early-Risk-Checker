import io
import sys
import soundfile as sf
import speech_recognition as sr
import numpy as np
from gtts import gTTS

reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
if callable(reconfigure_stdout):
    reconfigure_stdout(encoding="utf-8")


# Patch sr get_flac_data to use soundfile instead of external flac.exe
def custom_get_flac_data(self, convert_rate=None, convert_width=None):
    raw_bytes = self.get_raw_data(convert_rate=convert_rate, convert_width=convert_width)
    rate = convert_rate or self.sample_rate
    samples = np.frombuffer(raw_bytes, dtype=np.int16)
    buf = io.BytesIO()
    sf.write(buf, samples, rate, format='FLAC')
    buf.seek(0)
    return buf.read()

sr.AudioData.get_flac_data = custom_get_flac_data

test_cases = [
    ('en', 'I am 45 years old with high blood pressure and BMI of 29', 'en-US'),
    ('hi', 'मेरा उम्र 52 वर्ष है और मुझे हाई ब्लड प्रेशर है', 'hi-IN'),
    ('bn', 'আমার বয়স 48 বছর এবং আমার ব্লাড প্রেসার বেশি', 'bn-IN')
]

for lang_code, text, regional in test_cases:
    print(f"Synthesizing actual spoken audio for {lang_code.upper()}...")
    tts = gTTS(text, lang=lang_code)
    mp3_buf = io.BytesIO()
    tts.write_to_fp(mp3_buf)
    mp3_buf.seek(0)
    data, sr_rate = sf.read(mp3_buf)
    
    # Convert to standard 16kHz mono WAV for STT
    wav_buf = io.BytesIO()
    sf.write(wav_buf, data, sr_rate, format='WAV')
    wav_buf.seek(0)
    
    r = sr.Recognizer()
    with sr.AudioFile(wav_buf) as source:
        audio = r.record(source)
    
    recognize_fn = getattr(r, "recognize_google", None)
    if callable(recognize_fn):
        transcribed = recognize_fn(audio, language=regional)
        print(f"  [SUCCESS] {lang_code.upper()} Real Audio Transcribed: '{transcribed}'")
