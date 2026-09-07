import os
import sys
import io
import json
import urllib.request
from gtts import gTTS
import soundfile as sf

reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
if callable(reconfigure_stdout):
    reconfigure_stdout(encoding="utf-8")

scenarios = [
    {
        "lang": "en",
        "name": "English Native Speech Audio",
        "text": "I am 52 years old, height 175 cm, weight 92 kg, I have high blood pressure, high cholesterol, and I smoke."
    },
    {
        "lang": "hi",
        "name": "Hindi Native Speech Audio (Devanagari)",
        "text": "मेरा उम्र 52 वर्ष है, ऊंचाई 175 सेंटीमीटर, वजन 92 किलो, मुझे हाई ब्लड प्रेशर और हाई कोलेस्ट्रॉल है और मैं सिगरेट पीता हूँ"
    },
    {
        "lang": "bn",
        "name": "Bengali Native Speech Audio (Bangla)",
        "text": "আমার বয়স 52 বছর, উচ্চতা 175 সেন্টিমিটার, ওজন 92 কেজি, আমার হাই ব্লাড প্রেশার এবং হাই কোলেস্টেরল আছে এবং আমি ধূমপান করি"
    }
]

print("==========================================================================")
print("   TESTING REAL ACOUSTIC AUDIO CAPTURE -> STT -> PREDICTION PIPELINE     ")
print("==========================================================================")

results = []

for s in scenarios:
    print(f"\n--- Testing Scenario: {s['name']} [{s['lang'].upper()}] ---")
    
    # 1. Synthesize real acoustic audio
    print(f"1. Synthesizing spoken audio...")
    tts = gTTS(s["text"], lang=s["lang"])
    mp3_buf = io.BytesIO()
    tts.write_to_fp(mp3_buf)
    mp3_buf.seek(0)
    
    # Convert to WAV
    data, sr = sf.read(mp3_buf)
    wav_buf = io.BytesIO()
    sf.write(wav_buf, data, sr, format="WAV")
    wav_buf.seek(0)
    wav_bytes = wav_buf.read()
    print(f"   Audio size: {len(wav_bytes)} bytes WAV (sample rate: {sr} Hz)")
    
    # 2. Upload to /api/v1/voice/process-audio via multipart form
    print("2. Sending audio file to Voice AI Service (Port 8002: /api/v1/voice/process-audio)...")
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = io.BytesIO()
    
    # session_id
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"session_id\"\r\n\r\nreal_audio_{s['lang']}\r\n".encode("utf-8"))
    # disease_id
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"disease_id\"\r\n\r\ndiabetes\r\n".encode("utf-8"))
    # language
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{s['lang']}\r\n".encode("utf-8"))
    # file
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"speech_{s['lang']}.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode("utf-8"))
    body.write(wav_bytes)
    body.write(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    
    req = urllib.request.Request(
        "http://127.0.0.1:8002/api/v1/voice/process-audio",
        data=body.getvalue(),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_data = json.loads(resp.read().decode("utf-8"))
            
            print(f"3. Real STT Transcript: \"{resp_data.get('transcript')}\"")
            print(f"4. Detected Language: {resp_data.get('detected_language')}")
            print(f"5. Extracted Parameters: {resp_data.get('extracted_data')}")
            print(f"6. Ready for Prediction: {resp_data.get('is_ready_for_prediction')}")
            print(f"7. Actual Backend Prediction: {resp_data.get('prediction_result')}")
            print(f"8. Assistant Vocal Reply: \"{resp_data.get('assistant_reply')[:80]}...\"")
            print(f"9. TTS Audio Returned: {'Yes (' + str(len(resp_data.get('audio_base64', ''))) + ' b64 chars)' if resp_data.get('audio_base64') else 'No'}")
            
            results.append({
                "lang": s["lang"],
                "name": s["name"],
                "stt_ok": bool(resp_data.get('transcript') and "Could not transcribe" not in resp_data.get('transcript')),
                "transcript": resp_data.get('transcript'),
                "lang_match": resp_data.get('detected_language') == s["lang"],
                "extracted": resp_data.get('extracted_data'),
                "has_age": "Age" in resp_data.get('extracted_data', {}),
                "has_bp": "HighBP" in resp_data.get('extracted_data', {}),
                "has_bmi": "BMI" in resp_data.get('extracted_data', {}),
                "pred_ok": resp_data.get('prediction_result') is not None,
                "model_used": (resp_data.get('prediction_result') or {}).get("model_used"),
                "risk_probability": (resp_data.get('prediction_result') or {}).get("risk_probability"),
                "tts_ok": bool(resp_data.get('audio_base64'))
            })
    except Exception as ex:
        print(f"   [ERROR]: {ex}")
        results.append({"lang": s["lang"], "error": str(ex)})

print("\n==========================================================================")
print("                           SUMMARY OF RESULTS                             ")
print("==========================================================================")
for r in results:
    if "error" in r:
        print(f"[{r['lang'].upper()}] FAILED: {r['error']}")
    else:
        print(f"[{r['lang'].upper()}] STT: {'PASS' if r['stt_ok'] else 'FAIL'} | Lang: {'PASS' if r['lang_match'] else 'FAIL'} | Extracted: {list(r['extracted'].keys())} | ML/QML Pred: {'PASS' if r['pred_ok'] else 'FAIL'} ({r['model_used']}, risk={r['risk_probability']}) | TTS: {'PASS' if r['tts_ok'] else 'FAIL'}")
