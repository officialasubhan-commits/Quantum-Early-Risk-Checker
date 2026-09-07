# SIH26139 Standalone Local Voice AI Module

A 100% self-hosted, local Voice AI Assistant microservice built for the **SIH26139 Hybrid Quantum Machine Learning Platform**. 

The module listens to patient speech, converts speech to text locally (faster-whisper), extracts clinical parameters via natural language NLP, maintains multi-turn conversation state, queries the existing FastAPI prediction backend (`http://127.0.0.1:8001/`), formats patient-friendly clinical explanations, and synthesizes spoken audio responses locally using Kokoro-82M.

---

## 1. System Architecture

```text
Microphone / Audio Input
  │
  ▼
Local STT (faster-whisper / Whisper)
  │
  ▼
Medical Parameter Extractor (NLP / Pattern Matching)
  │
  ▼
Stateful Conversation Manager (Multi-Turn Dialogue & Missing Features)
  │
  ├──► Missing Critical Data? ──► Generate Follow-up Prompt ──► Kokoro TTS ──► Speaker
  │
  ▼
Collected Complete Feature Payload (GenericDiseasePredictionRequest)
  │
  ▼
Existing FastAPI Prediction Backend (http://127.0.0.1:8001/api/v1/predict/disease/{id}/hybrid)
  │
  ▼
Actual Hybrid ML/QML Model Risk Probability & SHAP Narrative
  │
  ▼
Local LLM Patient Explanation (Ollama Qwen2.5/Qwen3 / Local Clinical Engine)
  │
  ▼
Local TTS (Kokoro-82M / ONNX Synthesizer)
  │
  ▼
Spoken Audio Output & Base64 Stream
```

---

## 2. Installation & Setup

### Prerequisites
* Python 3.11.x
* Virtual environment (`.venv`)

### Installation Commands
```powershell
# 1. Activate project virtual environment
.\.venv\Scripts\Activate.ps1

# 2. Install Voice AI dependencies
pip install -r voice_ai/requirements.txt
```

### Optional Model Setup (Local STT & TTS)
```powershell
# Create local models directory
mkdir voice_ai/models

# Kokoro-82M ONNX models can be downloaded into voice_ai/models/
# kokoro-v0_19.onnx and voices.bin
```

---

## 3. Starting the Services

### Step 1: Start Existing Prediction Backend (Port 8001)
```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --port 8001
```

### Step 2: Start Voice AI Service (Port 8002)
```powershell
.\.venv\Scripts\python.exe -m uvicorn voice_ai.app.main:app --port 8002
```

---

## 4. REST & WebSocket Endpoints

* **REST Health**: `GET http://127.0.0.1:8002/api/v1/voice/health`
* **REST Text/Voice Chat**: `POST http://127.0.0.1:8002/api/v1/voice/chat`
* **REST Audio Upload**: `POST http://127.0.0.1:8002/api/v1/voice/process-audio`
* **Realtime WebSocket**: `ws://127.0.0.1:8002/ws/voice`

---

## 5. Test Procedure & Verification

Run the automated voice test suite:
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s voice_ai/tests
```

---

## 6. Example Request & Response JSON

### Turn 1: Initial Spoken Input
**Request (`POST /api/v1/voice/chat`)**:
```json
{
  "session_id": "demo_session_101",
  "user_input": "I am 45 years old. I have high blood pressure and I smoke.",
  "disease_id": "diabetes"
}
```

**Response**:
```json
{
  "session_id": "demo_session_101",
  "transcript": "I am 45 years old. I have high blood pressure and I smoke.",
  "extracted_data": {
    "age": 45.0,
    "Age": 6,
    "HighBP": 1,
    "Smoker": 1
  },
  "missing_features": ["BMI"],
  "is_ready_for_prediction": false,
  "assistant_reply": "Thank you. Could you share your height and weight, or your current Body Mass Index (BMI)?",
  "prediction_result": null,
  "audio_base64": "UklGRi...",
  "disclaimer": "AI-Assisted Early Disease Risk Decision Support: This output is an automated probabilistic risk estimation designed solely for decision support. It is NOT a medical diagnosis and should not replace professional clinical evaluation by a licensed healthcare practitioner."
}
```

### Turn 2: Providing Remaining Missing Parameter
**Request**:
```json
{
  "session_id": "demo_session_101",
  "user_input": "My weight is 88 kg, height is 175 cm, cholesterol is high.",
  "disease_id": "diabetes"
}
```

**Response**:
```json
{
  "session_id": "demo_session_101",
  "transcript": "My weight is 88 kg, height is 175 cm, cholesterol is high.",
  "extracted_data": {
    "age": 45.0,
    "Age": 6,
    "HighBP": 1,
    "Smoker": 1,
    "BMI": 28.7,
    "HighChol": 1,
    "CholCheck": 1
  },
  "missing_features": [],
  "is_ready_for_prediction": true,
  "assistant_reply": "Based on your clinical profile, our Hybrid Classical-Quantum Ensemble Model estimates an elevated risk score of 74.5% for Diabetes / Prediabetes. We recommend discussing these indicators with a healthcare professional for a complete clinical evaluation. Please note: AI-Assisted Early Disease Risk Decision Support: This output is an automated probabilistic risk estimation designed solely for decision support. It is NOT a medical diagnosis and should not replace professional clinical evaluation by a licensed healthcare practitioner.",
  "prediction_result": {
    "request_id": "req_voice_demo_session_101",
    "disease_id": "diabetes",
    "disease_name": "Diabetes / Prediabetes",
    "model_used": "Hybrid Classical-Quantum Ensemble Classifier",
    "risk_probability": 0.745,
    "is_high_risk": true
  },
  "audio_base64": "UklGRi..."
}
```

---

## 7. System Hardware Requirements

| Mode | Minimum RAM | CPU Cores | Optional GPU VRAM | Latency per Turn |
| :--- | :---: | :---: | :---: | :---: |
| **CPU Only (Standard)** | 8 GB | 4 Cores | N/A | ~0.8s - 1.5s |
| **GPU Accelerated (CUDA)** | 8 GB | 4 Cores | 4 GB VRAM | ~0.2s - 0.5s |

---

## 8. Implemented Components & Features

* [x] **Standalone Microservice**: Fully contained within `voice_ai/` on Port 8002.
* [x] **100% Local STT**: Powered by `faster-whisper` / `whisper`.
* [x] **Medical Entity Extractor**: Parses complex health expressions, numbers, units, and user corrections.
* [x] **Stateful Dialogue Manager**: Tracks parameters across turns and generates targeted follow-up prompts.
* [x] **Backend API Integration**: Connects dynamically to existing FastAPI prediction service (`http://127.0.0.1:8001/`).
* [x] **Local TTS Synthesizer**: Kokoro-82M / ONNX local speech synthesizer.
* [x] **WebSocket & REST APIs**: Supports streaming audio frames and JSON payloads.
* [x] **Mandatory Medical Disclaimer**: Prominently enforced on all prediction outputs.

---

## 9. Current Limitations

1. **Complex Unstructured Clinical Jargon**: Niche medical terms not present in training vocabulary require explicit spelling or numeric entry.
2. **Multilingual Speech**: Initial version is optimized for English medical terminology (`en-us`).
