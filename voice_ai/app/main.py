import os
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from voice_ai.app.config import config
from voice_ai.app.api.router import router as voice_router
from voice_ai.app.api.websocket import websocket_router

app = FastAPI(
    title="SIH26139 Standalone Local Voice AI Module",
    description="100% Self-Hosted Local Voice AI Assistant with Whisper STT, Natural Language Entity Extraction, Multi-Turn Conversation State Manager, and Local Kokoro TTS.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST & WebSocket Routers
app.include_router(voice_router)
app.include_router(websocket_router)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    print(f"Starting Voice AI Service on http://{config.VOICE_SERVICE_HOST}:{config.VOICE_SERVICE_PORT}...")
    uvicorn.run("voice_ai.app.main:app", host=config.VOICE_SERVICE_HOST, port=config.VOICE_SERVICE_PORT, reload=True)
