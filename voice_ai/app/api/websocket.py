import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from voice_ai.app.services.voice_pipeline import VoiceAIPipelineService

websocket_router = APIRouter()
pipeline_service = VoiceAIPipelineService()

@websocket_router.websocket("/ws/voice")
async def voice_websocket_endpoint(websocket: WebSocket):
    """
    Realtime WebSocket communication endpoint for streaming voice audio and text dialogue.
    """
    await websocket.accept()
    print("[VOICE AI - WS] Client connected to Voice WebSocket stream.")

    try:
        while True:
            raw_message = await websocket.receive_text()
            try:
                data = json.loads(raw_message)
                session_id = data.get("session_id", "ws_default_session")
                disease_id = data.get("disease_id", "diabetes")
                user_text = data.get("text", "")

                if user_text:
                    response = pipeline_service.process_text_turn(session_id, user_text, disease_id)
                    await websocket.send_json(response.model_dump())
                else:
                    await websocket.send_json({
                        "error": "Empty text frame received",
                        "session_id": session_id
                    })
            except json.JSONDecodeError:
                await websocket.send_json({"error": "Invalid JSON frame received"})
    except WebSocketDisconnect:
        print("[VOICE AI - WS] Client disconnected from Voice WebSocket stream.")
