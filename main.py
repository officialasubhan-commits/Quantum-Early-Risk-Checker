"""
Main application entrypoint for Vercel, ASGI servers, and local execution.
"""
import os
import sys

# Ensure project root is in Python sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.main import app as fastapi_app

# Explicit top-level variables for Vercel Serverless Function discovery
app = fastapi_app
application = fastapi_app
handler = fastapi_app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
