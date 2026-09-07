import os
import json
import urllib.parse
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 8080
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DIR = os.path.join(BASE_DIR, "public")

class PatientAppRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=PUBLIC_DIR, **kwargs)

def run_patient_app_server(port=PORT):
    server_address = ("", port)
    httpd = HTTPServer(server_address, PatientAppRequestHandler)
    print(f"=" * 70, flush=True)
    print(f" SIH26139 Patient-Facing Web Application Server Running ", flush=True)
    print(f" Access URL: http://localhost:{port}/", flush=True)
    print(f" Connecting to Backend API at: http://127.0.0.1:8001/", flush=True)
    print(f"=" * 70, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Patient Web Application server...", flush=True)
        httpd.server_close()

if __name__ == "__main__":
    run_patient_app_server()
