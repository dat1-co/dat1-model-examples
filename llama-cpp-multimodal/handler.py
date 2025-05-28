import os
import traceback, sys
from pathlib import Path
import subprocess
import requests
from fastapi import Request, FastAPI, Response
from sse_starlette.sse import EventSourceResponse
import sseclient
from requests import Response as RequestsResponse

ROOT_DIR = Path(__file__).parent

process = subprocess.Popen(
    ["/workspace/llama.cpp/build/bin/llama-server", "-m", str(ROOT_DIR / "model.gguf"), "--mmproj", str(ROOT_DIR / "mmproj.gguf"), "--ctx-size", "128000", "-ngl", "999999", "--port", "8080"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
    universal_newlines=True
)

# Wait for the "main: model loaded" line
for line in process.stdout:
    print(line, end='')
    if "main: model loaded" in line:
        print("Model is loaded.")
        break

app = FastAPI()

@app.get("/")
async def root(response: Response):
    try:
        response = requests.get("http://localhost:8080/health")
        if response.status_code == 200:
            return "OK"
        else:
            response.status_code = 500
            return response.text()
    except requests.exceptions.RequestException as e:
        response.status_code = 500
        return response.text()

def event_generator(response: RequestsResponse):
    client = sseclient.SSEClient(response)
    for event in client.events():
        yield {
            "event": event.event or "message",
            "data": event.data,
        }


@app.post("/infer")
async def infer(request: Request):
    body = await request.json()

    try:
        response = requests.post("http://localhost:8080/v1/chat/completions", json=body, stream=body.get("stream", False))
        if response.status_code == 200:
            if "text/event-stream" in response.headers.get("Content-Type", ""):
                return EventSourceResponse(event_generator(response), sep="\n")
            else:
                return response.json()
        else:
            return response.text
    except requests.exceptions.RequestException as e:
        return {"status": "ERROR", "message": str(e)}

@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    exc_type, exc_value, exc_tb = sys.exc_info()
    formatted_traceback = "".join(
        traceback.format_exception(exc_type, exc_value, exc_tb)
    )
    return Response(content=formatted_traceback, media_type="text/plain", status_code=500)
