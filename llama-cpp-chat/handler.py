import os
import traceback, sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent

os.environ["MODEL"]=f"{str(ROOT_DIR/'model.gguf')}"
os.environ["N_GPU_LAYERS"] = "-1"

from fastapi import Request, Response
from llama_cpp.server.app import create_app, create_chat_completion, llama_outer_lock, llama_inner_lock


app = create_app()

@app.get("/")
async def root():
    assert not llama_outer_lock.locked() and not llama_inner_lock.locked(), "Locks are not released"
    return "OK"

@app.post("/infer")(create_chat_completion)

@app.exception_handler(Exception)
async def debug_exception_handler(request: Request, exc: Exception):
    # Get the traceback from the current exception
    exc_type, exc_value, exc_tb = sys.exc_info()

    # Format the traceback to a string
    formatted_traceback = "".join(
        traceback.format_exception(exc_type, exc_value, exc_tb)
    )

    # Return it as a response with a 500 status code
    return Response(content=formatted_traceback, media_type="text/plain", status_code=500)
