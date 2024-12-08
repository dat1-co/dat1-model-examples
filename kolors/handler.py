from fastapi import Request, FastAPI, Response
import torch, sys
from base64 import b64encode
from io import BytesIO
from pathlib import Path

from diffusers.pipelines.kolors import ChatGLMModel, ChatGLMTokenizer, KolorsPipeline
from diffusers import UNet2DConditionModel, AutoencoderKL, EulerDiscreteScheduler

import traceback

ROOT_DIR = Path(__file__).parent


CKPT_DIR = f'{ROOT_DIR}/weights/Kolors'
TEXT_ENC = ChatGLMModel.from_pretrained(
    f'{CKPT_DIR}/text_encoder',
    torch_dtype=torch.float16).half()
TOKENIZER = ChatGLMTokenizer.from_pretrained(f'{CKPT_DIR}/text_encoder')
VAE = AutoencoderKL.from_pretrained(f"{CKPT_DIR}/vae", revision=None).half()
SCHEDULER = EulerDiscreteScheduler.from_pretrained(f"{CKPT_DIR}/scheduler")
UNET = UNet2DConditionModel.from_pretrained(f"{CKPT_DIR}/unet", revision=None).half()
PIPE = KolorsPipeline(
        vae=VAE,
        text_encoder=TEXT_ENC,
        tokenizer=TOKENIZER,
        unet=UNET,
        scheduler=SCHEDULER,
        force_zeros_for_empty_prompt=False)
PIPE = PIPE.to("cuda")


def kolors_pipe(prompt):
    image = PIPE(
        prompt=prompt,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=5.0,
        num_images_per_prompt=1,
        generator= torch.Generator(PIPE.device).manual_seed(66)).images[0]
    return image


app = FastAPI()

@app.get("/")
async def root():
    return "OK"

@app.post("/infer")
async def infer(request: Request):
    request = await request.json()
    prompts = request["prompt"]
    image = kolors_pipe(prompts)

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = b64encode(buffered.getvalue()).decode("utf-8")
    return {"response": f"{img_str}"}

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