from fastapi import APIRouter
import base64
import os

from models.generator import Generator, TEXT_TO_INDEX, current_index, generate_cgan_images
from models.upscaler import upscale_image
from models.diffusion import style_with_diffusion

router = APIRouter(prefix="/api")

@router.post("/generate")
async def generate_logo_api(label: str, apply_custom_lora: bool = False):
    try:
        base_image = generate_cgan_images(label, count=1)[0]
        upscaled = upscale_image(base_image, label)
        final_image = style_with_diffusion(upscaled, label, apply_custom_lora=apply_custom_lora)
        with open(final_image, "rb") as f:
            return {"status": "success", "image": base64.b64encode(f.read()).decode()}
    except Exception as e:
        return {"status": "error", "message": str(e)}
