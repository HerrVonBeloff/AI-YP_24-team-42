from diffusers import StableDiffusionImg2ImgPipeline
import torch
from shared.config import LORA_PATH
import logging
import streamlit as st
from PIL import Image as PILImage
import os
import time

logger = logging.getLogger("LogoGenerator")

@st.cache_resource
def load_diffusion_pipeline(lora_name=None):
    try:
        model_id = "runwayml/stable-diffusion-v1-5"
        logger.info(f"Загрузка модели: {model_id}")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            use_safetensors=True,
            safety_checker=None,
            requires_safety_checker=False
        ).to('cuda' if torch.cuda.is_available() else 'cpu')

        if lora_name:
            lora_path = os.path.join(LORA_PATH, lora_name)
            try:
                logger.info(f"Попытка загрузить LoRA веса из: {lora_path}")
                pipe.load_lora_weights(
                    os.path.dirname(lora_path),
                    weight_name=os.path.basename(lora_path)
                )
                logger.info(f"✅ LoRA стиль '{lora_name}' успешно применён")
            except Exception as e:
                logger.warning(f"⚠️ Не удалось загрузить LoRA стиль '{lora_name}': {str(e)}")

        return pipe
    except Exception as e:
        logger.error(f"❌ Ошибка при загрузке пайплайна: {str(e)}", exc_info=True)
        st.error(f"Не удалось загрузить модель: {str(e)}")
        raise

def style_with_diffusion(image_path, label, strength=0.7, steps=30, guidance=13.0, lora_name=None):
    try:
        init_image = PILImage.open(image_path).convert("RGB")
        if min(init_image.size) < 512:
            init_image = init_image.resize((512, 512))

        pipe = load_diffusion_pipeline(lora_name)
        prompt = f"профессиональный логотип: {label}, минимализм, векторная графика"
        negative_prompt = "размытость, артефакты, водяные знаки, низкое качество"

        result = pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance,
            negative_prompt=negative_prompt
        ).images[0]

        final_path = f"output/logo_{label}_{int(time.time())}.png"
        result.save(final_path)
        return final_path
    except Exception as e:
        logger.error(f"Ошибка стилизации: {str(e)}")
        return None
