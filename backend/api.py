import os
import logging
from logging.handlers import RotatingFileHandler
import spacy
import torch
from models import Generator
from models import TextTransformer
from fastapi import FastAPI, HTTPException, Body
from typing import List, Annotated
from pydantic import BaseModel
import numpy as np

# Создание папки для логов, если она не существует
if not os.path.exists('logs'):
    os.makedirs('logs')

# Настройка логирования
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)
handler = RotatingFileHandler('logs/logs.api', maxBytes=10 * 1024 * 1024, backupCount=5)  # 10 MB max per file, 5 backups
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

app = FastAPI()

# Логирование запуска приложения
logger.info("Starting the FastAPI application...")

spacy.prefer_gpu()
nlp = spacy.load("en_core_web_md")

# Подготовка генератора
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = Generator(z_dim=100).to(device)
generator.load_state_dict(torch.load(r"generator_15.pth", map_location=device))
generator.eval()

class TextDescription(BaseModel):
    description: str

class LogoImage(BaseModel):
    image: List[List[List[int]]]  # Предполагаем, что изображение возвращается в формате 3D списка (H x W x C)

@app.post("/generate/", response_model=LogoImage)
async def generate_logo(
    text: Annotated[TextDescription, Body()]
) -> LogoImage:
    logger.info(f"Received request to generate logo with description: {text.description}")
    try:
        input_tensor = preprocess_text(text.description)
        logger.debug(f"Input tensor shape: {input_tensor.shape}")

        with torch.no_grad():
            output = generator(input_tensor.to(device))
            logger.debug(f"Output shape from generator: {output.shape}")

        image = postprocess_output(output)
        logger.info("Logo generated successfully.")
        return {"image": image.tolist()}
    except Exception as e:
        logger.error(f"Error occurred while generating logo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Функция для преобразования вывода модели в изображение
def postprocess_output(output):
    logger.debug("Postprocessing output to convert to image.")
    image = (output.cpu().detach().numpy().squeeze(0) * 255).astype("uint8")
    image = np.transpose(image, (1, 2, 0))
    logger.debug(f"Image shape after postprocessing: {image.shape}")
    return image

# Функция для предобработки текста во вход генератора
def preprocess_text(text):
    logger.debug(f"Preprocessing text: {text}")
    z_dim = 100
    random_data = torch.randn(1, z_dim)
    text_transformer = TextTransformer(nlp_model=nlp)
    text_vector = text_transformer.transform(text)
    z = torch.cat((random_data, text_vector), dim=1)
    logger.debug(f"Generated latent vector shape: {z.shape}")
    return z

# Обработка глобальных ошибок приложения
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception occurred: {str(exc)}")
    return HTTPException(status_code=500, detail="Internal Server Error")
