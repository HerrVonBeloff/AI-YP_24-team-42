import os
import tempfile
import logging
from typing import List

RESULT_DIR = tempfile.mkdtemp()
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs("output", exist_ok=True)
LORA_PATH = "./lora"

logger = logging.getLogger("LogoGenerator")

def get_available_loras():
    if not os.path.exists(LORA_PATH):
        os.makedirs(LORA_PATH, exist_ok=True)
        return []
    loras = []
    for file in os.listdir(LORA_PATH):
        if file.endswith(('.safetensors', '.pt', '.bin')):
            loras.append(file)
    return loras
