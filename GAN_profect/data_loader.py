# data_loader.py

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import spacy
import torch.nn.functional as F
from config import *

class TransformDataset(Dataset):
    def __init__(self, df, new_size, nlp_model):
        self.df = df
        self.new_size = new_size
        self.transform = transforms.Compose([
            transforms.Resize(new_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.nlp_model = nlp_model

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Обработка изображения
        image = self.df[idx]["image"]
        image = image.convert("RGB")
        image = self.transform(image)

        # Обработка текста
        text = self.df[idx]["text"]
        doc = self.nlp_model(text)

        # 1. Лемматизация (убираем пунктуацию и пробелы)
        lemmatized_tokens = [
            token.lemma_ for token in doc
            if not token.is_punct and not token.is_space and not token.is_stop  # Исключаем стоп-слова
        ]

        # 2. Извлекаем эмбеддинги только для лемматизированных токенов
        vectors = [token.vector for token in doc if token.lemma_ in lemmatized_tokens]

        # Обработка случая, если векторы пустые
        if len(vectors) > 0:
            text_embedding = np.mean(vectors, axis=0)  # Средний эмбеддинг
        else:
            text_embedding = np.zeros(vect_size)  # Заполняем нулями

        return image, torch.tensor(text_embedding, dtype=torch.float32)


def text_plus_image(text, image):
    embedding = text.unsqueeze(-1).unsqueeze(-1)
    text_embedding_tensor = F.interpolate(embedding, size=image.shape[2:], mode="bilinear", align_corners=False)
    combined_input = torch.cat((text_embedding_tensor, image), dim=1)
    return combined_input