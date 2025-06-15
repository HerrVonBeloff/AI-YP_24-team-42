import torch
import torch.nn as nn
import torchvision.utils as vutils
import os
import tempfile
import torch.nn.functional as F

RESULT_DIR = tempfile.mkdtemp()
os.makedirs(RESULT_DIR, exist_ok=True)

class Generator(nn.Module):
    def __init__(self, z_dim=100, text_embedding_dim=128, output_channels=3):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim + text_embedding_dim, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, z: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([z, text_embedding], dim=1).view(z.size(0), -1, 1, 1)
        return self.model(combined)
