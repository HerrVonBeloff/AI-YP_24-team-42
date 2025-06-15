import torch
import torch.nn as nn
import torchvision.utils as vutils
from models.upscaler import upscale_image
from models.diffusion import style_with_diffusion

class Generator(nn.Module):
    def __init__(self, z_dim=100, text_embedding_dim=128, output_channels=3):
        super().__init__()
        self.text_embedding_dim = text_embedding_dim
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_dim + text_embedding_dim, 256, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(256), nn.ReLU(True),
            ...
        )
    def forward(self, z: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([z, text_embedding], dim=1).view(z.size(0), -1, 1, 1)
        return self.model(combined)

# Инициализация модели
device = 'cuda' if torch.cuda.is_available() else 'cpu'
generator = Generator().to(device)
...
