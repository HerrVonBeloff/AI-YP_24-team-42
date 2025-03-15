"""
Этот файл содержит определения моделей генератора и дискриминатора
"""

import torch.nn as nn
from config import *

class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        self.model = nn.Sequential(
            # Линейное преобразование латентного вектора в 4x4x512
            nn.Linear(z_dim + vect_size, 4 * 4 * 512),
            nn.ReLU(True),

            # Преобразуем в 4D тензор
            ViewLayer(512, 4, 4),

            # Первый блок: Upsample + Conv2d
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # Второй блок: Upsample + Conv2d
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # Третий блок: Upsample + Conv2d
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # Выходной слой: Conv2d без BatchNorm
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        z = self.model(z)
        return z

class ViewLayer(nn.Module):
    def __init__(self, channels, height, width):
        super(ViewLayer, self).__init__()
        self.channels = channels
        self.height = height
        self.width = width

    def forward(self, x):
        return x.view(x.size(0), self.channels, self.height, self.width)

# ДИСКРИМИНАТОР
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3 + vect_size, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img)
    
# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3 + vect_size, 64, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(128, 1, kernel_size=7, stride=1, padding=0, bias=False),
#         )

#     def forward(self, img):
#         validity = self.model(img)
#         return validity.mean([2, 3]).view(img.size(0), 1)  # Убедитесь, что размерность (batch_size, 1)