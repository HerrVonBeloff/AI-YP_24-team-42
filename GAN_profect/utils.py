import torch
import torch.nn.functional as F
from data_loader import text_plus_image
from config import model_save_path, batch_size, metrics_save_path
import pandas as pd
import os
from metrics import inception_score, calculate_fid
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.amp import GradScaler, autocast

scaler = GradScaler("cuda")

def calculate_discriminator_loss(discriminator, images, labels, adversarial_loss):
    outputs = discriminator(images)
    return adversarial_loss(outputs, labels)

def train_step(real_images, real_text, generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, device, z_dim):
    real_images = real_images.to(device)
    real_text = real_text.to(device)

    # Тренировка дискриминатора
    optimizer_D.zero_grad()
    current_batch_size = real_images.size(0)  # Динамический размер батча
    real_labels = torch.ones(current_batch_size, 1).to(device)
    fake_labels = torch.zeros(current_batch_size, 1).to(device)

    # Реальные изображения с текстом
    real_images_combined = text_plus_image(real_text, real_images)
    real_output = discriminator(real_images_combined)
    real_loss = adversarial_loss(real_output, real_labels)

    # Генерация фейковых изображений
    latent_vector = torch.cat([torch.randn(real_images.size(0), z_dim).to(device), real_text], 1)
    fake_images = generator(latent_vector)
    fake_images_combined = text_plus_image(real_text, fake_images)
    fake_output = discriminator(fake_images_combined.detach())
    fake_loss = adversarial_loss(fake_output, fake_labels)

    d_loss = real_loss + fake_loss

    scaler.scale(d_loss).backward()
    scaler.step(optimizer_D)
    scaler.update()

    # Тренировка генератора
    optimizer_G.zero_grad()

    fake_output = discriminator(fake_images_combined)
    g_loss = adversarial_loss(fake_output, real_labels)

    scaler.scale(g_loss).backward()
    scaler.step(optimizer_G)
    scaler.update()

    return d_loss.item(), g_loss.item(), fake_images

def train(epochs, train_loader, generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, device, z_dim, txt_to_generate, metrics_data):
    for epoch in range(epochs):
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{epochs}]", leave=False)

        for image, text in tqdm(progress_bar):
            d_loss, g_loss, fake_images = train_step(
                image, text, generator, discriminator, optimizer_G, optimizer_D, adversarial_loss, device, z_dim
            )

        if epoch % 10 == 0:
            print(
                    f"Epoch [{epoch}/{epochs - 1}] D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}"
                )
            
            torch.save(
                generator.state_dict(),
                os.path.join(model_save_path, f"generator_{epoch}.pth"),
            )
            torch.save(
                discriminator.state_dict(),
                os.path.join(model_save_path, f"discriminator_{epoch}.pth"),
            )

            generate_and_save_images(generator, epoch, txt_to_generate, device)

            is_mean, is_std = inception_score(
                images=fake_images, batch_size=batch_size, splits=10
            )

            print(f"Epoch {epoch} - Inception Score: {is_mean:.4f} ± {is_std:.4f}")

            fid_score = calculate_fid(
                real_images=image, generated_images=fake_images, batch_size=batch_size
            )

            print(f"Epoch {epoch} - FID Score: {fid_score:.4f}")

            new_row = pd.DataFrame(
                [
                    {
                        "epoch": epoch,
                        "g_loss": g_loss,
                        "d_loss": d_loss,
                        "inception_score_mean": is_mean,
                        "inception_score_std": is_std,
                        "fid_score": fid_score,
                    }
                ]
            )
            metrics_data = pd.concat([metrics_data, new_row], ignore_index=True)

def generate_and_save_images(model, epoch, txt_to_generate, device):
    z = torch.cat([torch.randn(16, 100), txt_to_generate], 1).to(device)
    generated_images = model(z).detach().cpu()

    plt.figure(figsize=(4, 4))

    for i in range(generated_images.size(0)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(
            generated_images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5
        )  # Денормализация
        plt.axis("off")

    plt.savefig(f"kaggle/working/image_at_epoch_{epoch}.png")
    plt.show()