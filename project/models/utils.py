import torch
from matplotlib import pyplot as plt

def train_step(real_images, noise_len, optimizers, models, loss_func, device):
    real_images = real_images.to(device)
    batch_size = real_images.size(0)
    
    optimizer_D, optimizer_G = optimizers
    discriminator, generator = models
    
    # Тренировка дискриминатора
    optimizer_D.zero_grad()

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    real_output = discriminator(real_images)
    real_loss = loss_func(real_output, real_labels)

    random_noise = torch.randn(batch_size, noise_len).to(device)
    fake_images = generator(random_noise)
    fake_output = discriminator(fake_images.detach())
    fake_loss = loss_func(fake_output, fake_labels)

    loss_D = real_loss + fake_loss
    loss_D.backward()
    optimizer_D.step()

    # Тренировка генератора
    optimizer_G.zero_grad()

    fake_output = discriminator(fake_images)
    loss_G = loss_func(fake_output, real_labels)

    loss_G.backward()
    optimizer_G.step()

    return loss_D.item(), loss_G.item()

def generate_and_save_images(model, epoch, noise_len, device):
    noise = torch.randn(16, noise_len).to(device)
    generated_images = model(noise).detach().cpu()*0.5 + 0.5

    plt.figure(figsize=(4, 4))

    for i in range(generated_images.size(0)):
        plt.subplot(4, 4, i+1)
        plt.imshow(generated_images[i].permute(1, 2, 0).numpy())
        plt.axis('off')

    plt.savefig(f'train_results/image_at_epoch_{epoch}.png')
    plt.show()