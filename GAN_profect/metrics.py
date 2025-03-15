# metrics.py

import torch
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision import transforms
import torchvision.models as models
from scipy.linalg import sqrtm
import numpy as np

def inception_score(images, batch_size=32, splits=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = len(images)

    inception_model = inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT, transform_input=False
    ).to(device)
    inception_model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((299, 299))
        ]
    )
    images = torch.stack([transform(img) for img in images])

    preds = []
    for i in range(0, N, batch_size):
        batch = images[i : i + batch_size].to(device)
        with torch.no_grad():
            preds.append(F.softmax(inception_model(batch), dim=1))
    preds = torch.cat(preds, dim=0).cpu().numpy()

    split_scores = []
    for k in range(splits):
        part = preds[k * (N // splits) : (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = [np.sum(p * (np.log(p) - np.log(py))) for p in part]
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)

def calculate_fid(real_images, generated_images, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inception_model = inception_v3(
        weights=models.Inception_V3_Weights.DEFAULT, transform_input=False
    ).to(device)
    inception_model.eval()

    def get_activations(images):
        activations = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size].to(device)
            with torch.no_grad():
                features = inception_model(batch).detach()
                activations.append(features.cpu())
        return torch.cat(activations, dim=0).numpy()

    transform = transforms.Compose(
        [
            transforms.Resize((299, 299))
        ]
    )
    real_images = torch.stack([transform(img) for img in real_images])
    generated_images = torch.stack([transform(img) for img in generated_images])

    act_real = get_activations(real_images)
    act_gen = get_activations(generated_images)

    mu_real, sigma_real = act_real.mean(axis=0), np.cov(act_real, rowvar=False)
    mu_gen, sigma_gen = act_gen.mean(axis=0), np.cov(act_gen, rowvar=False)

    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real @ sigma_gen).real
    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)

    return fid