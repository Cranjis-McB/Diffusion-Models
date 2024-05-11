# -*- coding: utf-8 -*-
"""
Created on Fri May 10 14:16:22 2024

@author: Vikram Sandu

Description: Performance Evaluation of the Generated Images.

"""

import torch
from torch.utils.data import DataLoader # Dataloader
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.inception import Inception_V3_Weights

from cfg.config import CONFIG
from dataset.mnist_dataset import CustomMnistDataset
from utils.metrics import calculate_activation_statistics, calculate_frechet_distance

def evaluate_generated_images():
    """
    Evaluates the performance of the Generated
    Images by trained diffusion model
    """
    
    # Get Configuration
    cfg = CONFIG()
    
    # Transform to Convert Output of CustomMnistDataset class to Inception format.
    transform_inception = transforms.Compose([
    transforms.Lambda(lambda x: (x + 1.0)/2.0), # [-1, 1] => [0, 1]
    transforms.ToPILImage(), # Tensor to PIL Image 
    transforms.Resize((299, 299)),
    transforms.Grayscale(num_output_channels=3),  # Convert to RGB format
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization

])
    
    # Load InceptionV3 Model
    model = models.inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    
    # Mean and Sigma For Generated Data
    mnist_ds = CustomMnistDataset(cfg.generated_csv_path, cfg.num_img_to_generate)
    mnist_dl = DataLoader(mnist_ds, cfg.batch_size//4, shuffle=False)
    mu1, sigma1 = calculate_activation_statistics(mnist_dl, model, preprocess = transform_inception, device='cpu')
    
    # Mean and Sigma for Test Data
    mnist_ds = CustomMnistDataset(cfg.test_csv_path, cfg.num_img_to_generate)
    mnist_dl = DataLoader(mnist_ds, cfg.batch_size//4, shuffle=False)
    mu2, sigma2 = calculate_activation_statistics(mnist_dl, model, preprocess = transform_inception, device='cpu')
    
    # Calculate FID
    fid = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    
    print(f'FID-Score: {fid}')
    
    return fid

#-------------------------------------------------------------------

if __name__ == '__main__':
    # Evaluate
    evaluate_generated_images()