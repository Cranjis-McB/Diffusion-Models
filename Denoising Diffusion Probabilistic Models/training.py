# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:13:33 2024

@author: Vikram Sandu

Training Code for Diffusion Model.
"""

# Imports
import numpy as np
import torch
from torch.utils.data import DataLoader # Dataloader
from tqdm import tqdm # Progress Bar

from diffusion_process.gaussian_process import DiffusionForwardProcess # Import Diffusion Forward Process
from model.unet import Unet # Import Model
from cfg.config import CONFIG # Import Configuration file
from dataset.mnist_dataset import CustomMnistDataset # Import Dataset Class


def train():
    
    # Extract the Configuration
    cfg = CONFIG()
    
    # Dataset and Dataloader
    mnist_ds = CustomMnistDataset(cfg.train_csv_path)
    mnist_dl = DataLoader(mnist_ds, cfg.batch_size, shuffle=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')
    
    # Initiate Model
    model = Unet(cfg).to(device)
    
    # Initialize Optimizer and Loss Function
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = torch.nn.MSELoss()
    
    # Diffusion Forward Process to add noise
    dfp = DiffusionForwardProcess()
    
    # Best Loss
    best_eval_loss = float('inf')
    
    # Train
    for epoch in range(cfg.num_epochs):
        
        # For Loss Tracking
        losses = []
        
        # Set model to train mode
        model.train()
        
        # Loop over dataloader
        for imgs in tqdm(mnist_dl):
            
            imgs = imgs.to(device)
            
            # Generate noise and timestamps
            noise = torch.randn_like(imgs).to(device)
            t = torch.randint(0, cfg.num_timesteps, (imgs.shape[0],)).to(device)
            
            # Add noise to the images using Forward Process
            noisy_imgs = dfp.add_noise(imgs, noise, t)
            
            # Avoid Gradient Accumulation
            optimizer.zero_grad()
            
            # Predict noise using U-net Model
            noise_pred = model(noisy_imgs, t)
            
            # Calculate Loss
            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            
            # Backprop + Update model params
            loss.backward()
            optimizer.step()
        
        # Mean Loss
        mean_epoch_loss = np.mean(losses)
        
        # Display
        print('Epoch:{} | Loss : {:.4f}'.format(
            epoch + 1,
            mean_epoch_loss,
        ))
        
        # Save based on train-loss
        if mean_epoch_loss < best_eval_loss:
            best_eval_loss = mean_epoch_loss
            torch.save(model, cfg.model_path)
            
    print('\nTraining completed.....')

#--------------------------------------------------------

if __name__ == '__main__':
    # Train
    train()