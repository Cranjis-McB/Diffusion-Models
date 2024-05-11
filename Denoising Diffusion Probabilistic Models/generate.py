# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:13:49 2024

@author: Vikram Sandu

This code is used to Generate Images using Trained
Diffusion Model.
"""

# Import
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

from diffusion_process.gaussian_process import DiffusionReverseProcess
from cfg.config import CONFIG

def generate_one_image(cfg):
    """
    Given Pretrained DDPM U-net model, Generate Real-life
    Images from noise by going backward step by step. i.e.,
    Mapping of Random Noise to Real-life images.
    """
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(f'Device: {device}\n')
    
    # Initialize Diffusion Reverse Process
    drp = DiffusionReverseProcess()
    
    # Set model to eval mode
    model = torch.load(cfg.model_path, map_location='cpu').to(device)
    model.eval()
    
    # Generate Noise sample from N(0, 1)
    xt = torch.randn(1, cfg.in_channels, cfg.img_size, cfg.img_size).to(device)
    
    # Denoise step by step by going backward.
    with torch.no_grad():
        for t in reversed(range(cfg.num_timesteps)):
            noise_pred = model(xt, torch.as_tensor(t).unsqueeze(0).to(device))
            xt, x0 = drp.sample_prev_timestep(xt, noise_pred, torch.as_tensor(t).to(device))

    # Convert the image to proper scale
    xt = torch.clamp(xt, -1., 1.).detach().cpu()
    xt = (xt + 1) / 2
    
    return xt

#-------------------------------------------------------------------

def generate_images():
    
    """
    Generate cfg.num_img_to_generate images
    and save them in cfg.generated_csv_path.
    """
    
    # Config
    cfg = CONFIG()
    
    # Generate
    generated_imgs = []
    for i in tqdm(range(cfg.num_img_to_generate)):
        xt = generate_one_image(cfg)
        xt = 255 * xt[0][0].numpy()
        generated_imgs.append(xt.astype(np.uint8).flatten())
    
    # Save Generated Data CSV
    generated_df = pd.DataFrame(generated_imgs, columns=[f'pixel{i}' for i in range(784)])
    generated_df.to_csv(cfg.generated_csv_path, index=False)
    
    return 0

#---------------------------------------------------------

if __name__ == '__main__':
    # Generate
    generate_images()