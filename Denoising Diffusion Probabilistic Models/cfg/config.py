# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:36:30 2024
@author: Vikram Sandu

Configuration class for "Denoising Diffusion Training Model"

Note: Please initialize this class appropriately before training.
"""

class CONFIG:
    
    # Training Configuration
    in_channels = 1 # Input Image Channels
    img_size = 28 # Input Image Size (Size X Size)
    num_epochs = 50 # Number of Epochs to train
    lr = 1e-4 # Learning Rate
    batch_size = 128 # Batch Size for dataloader
    
    # Diffusion Process Configuration
    num_timesteps = 1000 # Timesteps in Diffusion Process
    num_img_to_generate = 256 # Number of Images to Generate after training
    
    # Unet Model Configuration
    t_emb_dim = 128 # Time Embedding Dimension
    down_ch = [32, 64, 128, 256] # Number of channels in Down-Module
    mid_ch = [256, 256, 128] # Number of channels in Mid-Module
    up_ch = [256, 128, 64, 16] # Number of channels in Up-Module
    down_sample = [True, True, False] # whether to perform downsampling or not in Down-blocks
    num_layers_module = [2, 2, 2] # Number of layers in [DownC, MidC, UpC] module
    down_sampling_methods = ['conv', 'mpool'] # Uses both Conv2d and Maxpooling for downsampling
    up_sampling_methods = ['conv', 'upsample'] # Uses both Conv2D and nn.Upsampling for Upsampling
    
    
    # Path Configuration
    model_path = 'pretrained_models/ddpm_unet.pth' # Where to store the model
    train_csv_path = 'path_to_train_csv'
    test_csv_path = 'path_to_test_csv_for_evaluation'
    generated_csv_path = 'utils/mnist_generated_data.csv' # where to save generated images
    
