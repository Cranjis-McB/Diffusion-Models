# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:12:51 2024

@author: Vikram Sandu

Description: Visualize the Generated Images.
"""

# Imports
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from cfg.config import CONFIG


def visualize(n_rows = 8): 
    
    """
    Read n_rows * n_rows images from generated csv file
    and plot them.
    """
    
    # Config
    cfg = CONFIG()
    
    # Read file
    df = pd.read_csv(cfg.generated_csv_path)
    
    # Randomly choose n_rows*n_rows images from csv
    ims_to_plot = df.sample(n_rows * n_rows).values
    
    # Plot
    fig, axes = plt.subplots(n_rows, n_rows, figsize=(5, 5))
    
    # Plot each image in the corresponding subplot
    for i, ax in enumerate(axes.flat):
        ax.imshow(np.reshape(ims_to_plot[i], (cfg.img_size, cfg.img_size)), cmap='gray')  # You might need to adjust the colormap based on your images
        ax.axis('off')  # Turn off axis labels
    
    plt.tight_layout()  # Adjust spacing between subplots
    plt.show()
    
    return

#---------------------------------------------------------

if __name__ == '__main__':
    # Visualize
    visualize()