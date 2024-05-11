# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:13:07 2024

@author: Vikram Sandu

This code implements the Fréchet Inception Distance (FID) 
which is used to evaluate the performance of the Generated
images.

Code Reference: "https://github.com/mseitzer/pytorch-fid"
"""

# Imports
from scipy import linalg
import numpy as np
import torch
from tqdm import tqdm



def get_activation(dataloader, 
                   model, 
                   preprocess, # Preprocessing Transform for InceptionV3
                   device = 'cpu'
                  ):
    """
    Given Dataloader and Model, Generate N X 2048
    Dimensional activation map for N data points
    in dataloader.
    """
    
    # Set model to evaluation Mode
    model.to(device)
    model.eval()
    
    # Save activations
    pred_arr = np.zeros((len(dataloader.dataset), 2048))
    
    # Batch Size
    batch_size = dataloader.batch_size
    
    # Loop over Dataloader
    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            
            # Transform the Batch according to Inceptionv3 specification
            batch = torch.stack([preprocess(img) for img in batch]).to(device)
            
            # Predict
            pred = model(batch).cpu().numpy()
            
            # Store
            pred_arr[i*batch_size : i*batch_size + batch.size(0), :] = pred
            
    return pred_arr

#--------------------------------------------------------------------

def calculate_activation_statistics(dataloader, 
                                    model, 
                                    preprocess, 
                                    device='cpu'
                                   ):
    """
    Get mean vector and covariance matrix of the activation maps.
    """
    
    # Get activation maps
    act = get_activation(dataloader, 
                         model, 
                         preprocess, # Preprocessing Transform for InceptionV3
                         device
                       )
    # Mean
    mu = np.mean(act, axis=0)
    
    # Covariance Metric
    sigma = np.cov(act, rowvar=False)
    
    return mu, sigma

#----------------------------------------------------------------

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    
    """
    Given Mean and Sigma of Real and Generated Data,
    it calculates FID between them using:
     
     d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
     
    """
    # Make sure they have appropriate dims
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)
    
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    # Handle various cases
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

#-----------------------------------------------------------------------------------
    