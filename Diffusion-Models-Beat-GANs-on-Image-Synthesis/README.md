## Overview

* [Diffusion Models Beat GANs on Image Synthesis](https://openreview.net/pdf?id=AAWuCvzaVt) paper builds upon the earlier paper [Improved Denoising Diffusion Probabilistic Models](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf) by the same authors from OpenAI.

* [Improved Denoising Diffusion Probabilistic Models](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf) introduced several improvements over DDPM such as **Cosine Noise Schedular, Learnt Variance, and Improved Sampling Speed.** 

* Later [Diffusion Models Beat GANs on Image Synthesis](https://openreview.net/pdf?id=AAWuCvzaVt) proposes further improvements such as **Better Architecture and Classifier Guidance** which surpasses GANs at that time.

## Improvements over [DDPM](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf)

### 1. Cosine Noise Schedular

* DDPM utilizes a linear noise scheduler, wherein the parameter β controlling the noise is linearly sampled from the range **[1e-4, 0.02]**.

* The drawback of the Linear Noise scheduler is that it **deteriorates the image well before reaching the final time steps T**. Therefore, when observing images in the last few time steps, they mostly appear as noise.

* The author of the [paper](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf) also concluded that last 20% of the timesteps is not useful and can be skipped during reverse process without compromising the performance.

* To tackle this issue and make every timestep count, **Cosine Noise Schedular** is proposed which **prevents destroying the whole image information well before reaching the final timestep**.

![image.png](attachment:20f19b6f-81d3-4ef6-bdcd-82dde652f6b0.png)

                       Linear (Top) vs Cosine (Bottom) Schedular
                       
* Author designs the Cosine Noise Schedular as follows:

![image.png](attachment:070a6660-0ae0-45f2-b298-813f2e10f780.png)

* Then Noise Controlling parameter β is calculated by the following equation:

$$
\beta_{t} = 1 - \frac{\bar{\alpha}_{t}}{\bar{\alpha}_{t-1}}
$$
