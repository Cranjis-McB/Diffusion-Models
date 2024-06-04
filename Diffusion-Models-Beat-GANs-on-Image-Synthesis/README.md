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

![image](https://github.com/Cranjis-McB/Diffusion-Models/assets/82195207/f0aed4d0-5520-44e1-809b-d8ccdb6399dd)


                       Linear (Top) vs Cosine (Bottom) Schedular
                       
* Author designs the Cosine Noise Schedular as follows:

![image](https://github.com/Cranjis-McB/Diffusion-Models/assets/82195207/b67e7b3f-5554-45c4-ab3c-6054780e5c3c)

* Then Noise Controlling parameter β is calculated by the following equation:

$$
\beta_{t} = 1 - \frac{\bar{\alpha}_{t}}{\bar{\alpha}_{t-1}}
$$

### 2. Learnt Variance through Hybrid-Loss

* In the [DDPM paper](https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf), the variance Σ is kept **constant** and set to either forward process variance β<sub>t</sub> or forward posterior variance β<sup>~</sup><sub>t</sub>.

* Both choices of variance yielded almost similar results according to the DDPM paper which is bit unusual considering β<sub>t</sub> and β<sup>~</sup><sub>t</sub> are two opposite extremes.

* According to [this paper](https://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf), the reason behind this is that for simple MSE loss, the choice of Σ doesn't matter much. However, the choice of Σ can be crucial in improving the log-likelihood.

* To achieve this, author chooses to learn the variance Σ through neural network instead of setting it to the constant.

* Author chooses to parametrize the variance as an **interpolation of β<sub>t</sub> and β<sup>~</sup><sub>t</sub> in log domain**. (May be experimental?) and the **v** is learnt through neural network as shown below.

![image.png](attachment:ca31f187-ad39-4abf-9739-a11013cc1314.png)

* Since current objective of **Simple MSE Loss doesn't depend on the Σ** so author proposed the **hybrid loss which depends on the Σ also**. The hybrid loss is the weighted sum of MSE Loss and Variational Lower bound Loss as shown below.

### 3. Improved Sampling Speed

* As discussed in the last section, Improved sampling speed is the result of learnt variance through hybrid loss.

* It allows the diffusion model to sample with much fewer steps (250 in paper) than they were trained with (typically 1000).

* The question now is: which 250 steps should we use? A straightforward approach would be to choose 250 evenly spaced steps between 1 and 1000. However, this method is sub-optimal. The author suggests a more effective strategy: **taking larger steps at the beginning of the reverse process and progressively smaller steps as you approach x<sub>0</sub>.**

* The rationale behind this strategy is that at the beginning of the reverse process, there is more noise, allowing for skipping steps. However, as you get closer to x<sub>0</sub>, the noise decreases, and taking larger steps might remove important image information along with the noise.

* when sampling with fewer steps, the parameter β<sub>t</sub> also needs to be modified accordingly as shown below.

![image.png](attachment:f69a60b0-92bd-4d4d-87ee-8000246f4b19.png)



![image.png](attachment:701b85c0-280e-4282-9251-ab97f33ad968.png)

* According to the author, **Learning Variance through hybrid loss allowed the diffusion model to sample with fewer steps without much drop in the sample quality.**

* In paper, the author uses **250 reverse steps** during sampling as compared to the 1000 in the DDPM paper. (It mean the sampling time is reduced by 3/4th times). **Note that During training the complete 1000 forward steps are utilized but the sampling is done using 250 reverse steps only**.

### 4. Class Embedding + Adaptive GroupNorm in U-net

* As we have seen in the [DDPM implementation](https://www.kaggle.com/code/vikramsandu/ddpm-from-scratch-in-pytorch#Model-Architecture) that timesteps information was utilised in the Unet architecture for predicting noise.  

* In this paper, Author also utilizes **class label information** in the U-net. First the class labels are converted into embedding through nn.Embedding Layer and merged with Timestep embedding through addition.

* Author also uses **Adaptive Instance Normalization (AdaIn)** while merging the Embedding with features extracted (inside the Resnet Block of the U-net). AdaIn basically aligns the mean and variance of the features with Embeddings.

* The extracted features and Projection of the Embeddings (timesteps + label) are merged in the following manner. where scale and shift are obtained from Projection of the Embeddings.

$$\text{GroupNorm}(features) * (scale + 1) + shift$$

* Author mentioned that using this technique slightly improved the FID score.

* We will understand this more clearly during implementation of the same.

### 5. Classifier Guidance

* It is a process of guiding the diffusion process towards an arbitrary class label y through a pretrained classifier.

* In this process, first a classifier is trained on noisy images obtained from diffusion forward Process and labels.

* Then Diffusion Model is trained as usual. (No need to do anything extra)

* Finally they use **gradient of the pretrained classifier to guide the diffusion process during sampling**. The sampling is done using the following distribution.

$$ N(\mu + \Sigma g, \Sigma) $$

* So here they **shifted the mean of diffusion process by Σg**. where g is gradient of the image contributing to that particular class label y. Note that here µ and Σ are the estimated mean and variance of the diffusion reverse process.

* According to the paper, Classifier guidance improved the precision at the cost of recall. i.e. **generated better looking samples but less diverse**, thus introducing a trade-off in sample fidelity versus diversity.

* They also mentioned that it is also crucial to **scale the classfier gradient by a factor more than 1** to generate images of a given class label with high probability.
