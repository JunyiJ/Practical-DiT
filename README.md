# Practical-DiT

# DDPM Quick Summary

1. The Noise Schedule ($\beta$ and $\alpha$)
$\beta_t$ as the amount of noise added at each step $t$. One can 
start with basic linear schedule.

* Let $\alpha_t = 1 - \beta_t$
* Let $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$

2. The Forward Process: Jump to Timestamp $t$
Ho et al. prove we can sample the noisy image $x_t$ at any arbitrary timestep directly from the original image $x_0$

$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$

(where $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ is pure standard Gaussian noise).

3. The simplified loss function
$$L_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ ||\epsilon - \epsilon_\theta(x_t, t)||^2 \right]$$

* Basically you pass the image and generate a timestamp $t$ to get the noisy image $x_t$
* The DiT model takes $x_t$ and $t$ as input to predict the noise.
* We calculate the mean squared error (MSE) between true noise and predicted noise.