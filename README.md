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

3. Reparametirization the mean with noise
Basically the true target mean can be calculated by this formula

$$\tilde{\mu}_t = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1 - \bar{\alpha}_t} x_0$$

However, x_0 is unknown and we only have x_t. To fix this, we can derive x_0 based on this formula:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$$
So, $$x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}} (x_t - \sqrt{1 - \bar{\alpha}_t} \epsilon)$$.

Then it's clear to subsitute x_0 with the equation above and after simplification:
$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)$$

which only depends on x_t (known) and noise ($epsilon$, model prediction). As as result, we turn the model from predicting the distribution mean to predicting the noise.

4. The simplified loss function
$$L_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ ||\epsilon - \epsilon_\theta(x_t, t)||^2 \right]$$

* Basically you pass the image and generate a timestamp $t$ to get the noisy image $x_t$
* The DiT model takes $x_t$ and $t$ as input to predict the noise.
* We calculate the mean squared error (MSE) between true noise and predicted noise.


# Cmd cheatsheet

* Train model `python src/train.py`

* Run sampling: e.g. (run within `src/` dir):
```python sample.py --checkpoint ../checkpoints/model_checkpoint_20260310_144946_20260310_144946.pt \
  --class-label 3 --num-samples 8 --output ../outputs/cond_samples.png
```