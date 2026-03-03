import torch

class DDPM:
    """
    Denoising Diffusion Probabilistic Model utility class.
    Handles the noise schedule and loss computation.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        # TODO: Define your noise schedule (betas, alphas, alpha_bars)
        # You can start with a simple linear schedule.
        pass

    def add_noise(self, x_start, t, noise=None):
        """
        The forward process: q(x_t | x_0).
        """
        # TODO: Implement the mathematical formula to sample x_t 
        # given the original image x_start and timestep t.
        pass

    def compute_loss(self, model, x_start):
        """
        Samples random timesteps, adds noise, and computes the MSE loss
        between the true noise and the model's predicted noise.
        """
        # TODO: 
        # 1. Sample random timesteps t for the batch
        # 2. Generate random Gaussian noise
        # 3. Add noise to x_start to get x_t
        # 4. Pass x_t and t to the model to predict the noise
        # 5. Return the Mean Squared Error between true and predicted noise
        pass