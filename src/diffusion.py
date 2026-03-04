import torch

class DDPM:
    """
    Denoising Diffusion Probabilistic Model utility class.
    Handles the noise schedule and loss computation.
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cpu'):
        self.num_timesteps = num_timesteps
        self.device = device
        # Define noise schedule (betas, alphas, alpha_bars)
        # You can start with a simple linear schedule.
        self.betas = torch.linspace(beta_start, beta_end, self.num_timesteps).to(device)
        self.alphas = 1 - betas
        self.alpha_bars = torch.cumprod(self.alpha, dim=0)


    def add_noise(self, x_start, t, noise=None):
        """
        The forward process: q(x_t | x_0).
        """
        # Implement the mathematical formula to sample x_t 
        # given the original image x_start and timestep t.
        # x_start [B, C, H, W]
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bars[t])
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - self.alpha_bars[t])
        x_t = sqrt_alpha_bar_t.view(-1, 1, 1, 1) * x_start + sqrt_one_minus_alpha_bar_t.view(-1, 1, 1, 1) * noise
        return x_t

    def compute_loss(self, model, x_start):
        """
        Samples random timesteps, adds noise, and computes the MSE loss
        between the true noise and the model's predicted noise.
        """
        B, C, H, W = x_start.shape
        # 1. Sample random timesteps t for the batch (unique timestamp for EACH image)
        t = torch.randint(0, self.num_timesteps, (B, ), device=self.device)
        # 2. Generate random Gaussian noise
        noise = torch.randn_like(x_start)
        # 3. Add noise to x_start to get x_t
        x_t = self.add_noise(x_start, t, noise)
        # 4. Pass x_t and t to the model to predict the noise
        pred = model(x_t, t)
        # 5. Return the Mean Squared Error between true and predicted noise
        loss = F.mse_loss(pred, noise)
        return loss