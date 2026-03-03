import torch
import torch.nn as nn
import math

class TimestepEmbedder(nn.Module):
    """
    Embeds the diffusion timestep into a continuous vector.
    """
    def __init__(self, hidden_size):
        super().__init__()
        # TODO: Implement sinusoidal timestep embeddings 
        # followed by a 2-layer MLP (Linear -> SiLU -> Linear)
        pass

    def forward(self, t):
        # t: (N,) tensor of timesteps
        # Returns: (N, hidden_size)
        pass

class DiTBlock(nn.Module):
    """
    A Transformer block with Adaptive Layer Normalization (adaLN).
    """
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        # TODO: Initialize standard transformer components (Attention, MLP)
        # TODO: Initialize adaLN components to inject the timestep embedding
        # Hint: You'll need a linear layer that maps the timestep embedding 
        # to the scale and shift parameters for your LayerNorms.
        pass

    def forward(self, x, c):
        # x: (N, T, hidden_size) - image patches
        # c: (N, hidden_size) - timestep conditioning
        # TODO: Implement the forward pass. Apply adaLN before Attention and MLP.
        pass


class DiT(nn.Module):
    """
    The full Diffusion Transformer.
    """
    def __init__(self, image_size, patch_size, in_channels, hidden_size, depth, num_heads):
        super().__init__()
        # TODO: 1. Patchify the image (Conv2d is a standard shortcut here)
        # TODO: 2. Add learnable positional embeddings
        # TODO: 3. Initialize the TimestepEmbedder
        # TODO: 4. Create a sequential list of DiTBlocks
        # TODO: 5. Un-patchify (Linear layer to map back to patch_size*patch_size*in_channels)
        pass

    def forward(self, x, t):
        # x: (N, C, H, W) noisy images
        # t: (N,) timesteps
        # TODO: Combine all components to output the predicted noise.
        # Ensure the output shape matches the input image shape.
        pass