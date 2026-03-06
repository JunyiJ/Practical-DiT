import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import einops
import math


class TimestepEmbedder(nn.Module):
    """
    Embeds the diffusion timestep into a continuous vector.
    """
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        half = hidden_size // 2
        if half > 1:
            freqs = torch.exp(-math.log(10000) * torch.arange(half) / (half - 1))
        elif half == 1:
            freqs = torch.ones(1)
        else:
            freqs = torch.empty(0)
        self.register_buffer("freqs", freqs, persistent=False)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.SiLU(),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, t):
        # t: (N,) tensor of timesteps
        # Returns: (N, hidden_size)
        t = t.float()
        if self.freqs.numel() == 0:
            emb = torch.zeros((t.shape[0], 0), device=t.device, dtype=t.dtype)
        else:
            freqs = self.freqs.to(device=t.device, dtype=t.dtype)
            args = t[:, None] * freqs[None]
            emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.hidden_size % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)

class DiTBlock(nn.Module):
    """
    A Transformer block with Adaptive Layer Normalization (adaLN).
    """
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        # TODO: Initialize standard transformer components (Attention, MLP)
        self.num_heads = num_heads
        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size must be divisible by num_heads (got {hidden_size}, {num_heads})")
        self.head_dim = hidden_size // num_heads
        self.QKV = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.MLP = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size)
        )
        # TODO: Initialize adaLN components to inject the timestep embedding
        # Hint: You'll need a linear layer that maps the timestep embedding 
        # to the scale and shift parameters for your LayerNorms.
        # There are 2 layer norms for a transformer block and each layer is 
        # composed of 3 params, gamma (scale for LN), beta (shift for LN) and
        # alpha (residual scale applied after the block)
        # [gamma1, gamma2, beta1, beta2, alpha1, alpha2]

        # Turn off PyTorch's default learnable weights so they don't fight our dynamic ones
        self.ln1 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(hidden_size, elementwise_affine=False)
        # DiT usually adds a SiLU activation before projecting the timestep conditioning
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )
        # AdaLN-Zero Initialization: Zero out the final linear layer
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x, c):
        # x: (N, T, hidden_size) - image patches
        # c: (N, hidden_size) - timestep conditioning
        # TODO: Implement the forward pass. Apply adaLN before Attention and MLP.
        adaLN = self.adaLN(c).unsqueeze(1)
        gamma1, gamma2, beta1, beta2, alpha1, alpha2 = adaLN.chunk(6, dim=-1)
        N, T, _ = x.shape
        norm_x = self.ln1(x) * (1 + gamma1) + beta1
        qkv = self.QKV(norm_x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(N, T, self.num_heads, -1).transpose(1, 2)
        k = k.view(N, T, self.num_heads, -1).transpose(1, 2)
        v = v.view(N, T, self.num_heads, -1).transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        y = attn_output.transpose(1, 2).contiguous().view(N, T, -1)
        y = self.out_proj(y)
        y = alpha1 * y
        y = x + y
        norm_y = self.ln2(y) * (1 + gamma2) + beta2
        mlp_output = self.MLP(norm_y)
        mlp_output = alpha2 * mlp_output
        res = y + mlp_output
        return res


class DiT(nn.Module):
    """
    The full Diffusion Transformer.
    """
    def __init__(self, image_size, patch_size, in_channels, hidden_size, depth, num_heads):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_patches = (image_size // patch_size) ** 2
        # TODO: 1. Patchify the image (Conv2d is a standard shortcut here)
        self.x_embedder = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        # TODO: 2. Add learnable positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, hidden_size))
        # TODO: 3. Initialize the TimestepEmbedder
        self.timestep_embed = TimestepEmbedder(hidden_size)
        # TODO: 4. Create a sequential list of DiTBlocks
        self.blocks = nn.ModuleList([DiTBlock(hidden_size, num_heads) for _ in range(depth)])
        # TODO: 5. Un-patchify (Linear layer to map back to patch_size*patch_size*in_channels)
        self.final_layer = nn.Linear(hidden_size, patch_size * patch_size * in_channels)

    def forward(self, x, t):
        # x: (N, C, H, W) noisy images
        # t: (N,) timesteps
        # TODO: Combine all components to output the predicted noise.
        # Ensure the output shape matches the input image shape.
        
        # (N, C, H, W) -> (N, hidden_size, H/P, W/P)
        x = self.x_embedder(x)
        # (N, T=num_patches=H/P*W/P, hidden_size)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embedding
        c = self.timestep_embed(t)
        for block in self.blocks:
            x = block(x, c)
        # (N, T, p * p * in_channels)
        x = self.final_layer(x)
        # use einops instead of manual change.
        p = self.patch_size
        h = self.image_size // p
        x = einops.rearrange(
            x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=h, w=h, p1=p, p2=p, c=self.in_channels
        )
        # x = x.view(x.shape[0], h, w, p, p, self.in_channels)
        # x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        # x = x.view(x.shape[0], self.in_channels, h * p, w * p)
        return x
