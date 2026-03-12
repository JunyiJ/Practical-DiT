import argparse
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf

from diffusion import DDPM
from model import DiT


def load_model(model_config_path: str, checkpoint_path: str, device: torch.device) -> DiT:
    cfg = OmegaConf.load(model_config_path)
    model = DiT(**cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def sample_images(
    model: DiT,
    diffusion: DDPM,
    num_samples: int,
    device: torch.device,
    class_label: Optional[int] = None,
) -> torch.Tensor:
    # Reverse diffusion sampling with DDPM posterior mean/variance.
    # Return shape: (N, C, H, W)
    if class_label is not None:
        labels = torch.full((num_samples,), class_label, device=device, dtype=torch.long)
    else:
        labels = None

    x_t = torch.randn(
        num_samples,
        model.in_channels,
        model.image_size,
        model.image_size,
        device=device,
    )

    alphas = diffusion.alphas.to(device=device, dtype=x_t.dtype)
    alpha_bars = diffusion.alpha_bars.to(device=device, dtype=x_t.dtype)
    betas = diffusion.betas.to(device=device, dtype=x_t.dtype)

    with torch.no_grad():
        for i in range(diffusion.num_timesteps - 1, -1, -1):
            t = torch.full((num_samples,), i, device=device, dtype=torch.long)
            eps = model(x_t, t, labels)
            alpha_t = alphas[i]
            alpha_bar_t = alpha_bars[i]
            beta_t = betas[i]

            # Predict x0 from epsilon model and keep it in the training image range.
            x0_pred = (x_t - torch.sqrt(1 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
            x0_pred = x0_pred.clamp(0.0, 1.0)

            if i > 0:
                alpha_bar_prev = alpha_bars[i - 1]
            else:
                alpha_bar_prev = torch.tensor(1.0, device=device, dtype=x_t.dtype)

            coef_x0 = (torch.sqrt(alpha_bar_prev) * beta_t) / (1 - alpha_bar_t)
            coef_xt = (torch.sqrt(alpha_t) * (1 - alpha_bar_prev)) / (1 - alpha_bar_t)
            mean = coef_x0 * x0_pred + coef_xt * x_t

            if i > 0:
                posterior_var = beta_t * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                x_t = mean + torch.sqrt(posterior_var) * torch.randn_like(x_t)
            else:
                x_t = mean
    return x_t.clamp(0.0, 1.0)


def save_or_print(samples: torch.Tensor, output_path: Optional[str]) -> None:
    print(f"Samples: shape={tuple(samples.shape)}, min={samples.min().item():.3f}, max={samples.max().item():.3f}, mean={samples.mean().item():.3f}")
    if output_path:
        try:
            from torchvision.utils import save_image
        except ImportError as exc:
            raise RuntimeError("torchvision is required to save images") from exc

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Training uses ToTensor() without normalization, so clamp to [0, 1] for visualization.
        save_image(samples.clamp(0.0, 1.0), path, nrow=min(8, samples.shape[0]), normalize=False)
        print(f"Saved image grid to: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DiT sampling skeleton")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument(
        "--model-config",
        default="../conf/model/dit_tiny.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--class-label", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--output", default=None, help="Optional output image path")
    parser.add_argument(
        "--training-config",
        default="../conf/training/cifar10_default.yaml",
        help="Path to training config YAML (used for num_timesteps)",
    )
    parser.add_argument(
        "--num-timesteps",
        type=int,
        default=None,
        help="Override diffusion steps at sampling; should match training",
    )
    parser.add_argument("--beta-start", type=float, default=None, help="Override beta_start at sampling")
    parser.add_argument("--beta-end", type=float, default=None, help="Override beta_end at sampling")
    return parser.parse_args()


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    print(f"Using device: {device}")

    model = load_model(args.model_config, args.checkpoint, device)
    training_cfg = OmegaConf.load(args.training_config)
    if args.num_timesteps is not None:
        num_timesteps = args.num_timesteps
    else:
        num_timesteps = int(training_cfg.num_timesteps)
    if args.beta_start is not None:
        beta_start = args.beta_start
    else:
        beta_start = float(training_cfg.get("beta_start", 1.0e-4))
    if args.beta_end is not None:
        beta_end = args.beta_end
    else:
        beta_end = float(training_cfg.get("beta_end", 2.0e-2))
    diffusion = DDPM(
        num_timesteps=num_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        device=str(device),
    )
    print(
        f"Sampling with num_timesteps={num_timesteps}, "
        f"beta_start={beta_start}, beta_end={beta_end}"
    )

    print("Sampling unconditional...")
    samples = sample_images(model, diffusion, args.num_samples, device)
    save_or_print(samples, args.output)

    if args.class_label is not None:
        print(f"Sampling conditional label={args.class_label}...")
        cond_samples = sample_images(model, diffusion, args.num_samples, device, args.class_label)
        if args.output:
            output_path = Path(args.output)
            output_path = output_path.with_name(
                f"{output_path.stem}_cond{args.class_label}{output_path.suffix}"
            )
            save_or_print(cond_samples, str(output_path))
        else:
            save_or_print(cond_samples, None)


if __name__ == "__main__":
    main()
