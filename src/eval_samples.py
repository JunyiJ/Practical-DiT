import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

from diffusion import DDPM
from sample import load_model, resolve_device, sample_images


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def format_scale(scale: float) -> str:
    text = f"{scale:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def to_image_space(samples: torch.Tensor) -> torch.Tensor:
    return ((samples + 1.0) / 2.0).clamp(0.0, 1.0)


def save_case(samples: torch.Tensor, output_path: Path) -> dict[str, float]:
    image_samples = to_image_space(samples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_image(image_samples, output_path, nrow=min(8, samples.shape[0]), normalize=False)
    return {
        "model_min": samples.min().item(),
        "model_max": samples.max().item(),
        "model_mean": samples.mean().item(),
        "image_min": image_samples.min().item(),
        "image_max": image_samples.max().item(),
        "image_mean": image_samples.mean().item(),
    }


def append_manifest_line(lines: list[str], name: str, stats: dict[str, float]) -> None:
    lines.append(
        f"{name}: "
        f"model(min={stats['model_min']:.3f}, max={stats['model_max']:.3f}, mean={stats['model_mean']:.3f}) "
        f"image(min={stats['image_min']:.3f}, max={stats['image_max']:.3f}, mean={stats['image_mean']:.3f})"
    )


def seed_case(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_diffusion(training_cfg_path: str, device: torch.device) -> DDPM:
    training_cfg = OmegaConf.load(training_cfg_path)
    return DDPM(
        num_timesteps=int(training_cfg.num_timesteps),
        beta_start=float(training_cfg.get("beta_start", 1.0e-4)),
        beta_end=float(training_cfg.get("beta_end", 2.0e-2)),
        device=str(device),
    )


def sample_case(
    model,
    diffusion: DDPM,
    num_samples: int,
    device: torch.device,
    num_classes: int,
    seed: int,
    class_label: int | None,
    guidance_scale: float,
    clip_x0: bool,
) -> torch.Tensor:
    seed_case(seed)
    return sample_images(
        model,
        diffusion,
        num_samples,
        device,
        num_classes,
        class_label,
        guidance_scale,
        clip_x0=clip_x0,
        trace_every=0,
    )


def evaluate_checkpoint(
    checkpoint_path: str,
    model_config_path: str,
    training_config_path: str,
    device: torch.device,
    output_dir: str,
    num_samples: int,
    seed: int,
    class_labels: list[int],
    guidance_scales: list[float],
    clip_x0: bool,
) -> tuple[Path, dict[str, Path], list[str]]:
    model_cfg = OmegaConf.load(model_config_path)
    num_classes = int(model_cfg.num_classes)

    model = load_model(model_config_path, checkpoint_path, device)
    diffusion = build_diffusion(training_config_path, device)

    checkpoint_stem = Path(checkpoint_path).stem
    checkpoint_output_dir = Path(output_dir) / checkpoint_stem
    checkpoint_output_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = [
        f"checkpoint: {Path(checkpoint_path)}",
        f"device: {device}",
        f"seed: {seed}",
        f"num_samples: {num_samples}",
        f"class_labels: {class_labels}",
        f"guidance_scales: {guidance_scales}",
        f"clip_x0: {clip_x0}",
        "",
    ]
    case_paths: dict[str, Path] = {}

    print("Evaluating unconditional samples...")
    uncond_samples = sample_case(
        model,
        diffusion,
        num_samples,
        device,
        num_classes,
        seed,
        None,
        1.0,
        clip_x0,
    )
    uncond_path = checkpoint_output_dir / "unconditional.png"
    stats = save_case(uncond_samples, uncond_path)
    append_manifest_line(manifest_lines, "unconditional", stats)
    case_paths["unconditional"] = uncond_path
    print(f"Saved {uncond_path}")

    for class_label in class_labels:
        for guidance_scale in guidance_scales:
            print(f"Evaluating class={class_label}, guidance_scale={guidance_scale}...")
            samples = sample_case(
                model,
                diffusion,
                num_samples,
                device,
                num_classes,
                seed,
                class_label,
                guidance_scale,
                clip_x0,
            )
            scale_tag = format_scale(guidance_scale)
            case_name = f"class_{class_label}_cfg_{guidance_scale}"
            output_path = checkpoint_output_dir / f"class_{class_label}_cfg_{scale_tag}.png"
            stats = save_case(samples, output_path)
            append_manifest_line(manifest_lines, case_name, stats)
            case_paths[case_name] = output_path
            print(f"Saved {output_path}")

    manifest_path = checkpoint_output_dir / "manifest.txt"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="ascii")
    print(f"Saved manifest to: {manifest_path}")
    return checkpoint_output_dir, case_paths, manifest_lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed evaluation sample grids for a checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument(
        "--model-config",
        default="conf/model/dit_tiny.yaml",
        help="Path to model config YAML",
    )
    parser.add_argument(
        "--training-config",
        default="conf/training/cifar10_default.yaml",
        help="Path to training config YAML",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-labels",
        default="0,3,8",
        help="Comma-separated class labels for conditional evaluation",
    )
    parser.add_argument(
        "--guidance-scales",
        default="1.0,3.0",
        help="Comma-separated CFG scales to evaluate for each class label",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/eval",
        help="Directory to write evaluation grids and manifest",
    )
    parser.add_argument("--clip-x0", action="store_true", help="Enable x0 clipping during sampling")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    class_labels = parse_csv_ints(args.class_labels)
    guidance_scales = parse_csv_floats(args.guidance_scales)
    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        model_config_path=args.model_config,
        training_config_path=args.training_config,
        device=device,
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        seed=args.seed,
        class_labels=class_labels,
        guidance_scales=guidance_scales,
        clip_x0=args.clip_x0,
    )


if __name__ == "__main__":
    main()
