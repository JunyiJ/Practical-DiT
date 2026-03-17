import argparse
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from eval_samples import evaluate_checkpoint, parse_csv_floats, parse_csv_ints
from sample import resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints and build comparison sheets")
    parser.add_argument("--checkpoint-dir", default="checkpoints", help="Directory containing checkpoint files")
    parser.add_argument("--checkpoint-glob", default="*.pt", help="Glob pattern within checkpoint-dir")
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
    parser.add_argument("--class-labels", default="0,3,8")
    parser.add_argument("--guidance-scales", default="1.0,3.0")
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--clip-x0", action="store_true")
    return parser.parse_args()


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"_epoch(\d+)", path.stem)
    epoch = int(match.group(1)) if match else math.inf
    return (epoch, path.stem)


def load_font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def create_case_sheet(
    case_name: str,
    entries: list[tuple[str, Path]],
    output_path: Path,
    title: str,
    columns: int = 4,
) -> None:
    font = load_font()
    opened = [(label, Image.open(path).convert("RGB")) for label, path in entries]
    try:
        tile_width = max(image.width for _, image in opened)
        tile_height = max(image.height for _, image in opened)
        rows = math.ceil(len(opened) / columns)
        text_height = 16
        header_height = 24
        padding = 8
        width = columns * tile_width + (columns + 1) * padding
        height = header_height + rows * (tile_height + text_height + padding) + padding
        canvas = Image.new("RGB", (width, height), color="white")
        draw = ImageDraw.Draw(canvas)
        draw.text((padding, 6), title, fill="black", font=font)

        for index, (label, image) in enumerate(opened):
            row = index // columns
            col = index % columns
            x = padding + col * (tile_width + padding)
            y = header_height + row * (tile_height + text_height + padding)
            canvas.paste(image, (x, y))
            draw.text((x, y + tile_height + 2), label, fill="black", font=font)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(output_path)
    finally:
        for _, image in opened:
            image.close()


def create_overview_sheet(
    checkpoint_names: list[str],
    case_names: list[str],
    case_images: dict[str, list[Path]],
    output_path: Path,
) -> None:
    font = load_font()
    first_case = case_names[0]
    first_image = Image.open(case_images[first_case][0]).convert("RGB")
    try:
        tile_width, tile_height = first_image.size
    finally:
        first_image.close()

    label_col_width = 120
    header_height = 28
    row_label_height = 16
    padding = 8
    total_width = label_col_width + len(checkpoint_names) * (tile_width + padding) + padding
    total_height = header_height + len(case_names) * (tile_height + row_label_height + padding) + padding

    canvas = Image.new("RGB", (total_width, total_height), color="white")
    draw = ImageDraw.Draw(canvas)
    draw.text((padding, 6), "Checkpoint comparison overview", fill="black", font=font)

    for col, checkpoint_name in enumerate(checkpoint_names):
        x = label_col_width + padding + col * (tile_width + padding)
        draw.text((x, 6), checkpoint_name, fill="black", font=font)

    for row, case_name in enumerate(case_names):
        y = header_height + row * (tile_height + row_label_height + padding)
        draw.text((padding, y + tile_height // 2), case_name, fill="black", font=font)
        for col, image_path in enumerate(case_images[case_name]):
            x = label_col_width + padding + col * (tile_width + padding)
            image = Image.open(image_path).convert("RGB")
            try:
                canvas.paste(image, (x, y))
            finally:
                image.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoints = sorted(checkpoint_dir.glob(args.checkpoint_glob), key=checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir} matching {args.checkpoint_glob}")

    class_labels = parse_csv_ints(args.class_labels)
    guidance_scales = parse_csv_floats(args.guidance_scales)
    checkpoint_results: list[tuple[str, dict[str, Path]]] = []

    for checkpoint_path in checkpoints:
        print(f"Evaluating checkpoint: {checkpoint_path}")
        _, case_paths, _ = evaluate_checkpoint(
            checkpoint_path=str(checkpoint_path),
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
        checkpoint_results.append((checkpoint_path.stem, case_paths))

    case_names = list(checkpoint_results[0][1].keys())
    summary_dir = Path(args.output_dir) / "_summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for case_name in case_names:
        entries = [(checkpoint_name, case_paths[case_name]) for checkpoint_name, case_paths in checkpoint_results]
        create_case_sheet(
            case_name=case_name,
            entries=entries,
            output_path=summary_dir / f"{case_name}.png",
            title=case_name,
        )

    overview_case_images = {
        case_name: [case_paths[case_name] for _, case_paths in checkpoint_results]
        for case_name in case_names
    }
    create_overview_sheet(
        checkpoint_names=[checkpoint_name for checkpoint_name, _ in checkpoint_results],
        case_names=case_names,
        case_images=overview_case_images,
        output_path=summary_dir / "overview.png",
    )
    print(f"Saved summary sheets to: {summary_dir}")


if __name__ == "__main__":
    main()
