from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


def save_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Optional[str] = None,
    epoch: Optional[int] = None,
    ema_model: Optional[torch.nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    config: Optional[dict] = None,
) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not checkpoint_path:
        repo_root = Path(__file__).resolve().parents[1]
        checkpoint_path = repo_root / "checkpoints" / "model_checkpoint.pt"
    path = Path(checkpoint_path)
    epoch_part = f"_epoch{epoch}" if epoch is not None else ""
    suffix_part = f"{epoch_part}_{timestamp}"
    if path.suffix:
        path = path.with_name(f"{path.stem}{suffix_part}{path.suffix}")
    else:
        path = path.with_name(f"{path.name}{suffix_part}")
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model.state_dict(),
        "epoch": epoch,
    }
    if ema_model is not None:
        checkpoint["ema_model"] = ema_model.state_dict()
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if config is not None:
        checkpoint["config"] = config
    torch.save(checkpoint, path)
    return path

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    ema_params = dict(ema_model.named_parameters())
    model_params = dict(model.named_parameters())

    for name, p in model_params.items():
        ema_params[name].mul_(decay).add_(p, alpha=1 - decay)
