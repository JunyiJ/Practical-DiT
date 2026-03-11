from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


def save_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Optional[str] = None,
    epoch: Optional[int] = None,
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
    torch.save(model.state_dict(), path)
    return path
