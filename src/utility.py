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
        epoch_part = f"_epoch{epoch}" if epoch is not None else ""
        checkpoint_path = repo_root / "checkpoints" / f"model_checkpoint{epoch_part}_{timestamp}.pt"
    path = Path(checkpoint_path)
    if path.suffix:
        epoch_part = f"_epoch{epoch}" if epoch is not None else ""
        path = path.with_name(f"{path.stem}{epoch_part}_{timestamp}{path.suffix}")
    else:
        epoch_part = f"_epoch{epoch}" if epoch is not None else ""
        path = path.with_name(f"{path.name}{epoch_part}_{timestamp}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path
