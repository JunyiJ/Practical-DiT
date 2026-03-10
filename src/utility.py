from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


def save_model_checkpoint(model: torch.nn.Module, checkpoint_path: Optional[str] = None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not checkpoint_path:
        repo_root = Path(__file__).resolve().parents[1]
        checkpoint_path = repo_root / "checkpoints" / f"model_checkpoint_{timestamp}.pt"
    path = Path(checkpoint_path)
    if path.suffix:
        path = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    else:
        path = path.with_name(f"{path.name}_{timestamp}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path
