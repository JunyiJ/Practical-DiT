import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model import DiT
from diffusion import DDPM
from data import get_cifar10_dataloader # You'll define this in data.py
from utility import save_model_checkpoint

def get_device(device_cfg):
    if device_cfg == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_cfg)

@hydra.main(version_base="1.3", config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    device = get_device(cfg.device)
    print(f"Using device: {device}")

    # 1. Initialize data
    dataloader = get_cifar10_dataloader(
        batch_size=cfg.training.batch_size,
        num_workers=cfg.training.num_workers
    )

    # 2. Initialize Model & Diffusion
    model = DiT(**cfg.model).to(device)
    diffusion = DDPM(
        num_timesteps=cfg.training.num_timesteps,
        beta_start=cfg.training.get("beta_start", 1.0e-4),
        beta_end=cfg.training.get("beta_end", 2.0e-2),
        device=str(device),
    )

    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr)

    # 4. Training Loop
    model.train()
    checkpoint_every = cfg.training.get("checkpoint_every", 0)
    last_checkpoint_epoch = None
    for epoch in range(cfg.training.epochs):
        epoch_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = diffusion.compute_loss(model, images, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{cfg.training.epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
        if checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = save_model_checkpoint(
                model,
                cfg.training.get("checkpoint_path"),
                epoch=epoch + 1,
            )
            print(f"Checkpoint saved to: {checkpoint_path}")
            last_checkpoint_epoch = epoch + 1
    if last_checkpoint_epoch != cfg.training.epochs:
        checkpoint_path = save_model_checkpoint(
            model,
            cfg.training.get("checkpoint_path"),
            epoch=cfg.training.epochs,
        )
        print(f"Checkpoint saved to: {checkpoint_path}")

if __name__ == "__main__":
    main()
