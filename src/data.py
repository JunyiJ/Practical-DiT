from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloader(batch_size=8, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # make sure data in range [-1, 1] instead of (0, 1) given diffusion model is using gaussian
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
