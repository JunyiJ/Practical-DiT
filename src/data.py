from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_cifar10_dataloader(batch_size=8, num_workers=0):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
