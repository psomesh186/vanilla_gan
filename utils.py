from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(batch_size=256, num_workers=4):
    """Create MNIST handwritten digits dataloader
    
        Args:
            batch_size (int): Size of mini-batches of dataloader. (Default: 256)
            num_workers (int): Number of parallel workers. (Default: 4)
        
        Returns:
            Dataloader object
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5,], std=[0.5,])
    ])
    dataset = datasets.MNIST(
        root='data/',
        train=True,
        transform=transform,
        download=True
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return dataloader