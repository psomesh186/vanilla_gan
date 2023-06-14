import torch
import glob

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable
from PIL import Image


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

def noise(n, n_features=128):
    return Variable(torch.randn(n, n_features))

def make_ones(size):
    data = Variable(torch.ones(size, 1))
    return data

def make_zeros(size):
    data = Variable(torch.zeros(size, 1))
    return data

def make_training_gif():
    imgs = glob.glob("results/images/*.png")
    imgs.sort()
    frames = [Image.open(img) for img in imgs]
    frames[0].save(
        "results/training.gif",
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        loop=0,
        duration=500
    )