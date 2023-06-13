import torch
import matplotlib.pyplot as plt

from tqdm import tqdm
from torchvision.utils import make_grid
from utils import get_dataloader, make_ones, make_zeros, noise
from models import Generator, Discriminator


def train_generator_step(
        optimizer,
        discriminator,
        fake_data,
        criterion,
        device
    ):
    """Perform one step of generator training.
    
    Args:
        optimizer: Generator optimzer
        discriminator: Discriminator model
        fake_data: Fake images given by generator
        criterion: Loss function
        device: Device to run model on

    Returns:
        Generator loss
    """
    optimizer.zero_grad()
    prediction = discriminator(fake_data)
    loss = criterion(prediction, make_ones(prediction.shape[0]).to(device))
    loss.backward()
    optimizer.step()

    return loss.item()

def train_discriminator_step(
        optimizer,
        discriminator,
        real_data,
        fake_data,
        criterion,
        device
    ):
    """Perform one step of discriminator training.
    
    Args:
        optimizer: Discriminator optimizer
        discriminator: Discriminator model
        real_data: Real images from the dataset
        fake_data: Generated images from generator model
        device: Device to run model on

    Returns:
        Discriminator loss
    """
    # Compute loss for real images
    optimizer.zero_grad()
    prediction = discriminator(real_data)
    loss_real = criterion(prediction, make_ones(prediction.shape[0]).to(device))
    loss_real.backward()

    # Compute loss for fake images
    prediction = discriminator(fake_data)
    loss_fake = criterion(
        prediction,
        make_zeros(prediction.shape[0]).to(device)
    )
    loss_fake.backward()
    optimizer.step()

    return loss_real.item() + loss_fake.item()

def trainGAN(device, batch_size, patience=10, numEpochs=100, noise_size=128):
    """Train the GAN according to given hyperparameters.
    
    Args:
        device (str): Device to train the model on.
        batch_size (int): Mini-batch size used during training.
        patience (int): Number of epochs to wait without improvement in loss. 
        (Default: 10)
        numEpochs (int): Epochs to perform training for. (Default: 100)
        noise_size (int): Size of input noise for generator. (Default: 128)
    """
    # Load images
    trainloader = get_dataloader(batch_size=batch_size, num_workers=0)

    # Graph setup
    loss_fig, loss_ax = plt.subplots()
    loss_ax.set_title("Loss plot")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    plt.ion()
    plt.tight_layout()
    plt.show()

    test_fig, test_ax = plt.subplots(figsize=(9, 9))
    test_ax.set_xticks([])
    test_ax.set_yticks([])
    plt.show()

    # Train setup
    device = torch.device(device)
    generator = Generator(noise_size=noise_size)
    generator.to(device)
    discriminator = Discriminator()
    discriminator.to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=2e-4)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=2e-4)
    criterion = torch.nn.BCELoss()
    best_loss = torch.inf
    patience_counter = 0
    test_noise = noise(64).to(device)
    k = 1

    # Training
    for epoch in range(numEpochs):
        # Train section
        generator.train()
        discriminator.train()
        g_loss = 0
        d_loss = 0
        for i, (images, _) in enumerate(
            tqdm(trainloader, desc=f"Epoch {epoch + 1}")
            ):
            images = images.to(device)
            for _ in range(k):
                fake_data = generator(noise(images.shape[0]).to(device))
                d_loss += train_discriminator_step(
                    optimizer=d_optimizer,
                    discriminator=discriminator,
                    real_data=images,
                    fake_data=fake_data,
                    criterion=criterion,
                    device=device
                )
            fake_data = generator(noise(images.shape[0]).to(device))
            g_loss += train_generator_step(
                optimizer=g_optimizer,
                discriminator=discriminator,
                fake_data=fake_data,
                criterion=criterion,
                device=device
            )
        g_loss /= (i + 1)
        d_loss /= (i + 1)

        # Test generator performance
        generator.eval()
        with torch.no_grad():
            fake_img = generator(test_noise).cpu()
            fake_img = make_grid(fake_img)
        
        # Report metrics
        print(f"Generator loss: {g_loss}, Discriminator loss: {d_loss}")

        # TODO: Plot graphs and images, early stopping, model saving
