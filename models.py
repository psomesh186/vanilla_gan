import torch.nn as nn


class Generator(nn.Module):
    """Generator model."""

    def __init__(self, noise_size):
        """
            Args:
                noise_size (int): Size of input noise.
        """
        super(Generator, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=noise_size, out_features=256),
            nn.LeakyReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.LeakyReLU(),
            nn.Linear(in_features=1024, out_features=28*28),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.reshape(-1, 1, 28, 28)
        return x
    
class Discriminator(nn.Module):
    """Discriminator model."""

    def __init__(self):
        super(Discriminator, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28*28, out_features=1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=1024, out_features=512),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=512, out_features=256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features=256, out_features=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.layers(x)
        return x