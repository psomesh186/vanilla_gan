import argparse
import os
import torch

from train import trainGAN
from utils import make_training_gif


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Running image classifier model"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=200,
        help="Number of epochs to train model for")
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=256,
        help="Size of the mini-batches used in training"
    )
    parser.add_argument(
        "-p",
        "--patience",
        type=int,
        default=100,
        help="Patience for early stopping of training"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device to use for running the model"
    )
    args = parser.parse_args()
    
    # Create relevant folders
    if not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")

    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists("results/images"):
        os.mkdir("results/images")
    
    # Get GPU device if available
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    trainGAN(
        device=device,
        batch_size=args.batch_size,
        patience=args.patience,
        numEpochs=args.epochs,
    )
    make_training_gif()