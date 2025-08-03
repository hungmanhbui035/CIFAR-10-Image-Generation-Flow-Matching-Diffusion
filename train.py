import os
import argparse
import wandb

import torch
import torch.nn as nn

from utils import GaussianConditionalProbabilityPath, MNISTSampler, CIFAR10Sampler, LinearAlpha, LinearBeta, CFGTrainer
from models import UNet

def parse_args():
    parser = argparse.ArgumentParser()
    
    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10' ,choices=['mnist', 'cifar10'])
    # model
    parser.add_argument('--channels', nargs='+', type=int, default=[64, 128, 256, 512])
    parser.add_argument('--num-residual-layers', type=int, default=2)
    parser.add_argument('--t-embed-dim', type=int, default=128)
    parser.add_argument('--y-embed-dim', type=int, default=128)
    # trainer
    parser.add_argument('--eta', type=float, default=0.1)
    # train
    parser.add_argument('--num-epochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=1024)

    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize probability path
    if args.dataset == 'cifar10':
        p_data = CIFAR10Sampler()
        in_channel = 3
    elif args.dataset == 'mnist':
        p_data = MNISTSampler()
        in_channel = 1
    
    path = GaussianConditionalProbabilityPath(
        p_data = p_data,
        p_simple_shape = [in_channel, 32, 32],
        alpha = LinearAlpha(),
        beta = LinearBeta()
    ).to(device)

    # Initialize model
    unet = UNet(
        in_channel = in_channel,
        num_classes = 10,
        channels = args.channels,
        num_residual_layers = args.num_residual_layers,
        t_embed_dim = args.t_embed_dim,
        y_embed_dim = args.y_embed_dim
    )
    unet = nn.DataParallel(unet)
    model_path = f'./models/{args.dataset}_unet.pth'
    
    # Initialize trainer
    trainer = CFGTrainer(path = path, model = unet, eta=args.eta)

    # wandb
    wandb.login()
    wandb.init(project='CIFAR-10-Image-Generation-Flow-Matching-Diffusion')

    # Train!
    trainer.train(num_epochs = args.num_epochs, device=device, lr=args.lr, batch_size=args.batch_size)
    wandb.finish()

    # Save
    if not os.path.exists('./models'):
        os.makedirs('./models', exist_ok=True)
    torch.save(trainer.model.module.state_dict(), model_path)

if __name__ == '__main__':
    main()