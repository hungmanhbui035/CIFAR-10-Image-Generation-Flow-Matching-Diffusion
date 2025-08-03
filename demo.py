import argparse
import os

from matplotlib import pyplot as plt
import torch
from torchvision import transforms

from utils import CIFAR10Sampler, MNISTSampler, GaussianConditionalProbabilityPath, LinearAlpha, LinearBeta, CFGVectorFieldODE, EulerSimulator
from models import UNet

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def parse_args():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar10' ,choices=['mnist', 'cifar10'])
    # model
    parser.add_argument('--channels', nargs='+', type=int, default=[64, 128, 256, 512])
    parser.add_argument('--num-residual-layers', type=int, default=2)
    parser.add_argument('--t-embed-dim', type=int, default=128)
    parser.add_argument('--y-embed-dim', type=int, default=128)
    parser.add_argument('--model-path', type=str, required=True)
    # simulate
    parser.add_argument('--y', type=str, required=True, choices=classes)
    parser.add_argument('--w', type=float, default=3.0)
    parser.add_argument('--num-timesteps', type=int, default=1000)

    return parser.parse_args()

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    unet = UNet(
        in_channel = in_channel,
        num_classes = 10,
        channels = args.channels,
        num_residual_layers = args.num_residual_layers,
        t_embed_dim = args.t_embed_dim,
        y_embed_dim = args.y_embed_dim
    )
    unet.load_state_dict(torch.load(args.model_path, map_location=device))

    ode = CFGVectorFieldODE(unet, guidance_scale=args.w)
    simulator = EulerSimulator(ode)

    y = classes.index(args.y)
    y = torch.tensor([y]).to(device)
    x0, _ = path.p_simple.sample(1)

    ts = torch.linspace(0,1,args.num_timesteps).view(1, -1, 1, 1, 1).expand(1, -1, 1, 1, 1).to(device)
    x1 = simulator.simulate(x0, ts, y=y).view(in_channel, 32, 32)

    mean = torch.tensor(0.5)
    std = torch.tensor(0.5)
    denormalize = transforms.Normalize(mean = -mean/std, std = 1/std)

    x1 = denormalize(x1)
    plt.axis("off")
    if args.dataset == 'cifar10':
        cmap = None
    elif args.dataset == 'mnist':
        cmap = 'gray'
    plt.imshow(x1.permute(1,2,0).cpu(), cmap=cmap)

    if not os.path.exists('./generated'):
        os.makedirs('./generated')
    plt.savefig(f'./generated/{args.y}.png')

if __name__ == '__main__':
    main()