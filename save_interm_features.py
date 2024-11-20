import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
import argparse

from model import HybridNet
from dataset import *
from utils import *

def get_interm_features(model, num_layers, device, loader):
    interm_features = {}
    for i in range(num_layers - 1):
        interm_features[i] = []

    model.eval()

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        with torch.no_grad():
            _, layers_out = model(inputs)

        for i in range(num_layers - 1):
            interm_features[i].append(layers_out[i])

    return interm_features


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_trials', type=int, default=10)

    # Model selection
    parser.add_argument('--model_path', type=str, default=None, help='Path to saved model weights')
    parser.add_argument('--hidden_dim', type=int, default=32)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--nonlinear_depth', type=int, default=1)
    parser.add_argument('--activation', type=str, default='relu', help='Activation function',
                        choices=['relu', 'sigmoid', 'quadratic', 'gelu', 'elu', 'leaky-relu', 'step'])

    # Initialization method
    parser.add_argument('--init', type=str, default='gaussian', choices=['gaussian', 'uniform', 'default'])
    parser.add_argument('--init_var', type=float, default=1e-2)

    # Random seed
    parser.add_argument('--seed', type=int, default=0)

    # Data
    parser.add_argument('--data_type', type=str, default='uos', choices=['uos', 'cifar10', 'cifar10_mcr2'])
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--samples_per_class', type=int, default=100)
    parser.add_argument('--data_dim', type=int, default=16)
    parser.add_argument('--rank', type=int, default=4, help='Rank of subspaces. Only used when --data_type==uos')
    parser.add_argument('--angle', type=float, default=0, help='Principal angle (in degrees) between pairs of subspaces. Only used when data_type==uos')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)


    # Dataset settings
    data_type = args.data_type
    N_k = args.samples_per_class
    K = args.num_classes
    N = N_k * K
    d = args.data_dim
    batch_size = args.batch_size

    if data_type == 'uos':
        r = args.rank
        angle = args.angle
        train_set, train_loader = get_uos_dataset(N_k, K, d, r, batch_size=batch_size, angle=angle)
        val_set, val_loader = get_uos_dataset(N_k, K, d, r, batch_size=batch_size, angle=angle)
    elif data_type == 'cifar10_mcr2':
        train_set, train_loader = get_cifar10_mcr2_dataset(N_k, K, root='./datasets/cifar10_mcr2/', features_fname='train_features.npy')
        val_set, val_loader = get_cifar10_mcr2_dataset(N_k, K, root='./datasets/cifar10_mcr2/', features_fname='val_features.npy')
    elif data_type == 'cifar10':
        train_set, train_loader = get_cifar10_dataset(N_k, K, '/scratch/qingqu_root/qingqu1/alecx/cifar10/', batch_size, train=True)
        val_set, val_loader = get_cifar10_dataset(N_k, K, '/scratch/qingqu_root/qingqu1/alecx/cifar10/', batch_size, train=False)


    # Model settings
    D = args.hidden_dim
    L = args.depth
    nonlinear_L = args.nonlinear_depth
    activation_str = args.activation
    activation = activations[activation_str]
    init_ = args.init
    var = args.init_var


    # Load best saved state from training
    model = HybridNet(in_dim=d, hidden_dim=D, num_classes=K,
                      num_layers=L, num_nonlinear_layers=nonlinear_L,
                      activation=activation, init_method=init_, var=var)

    model_path = args.model_path
    if model_path is not None:
        load_path = os.path.join(model_path, 'best.pth')
        ckpt = torch.load(load_path, map_location=device)
        print("Loading saved model.")
        model.load_state_dict(ckpt['state_dict'])
    else:
        print("Using untrained model.")

    model = model.to(device)


    # Extract and save intermediate layer features
    interm_features = get_interm_features(model, L, device, train_loader)
    for l in range(L - 1):
        interm_features[l] = torch.cat(interm_features[l], 0).detach().cpu()

    save_state = {'interm_features': interm_features}
    save_path = os.path.join(model_path, 'interm_features.pth')
    torch.save(save_state, save_path)

if __name__ == "__main__":
    main()
