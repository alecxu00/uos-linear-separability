import torch
import torch.nn as nn

import numpy as np
import os
import argparse

from model import HybridNet
from dataset import *
from utils import *


def test(model, device, loader, num_samples, criterion):
    model.eval()

    running_loss = 0.0
    num_correct = 0

    for _, (data, targets) in enumerate(loader):
        targets = targets.type(torch.LongTensor)
        data, targets = data.to(device), targets.to(device)
        out, _ = model(data)

        running_loss += criterion(out, targets).item() * data.size(0)

        net_pred = torch.argmax(out, dim=1)
        num_correct += torch.sum(net_pred==targets).item()
    
    test_loss = running_loss / len(loader.sampler)
    test_accuracy = 100. * num_correct / num_samples

    print("Test loss: {}".format(test_loss))
    print("Test accuracy: {}".format(test_accuracy))
    
    return test_loss, test_accuracy


# Parse command line arguments
def parse_test_args():
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
    parser.add_argument('--angle', type=float, default=0, help='Principal angle (in degrees) between pairs of subspaces. Only used when --data_type==uos and K must be even.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    args = parser.parse_args()

    return args


# Main function
def main():
    args = parse_test_args()
    print(args)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create dataset
    data_type = args.data_type
    N_k = args.samples_per_class
    K = args.num_classes
    N = N_k * K
    d = args.data_dim
    batch_size = args.batch_size

    if data_type == 'uos':
        r = args.rank
        angle = args.angle
        test_set, test_loader = get_uos_dataset(N_k, K, d, r, batch_size=batch_size, angle=angle)
    elif data_type == 'cifar10_mcr2':
        test_set, test_loader = get_cifar10_mcr2_dataset(N_k, K, root='./datasets/cifar10_mcr2/', features_fname='test_features.npy', labels_fname='test_labels.npy', batch_size=batch_size)

    # Initialize model
    D = args.hidden_dim
    L = args.depth
    nonlinear_L = args.nonlinear_depth
    activation_str = args.activation
    activation = activations[activation_str]
    init_ = args.init
    var = args.init_var

    model = HybridNet(in_dim=d, hidden_dim=D, num_classes=K,
                     num_layers=L, num_nonlinear_layers=nonlinear_L,
                     activation=activation, init_method=init_, var=var)


    num_trials = args.num_trials
    trial_test_accs = []
    for i in range(num_trials):
        # Load best saved state from training
        model_path = args.model_path
        if model_path is not None:
            load_path = os.path.join(model_path, 'trial_' + str(i), 'best.pth')
            ckpt = torch.load(load_path)
            print("Loading saved model.")
            model.load_state_dict(ckpt['state_dict'])
        else:
            print("Using untrained model.")

        model = model.to(device)

        # Test model
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = test(model, device, test_loader, N, criterion)
        trial_test_accs.append(test_acc)
        test_state = {
            'state_dict': model.state_dict(),
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'test_set': test_set.samples,
            'test_labels': test_set.labels
        }

        print("Saving test results\n")
        save_path = os.path.join(model_path, 'trial_' + str(i), 'test.pth') # Save in same directory as saved model
        torch.save(test_state, save_path)


    print("Test accuracy mean and stdev = ", np.mean(trial_test_accs), np.std(trial_test_accs))

if __name__ == "__main__":
    main()
