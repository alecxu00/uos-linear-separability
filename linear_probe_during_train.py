import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
import argparse

from model import HybridNet, LinearClassifier
from dataset import *
from utils import *


# Train function
def train_epoch(model, device, loader, optimizer, num_samples, criterion, epoch):
    model.train()

    running_loss = 0.0
    num_correct = 0
    for _, (data, targets) in enumerate(loader):
        # Get model predictions
        targets = targets.type(torch.LongTensor)
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        out, _ = model(data)
        loss = criterion(out, targets)

        # Backwards pass
        loss.backward()
        optimizer.step()

        # Compute train performance
        running_loss += loss.item() * data.size(0)
        pred = torch.argmax(out, dim=1)
        num_correct += torch.sum(pred==targets).item()

    epoch_loss = running_loss / len(loader.sampler)
    epoch_accuracy = 100. * num_correct / num_samples

    return epoch_loss, epoch_accuracy



# Cross validation function
def eval_epoch(model, device, loader, num_samples, criterion, epoch):
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
    
    epoch_loss = running_loss / len(loader.sampler)
    epoch_accuracy = 100. * num_correct / num_samples
    
    return epoch_loss, epoch_accuracy
    


# Linear probe train epoch
def train_epoch_linear(model, linear_model, layer_idx, device, loader, optimizer, num_samples, criterion):
    model.eval()
    linear_model.train()

    running_loss = 0.0
    num_correct = 0

    for _, (data, targets) in enumerate(loader):
        # Hybrid model intermediate features
        data, targets = data.to(device), targets.to(device)
        _, layer_outs = model(data)
        interm = layer_outs[layer_idx]

        # Linear probe intermediate features
        out = linear_model(interm)
        loss = criterion(out, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save loss and accuracy
        running_loss += loss.item() * data.size(0)
        pred = torch.argmax(out, dim=1)
        num_correct += sum(pred==targets).item()


    linear_train_loss = running_loss / len(loader.sampler)
    linear_train_acc = 100. * num_correct / num_samples

    return linear_train_loss, linear_train_acc


# Linear probe val epoch
def eval_epoch_linear(model, linear_model, layer_idx, device, loader, num_samples, criterion):
    model.eval()
    linear_model.eval()

    running_loss = 0.0
    num_correct = 0

    for _, (data, targets) in enumerate(loader):
        # Get model intermediate features
        data, targets = data.to(device), targets.to(device)
        _, layer_out = model(data)
        interm = layer_out[layer_idx]

        # Evaluate linear probe
        out = linear_model(interm)
        loss = criterion(out, targets)

        running_loss += loss.item() * data.size(0)
        pred = torch.argmax(out, dim=1)
        num_correct += sum(pred==targets).item()

    linear_eval_loss = running_loss / len(loader.sampler)
    linear_eval_acc = 100. * num_correct / num_samples

    return linear_eval_loss, linear_eval_acc


# Linear probe
def linear_probe(model, epoch, layers_to_probe, device, linear_lr, linear_epochs, linear_patience, train_loader, val_loader, checkpoint_dir, D, K, N):
    for layer_idx in range(layers_to_probe):
        linear_model = LinearClassifier(D, K)
        linear_model = linear_model.to(device)

        linear_criterion = nn.CrossEntropyLoss()
        linear_optimizer = optim.SGD(linear_model.parameters(), lr=linear_lr, momentum=0.9, weight_decay=1e-4)
        linear_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(linear_optimizer, T_0=linear_epochs, eta_min=linear_lr/1000)
        linear_early_stopping = EarlyStopping(patience=linear_patience)

        min_linear_val_acc = float('Inf')
        for linear_epoch in range(1, linear_epochs+1):
            # Linear probe training
            linear_train_loss, linear_train_acc = train_epoch_linear(model, linear_model, layer_idx, device, train_loader, linear_optimizer, N, linear_criterion)
            linear_scheduler.step()

            # Validation
            linear_val_loss, linear_val_acc = eval_epoch_linear(model, linear_model, layer_idx, device, val_loader, N, linear_criterion)

            # Early stopping criteria
            last_linear_state = {
                'train_accuracy': linear_train_acc,
                'val_accuracy': linear_val_acc
            }

            if linear_val_acc < min_linear_val_acc:
                min_linear_val_acc = linear_val_acc
                best_linear_state = last_linear_state

            if linear_early_stopping(linear_val_acc):
                print("Done linear probing in {} epochs".format(linear_epoch))
                break

        # Save linear probe results
        save_path = os.path.join(checkpoint_dir, 'layer_' + str(layer_idx) + '_best_epoch_' + str(epoch) + '_probe.pth')
        torch.save(best_linear_state, save_path)

        save_path = os.path.join(checkpoint_dir, 'layer_' + str(layer_idx) + '_last_epoch_' + str(epoch) + '_probe.pth')
        torch.save(last_linear_state, save_path)



# Parse command line arguments
def parse_train_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_trials', type=int, default=10)

    # Model selection
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
    parser.add_argument('--angle', type=float, default=None, help='Principal angle (in degrees) between pairs of subspaces. Only used when --data_type==uos and K must be even.')
    parser.add_argument('--noise_std', type=float, default=0.0, help='Stdev of noise in data. Only used when --data_type==uos')

    # Training settings
    parser.add_argument('--epochs', type=int, default=1000, help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')


    # Linear probing settings
    parser.add_argument('--layers_to_probe', type=int, default=1, help='Number of layers to linear probe.')
    parser.add_argument('--linear_epochs', type=int, default=1000, help='Max linear probing epochs')
    parser.add_argument('--linear_batch_size', type=int, default=128, help='Linear probing batch size')
    parser.add_argument('--linear_lr', type=float, default=1e-2, help='Initial linear probing learning rate')
    parser.add_argument('--linear_patience', type=int, default=100, help='Early stopping patience during linear probing')

    # Save directory
    parser.add_argument('--save_dir', type=str, default='./save/linear_probe_during_train')

    args = parser.parse_args()

    return args


# Main function
def main():
    args = parse_train_args()
    print(args)

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
    print("Number of samples per trial = ", N, "\n")
    d = args.data_dim
    batch_size = args.batch_size
    noise_std = args.noise_std

    # Model settings
    D = args.hidden_dim
    L = args.depth
    nonlinear_L = args.nonlinear_depth
    activation_str = args.activation
    activation = activations[activation_str]
    init_ = args.init
    init_var = args.init_var

    # Train num_trials different models
    num_trials = args.num_trials
    epochs = args.epochs
    assert epochs >= 0
    lr = args.lr
    patience = args.patience

    layers_to_probe = args.layers_to_probe
    linear_epochs = args.linear_epochs
    linear_lr = args.linear_lr
    linear_batch_size = args.linear_batch_size
    linear_patience = args.linear_patience


    trial_train_accs = []
    trial_val_accs = []
    for i in range(num_trials):
        print("TRIAL " + str(i))

        # Get/create dataset
        if data_type == 'uos':
            r = args.rank
            angle = args.angle
            train_set, train_loader = get_uos_dataset(N_k, K, d, r, batch_size=batch_size, angle=angle, noise_std=noise_std)
            val_set, val_loader = get_uos_dataset(N_k, K, d, r, batch_size=batch_size, angle=angle, noise_std=noise_std)
        elif data_type == 'cifar10_mcr2':
            train_set, train_loader = get_cifar10_mcr2_dataset(N_k, K, root='./datasets/cifar10_mcr2/', features_fname='train_features.npy', labels_fname='train_labels.npy', batch_size=batch_size)
            val_set, val_loader = get_cifar10_mcr2_dataset(N_k, K, root='./datasets/cifar10_mcr2/', features_fname='val_features.npy', labels_fname='val_labels.npy', batch_size=batch_size)
        elif data_type == 'cifar10':
            train_set, train_loader = get_cifar10_dataset(N_k, K, '/scratch/qingqu_root/qingqu1/alecx/cifar10/', batch_size, train=True)
            val_set, val_loader = get_cifar10_dataset(N_k, K, '/scratch/qingqu_root/qingqu1/alecx/cifar10/', batch_size, train=False)

        # Set up save path for trial
        save_dir = args.save_dir
        if data_type == 'uos':
            save_subdir = f"width_{D}_depth_{L}_nonlinear_depth_{nonlinear_L}_{K}_classes_rank_{r}_{activation_str}_activation_seed_{seed}"
        else:
            save_subdir = f"width_{D}_depth_{L}_nonlinear_depth_{nonlinear_L}_{K}_classes_{activation_str}_activation_seed_{seed}"

        checkpoint_dir = os.path.join(save_dir, save_subdir, 'trial_' + str(i))
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Initialize model
        model = HybridNet(in_dim=d, hidden_dim=D, num_classes=K,
                         num_layers=L, num_nonlinear_layers=nonlinear_L,
                         activation=activation, init_method=init_, var=init_var)
        model = model.to(device)

        # Set up loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(),
                              lr=lr,
                              momentum=0.9,
                              weight_decay=1e-4)

        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//3, 2*epochs//3], gamma=0.1)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, eta_min=lr/100)
        early_stopping = EarlyStopping(patience=patience)

        # Linear probe model initialization
        linear_probe(model, 0, layers_to_probe, device, linear_lr, linear_epochs, linear_patience, train_loader, val_loader, checkpoint_dir, D, K, N)

        # Training
        min_val_loss = float('Inf')
        for epoch in range(1, epochs+1):
            # Training
            train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, N, criterion, epoch)
            scheduler.step()

            # Cross-validation
            val_loss, val_acc = eval_epoch(model, device, val_loader, N, criterion, epoch)

            # Linear probe early-layer features
            linear_probe(model, epoch, layers_to_probe, device, linear_lr, linear_epochs, linear_patience, train_loader, val_loader, checkpoint_dir, D, K, N)

            # Print training progress
            if epoch % 25 == 0:
                print("Finish Epoch {}".format(epoch))
                print("Train loss: {}, train accuracy: {}".format(train_loss, train_acc))
                print("Val loss: {}, val accuracy: {}".format(val_loss, val_acc))

            if val_loss < min_val_loss: # Best training state
                min_val_loss = val_loss

            # Stopping criteria:
            if early_stopping(val_loss): #train_loss < 1e-10 and early_stopping(val_loss):
                print("Done training in {} epochs".format(epoch))
                break

if __name__ == "__main__":
    main()
