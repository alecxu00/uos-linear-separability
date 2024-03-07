import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
import argparse

from model import HybridNet
from dataset import *
from utils import *


# Train function
def train_epoch(model, device, loader, optimizer, num_samples, criterion, epoch):
    model.train()

    running_loss = 0.0
    num_correct = 0
    for _, (data, targets) in enumerate(loader):
        # Get model predictions
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
        data, targets = data.to(device), targets.to(device)
        out, _ = model(data)

        running_loss += criterion(out, targets).item() * data.size(0)

        net_pred = torch.argmax(out, dim=1)
        num_correct += torch.sum(net_pred==targets).item()
    
    epoch_loss = running_loss / len(loader.sampler)
    epoch_accuracy = 100. * num_correct / num_samples
    
    return epoch_loss, epoch_accuracy
    


# Parse command line arguments
def parse_train_args():
    parser = argparse.ArgumentParser()

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
    parser.add_argument('--data_type', type=str, default='uos', choices=['uos', 'mog'])
    parser.add_argument('--num_classes', type=int, default=5)
    parser.add_argument('--samples_per_class', type=int, default=100)
    parser.add_argument('--data_dim', type=int, default=16)
    parser.add_argument('--rank', type=int, default=4, help='Rank of subspaces. Only used when --data_type==uos')
    parser.add_argument('--angle', type=float, default=0, help='Principal angle (in degrees) between pairs of subspaces. Only used when --data_type==uos and K must be even.')
    
    # Training settings
    parser.add_argument('--epochs', type=int, default=1000, help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')

    # Save directory
    parser.add_argument('--save_dir', type=str, default='./save/hybrid')

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
        train_set, train_loader = uos_dataset(N_k, K, d, r, batch_size=batch_size, angle=angle)
        val_set, val_loader = uos_dataset(N_k, K, d, r, batch_size=batch_size, angle=angle)
    elif data_type == 'mog':
        train_set, train_loader = mog_dataset(N_k, K, d, batch_size=batch_size)
        val_set, val_loader = mog_dataset(N_k, K, d, batch_size=batch_size)

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
    #model.layers[0].requires_grad_(False)
    for param in model.layers[0].parameters(): # Freeze first weight matrix
        param.requires_grad = False
    model = model.to(device)

    # Set up loss and optimizer
    epochs = args.epochs
    lr = args.lr

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0)
    #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, eta_min=lr/1000)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs//3, 2*epochs//3], gamma=0.1)
    # Training
    patience = args.patience
    early_stopping = EarlyStopping(patience=patience)

    train_losses = []
    train_accs = []

    val_losses = []
    val_accs = []

    min_val_loss = float('Inf')
    for epoch in range(1, epochs+1):
        # Training
        train_loss, train_acc = train_epoch(model, device, train_loader, optimizer, N, criterion, epoch)
        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)

        # Cross-validation
        val_loss, val_acc = eval_epoch(model, device, val_loader, N, criterion, epoch)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        # Print training progress
        if epoch % 100 == 0:
            print("Finish Epoch {}".format(epoch))
            print("Train loss: {}, train accuracy: {}".format(train_loss, train_acc))
            print("Val loss: {}, val accuracy: {}".format(val_loss, val_acc))

        # Save training state
        last_state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_losses': train_losses,
            'train_accuracies': train_accs,
            'val_losses': val_losses,
            'val_accuracies': val_accs,
            'train_set': train_set,
            'train_loader': train_loader,
            'val_set': val_set,
            'val_loader': val_loader
        }

        if val_loss < min_val_loss: # Best training state
            min_val_loss = val_loss
            best_state = last_state

        # Stopping criteria:
        if early_stopping(val_loss): #train_loss < 1e-10 and early_stopping(val_loss):
            print("Done training in {} epochs".format(epoch))
            break


    # Save training results
    save_dir = args.save_dir
    if data_type == 'uos':
        angle_save = int(angle)
        save_subdir = f"width_{D}_depth_{L}_nonlinear_depth_{nonlinear_L}_{init_}_init_{data_type}_data_{K}_classes_rank_{r}_angle_{angle_save}_{activation_str}_activation_seed_{seed}"
    elif data_type == 'mog':
        save_subdir = f"width_{D}_depth_{L}_nonlinear_depth_{nonlinear_L}_{init_}_init_{data_type}_data_{K}_classes_{activation_str}_activation_seed_{seed}"
    checkpoint_dir = os.path.join(save_dir, save_subdir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    print("Saving best model")
    save_path = os.path.join(checkpoint_dir, 'best.pth')
    torch.save(best_state, save_path)

    print("Saving last model\n")
    save_path = os.path.join(checkpoint_dir, 'last.pth')
    torch.save(last_state, save_path)


if __name__ == "__main__":
    main()
