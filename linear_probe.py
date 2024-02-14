import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
import argparse

from model import *
from dataset import *
from utils import *


def train_epoch_linear(model, linear_model, layer_idx, device, loader, optimizer, num_samples, criterion, epoch):
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



# Save model's intermediate layer features
def save_interm_features(model, depth, device, loader):
    interm_features = {}
    for i in range(depth):
        interm_features[i] = []
    
    model.eval()

    # One loop through data
    for _, (data, targets) in enumerate(loader):
        data, targets = data.to(device), targets.to(device)

        with torch.no_grad():
            _, layers_out = model(data)
        
        for i in range(depth):
            interm_features[i].append(layers_out[i])

    return interm_features



# Parse command line arguments
def parse_probe_args():
    parser = argparse.ArgumentParser()

    # Model selection
    parser.add_argument('--model_path', type=str, help='Path to saved model weights')
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

    # Training settings
    parser.add_argument('--epochs', type=int, default=500, help='Max training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-2, help='Initial learning rate')
    parser.add_argument('--patience', type=int, default=100, help='Early stopping patience')

    # Save directory
    parser.add_argument('--save_dir', type=str, default='./save/linear_probe')

    args = parser.parse_args()

    return args


# Main function
def main():
    args = parse_probe_args()
    print(args)

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Random seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Command line arguments
    D = args.hidden_dim
    L = args.depth

    data_type = args.data_type
    K = args.num_classes
    N_k = args.samples_per_class
    N = N_k * K
    d = args.data_dim

    nonlinear_L = args.nonlinear_depth
    activation_str = args.activation
    activation = activations[activation_str]
    init_ = args.init
    var = args.init_var

    # Initialize model
    model = HybridNet(in_dim=d, hidden_dim=D, num_classes=K,
                     num_layers=L, num_nonlinear_layers=nonlinear_L,
                     activation=activation, init_method=init_, var=var)

    # Load saved data loaders and model weights
    model_path = args.model_path
    load_path = os.path.join(model_path, 'best.pth')
    ckpt = torch.load(load_path)

    train_loader = ckpt['train_loader'] # Train set loader
    val_loader = ckpt['val_loader'] # Val set loader

    print("Loading saved model.")
    model.load_state_dict(ckpt['state_dict']) # Model weights
    model = model.to(device)

    # Set up save directory
    save_dir = args.save_dir
    save_subdir = f"linear_probe_width_{D}_depth_{L}_nonlinear_depth_{nonlinear_L}_{init_}_init_{data_type}_data_{activation_str}_activation_seed_{seed}"
    checkpoint_dir = os.path.join(save_dir, save_subdir)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # Linear probe intermediate layer features
    epochs = args.epochs
    lr = args.lr
    patience = args.patience

    layer_idx = list(range(L-1))
    for idx in layer_idx:
        # Initialize linear probe
        linear_model = LinearClassifier(D, K)
        linear_model = linear_model.to(device)

        # Set up loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(linear_model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epochs, eta_min=lr/1000)
        early_stopping = EarlyStopping(patience=patience)

        min_val_loss = float('Inf')
        for epoch in range(1, epochs+1):
            # Training
            train_loss, train_acc = train_epoch_linear(model, linear_model, idx, device, train_loader, optimizer, N, criterion, epoch)
            scheduler.step()
        
            # Cross-validation
            val_loss, val_acc = eval_epoch_linear(model, linear_model, idx, device, val_loader, N, criterion)
            
            # Print training progress
            if epoch % 100 == 0:
                print("Finish Epoch {}".format(epoch))
                print("Train loss: {}, train accuracy: {}".format(train_loss, train_acc))
                print("Val loss: {}, val accuracy: {}".format(val_loss, val_acc))
            
            # Save training state
            last_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'linear_state_dict': linear_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }

            if val_loss < min_val_loss: # Best training state 
                min_val_loss = val_loss
                best_state = last_state

            # Stopping criteria
            if train_loss < 1e-10 and early_stopping(val_loss):
                print("Done training in {} epochs".format(epoch))
                break
    
        # Save training results
        print("Saving best linear model")
        save_fname = 'layer_' + str(idx) + '_best.pth'
        save_path = os.path.join(checkpoint_dir, save_fname)
        torch.save(best_state, save_path)

        print("Saving last linear model")
        save_fname = 'layer_' + str(idx) + '_last.pth'
        save_path = os.path.join(checkpoint_dir, save_fname)
        torch.save(last_state, save_path)

    
    # Post-processing: save intermediate layer features
    interm_features = save_interm_features(model, L-1, device, train_loader)
    for i in range(L-1):
        interm_features[i] = torch.cat(interm_features[i], 0).detach().cpu()
    state = { 'interm_features': interm_features }
    save_path = os.path.join(checkpoint_dir, "intermediate_features.pth")
    print("Saving intermediate feature layers\n")
    torch.save(state, save_path)
    


if __name__ == "__main__":
    main()
