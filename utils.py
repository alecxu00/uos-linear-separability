import torch
import torch.nn as nn
import numpy as np

# Activation functions
class Quadratic(nn.Module):
    def __init__(self):
        super(Quadratic, self).__init__()
    def forward(self, x):
        return torch.pow(x, 2)

class Step(nn.Module):
    def __init__(self):
        super(Step, self).__init__()
    def forward(self, x):
        return (x > 0).float()


activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'quadratic': Quadratic(),
    'gelu': nn.GELU(),
    'elu': nn.ELU(),
    'leaky-relu': nn.LeakyReLU(),
    'step': Step()
}


# Early stopping
class EarlyStopping:
    def __init__(self, patience=100):

        self.patience = patience
        self.counter = 0
        self.min_val_loss = float('Inf')


    def __call__(self, val_loss):
        if (val_loss < self.min_val_loss):
            self.counter = 0
            self.min_val_loss = val_loss
        elif (val_loss > self.min_val_loss):
            self.counter += 1
            if self.counter > self.patience:
                return True

        return False
