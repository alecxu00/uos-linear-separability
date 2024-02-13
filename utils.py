import torch
import torch.nn as nn
import numpy as np

# Activation functions
def quadratic(x: torch.Tensor) -> torch.Tensor: return x**2
def step(x: torch.Tensor) -> torch.Tensor: return x > 0

activations = {
    'relu': nn.ReLU(),
    'sigmoid': nn.Sigmoid(),
    'quadratic': quadratic,
    'gelu': nn.GELU(),
    'elu': nn.ELU(),
    'leaky-relu': nn.LeakyReLU(),
    'step': step
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