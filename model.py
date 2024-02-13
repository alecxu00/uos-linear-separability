from typing import Callable
import numpy as np
import torch
import torch.nn as nn 


# Hybrid network implementation
class HybridNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes, num_layers, num_nonlinear_layers=1,
                 activation: nn.Module | Callable[[torch.Tensor], torch.Tensor]=nn.ReLU(),
                 bias=False, init_method='gaussian', var=1e-2):

        super(HybridNet, self).__init__()

        # Dimensions of weight matrices
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.bias = bias

        # Number of layers
        self.num_nonlinear_layers = num_nonlinear_layers
        self.num_linear_layers = num_layers - self.num_nonlinear_layers

        # Create layers
        self.activation = activation
        self.create_layers()
        
        # Initialize weights
        self.init_method = init_method
        self.var = var
        self.init_weights()
    

    # Helper function to create layers
    def create_layers(self):
        layers_list = [ nn.Sequential(nn.Linear(self.in_dim, self.hidden_dim, bias=self.bias), self.activation) ]

        # Nonlinear layers
        for _ in range(self.num_nonlinear_layers - 1):
            layers_list.append( nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias), self.activation) )
        
        # Linear layers
        for _ in range(self.num_linear_layers - 1):
            layers_list.append( nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim, bias=self.bias)) )
        
        self.layers = nn.ModuleList(layers_list)

        # Last layer linear classifier
        self.fc = nn.Linear(self.hidden_dim, self.num_classes, bias=self.bias)

    
    # Helper function to initialize weights
    def init_weights(self):
        print(f"Initializing weights using {self.init_method} method")

        # default: PyTorch default initialization method
        if self.init_method == 'default':
            pass

        # uniform: Uniform initialization with indicated variance
        elif self.init_method == 'uniform':
            bound_ = np.sqrt(3) * self.var
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    nn.init.uniform_(layer.weight, a=-bound_, b=bound_)
            nn.init.uniform_(self.fc.weight, a=-bound_, b=bound_)

        # gaussian: Gaussian initialization with indicated variance
        elif self.init_method == 'gaussian':
            for layer in self.layers:
                if isinstance(layer, nn.Linear):
                    nn.init.normal_(layer.weight, mean=0.0, std=np.sqrt(self.var))
            nn.init.normal_(self.fc.weight, mean=0.0, std=np.sqrt(self.var))
    
    # Forward pass
    def forward(self, x):
        # Store each layer output
        out_list = []
        for layer in self.layers:
            x = layer(x)
            out_list.append(x.clone().detach())
        
        out = self.fc(x)
        return out, out_list


# Linear classifier
class LinearClassifier(nn.Module):
    def __init__(self, in_dim, num_classes, bias=False):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(in_dim, num_classes, bias=bias)
    
    def forward(self, x):
        return self.fc(x)
