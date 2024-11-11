import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any

class MLPSimpleShortcut(nn.Module):
    def __init__(self, width, activation='relu', kernel_init_std=0.1, bias_init_std=0.1):
        super().__init__()
        self.linear = nn.Linear(width, width)
        self.activation = getattr(F, activation)
        
        # Initialize weights
        nn.init.normal_(self.linear.weight, std=kernel_init_std)
        nn.init.normal_(self.linear.bias, std=bias_init_std)
        
    def forward(self, x):
        identity = x
        out = self.linear(x)
        out = self.activation(out)
        return out + identity
