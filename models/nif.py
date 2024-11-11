import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Optional
import torch.nn.functional as F

class MLPSimpleShortcut(nn.Module):
    def __init__(self, width: int, activation: str = 'relu', 
                 kernel_init_std: float = 0.1,
                 bias_init_std: float = 0.1):
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

class NIFModule(pl.LightningModule):
    def __init__(self, cfg_shape_net: Dict[str, Any], 
                 cfg_parameter_net: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize network parameters
        self.n_st = cfg_parameter_net.get('width', 64)
        self.l_st = cfg_parameter_net.get('depth', 4)
        self.pi_hidden = cfg_parameter_net.get('bottleneck_size', 32)
        self.po_dim = self._calculate_output_dim(cfg_shape_net)
        
        # Build parameter network
        self.pnet = self._build_parameter_net(cfg_parameter_net)
        
    def _build_parameter_net(self, cfg):
        layers = []
        
        # Input layer
        layers.append(nn.Linear(cfg.get('input_dim'), self.n_st))
        layers.append(getattr(nn, cfg.get('activation', 'ReLU'))())
        
        # Hidden layers with shortcut connections
        for i in range(self.l_st):
            layers.append(
                MLPSimpleShortcut(
                    width=self.n_st,
                    activation=cfg.get('activation', 'relu').lower(),
                    kernel_init_std=0.1,
                    bias_init_std=0.1
                )
            )
            
        # Bottleneck layer
        layers.append(nn.Linear(self.n_st, self.pi_hidden))
        
        # Output layer
        layers.append(nn.Linear(self.pi_hidden, self.po_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.pnet(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
    
    def _calculate_output_dim(self, cfg_shape_net: Dict[str, Any]) -> int:
        # Implement based on your shape network configuration
        return cfg_shape_net.get('output_dim', 1)
