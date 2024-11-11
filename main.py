import torch
from train import train_nif

# Configuration dictionary
cfg = {
    'shape_net': {
        'output_dim': 3  # Example: 3D output
    },
    'parameter_net': {
        'input_dim': 3,  # Example: 3D input
        'width': 64,
        'depth': 4,
        'bottleneck_size': 32,
        'activation': 'ReLU'
    },
    'data_path': 'sphere_data.npy',  # Use the generated data would otherwise be 'path/to/your/data.npy'
    'n_feature': 3,  # Number of input features
    'n_target': 3,   # Number of target features
    'batch_size': 32,
    'epochs': 10
}

if __name__ == "__main__":
    train_nif(cfg)
