# Neural Implicit Functions (NIFv2)

This project implements a Neural Implicit Function (NIF) to predict surface normals from 3D point cloud data. The implementation uses PyTorch Lightning and demonstrates the capability on a spherical dataset.

## Setup

1. Create a virtual environment:

python -m venv nif_env
source nif_env/bin/activate # On Unix/macOS

2. Install dependencies:

pip install -r requirements.txt

## Usage

1. Generate synthetic sphere data:

python generate_data.py

2. Train the model:

python main.py

3. Visualize results:

python visualize_results.py

## Results
![Results](results.png)

The visualization shows:
- Left: Input point cloud
- Middle: True surface normals
- Right: Predicted surface normals

## Model Architecture
- Parameter Network with shortcut connections
- Input dimension: 3 (x, y, z coordinates)
- Output dimension: 3 (surface normal vectors)
- Hidden layers: 4 layers with width 64
- Bottleneck size: 32

## Training Details
- Optimizer: Adam
- Loss: MSE
- Epochs: 10
- Batch size: 32
- Device: MPS (Apple Silicon) (Was initially using GPU but found MPS to be faster)
