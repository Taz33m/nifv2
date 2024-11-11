import torch
import numpy as np
import matplotlib.pyplot as plt
from nif_torch.models.nif import NIFModule
from mpl_toolkits.mplot3d import Axes3D

def load_and_visualize():
    # Load the test data
    data = np.load('sphere_data.npy')
    test_points = torch.FloatTensor(data[:1000, :3])  # Take first 1000 points
    true_normals = data[:1000, 3:]

    # Model configuration
    cfg = {
        'shape_net': {'output_dim': 3},
        'parameter_net': {
            'input_dim': 3,
            'width': 64,
            'depth': 4,
            'bottleneck_size': 32,
            'activation': 'ReLU'
        }
    }
    
    # Load the latest checkpoint - note the change here
    checkpoint_path = 'checkpoints/epoch=9-train_loss=0.00.ckpt'  # Adjust filename to match your checkpoint
    model = NIFModule.load_from_checkpoint(checkpoint_path, 
                                         cfg_shape_net=cfg['shape_net'],
                                         cfg_parameter_net=cfg['parameter_net'])
    model.eval()

    # Generate predictions
    with torch.no_grad():
        pred_normals = model(test_points).numpy()

    # Create 3D visualization
    fig = plt.figure(figsize=(15, 5))
    
    # Plot input points
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(test_points[:, 0], test_points[:, 1], test_points[:, 2], c='b', marker='.')
    ax1.set_title('Input Points')
    
    # Plot true normals
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.quiver(test_points[:, 0], test_points[:, 1], test_points[:, 2],
              true_normals[:, 0], true_normals[:, 1], true_normals[:, 2],
              length=0.2, normalize=True)
    ax2.set_title('True Normals')
    
    # Plot predicted normals
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.quiver(test_points[:, 0], test_points[:, 1], test_points[:, 2],
              pred_normals[:, 0], pred_normals[:, 1], pred_normals[:, 2],
              length=0.2, normalize=True)
    ax3.set_title('Predicted Normals')
    
    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()

if __name__ == "__main__":
    load_and_visualize()
