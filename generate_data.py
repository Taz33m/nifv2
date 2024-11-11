import numpy as np

# Generate synthetic training data
def generate_sphere_data(n_samples=10000):
    # Generate random angles
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    phi = np.random.uniform(0, np.pi, n_samples)
    
    # Generate random radii (slightly noisy sphere)
    r = 1.0 + np.random.normal(0, 0.1, n_samples)
    
    # Convert to Cartesian coordinates (input features)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    # Input features (3D points)
    features = np.stack([x, y, z], axis=1)
    
    # Target values (normalized surface normals)
    targets = features / np.linalg.norm(features, axis=1, keepdims=True)
    
    # Combine features and targets
    data = np.concatenate([features, targets], axis=1)
    
    return data

if __name__ == "__main__":
    # Generate data
    data = generate_sphere_data(n_samples=10000)
    
    # Save the data
    np.save('sphere_data.npy', data)
    
    # Print data shape and statistics
    print(f"Data shape: {data.shape}")
    print(f"Features (first 3 columns) range: [{data[:,:3].min():.3f}, {data[:,:3].max():.3f}]")
    print(f"Targets (last 3 columns) range: [{data[:,3:].min():.3f}, {data[:,3:].max():.3f}]")
    
    # Verify data is saved correctly
    loaded_data = np.load('sphere_data.npy')
    print(f"\nLoaded data shape: {loaded_data.shape}")
