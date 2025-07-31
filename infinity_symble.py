import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_swiss_roll
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend to avoid PyCharm compatibility issues
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


class StandardNN(nn.Module):
    """Standard Neural Network"""

    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.network(x)


class ICNN(nn.Module):
    """Input Convex Neural Network"""

    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # First layer: normal linear layer
        self.first_layer = nn.Linear(input_dim, hidden_dim)

        # Hidden layers: weights must be non-negative
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)
        ])

        # Skip connection layers (from input directly to each hidden layer)
        self.skip_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(2)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.output_skip = nn.Linear(input_dim, 1)

    def forward(self, x):
        # First layer
        h = torch.relu(self.first_layer(x))

        # Hidden layers (ensure weights are non-negative)
        for i, (hidden_layer, skip_layer) in enumerate(zip(self.hidden_layers, self.skip_layers)):
            # Ensure hidden layer weights are non-negative
            with torch.no_grad():
                hidden_layer.weight.data = torch.clamp(hidden_layer.weight.data, min=0)

            h = torch.relu(hidden_layer(h) + skip_layer(x))

        # Output layer (ensure weights are non-negative)
        with torch.no_grad():
            self.output_layer.weight.data = torch.clamp(self.output_layer.weight.data, min=0)

        return self.output_layer(h) + self.output_skip(x)


class IWCNN(nn.Module):
    """Input Weakly Convex Neural Network"""

    def __init__(self, input_dim=2, hidden_dim=128, lambda_reg=0.1):
        super().__init__()
        self.lambda_reg = lambda_reg

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Quadratic term for computing weak convexity regularization
        self.quadratic = nn.Linear(input_dim, 1, bias=False)
        with torch.no_grad():
            # Initialize as diagonal matrix weights
            self.quadratic.weight.data = torch.ones(1, input_dim) * 0.1

    def forward(self, x):
        network_out = self.network(x)
        quadratic_out = 0.5 * torch.sum(x * (self.quadratic.weight * x), dim=1, keepdim=True)
        return network_out + self.lambda_reg * quadratic_out


def generate_infinity_symbol_data():
    """Generate infinity symbol (lemniscate) data while preserving curve structure"""
    n_samples = 1000
    noise_level = 0.1

    # Parameter range: t from 0 to 2π for complete infinity symbol
    t = np.random.uniform(0, 2 * np.pi, n_samples)

    # Lemniscate of Bernoulli parametric equations
    # x = a * cos(t) / (1 + sin²(t))
    # y = a * sin(t) * cos(t) / (1 + sin²(t))
    a = 3.0  # Scale factor for the infinity symbol

    denominator = 1 + np.sin(t) ** 2
    x = a * np.cos(t) / denominator
    y = a * np.sin(t) * np.cos(t) / denominator

    # Create 2D coordinates
    X_2d = np.column_stack([x, y])

    # Center the data
    X_2d = X_2d - X_2d.mean(axis=0)

    # Add noise
    noise = np.random.normal(0, noise_level, X_2d.shape)
    X_noisy = X_2d + noise

    # Save t parameter for subsequent analysis
    t_sorted_idx = np.argsort(t)

    return (torch.tensor(X_noisy, dtype=torch.float32),
            torch.tensor(X_2d, dtype=torch.float32),
            t, t_sorted_idx)


def compute_manifold_distances(X_clean):
    """Compute geodesic distances on manifold (using k-nearest neighbor graph approximation)"""
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import csr_matrix

    # Build k-nearest neighbor graph
    k = 15  # Increase number of neighbors to ensure connectivity
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_clean)
    distances, indices = nbrs.kneighbors(X_clean)

    # Build symmetric sparse distance matrix
    n = X_clean.shape[0]
    row_ind = []
    col_ind = []
    data = []

    for i in range(n):
        for j in range(k):
            neighbor_idx = indices[i, j]
            dist = distances[i, j]

            # Add bidirectional edges
            row_ind.extend([i, neighbor_idx])
            col_ind.extend([neighbor_idx, i])
            data.extend([dist, dist])

    distance_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))

    # Compute shortest paths (geodesic distances)
    geodesic_distances = shortest_path(distance_matrix, directed=False)

    # Handle infinite values
    geodesic_distances[np.isinf(geodesic_distances)] = geodesic_distances[np.isfinite(geodesic_distances)].max() * 2

    return geodesic_distances


def train_distance_function(model, X_data, X_clean, t_params, t_sorted_idx, epochs=1000, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Computing manifold distances...")

    # Using the rightmost point (where cos(t) is maximum and sin(t) ≈ 0)
    reference_t = 0.0  # This gives the rightmost point of the infinity symbol
    reference_idx = np.argmin(np.abs(t_params - reference_t))

    # Use geodesic distance based on the curve structure
    manifold_distances = compute_manifold_distances(X_clean.numpy())
    target_distances = torch.tensor(manifold_distances[reference_idx], dtype=torch.float32).unsqueeze(1)
    target_distances = target_distances / target_distances.max()

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        pred_distances = model(X_data)
        loss_mse = nn.MSELoss()(pred_distances, target_distances)

        X_mirrored = X_data.clone()
        X_mirrored[:, 0] = -X_mirrored[:, 0]  # Mirror across y-axis
        pred_distances_mirrored = model(X_mirrored)
        symmetry_loss = torch.mean((pred_distances - pred_distances_mirrored) ** 2)

        total_loss = loss_mse + symmetry_loss

        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, MSE Loss: {loss_mse.item():.4f}")

    return model


def plot_neural_network_contour(model, X_data, title, xlim=(-4, 4), ylim=(-2, 2)):
    """Create contour plot (specifically for subplots)"""
    model.eval()

    # Create grid
    x_range = np.linspace(xlim[0], xlim[1], 100)
    y_range = np.linspace(ylim[0], ylim[1], 100)
    xx, yy = np.meshgrid(x_range, y_range)

    # Compute distance function values for grid points
    grid_points = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32)

    with torch.no_grad():
        z_values = model(grid_points).numpy().reshape(xx.shape)

    # Draw contour lines
    levels = np.linspace(z_values.min(), z_values.max(), 15)
    plt.contourf(xx, yy, z_values, levels=levels, cmap='RdBu_r', alpha=0.8)
    plt.contour(xx, yy, z_values, levels=levels, colors='white', alpha=0.3, linewidths=0.5)

    # Draw data points
    X_np = X_data.numpy()
    plt.scatter(X_np[:, 0], X_np[:, 1], c='darkblue', s=20, alpha=0.8, edgecolors='white', linewidth=0.5)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim(xlim)
    plt.ylim(ylim)


def compute_geodesic_distance(grid_points, manifold_points, n_neighbors=15):
    all_points = np.vstack((grid_points, manifold_points))
    n_grid = grid_points.shape[0]
    knn_graph = kneighbors_graph(all_points, n_neighbors=n_neighbors, mode='distance')
    geo_dist_matrix = shortest_path(knn_graph, method='auto')
    distances = geo_dist_matrix[:n_grid, n_grid:]
    return np.min(distances, axis=1).reshape(-1)


def plot_true_manifold_distance(X_data, X_clean, t_params, t_sorted_idx, title, xlim=(-4, 4), ylim=(-2, 2),
                                resolution=100):
    """Draw true manifold distance map using graph distance, improved as geodesic-based True panel"""
    from scipy.ndimage import gaussian_filter

    # Create grid points
    x_range = np.linspace(xlim[0], xlim[1], resolution)
    y_range = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # Use geodesic distance to compute shortest graph distance from each grid point to manifold
    geodesic_dists = compute_geodesic_distance(grid_points, X_clean.numpy(), n_neighbors=10)
    z = geodesic_dists.reshape(resolution, resolution)
    z = gaussian_filter(z, sigma=1.0)

    # Plot
    levels = np.linspace(np.nanmin(z), np.nanmax(z), 15)
    plt.contourf(xx, yy, z, levels=levels, cmap='RdBu_r', alpha=0.7)
    plt.contour(xx, yy, z, levels=levels, colors='white', alpha=0.3, linewidths=0.5)

    # Visualize original points
    X_np = X_data.numpy()
    plt.scatter(X_np[:, 0], X_np[:, 1], c='darkblue', s=12, alpha=0.8,
                edgecolors='white', linewidth=0.3)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim(xlim)
    plt.ylim(ylim)


def main():
    # Generate data
    print("Generating infinity symbol data...")
    X_noisy, X_clean, t_params, t_sorted_idx = generate_infinity_symbol_data()

    # Compute true manifold distances (based on reference point)
    print("Computing true manifold distances...")
    reference_t = 0.0
    reference_idx = np.argmin(np.abs(t_params - reference_t))

    # For visualization: compute parameter-based distances
    param_distances = np.minimum(np.abs(t_params - reference_t),
                                 2 * np.pi - np.abs(t_params - reference_t))
    true_distances = param_distances / param_distances.max()

    # Initialize models
    models = {
        'NN': StandardNN(),
        'ICNN': ICNN(),
        'IWCNN': IWCNN()
    }

    # Train models
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        trained_models[name] = train_distance_function(model, X_noisy, X_clean, t_params, t_sorted_idx)

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Infinity Symbol Distance Function Approximation Comparison', fontsize=16, fontweight='bold')

    # Compute unified xlim and ylim for infinity symbol
    X_np = X_noisy.numpy()
    margin = 0.5
    xlim = (X_np[:, 0].min() - margin, X_np[:, 0].max() + margin)
    ylim = (X_np[:, 1].min() - margin, X_np[:, 1].max() + margin)

    # True distance (top left)
    plt.subplot(2, 2, 1)
    plot_true_manifold_distance(X_noisy, X_clean, t_params, t_sorted_idx, 'True', xlim, ylim)

    # NN approximation (top right)
    plt.subplot(2, 2, 2)
    plot_neural_network_contour(trained_models['NN'], X_noisy, 'NN approximation', xlim, ylim)

    # ICNN approximation (bottom left)
    plt.subplot(2, 2, 3)
    plot_neural_network_contour(trained_models['ICNN'], X_noisy, 'ICNN approximation', xlim, ylim)

    # IWCNN approximation (bottom right)
    plt.subplot(2, 2, 4)
    plot_neural_network_contour(trained_models['IWCNN'], X_noisy, 'IWCNN approximation', xlim, ylim)

    plt.tight_layout()

    plt.savefig('infinity_symbol_distance_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'infinity_symbol_distance_comparison.png'")

    print(
        "\nTraining completed! The plots show the comparison between different neural network architectures on infinity symbol:")
    print("- True: Ground truth manifold distance based on geodesic distance")
    print("- NN: Standard neural network approximation")
    print("- ICNN: Input Convex Neural Network approximation")
    print("- IWCNN: Input Weakly Convex Neural Network approximation")

    # Visualize infinity symbol structure
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(X_noisy.numpy()[:, 0], X_noisy.numpy()[:, 1],
                c=t_params, cmap='viridis', s=20, alpha=0.7)
    plt.colorbar(label='Parameter t')
    plt.title('Infinity Symbol colored by parameter t')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.axis('equal')

    plt.subplot(1, 3, 2)
    colors = param_distances
    plt.scatter(X_noisy.numpy()[:, 0], X_noisy.numpy()[:, 1],
                c=colors, cmap='RdBu_r', s=20, alpha=0.7)
    plt.colorbar(label='Distance from reference')
    plt.title('Infinity Symbol colored by distance from reference point')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.axis('equal')

    # Mark reference point
    ref_point = X_noisy[reference_idx].numpy()
    plt.scatter(ref_point[0], ref_point[1], c='red', s=100, marker='*',
                edgecolors='black', linewidth=2, label='Reference Point')
    plt.legend()

    plt.subplot(1, 3, 3)
    # Show the clean infinity symbol shape
    plt.plot(X_clean.numpy()[:, 0], X_clean.numpy()[:, 1], 'b-', alpha=0.3, linewidth=1)
    plt.scatter(X_clean.numpy()[:, 0], X_clean.numpy()[:, 1],
                c=t_params, cmap='plasma', s=15, alpha=0.8)
    plt.colorbar(label='Parameter t')
    plt.title('Clean Infinity Symbol (Lemniscate)')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig('infinity_symbol_structure_analysis.png', dpi=300, bbox_inches='tight')
    print("Structure analysis saved as 'infinity_symbol_structure_analysis.png'")

    # Print statistics
    reference_point = X_noisy[reference_idx].numpy()
    print(f"Reference point coordinates: ({reference_point[0]:.2f}, {reference_point[1]:.2f})")
    print(f"Parameter t for reference point: {t_params[reference_idx]:.2f}")
    print(f"Parameter t range: {t_params.min():.2f} to {t_params.max():.2f}")
    print(f"Infinity symbol dimensions: X ∈ [{X_clean.numpy()[:, 0].min():.2f}, {X_clean.numpy()[:, 0].max():.2f}], "
          f"Y ∈ [{X_clean.numpy()[:, 1].min():.2f}, {X_clean.numpy()[:, 1].max():.2f}]")


if __name__ == "__main__":
    main()
