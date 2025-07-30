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


def generate_double_spiral_data():
    """Generate double spiral data while preserving spiral structure"""
    n_samples_per_swiss_roll = 500
    noise_level = 0.1

    # Generate first spiral
    t1 = np.random.uniform(1.5 * np.pi, 4.5 * np.pi, n_samples_per_swiss_roll)
    height1 = np.random.uniform(0, 20, n_samples_per_swiss_roll)

    # 3D coordinates for first swiss roll
    x1 = t1 * np.cos(t1)
    y1 = height1
    z1 = t1 * np.sin(t1)

    # Take x and z coordinates as 2D projection
    X1_2d = np.column_stack([x1, z1])

    # Generate second swiss roll
    t2 = np.random.uniform(1.5 * np.pi, 4.5 * np.pi, n_samples_per_swiss_roll)
    height2 = np.random.uniform(0, 20, n_samples_per_swiss_roll)

    # 3D coordinates for second spiral
    x2 = t2 * np.cos(t2)
    y2 = height2
    z2 = t2 * np.sin(t2)

    # Transform second spiral
    X2_2d = np.column_stack([x2, z2])
    X2_2d[:, 0] = -X2_2d[:, 0]  # Mirror along x-axis
    X2_2d[:, 0] += 45  # Translate

    # Rotate second spiral
    angle = np.pi
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    X2_2d = X2_2d @ rotation_matrix.T

    # Combine both spirals
    X_2d = np.vstack([X1_2d, X2_2d])
    t_combined = np.concatenate([t1, t2])
    spiral_labels = np.concatenate([np.zeros(n_samples_per_swiss_roll), np.ones(n_samples_per_swiss_roll)])

    # Center the data
    x_center = (X_2d[:, 0].min() + X_2d[:, 0].max()) / 2
    X_2d[:, 0] -= x_center

    # Scale the data
    X_2d = X_2d * 0.3

    # Add noise
    noise = np.random.normal(0, noise_level, X_2d.shape)
    X_noisy = X_2d + noise

    # Find spiral starting points (points with minimum t values for each spiral)
    spiral1_mask = spiral_labels == 0
    spiral2_mask = spiral_labels == 1

    spiral1_start_idx = np.where(spiral1_mask)[0][np.argmin(t1)]
    spiral2_start_idx = np.where(spiral2_mask)[0][np.argmin(t2)]

    return (torch.tensor(X_noisy, dtype=torch.float32),
            torch.tensor(X_2d, dtype=torch.float32),
            t_combined, spiral_labels,
            spiral1_start_idx, spiral2_start_idx)


def compute_dual_reference_distances(X_clean, ref_idx1, ref_idx2):
    """
    Compute distances from dual reference points
    """

    X_np = X_clean.numpy()

    # Compute geodesic distances from both reference points
    manifold_distances = compute_manifold_distances(X_np)

    distances1 = manifold_distances[ref_idx1]
    distances2 = manifold_distances[ref_idx2]

    combined_distances = np.minimum(distances1, distances2)  # minimum distance to either reference point

    return combined_distances


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


def train_distance_function(model, X_data, X_clean, spiral1_start, spiral2_start,
                            epochs=1000, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Get combined distances from both reference points
    combined_distances = compute_dual_reference_distances(
        X_clean, spiral1_start, spiral2_start
    )

    target_distances = torch.tensor(combined_distances, dtype=torch.float32).unsqueeze(1)
    target_distances = target_distances / target_distances.max()

    model.train()

    for epoch in range(epochs):
        optimizer.zero_grad()

        pred_distances = model(X_data)
        loss_mse = nn.MSELoss()(pred_distances, target_distances)

        total_loss = loss_mse

        total_loss.backward()
        optimizer.step()

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, MSE Loss: {loss_mse.item():.4f}")

    return model


def plot_neural_network_contour(model, X_data, title, xlim=(-20, 20), ylim=(-15, 15)):
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


def plot_true_manifold_distance(X_data, X_clean, title, xlim=(-20, 20), ylim=(-15, 15), resolution=100):
    """Draw true manifold distance map using dual reference points"""
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
    print("Generating double spiral data...")
    X_noisy, X_clean, t_params, spiral_labels, spiral1_start, spiral2_start = generate_double_spiral_data()

    print(f"Spiral 1 start index: {spiral1_start}")
    print(f"Spiral 2 start index: {spiral2_start}")

    # Compute true manifold distances for visualization
    print("Computing true manifold distances...")
    combined_distances = compute_dual_reference_distances(
        X_clean, spiral1_start, spiral2_start)

    # Initialize models
    models = {
        'NN': StandardNN(),
        'ICNN': ICNN(),
        'IWCNN': IWCNN()
    }

    # Train models with dual reference points
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        trained_models[name] = train_distance_function(
            model, X_noisy, X_clean, spiral1_start, spiral2_start,
        )

    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'Double Swiss Roll Distance Function)',
                 fontsize=16, fontweight='bold')

    # Compute unified xlim and ylim
    X_np = X_noisy.numpy()
    margin = 3
    xlim = (X_np[:, 0].min() - margin, X_np[:, 0].max() + margin)
    ylim = (X_np[:, 1].min() - margin, X_np[:, 1].max() + margin)

    # True distance (top left)
    plt.subplot(2, 2, 1)
    plot_true_manifold_distance(X_noisy, X_clean, 'True',
                                xlim, ylim)

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

    plt.savefig(f'double_swiss_roll_distance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Plot saved as 'double_swiss_roll_distance_comparison.png'")


if __name__ == "__main__":
    main()
