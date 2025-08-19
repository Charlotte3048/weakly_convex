import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_swiss_roll
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)


class StandardNN(nn.Module):
    """Standard Neural Network"""

    def __init__(self, input_dim=2, hidden_dim=128, num_layers=8):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]

        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

        layers.append(nn.Linear(hidden_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ICNN(nn.Module):
    """Input Convex Neural Network"""

    def __init__(self, input_dim=2, hidden_dim=128, num_layers=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # First layer: normal linear layer
        self.first_layer = nn.Linear(input_dim, hidden_dim)

        # Hidden layers: weights must be non-negative
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        ])

        # Skip connection layers (from input directly to each hidden layer)
        self.skip_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_layers - 2)
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


class NeuralSpline(nn.Module):
    """Neural Spline Network (NS) - Standard NN with smooth activations"""

    def __init__(self, input_dim=2, hidden_dim=128, num_layers=4, activation='swish'):
        super().__init__()
        self.activation_name = activation

        # Choose smooth activation function
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'softplus':
            self.activation = nn.Softplus()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = [nn.Linear(input_dim, hidden_dim), self.activation]

        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), self.activation])

        # Output layer - no activation (linear output)
        layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ICNN_Component(nn.Module):
    """Input Convex Neural Network Component for IWCNN"""

    def __init__(self, input_dim, hidden_dim=128, num_layers=4):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # First layer: normal linear layer (from NS output)
        self.first_layer = nn.Linear(hidden_dim, hidden_dim)

        # Hidden layers: weights must be non-negative
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        ])

        # Skip connection layers (from original input directly to each hidden layer)
        self.skip_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_layers - 1)
        ])

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.output_skip = nn.Linear(input_dim, 1)

    def forward(self, ns_output, original_input):
        # First layer from NS output
        h = torch.relu(self.first_layer(ns_output))

        # Hidden layers with skip connections from original input
        for i, (hidden_layer, skip_layer) in enumerate(zip(self.hidden_layers, self.skip_layers)):
            # Ensure hidden layer weights are non-negative
            with torch.no_grad():
                hidden_layer.weight.data = torch.clamp(hidden_layer.weight.data, min=0)

            h = torch.relu(hidden_layer(h) + skip_layer(original_input))

        # Output layer (ensure weights are non-negative)
        with torch.no_grad():
            self.output_layer.weight.data = torch.clamp(self.output_layer.weight.data, min=0)

        return self.output_layer(h) + self.output_skip(original_input)


class IWCNN(nn.Module):
    """
    Input Weakly Convex Neural Network: IWCNN(x) = ICNN(NS(x))
    where NS is a Neural Spline (standard NN with smooth activations)
    and ICNN is an Input Convex Neural Network
    """

    def __init__(self, input_dim=2, hidden_dim=128, total_layers=8,
                 ns_layers=4, activation='relu', scenario='equal'):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.total_layers = total_layers
        self.scenario = scenario

        # Determine layer distribution based on scenario
        if scenario == 'equal':
            self.ns_layers = total_layers // 2
            self.icnn_layers = total_layers // 2
        elif scenario == 'shallow_ns_deep_icnn':
            self.ns_layers = 2
            self.icnn_layers = total_layers - 2
        elif scenario == 'deep_ns_shallow_icnn':
            self.ns_layers = total_layers - 2
            self.icnn_layers = 2
        else:
            # Custom distribution
            self.ns_layers = ns_layers
            self.icnn_layers = total_layers - ns_layers

        print(f"IWCNN {scenario}: NS layers = {self.ns_layers}, ICNN layers = {self.icnn_layers}")

        # Neural Spline component
        self.neural_spline = NeuralSpline(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=self.ns_layers,
            activation=activation
        )

        # ICNN component
        self.icnn_component = ICNN_Component(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=self.icnn_layers
        )

    def forward(self, x):
        # First pass through Neural Spline
        ns_output = self.neural_spline(x)

        # Then pass through ICNN component with skip connections from original input
        icnn_output = self.icnn_component(ns_output, x)

        return icnn_output


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


def train_distance_function(model, X_data, X_clean, spiral1_start_idx, spiral2_start_idx,
                            epochs=1000, lr=0.002, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Computing dual reference distances...")
    combined_distances = compute_dual_reference_distances(X_clean, spiral1_start_idx, spiral2_start_idx)
    target_distances = torch.tensor(combined_distances, dtype=torch.float32).unsqueeze(1)
    target_distances = target_distances / target_distances.max()

    model.train()
    dataset = torch.utils.data.TensorDataset(X_data, target_distances)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            pred_distances = model(batch_X)
            loss_mse = nn.MSELoss()(pred_distances, batch_y)
            loss_mse.backward()
            optimizer.step()
            epoch_loss += loss_mse.item()

        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)

        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, MSE Loss: {avg_loss:.4f}")

    return model, loss_history


def plot_loss_curve(loss_history, title, save_path):
    plt.figure()
    plt.plot(loss_history, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[Viz] Saved loss curve to {save_path}")



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


def plot_true_manifold_distance(X_data, X_clean, spiral1_start_idx, spiral2_start_idx, title, xlim=(-20, 20),
                                ylim=(-15, 15),
                                resolution=100):
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
    """Main function: compare three IWCNN scenarios on double spiral data"""


    # Check for GPU availability and set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")


    # Generate double spiral data
    print("Generating double spiral data...")
    X_noisy, X_clean, t_params, spiral_labels, spiral1_start_idx, spiral2_start_idx = generate_double_spiral_data()

    X_noisy = X_noisy.to(device)
    X_clean = X_clean.to(device)

    print(f"Spiral 1 start index: {spiral1_start_idx}")
    print(f"Spiral 2 start index: {spiral2_start_idx}")

    # Initialize models with different scenarios
    models = {
        'NN': StandardNN(num_layers=8),
        'ICNN': ICNN(num_layers=8),
        'IWCNN_Equal': IWCNN(scenario='equal', total_layers=8),
        'IWCNN_Shallow_NS': IWCNN(scenario='shallow_ns_deep_icnn', total_layers=8),
        'IWCNN_Deep_NS': IWCNN(scenario='deep_ns_shallow_icnn', total_layers=8)
    }

    for name, model in models.items():
        models[name] = model.to(device)

    print("\nModel configurations:")
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{name}: {total_params:,} parameters")

    # Train models
    trained_models = {}
    loss_histories = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        trained_models[name], loss_histories[name] = train_distance_function(
            model, X_noisy, X_clean, spiral1_start_idx, spiral2_start_idx
        )
        plot_loss_curve(loss_histories[name], title=f"{name} Loss Curve", save_path=f"{name.lower()}_loss.png")

    # Create comparison figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('IWCNN Architecture Comparison on Double Spiral: Layer Distribution Effects', fontsize=16,
                 fontweight='bold')

    # Compute unified xlim and ylim for double spiral
    X_np = X_noisy.numpy()
    margin = 3
    xlim = (X_np[:, 0].min() - margin, X_np[:, 0].max() + margin)
    ylim = (X_np[:, 1].min() - margin, X_np[:, 1].max() + margin)

    # Plot results
    plot_configs = [
        ('True', None),
        ('Standard NN', 'NN'),
        ('ICNN', 'ICNN'),
        ('IWCNN Equal (4+4)', 'IWCNN_Equal'),
        ('IWCNN Shallow NS (2+6)', 'IWCNN_Shallow_NS'),
        ('IWCNN Deep NS (6+2)', 'IWCNN_Deep_NS')
    ]

    for i, (title, model_key) in enumerate(plot_configs):
        row, col = i // 3, i % 3
        plt.subplot(2, 3, i + 1)

        if model_key is None:
            # True distance plot
            plot_true_manifold_distance(X_noisy, X_clean, spiral1_start_idx, spiral2_start_idx, title, xlim, ylim)
        else:
            # Model prediction plot
            plot_neural_network_contour(trained_models[model_key], X_noisy, title, xlim, ylim)

    plt.tight_layout()
    plt.savefig('double_spiral_iwcnn_comparison.png', dpi=300, bbox_inches='tight')
    print("Comparison plot saved as 'double_spiral_iwcnn_comparison.png'")



if __name__ == "__main__":
    main()