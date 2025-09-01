# infinity_symble.py
# Cambridge-style: models, training pipeline, visualization, loss curves (for Infinity symbol)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib

matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from scipy.sparse.csgraph import shortest_path
from scipy.sparse import csr_matrix

# ------------------ seeds ------------------
torch.manual_seed(42)
np.random.seed(42)


# ------------------ Models (identical to Cambridge_double_swiss_roll.py) ------------------
class StandardNN(nn.Module):
    """Standard Neural Network (same depth/width as Cambridge)"""

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
    """Input Convex Neural Network (Cambridge style: non-neg hidden & output weights + input skips)"""

    def __init__(self, input_dim=2, hidden_dim=128, num_layers=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.first_layer = nn.Linear(input_dim, hidden_dim)

        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        ])
        self.skip_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_layers - 2)
        ])

        self.output_layer = nn.Linear(hidden_dim, 1)
        self.output_skip = nn.Linear(input_dim, 1)

    def forward(self, x):
        h = torch.relu(self.first_layer(x))
        for hidden_layer, skip_layer in zip(self.hidden_layers, self.skip_layers):
            with torch.no_grad():
                hidden_layer.weight.data.clamp_(min=0)
            h = torch.relu(hidden_layer(h) + skip_layer(x))
        with torch.no_grad():
            self.output_layer.weight.data.clamp_(min=0)
        return self.output_layer(h) + self.output_skip(x)


class NeuralSpline(nn.Module):
    """Smooth sub-network (NS) with smooth activations (use ReLU here to match Cambridge options)"""

    def __init__(self, input_dim=2, hidden_dim=128, num_layers=4, activation='relu'):
        super().__init__()
        if activation == 'elu':
            act = nn.ELU()
        elif activation == 'softplus':
            act = nn.Softplus()
        else:
            act = nn.ReLU()

        layers = [nn.Linear(input_dim, hidden_dim), act]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act])
        layers.append(nn.Linear(hidden_dim, hidden_dim))  # latent z
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class ICNN_Component(nn.Module):
    """ICNN head that takes z=f_NS(x) and also receives x via skips (Cambridge style)"""

    def __init__(self, input_dim=2, hidden_dim=128, num_layers=4):
        super().__init__()
        self.first_layer = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        ])
        self.skip_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(num_layers - 1)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.output_skip = nn.Linear(input_dim, 1)

    def forward(self, ns_output, x):
        h = torch.relu(self.first_layer(ns_output))
        for hidden_layer, skip_layer in zip(self.hidden_layers, self.skip_layers):
            with torch.no_grad():
                hidden_layer.weight.data.clamp_(min=0)
            h = torch.relu(hidden_layer(h) + skip_layer(x))
        with torch.no_grad():
            self.output_layer.weight.data.clamp_(min=0)
        return self.output_layer(h) + self.output_skip(x)


class IWCNN(nn.Module):
    """
    IWCNN(x) = ICNN( NS(x) ), weakly-convex via composition (Cambridge style)
    scenario in {'equal','shallow_ns_deep_icnn','deep_ns_shallow_icnn'}
    """

    def __init__(self, input_dim=2, hidden_dim=128, total_layers=8,
                 ns_layers=4, activation='relu', scenario='equal'):
        super().__init__()
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
            self.ns_layers = ns_layers
            self.icnn_layers = total_layers - ns_layers

        print(f"IWCNN {scenario}: NS layers = {self.ns_layers}, ICNN layers = {self.icnn_layers}")

        self.neural_spline = NeuralSpline(input_dim=input_dim, hidden_dim=hidden_dim,
                                          num_layers=self.ns_layers, activation=activation)
        self.icnn_component = ICNN_Component(input_dim=input_dim, hidden_dim=hidden_dim,
                                             num_layers=self.icnn_layers)

    def forward(self, x):
        z = self.neural_spline(x)
        return self.icnn_component(z, x)


# ------------------ Infinity symbol data ------------------
def generate_infinity_symbol_data(n_samples=1000, noise_level=0.1):
    """
    Generate infinity symbol (lemniscate of Bernoulli) dataset
    with a single reference point at the right endpoint.
    """
    # sample parameter t
    t = np.random.uniform(0, 2 * np.pi, n_samples)

    # parametric equations
    a = 3.0
    denominator = 1 + np.sin(t) ** 2
    x = a * np.cos(t) / denominator
    y = a * np.sin(t) * np.cos(t) / denominator

    # clean coordinates
    X_clean = np.column_stack([x, y])
    X_clean = X_clean - X_clean.mean(axis=0)  # center data

    # add noise
    noise = np.random.normal(0, noise_level, X_clean.shape)
    X_noisy = X_clean + noise

    # find reference point (right endpoint, t = 0)
    reference_t = 0.0
    ref_idx = np.argmin(np.abs(t - reference_t))
    reference_indices = [ref_idx]

    # convert to torch
    return (torch.tensor(X_noisy, dtype=torch.float32),
            torch.tensor(X_clean, dtype=torch.float32),
            reference_indices)



# ------------------ Geodesic distance utilities (same spirit as Cambridge) ------------------
def compute_manifold_distances(X_clean_np, k=15):
    """All-pairs geodesic (shortest path) distances on kNN graph of clean points"""
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_clean_np)
    distances, indices = nbrs.kneighbors(X_clean_np)

    n = X_clean_np.shape[0]
    row_ind, col_ind, data = [], [], []
    for i in range(n):
        for j in range(k):
            jidx = indices[i, j]
            d = distances[i, j]
            row_ind.extend([i, jidx])
            col_ind.extend([jidx, i])
            data.extend([d, d])
    G = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))
    D = shortest_path(G, directed=False)
    D[np.isinf(D)] = np.nanmax(D[np.isfinite(D)]) * 2
    return D


def compute_dual_reference_distances(X_clean, ref_idx1, ref_idx2):
    """min( geodesic_dist_to_ref1, geodesic_dist_to_ref2 )"""
    X_np = X_clean.numpy()
    D = compute_manifold_distances(X_np)  # [N,N]
    d1 = D[ref_idx1]
    d2 = D[ref_idx2]
    return np.minimum(d1, d2)


# ------------------ Training (same as Cambridge) ------------------
def train_distance_function(model, X_data, X_clean, ref_idx1, ref_idx2,
                            epochs=1000, lr=0.002, batch_size=128):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Computing dual reference distances (geodesic)...")
    combined = compute_dual_reference_distances(X_clean, ref_idx1, ref_idx2)
    target = torch.tensor(combined, dtype=torch.float32).unsqueeze(1)
    target = target / (target.max() + 1e-12)

    ds = torch.utils.data.TensorDataset(X_data, target)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    loss_hist = []
    for epoch in range(epochs):
        ep_loss = 0.0
        for xb, yb in dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = nn.MSELoss()(pred, yb)
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
        avg = ep_loss / len(dl)
        loss_hist.append(avg)
        if epoch % 100 == 0 or epoch == epochs - 1:
            print(f"[{model.__class__.__name__}] epoch {epoch}  loss {avg:.4f}")
    return model, loss_hist


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


# ------------------ Visualization (same layout as Cambridge) ------------------
def compute_geodesic_distance_to_set(grid_points, manifold_points, n_neighbors=10):
    all_pts = np.vstack((grid_points, manifold_points))
    n_grid = grid_points.shape[0]
    G = kneighbors_graph(all_pts, n_neighbors=n_neighbors, mode='distance')
    D = shortest_path(G, method='auto')
    dgrid2man = D[:n_grid, n_grid:]
    return np.min(dgrid2man, axis=1).reshape(-1)


def plot_true_manifold_distance(X_data, X_clean, title, xlim, ylim, resolution=100):
    from scipy.ndimage import gaussian_filter
    xr = np.linspace(xlim[0], xlim[1], resolution)
    yr = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(xr, yr)
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    z = compute_geodesic_distance_to_set(grid, X_clean.numpy(), n_neighbors=10).reshape(resolution, resolution)
    z = gaussian_filter(z, sigma=1.0)

    levels = np.linspace(np.nanmin(z), np.nanmax(z), 15)
    plt.contourf(xx, yy, z, levels=levels, cmap='RdBu_r', alpha=0.7)
    plt.contour(xx, yy, z, levels=levels, colors='white', alpha=0.3, linewidths=0.5)

    X_np = X_data.numpy()
    plt.scatter(X_np[:, 0], X_np[:, 1], c='darkblue', s=12, alpha=0.8, edgecolors='white', linewidth=0.3)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim(xlim)
    plt.ylim(ylim)


def plot_neural_network_contour(model, X_data, title, xlim, ylim):
    model.eval()
    xr = np.linspace(xlim[0], xlim[1], 100)
    yr = np.linspace(ylim[0], ylim[1], 100)
    xx, yy = np.meshgrid(xr, yr)
    grid = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32)
    with torch.no_grad():
        z = model(grid).cpu().numpy().reshape(xx.shape)

    levels = np.linspace(z.min(), z.max(), 15)
    plt.contourf(xx, yy, z, levels=levels, cmap='RdBu_r', alpha=0.8)
    plt.contour(xx, yy, z, levels=levels, colors='white', alpha=0.3, linewidths=0.5)

    X_np = X_data.numpy()
    plt.scatter(X_np[:, 0], X_np[:, 1], c='darkblue', s=20, alpha=0.8, edgecolors='white', linewidth=0.5)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim(xlim)
    plt.ylim(ylim)


# ------------------ Main (Cambridge-style pipeline on Infinity) ------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Generating infinity symbol data...")
    # 原来是 6 个返回值 -> 改成 3 个
    # X_noisy, X_clean, t_params, labels, ref_left, ref_right = generate_infinity_symbol_data()
    X_noisy, X_clean, ref_indices = generate_infinity_symbol_data()

    # 单参考点（右端点）同时作为 ref1 和 ref2，等价于单参考点监督
    ref_idx = int(ref_indices[0])
    ref_left = ref_idx
    ref_right = ref_idx

    X_noisy = X_noisy.to(device)
    X_clean = X_clean.to(device)

    print(f"Ref index (right endpoint): {ref_idx}")
    print(f"Ref coords (clean): {X_clean[ref_idx].tolist()}")

    # -------- models --------
    models = {
        'NN': StandardNN(num_layers=8).to(device),
        'ICNN': ICNN(num_layers=8).to(device),
        'IWCNN Equal (4+4)': IWCNN(scenario='equal', total_layers=8).to(device),
        'IWCNN Shallow NS (2+6)': IWCNN(scenario='shallow_ns_deep_icnn', total_layers=8).to(device),
        'IWCNN Deep NS (6+2)': IWCNN(scenario='deep_ns_shallow_icnn', total_layers=8).to(device),
    }

    print("\nModel configs (parameter counts):")
    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name}: {n_params:,}")

    # -------- train all & collect losses --------
    trained, losses = {}, {}
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        mdl, hist = train_distance_function(
            model, X_noisy, X_clean, ref_left, ref_right,  # 这里传同一个索引两次
            epochs=1000, lr=0.002, batch_size=128
        )
        trained[name] = mdl
        losses[name] = hist

    # -------- unified loss plot（保留你之前的一张图五条曲线）--------
    plt.figure(figsize=(8, 6))
    for name, hist in losses.items():
        plt.plot(hist, label=name)
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss")
    plt.title("Loss Curves on Infinity Symbol (Cambridge)")
    plt.legend(); plt.grid(alpha=0.25); plt.tight_layout()
    plt.savefig("infinity_symbol_all_losses.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved combined loss curves as 'infinity_symbol_all_losses.png'")

    # -------- visualization（2×3）--------
    X_np = X_noisy.detach().cpu().numpy()
    margin = 0.5
    xlim = (X_np[:, 0].min() - margin, X_np[:, 0].max() + margin)
    ylim = (X_np[:, 1].min() - margin, X_np[:, 1].max() + margin)

    plt.figure(figsize=(18, 12))
    plt.suptitle('IWCNN Architecture Comparison on Infinity Symbol: Layer Distribution Effects',
                 fontsize=16, fontweight='bold')
    panels = [
        ('True', None),
        ('Standard NN', 'NN'),
        ('ICNN', 'ICNN'),
        ('IWCNN Equal (4+4)', 'IWCNN Equal (4+4)'),
        ('IWCNN Shallow NS (2+6)', 'IWCNN Shallow NS (2+6)'),
        ('IWCNN Deep NS (6+2)', 'IWCNN Deep NS (6+2)'),
    ]
    for i, (title, key) in enumerate(panels):
        plt.subplot(2, 3, i + 1)
        if key is None:
            plot_true_manifold_distance(X_noisy, X_clean, title, xlim, ylim, resolution=100)
        else:
            plot_neural_network_contour(trained[key], X_noisy, title, xlim, ylim)
    plt.tight_layout()
    plt.savefig('infinity_symbol_iwcnn_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Comparison plot saved as 'infinity_symbol_iwcnn_comparison.png'")


    # ----




if __name__ == "__main__":
    main()
