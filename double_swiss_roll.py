import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_swiss_roll
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端避免PyCharm兼容性问题
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import shortest_path

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate batchable Swiss Roll data
batch_size = 32

class StandardNN(nn.Module):
    """标准神经网络"""

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
    """输入凸神经网络 (Input Convex Neural Network)"""

    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # 第一层：正常的线性层
        self.first_layer = nn.Linear(input_dim, hidden_dim)

        # 中间层：权重必须为非负
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(2)
        ])

        # 跳跃连接层（从输入直接到每个隐藏层）
        self.skip_layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim) for _ in range(2)
        ])

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.output_skip = nn.Linear(input_dim, 1)

    def forward(self, x):
        # 第一层
        h = torch.relu(self.first_layer(x))

        # 中间层（确保权重非负）
        for i, (hidden_layer, skip_layer) in enumerate(zip(self.hidden_layers, self.skip_layers)):
            # 确保隐藏层权重非负
            with torch.no_grad():
                hidden_layer.weight.data = torch.clamp(hidden_layer.weight.data, min=0)

            h = torch.relu(hidden_layer(h) + skip_layer(x))

        # 输出层（确保权重非负）
        with torch.no_grad():
            self.output_layer.weight.data = torch.clamp(self.output_layer.weight.data, min=0)

        return self.output_layer(h) + self.output_skip(x)


class IWCNN(nn.Module):
    """输入弱凸神经网络 (Input Weakly Convex Neural Network)"""

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

        # 用于计算弱凸性正则化的二次项
        self.quadratic = nn.Linear(input_dim, 1, bias=False)
        with torch.no_grad():
            # 初始化为对角矩阵的权重
            self.quadratic.weight.data = torch.ones(1, input_dim) * 0.1

    def forward(self, x):
        network_out = self.network(x)
        quadratic_out = 0.5 * torch.sum(x * (self.quadratic.weight * x), dim=1, keepdim=True)
        return network_out + self.lambda_reg * quadratic_out


def generate_swiss_roll_data():
    """生成双螺旋瑞士卷数据"""

    n_samples = 1000
    noise_level = 0.1

    # 生成第一个瑞士卷
    X1, _ = make_swiss_roll(n_samples // 2, noise=0, random_state=42)
    X1_2d = X1[:, [0, 2]]

    # 生成第二个瑞士卷
    X2, _ = make_swiss_roll(n_samples // 2, noise=0, random_state=42)
    X2_2d = X2[:, [0, 2]]

    # 变换第二个螺旋
    X2_2d[:, 0] = -X2_2d[:, 0]
    X2_2d[:, 0] += 45

    angle = np.pi
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    X2_2d = X2_2d @ rotation_matrix.T

    # 合并数据
    X = np.vstack((X1_2d, X2_2d))
    # 数据中心化
    x_coords = X[:, 0]
    x_center = (x_coords.min() + x_coords.max()) / 2
    X[:, 0] -= x_center

    # 添加噪声
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise

    return torch.tensor(X_noisy, dtype=torch.float32), torch.tensor(X, dtype=torch.float32)


def compute_manifold_distances(X_clean):
    """计算流形上的测地距离（使用k近邻图近似）"""
    from sklearn.neighbors import NearestNeighbors
    from scipy.sparse.csgraph import shortest_path
    from scipy.sparse import csr_matrix

    # 构建k近邻图
    k = 15  # 增加邻居数量以确保连通性
    nbrs = NearestNeighbors(n_neighbors=k).fit(X_clean)
    distances, indices = nbrs.kneighbors(X_clean)

    # 构建对称的稀疏距离矩阵
    n = X_clean.shape[0]
    row_ind = []
    col_ind = []
    data = []

    for i in range(n):
        for j in range(k):
            neighbor_idx = indices[i, j]
            dist = distances[i, j]

            # 添加双向边
            row_ind.extend([i, neighbor_idx])
            col_ind.extend([neighbor_idx, i])
            data.extend([dist, dist])

    distance_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(n, n))

    # 计算最短路径（测地距离）
    geodesic_distances = shortest_path(distance_matrix, directed=False)

    # 处理无穷大值
    geodesic_distances[np.isinf(geodesic_distances)] = geodesic_distances[np.isfinite(geodesic_distances)].max() * 2

    return geodesic_distances


def train_distance_function(model, X_data, X_clean, epochs=1000, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("Computing manifold distances...")
    manifold_distances = compute_manifold_distances(X_clean.numpy())
    center_idx = np.argmin(np.sum((X_clean.numpy() - X_clean.numpy().mean(axis=0)) ** 2, axis=1))
    target_distances = torch.tensor(manifold_distances[center_idx], dtype=torch.float32).unsqueeze(1)

    max_dist = target_distances.max()
    if max_dist > 0 and torch.isfinite(max_dist):
        target_distances = target_distances / max_dist
    else:
        center_point = X_clean[center_idx:center_idx + 1]
        target_distances = torch.norm(X_data - center_point, dim=1, keepdim=True)
        target_distances = target_distances / target_distances.max()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        pred_distances = model(X_data)
        loss_mse = nn.MSELoss()(pred_distances, target_distances)

        # 对称性损失：沿 x=0 轴镜像
        # X_mirrored = X_data.clone()
        # X_mirrored[:, 0] = -X_mirrored[:, 0]  # 沿 x=0 镜像
        # pred_distances_mirrored = model(X_mirrored)
        # symmetry_loss = torch.mean((pred_distances - pred_distances_mirrored) ** 2)

        # 总损失
        total_loss = loss_mse #+ 0.005 * symmetry_loss  # 0.01 为对称性损失权重，可调整

        total_loss.backward()
        optimizer.step()

        if epoch % 200 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}, MSE Loss: {loss_mse.item():.4f}")

    return model


def plot_neural_network_contour(model, X_data, title, xlim=(-30, 30), ylim=(-30, 30)):
    """创建等高线图（专门用于子图）"""
    model.eval()

    # 创建网格
    x_range = np.linspace(xlim[0], xlim[1], 100)
    y_range = np.linspace(ylim[0], ylim[1], 100)
    xx, yy = np.meshgrid(x_range, y_range)

    # 计算网格点的距离函数值
    grid_points = torch.tensor(np.column_stack([xx.ravel(), yy.ravel()]), dtype=torch.float32)

    with torch.no_grad():
        z_values = model(grid_points).numpy().reshape(xx.shape)

    # 绘制等高线
    levels = np.linspace(z_values.min(), z_values.max(), 15)
    plt.contourf(xx, yy, z_values, levels=levels, cmap='RdBu_r', alpha=0.8)
    plt.contour(xx, yy, z_values, levels=levels, colors='white', alpha=0.3, linewidths=0.5)

    # 绘制数据点
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


def plot_true_manifold_distance(X_data, X_clean, title, xlim=(-20, 40), ylim=(-30, 30), resolution=100):
    """使用图距离绘制真实 manifold 距离图，改进为 geodesic-based True 面板"""
    from scipy.ndimage import gaussian_filter

    # 创建网格点
    x_range = np.linspace(xlim[0], xlim[1], resolution)
    y_range = np.linspace(ylim[0], ylim[1], resolution)
    xx, yy = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    # 使用 geodesic 距离计算每个 grid 点到流形的最短图距离
    geodesic_dists = compute_geodesic_distance(grid_points, X_clean.numpy(), n_neighbors=10)
    z = geodesic_dists.reshape(resolution, resolution)
    z = gaussian_filter(z, sigma=1.0)

    # 绘图
    levels = np.linspace(np.nanmin(z), np.nanmax(z), 15)
    plt.contourf(xx, yy, z, levels=levels, cmap='RdBu_r', alpha=0.7)
    plt.contour(xx, yy, z, levels=levels, colors='white', alpha=0.3, linewidths=0.5)

    # 原始点可视化
    X_np = X_data.numpy()
    plt.scatter(X_np[:, 0], X_np[:, 1], c='darkblue', s=12, alpha=0.8,
                edgecolors='white', linewidth=0.3)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlim(xlim)
    plt.ylim(ylim)


def main():
    """主函数：复现Figure 1"""
    # 生成数据
    print("Generating Swiss roll data...")
    X_noisy, X_clean = generate_swiss_roll_data()

    # 计算真实的流形距离（用于True面板）
    print("Computing true manifold distances...")
    manifold_distances = compute_manifold_distances(X_clean.numpy())
    center_idx = np.argmin(np.sum((X_clean.numpy() - X_clean.numpy().mean(axis=0)) ** 2, axis=1))
    true_distances = manifold_distances[center_idx]

    # 安全归一化
    max_dist = np.max(true_distances[np.isfinite(true_distances)])
    if max_dist > 0:
        true_distances = true_distances / max_dist
    else:
        # 备选方案：使用欧几里得距离
        center_point = X_clean.numpy()[center_idx]
        true_distances = np.linalg.norm(X_clean.numpy() - center_point, axis=1)
        true_distances = true_distances / true_distances.max()

    # 初始化模型
    models = {
        'NN': StandardNN(),
        'ICNN': ICNN(),
        'IWCNN': IWCNN()
    }

    # 训练模型
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        trained_models[name] = train_distance_function(model, X_noisy, X_clean)

    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Distance Function Approximation Comparison', fontsize=16, fontweight='bold')

    # 计算统一的xlim和ylim
    X_np = X_noisy.numpy()
    xlim = (X_np[:, 0].min() - 5, X_np[:, 0].max() + 5)
    ylim = (X_np[:, 1].min() - 5, X_np[:, 1].max() + 5)

    # True距离（左上）
    plt.subplot(2, 2, 1)
    plot_true_manifold_distance(X_noisy, X_clean, 'True', xlim, ylim)

    # NN approximation（右上）
    plt.subplot(2, 2, 2)
    plot_neural_network_contour(trained_models['NN'], X_noisy, 'NN approximation', xlim, ylim)

    # ICNN approximation（左下）
    plt.subplot(2, 2, 3)
    plot_neural_network_contour(trained_models['ICNN'], X_noisy, 'ICNN approximation', xlim, ylim)

    # IWCNN approximation（右下）
    plt.subplot(2, 2, 4)
    plot_neural_network_contour(trained_models['IWCNN'], X_noisy, 'IWCNN approximation', xlim, ylim)

    plt.tight_layout()

    plt.savefig('double_swiss_roll_distance_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'double_swiss_roll_distance_comparison.png'")

    print("\nTraining completed! The plots show the comparison between different neural network architectures:")
    print("- True: Ground truth manifold distance")
    print("- NN: Standard neural network approximation")
    print("- ICNN: Input Convex Neural Network approximation")
    print("- IWCNN: Input Weakly Convex Neural Network approximation")


if __name__ == "__main__":
    main()
