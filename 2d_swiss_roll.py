import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)

# 步骤 1：生成2D双螺旋数据（沿 y 轴对称）
n_samples = 1000
noise_level = 0.1
random_state = 42
# 生成第一个瑞士卷
X1, _ = make_swiss_roll(n_samples//2, noise=0, random_state=random_state)
X1_2d = X1[:, [0, 2]]  # 取 x 和 z 坐标作为 2D 表示

# 生成第二个瑞士卷（使用不同的随机种子避免完全相同）
X2, _ = make_swiss_roll(n_samples//2, noise=0, random_state=random_state)
X2_2d = X2[:, [0, 2]]
# 第二个螺旋的变换（消除重叠）
X2_2d[:, 0] = -X2_2d[:, 0]  # 沿 x 轴镜像
X2_2d[:, 0] += 35  # 增大y轴偏移量，确保不重叠

angle = np.pi   # 60度旋转
rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                           [np.sin(angle), np.cos(angle)]])
X2_2d = X2_2d @ rotation_matrix.T

# 合并成双螺旋
X = np.vstack((X1_2d, X2_2d))
# 转换为tensor
X = torch.tensor(X, dtype=torch.float32)
# 添加噪声
noise = torch.normal(0, noise_level, X.shape)
X_noisy = X + noise

# 步骤 2：定义正则化器模型（2D输入）
class ICNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(ICNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.layer1.weight.data = torch.abs(self.layer1.weight.data)
        self.layer2.weight.data = torch.abs(self.layer2.weight.data)
        self.layer3.weight.data = torch.abs(self.layer3.weight.data)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class NN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(NN, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class IWCNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(IWCNN, self).__init__()
        self.smooth = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.icnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        for layer in self.icnn:
            if isinstance(layer, nn.Linear):
                layer.weight.data = torch.abs(layer.weight.data)
    def forward(self, x):
        x = self.smooth(x)
        x = self.icnn(x)
        return x

# 步骤 3：训练正则化器（每10个epoch打印一次损失，添加分组约束）
def train_regularizer(model, X, X_noisy, alpha=0.1, beta=0.05, epochs=100, lr=1e-3):
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    n_samples = X.shape[0] // 2  # 每个瑞士卷有 1000 个样本
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_noisy)
        # 基本损失：MSE + 正则化
        loss_mse = criterion(X_noisy, X)
        loss_reg = alpha * output.mean()
        loss_base = 0.5 * loss_mse + loss_reg
        # 分组一致性损失：鼓励 X1 和 X2 对称输出
        output1 = model(X_noisy[:n_samples])  # 第一个瑞士卷的输出
        output2 = model(X_noisy[n_samples:])  # 第二个瑞士卷的输出
        # 计算镜像对称性损失：X2 的 y 轴镜像应与 X1 的输出一致
        symmetry_loss = criterion(output1, -output2)  # 假设 y 轴对称性反映在输出上
        loss = loss_base + beta * symmetry_loss
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, MSE: {loss_mse.item():.4f}, Symmetry: {symmetry_loss.item():.4f}")
    return model

# 初始化并训练模型
icnn = ICNN()
nn_model = NN()
iwcnn = IWCNN()
icnn = train_regularizer(icnn, X, X_noisy, alpha=0.1, beta=0.05, epochs=100, lr=5e-5)
nn_model = train_regularizer(nn_model, X, X_noisy, alpha=0.1, beta=0.05, epochs=100, lr=5e-5)
iwcnn = train_regularizer(iwcnn, X, X_noisy, alpha=0.1, beta=0.05, epochs=100, lr=5e-5)

# 步骤 4：计算流形距离
def compute_manifold_distance(X, X_query):
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(X.numpy())
    distances, _ = nbrs.kneighbors(X_query.numpy())
    return distances.flatten()

# 步骤 5：创建2D网格以绘制切面图
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_2d = np.c_[xx.ravel(), yy.ravel()]
grid_2d = torch.tensor(grid_2d, dtype=torch.float32)

# 计算每种方法的预测值
manifold_dist_2d = compute_manifold_distance(X, grid_2d).reshape(xx.shape)
icnn_values_2d = icnn(grid_2d).detach().numpy().flatten().reshape(xx.shape)
nn_values_2d = nn_model(grid_2d).detach().numpy().flatten().reshape(xx.shape)
iwcnn_values_2d = iwcnn(grid_2d).detach().numpy().flatten().reshape(xx.shape)

# 步骤 6：生成二维切面图
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X[:, 0], X[:, 1], c='red', s=5, label='Double 2D Swiss Roll')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('2D Cross Section of Double Swiss Roll')
ax.legend()
plt.savefig('swiss_roll_2d_cross_section.png', dpi=600, bbox_inches='tight')
plt.close()

# 步骤 7：生成四种方法的二维等高线图（单张画面）
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

# True (流形距离)
cf0 = axs[0].contourf(xx, yy, manifold_dist_2d, levels=20, cmap='viridis')
axs[0].scatter(X[:, 0], X[:, 1], c='red', s=5, label='Double 2D Swiss Roll')
axs[0].set_title('True Manifold Distance')
plt.colorbar(cf0, ax=axs[0])
axs[0].legend()

# ICNN
cf1 = axs[1].contourf(xx, yy, icnn_values_2d, levels=20, cmap='viridis')
axs[1].scatter(X[:, 0], X[:, 1], c='red', s=5, label='Double 2D Swiss Roll')
axs[1].set_title('ICNN')
plt.colorbar(cf1, ax=axs[1])
axs[1].legend()

# NN
cf2 = axs[2].contourf(xx, yy, nn_values_2d, levels=20, cmap='viridis')
axs[2].scatter(X[:, 0], X[:, 1], c='red', s=5, label='Double 2D Swiss Roll')
axs[2].set_title('NN')
plt.colorbar(cf2, ax=axs[2])
axs[2].legend()

# IWCNN
cf3 = axs[3].contourf(xx, yy, iwcnn_values_2d, levels=20, cmap='viridis')
axs[3].scatter(X[:, 0], X[:, 1], c='red', s=5, label='Double 2D Swiss Roll')
axs[3].set_title('IWCNN')
plt.colorbar(cf3, ax=axs[3])
axs[3].legend()

plt.tight_layout()
plt.savefig('swiss_roll_2d_methods.png', dpi=600, bbox_inches='tight')
plt.close()