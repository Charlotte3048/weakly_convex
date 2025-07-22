import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generate batchable Swiss Roll data
n_samples = 1000
noise_level = 0.2
batch_size = 32

X1, _ = make_swiss_roll(n_samples // 2, noise=0)
X2, _ = make_swiss_roll(n_samples // 2, noise=0)
X1_2d = X1[:, [0, 2]]
X2_2d = X2[:, [0, 2]]
X2_2d[:, 0] = -X2_2d[:, 0] + 35
angle = np.pi
R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
X2_2d = X2_2d @ R.T
X = np.vstack((X1_2d, X2_2d)).astype(np.float32)
X = torch.tensor(X).to(device)

# Center the data
x_coords = X[:, 0]
x_center = (x_coords.min() + x_coords.max()) / 2
X[:, 0] += 15

# Add noise
noise = torch.normal(0, noise_level, X.shape).to(device)
X_noisy = X + noise

# Compute distance from each noisy point to the clean manifold
nbrs_noise = NearestNeighbors(n_neighbors=1).fit(X.cpu().numpy())
dists_noise, _ = nbrs_noise.kneighbors(X_noisy.cpu().numpy())

# Identify noise points based on a threshold
threshold = 0.05
is_noise = (dists_noise > threshold).astype(np.bool_).reshape(-1)  # [N]

# Log number and ratio of noise points
noise_indices = np.where(is_noise)[0]
print(f"Detected noise points: {len(noise_indices)} / {len(X_noisy)} ({len(noise_indices)/len(X_noisy)*100:.2f}%)")

# Create batched dataset
dataset = torch.utils.data.TensorDataset(X_noisy, X)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Proximal operator for training
def prox_operator(x_init, model, alpha, steps=5, lr=0.05):
    x = x_init.clone().detach().requires_grad_(True).to(device)
    for _ in range(steps):
        energy = ((x - x_init) ** 2).sum(dim=1, keepdim=True) + alpha * model(x)
        grad = torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
        x = x - lr * grad
    return x

# Training loop supporting batches
def train_with_prox(model, dataloader, alpha=0.1, epochs=100, lr=1e-3, model_name='Model'):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x_noisy, x_clean in dataloader:
            x_noisy, x_clean = x_noisy.to(device), x_clean.to(device)
            optimizer.zero_grad()
            prox_points = prox_operator(x_noisy, model, alpha)
            mse_loss = ((prox_points - x_clean) ** 2).mean()
            loss = mse_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.6f}")

    return model

# Simple fully connected regressor
class NN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128):
        super(NN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)

    def forward(self, x):
        return self.network(x)

# Input Convex Neural Network
class ICNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(ICNN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim).to(device)
        self.hidden_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim, bias=False).to(device) for _ in range(2)])
        self.input_skips = nn.ModuleList([nn.Linear(input_dim, hidden_dim).to(device) for _ in range(2)])
        self.output_layer = nn.Linear(hidden_dim, 1, bias=False).to(device)
        self.output_skip = nn.Linear(input_dim, 1, bias=True).to(device)
        self.activation = nn.ReLU()

    def forward(self, x):
        z = self.activation(self.input_layer(x))
        for Wz, Wx in zip(self.hidden_layers, self.input_skips):
            Wz.weight.data.clamp_(min=0)
            z = self.activation(Wz(z) + Wx(x))
        self.output_layer.weight.data.clamp_(min=0)
        return self.output_layer(z) + self.output_skip(x)

# IWCNN: Input Weakly Convex Network
class IWCNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, lambda_reg=0.2):
        super(IWCNN, self).__init__()
        self.lambda_reg = lambda_reg
        self.smooth = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden_dim, hidden_dim)
        ).to(device)
        self.icnn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.05),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        for layer in self.icnn:
            if isinstance(layer, nn.Linear):
                layer.weight.data.clamp_(min=0)
        self.quadratic = nn.Linear(input_dim, 1, bias=False).to(device)
        with torch.no_grad():
            self.quadratic.weight.data = torch.ones_like(self.quadratic.weight.data) * 0.1

    def forward(self, x):
        smooth_out = self.smooth(x)
        convex_out = self.icnn(smooth_out)
        quad_term = 0.5 * torch.sum(x * self.quadratic.weight * x, dim=1, keepdim=True)
        return convex_out + self.lambda_reg * quad_term

# Train all models
nn_model = NN().to(device)
icnn_model = ICNN().to(device)
iwcnn_model = IWCNN().to(device)

nn_model = train_with_prox(nn_model, dataloader, alpha=0.2, epochs=200, lr=5e-4, model_name='NN')
icnn_model = train_with_prox(icnn_model, dataloader, alpha=0.1, epochs=100, lr=1e-3, model_name='ICNN')
iwcnn_model = train_with_prox(iwcnn_model, dataloader, alpha=0.1, epochs=200, lr=1e-3, model_name='IWCNN')

# Visualization
x_min, x_max = X[:, 0].min().item() - 1, X[:, 0].max().item() + 1
y_min, y_max = X[:, 1].min().item() - 1, X[:, 1].max().item() + 1
xx, yy = np.meshgrid(np.linspace(-35, 35, 100), np.linspace(y_min, y_max, 100))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32).to(device)

# Compute true manifold distance for visualization
nbrs = NearestNeighbors(n_neighbors=1).fit(X.cpu().numpy())
dists, _ = nbrs.kneighbors(grid.cpu().numpy())
dist_true = dists.reshape(xx.shape)

# Evaluate trained models on grid
nn_model.eval()
icnn_model.eval()
iwcnn_model.eval()
with torch.no_grad():
    reg_pred = nn_model(grid).cpu().numpy().reshape(xx.shape)
    icnn_pred = icnn_model(grid).cpu().numpy().reshape(xx.shape)
    iwcnn_pred = iwcnn_model(grid).cpu().numpy().reshape(xx.shape)

# Plot results
fig, axs = plt.subplots(2, 2, figsize=(16, 10))
cf0 = axs[0, 0].contourf(xx, yy, gaussian_filter(dist_true, sigma=1), levels=10, cmap='RdBu_r')
axs[0, 0].scatter(X[:, 0].cpu().numpy(), X[:, 1].cpu().numpy(), c='red', s=5)
axs[0, 0].set_title('True Manifold')
plt.colorbar(cf0, ax=axs[0, 0])

cf1 = axs[0, 1].contourf(xx, yy, gaussian_filter(reg_pred, sigma=1), levels=10, cmap='RdBu_r')
axs[0, 1].scatter(X[:, 0].cpu().numpy(), X[:, 1].cpu().numpy(), c='red', s=5)
axs[0, 1].set_title('NN Prox')
plt.colorbar(cf1, ax=axs[0, 1])

cf2 = axs[1, 0].contourf(xx, yy, gaussian_filter(icnn_pred, sigma=1), levels=10, cmap='RdBu_r')
axs[1, 0].scatter(X[:, 0].cpu().numpy(), X[:, 1].cpu().numpy(), c='red', s=5)
axs[1, 0].set_title('ICNN Prox')
plt.colorbar(cf2, ax=axs[1, 0])

cf3 = axs[1, 1].contourf(xx, yy, gaussian_filter(iwcnn_pred, sigma=1), levels=10, cmap='RdBu_r')
axs[1, 1].scatter(X[:, 0].cpu().numpy(), X[:, 1].cpu().numpy(), c='red', s=5)
axs[1, 1].set_title('IWCNN Prox')
plt.colorbar(cf3, ax=axs[1, 1])

plt.tight_layout()
plt.savefig('swiss_roll_result.png', dpi=300)
plt.close()
