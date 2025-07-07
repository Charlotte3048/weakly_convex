import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.neighbors import NearestNeighbors

# Set seeds
np.random.seed(42)
torch.manual_seed(42)

# Generate Swiss Roll Data
n_samples = 1000
noise_level = 0.1

X1, _ = make_swiss_roll(n_samples//2, noise=0)
X2, _ = make_swiss_roll(n_samples//2, noise=0)
X1_2d = X1[:, [0, 2]]
X2_2d = X2[:, [0, 2]]
X2_2d[:, 0] = -X2_2d[:, 0] + 35
angle = np.pi
R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
X2_2d = X2_2d @ R.T
X = np.vstack((X1_2d, X2_2d)).astype(np.float32)
X = torch.tensor(X)
noise = torch.normal(0, noise_level, X.shape)
X_noisy = X + noise

# Proximal Operator
def prox_operator(x_init, model, alpha, steps=10, lr=0.1):
    x = x_init.clone().detach().requires_grad_(True)
    for _ in range(steps):
        energy = ((x - x_init)**2).sum(dim=1, keepdim=True) + alpha * model(x)
        grad = torch.autograd.grad(energy.sum(), x, create_graph=True)[0]
        x = x - lr * grad
    return x

# Training Function with WandB
def train_with_prox(model, X_clean, X_noisy, alpha=0.1, epochs=100, lr=1e-3, model_name='Model'):
    wandb.init(project="weakly-convex-regularizer", name=model_name, config={
        "alpha": alpha,
        "learning_rate": lr,
        "epochs": epochs,
        "model": model_name
    }, reinit=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        prox_points = prox_operator(X_noisy, model, alpha)
        mse_loss = ((prox_points - X_clean) ** 2).mean()
        loss = mse_loss

        loss.backward()
        optimizer.step()

        wandb.log({
            "epoch": epoch,
            "total_loss": loss.item(),
            "mse_loss": mse_loss.item()
        })

        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

    wandb.finish()
    return model

# Define Models
class Regularizer(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        out = self.fc3(out)
        return out

class ICNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64):
        super(ICNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.fc3 = nn.Linear(hidden_dim, 1, bias=False)
        self.act = nn.ReLU()
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.uniform_(layer.weight, a=0.01, b=0.1)

    def forward(self, x):
        self.fc1.weight.data.clamp_(min=0)
        self.fc2.weight.data.clamp_(min=0)
        self.fc3.weight.data.clamp_(min=0)
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        out = self.fc3(out)
        return out

class IWCNN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, lambda_reg=0.1):
        super(IWCNN, self).__init__()
        self.lambda_reg = lambda_reg
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.act = nn.Softplus()
        self.quadratic = nn.Linear(input_dim, 1, bias=False)
        nn.init.constant_(self.quadratic.weight, 0.1)

    def forward(self, x):
        out = self.act(self.fc1(x))
        out = self.act(self.fc2(out))
        out = self.fc3(out)
        quad = 0.5 * torch.sum(x * (self.quadratic.weight * x), dim=1, keepdim=True)
        return out + self.lambda_reg * quad

# Train All Models
reg_model = Regularizer()
icnn_model = ICNN()
iwcnn_model = IWCNN()

reg_model = train_with_prox(reg_model, X, X_noisy, alpha=0.1, epochs=100, lr=1e-3, model_name='NN')
icnn_model = train_with_prox(icnn_model, X, X_noisy, alpha=0.1, epochs=100, lr=1e-3, model_name='ICNN')
iwcnn_model = train_with_prox(iwcnn_model, X, X_noisy, alpha=0.1, epochs=100, lr=1e-3, model_name='IWCNN')

# Visualization
x_min, x_max = X[:,0].min()-1, X[:,0].max()+1
y_min, y_max = X[:,1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)

nbrs = NearestNeighbors(n_neighbors=1).fit(X.numpy())
dists, _ = nbrs.kneighbors(grid.numpy())
dist_true = dists.reshape(xx.shape)

reg_pred = reg_model(grid).detach().numpy().reshape(xx.shape)
icnn_pred = icnn_model(grid).detach().numpy().reshape(xx.shape)
iwcnn_pred = iwcnn_model(grid).detach().numpy().reshape(xx.shape)

fig, axs = plt.subplots(2, 2, figsize=(12, 10))
cf0 = axs[0,0].contourf(xx, yy, gaussian_filter(dist_true, sigma=1), levels=20, cmap='RdBu_r')
axs[0,0].scatter(X[:,0], X[:,1], c='red', s=5)
axs[0,0].set_title('True Manifold')
plt.colorbar(cf0, ax=axs[0,0])

cf1 = axs[0,1].contourf(xx, yy, gaussian_filter(reg_pred, sigma=1), levels=20, cmap='RdBu_r')
axs[0,1].scatter(X[:,0], X[:,1], c='red', s=5)
axs[0,1].set_title('NN Prox')
plt.colorbar(cf1, ax=axs[0,1])

cf2 = axs[1,0].contourf(xx, yy, gaussian_filter(icnn_pred, sigma=1), levels=20, cmap='RdBu_r')
axs[1,0].scatter(X[:,0], X[:,1], c='red', s=5)
axs[1,0].set_title('ICNN Prox')
plt.colorbar(cf2, ax=axs[1,0])

cf3 = axs[1,1].contourf(xx, yy, gaussian_filter(iwcnn_pred, sigma=1), levels=20, cmap='RdBu_r')
axs[1,1].scatter(X[:,0], X[:,1], c='red', s=5)
axs[1,1].set_title('IWCNN Prox')
plt.colorbar(cf3, ax=axs[1,1])

plt.tight_layout()
plt.savefig('swiss_roll_result.png', dpi=300)
