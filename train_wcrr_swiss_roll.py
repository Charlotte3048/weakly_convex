import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.wc_conv_net import WCvxConvNet
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# ---------- Swiss Roll Data Generator ----------
def generate_swiss_roll(n_samples=1000, noise_level=0.1, grid_size=25):
    from sklearn.datasets import make_swiss_roll

    # Set n_samples to grid_size ** 2 to match the reshape size
    n_samples = grid_size ** 2
    # Adjust samples for X1 and X2 to ensure total equals n_samples
    n_samples_x1 = (n_samples + 1) // 2  # Ceiling division for X1
    n_samples_x2 = n_samples - n_samples_x1  # Remaining samples for X2
    # Generate two Swiss Roll datasets without noise
    X1, _ = make_swiss_roll(n_samples_x1, noise=0)
    X2, _ = make_swiss_roll(n_samples_x2, noise=0)

    # Extract x and z coordinates (2D projection)
    X1_2d = X1[:, [0, 2]]
    X2_2d = X2[:, [0, 2]]
    # Flip and shift X2 along x-axis for separation
    X2_2d[:, 0] = -X2_2d[:, 0] + 35

    # Define rotation matrix for X2 (rotate by pi radians)
    angle = np.pi
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    # Apply rotation to X2_2d
    X2_2d = X2_2d @ R.T

    # Stack X1_2d and X2_2d, convert to float32 for PyTorch
    X = np.vstack((X1_2d, X2_2d)).astype(np.float32)
    # Convert to PyTorch tensor
    X = torch.tensor(X)

    # Center the x-coordinates
    x_center = (X[:, 0].min() + X[:, 0].max()) / 2
    X[:, 0] -= x_center

    # Add Gaussian noise with specified noise level
    noise = torch.normal(0, noise_level, X.shape)
    X_noisy = X + noise

    # Reshape data into grid format [grid_size, grid_size, 2]
    X_clean = X[:grid_size**2].reshape(grid_size, grid_size, 2)
    X_noisy = X_noisy[:grid_size**2].reshape(grid_size, grid_size, 2)

    # Return noisy and clean data tensors
    return X_noisy, X_clean

# Ensure model parameters are contiguous and independent
def ensure_no_shared_storage(model):
    for name, param in model.named_parameters():
        param.data = param.data.clone().contiguous()

# ---------- Training Function ----------
def train_model(model, dataloader, epochs=300, lr=1e-3, save_path="trained_models/wcrr_swiss_roll.pth"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ensure_no_shared_storage(model)

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for x_noisy, x_clean in dataloader:
            x_noisy = x_noisy.contiguous()
            x_clean = x_clean.contiguous()
            sigma = torch.full_like(x_noisy[:, :1, :, :], 0.2).clone()

            optimizer.zero_grad()
            output = model.grad_denoising(x_noisy, x_noisy, sigma=sigma)
            loss = criterion(output, x_clean)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 100 == 0 or epoch == epochs - 1:
            print(f"epoch {epoch + 1}, loss: {total_loss:.6f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# ---------- Visualization Function ----------
def visualize(model, X_noisy, X_clean):
    grid_size = X_noisy.shape[0]

    # Create meshgrid for evaluation
    x = torch.linspace(-35, 35, grid_size)
    y = torch.linspace(-20, 20, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)  # [H, W, 2]
    grid_tensor = grid.permute(2, 0, 1).unsqueeze(0)  # [1, 2, H, W]
    sigma_val = torch.full((1, 1, grid_size, grid_size), 0.2)

    # Inference
    model.eval()
    with torch.no_grad():
        prox = model.grad_denoising(grid_tensor, grid_tensor, sigma=sigma_val)

    prox_np = prox.squeeze().permute(1, 2, 0).reshape(-1, 2).numpy()
    gt_np = X_clean.reshape(-1, 2).numpy()

    # Nearest neighbor distance to clean manifold
    nbrs_true = NearestNeighbors(n_neighbors=1).fit(gt_np)
    dists_true, _ = nbrs_true.kneighbors(grid_tensor.squeeze().permute(1, 2, 0).reshape(-1, 2).numpy())
    dists_true = dists_true.reshape(grid_size, grid_size)

    nbrs_model = NearestNeighbors(n_neighbors=1).fit(prox_np)
    dists_model, _ = nbrs_model.kneighbors(grid_tensor.squeeze().permute(1, 2, 0).reshape(-1, 2).numpy())
    dists_model = dists_model.reshape(grid_size, grid_size)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    im1 = axes[0].contourf(xx, yy, dists_true, levels=10, cmap='RdBu_r')
    axes[0].scatter(X_noisy[..., 0], X_noisy[..., 1], s=5, color='r')
    axes[0].set_title("Ground Truth Distance")
    fig.colorbar(im1, ax=axes[0])

    im2 = axes[1].contourf(xx, yy, dists_model, levels=10, cmap='RdBu_r')
    axes[1].scatter(X_noisy[..., 0], X_noisy[..., 1], s=5, color='r')
    axes[1].set_title("Fitted Manifold by WCRR")
    fig.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig("EPFL_swiss_roll.png")

# ---------- Main ----------
if __name__ == "__main__":
    size = 63
    X_noisy, X_clean = generate_swiss_roll(grid_size=size)

    # Use full 2D field as input/output
    X_noisy_img = X_noisy.permute(2, 0, 1).unsqueeze(0)  # [1, 2, H, W]
    X_clean_img = X_clean.permute(2, 0, 1).unsqueeze(0)

    dataset = TensorDataset(X_noisy_img, X_clean_img)
    dataloader = DataLoader(dataset, batch_size=1)

    param_multi_conv = {
        "size_kernels": [5, 5, 5],
        "num_channels": [2, 4, 8, 60]
    }

    param_spline_activation = {
        "num_activations": 60,
        "num_knots": 100,
        "x_min": -2.0,
        "x_max": 2.0,
        "init": "leaky_relu",
        "slope_min": 0.01,
        "slope_max": None,
        "antisymmetric": False,
        "clamp": True
    }

    param_spline_scaling = {
        "num_activations": 60,
        "num_knots": 100,
        "x_min": -2.0,
        "x_max": 2.0,
        "init": "identity",
        "slope_min": 0.01,
        "slope_max": None,
        "antisymmetric": False,
        "clamp": True
    }

    model = WCvxConvNet(
        param_multi_conv=param_multi_conv,
        param_spline_activation=param_spline_activation,
        param_spline_scaling=param_spline_scaling,
        rho_wcvx=1.0
    )
    model.use_wx_for_scaling = True

    print("Multi convolutionnal layer: ", param_multi_conv)
    train_model(model, dataloader, epochs=3000, lr=1e-3, save_path="trained_models/wcrr_swiss_roll.pth")
    visualize(model, X_noisy, X_clean)
