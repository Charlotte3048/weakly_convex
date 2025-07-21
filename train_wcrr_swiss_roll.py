import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models.wc_conv_net import WCvxConvNet
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter

# ---------- Swiss Roll Data Generator ----------
def generate_swiss_roll(n_samples=1000, noise_level=0.2, grid_size=63):
    from sklearn.datasets import make_swiss_roll

    X1, _ = make_swiss_roll(n_samples // 2, noise=0)
    X2, _ = make_swiss_roll(n_samples // 2, noise=0)

    X1_2d = X1[:, [0, 2]]
    X2_2d = X2[:, [0, 2]]
    X2_2d[:, 0] = -X2_2d[:, 0] + 35

    angle = np.pi
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    X2_2d = X2_2d @ R.T

    X = np.vstack((X1_2d, X2_2d)).astype(np.float32)
    X = torch.tensor(X)

    x_center = (X[:, 0].min() + X[:, 0].max()) / 2
    X[:, 0] -= x_center

    noise = torch.normal(0, noise_level, X.shape)
    X_noisy = X + noise

    # Reshape into grid
    X_clean = X[:grid_size**2].reshape(grid_size, grid_size, 2)
    X_noisy = X_noisy[:grid_size**2].reshape(grid_size, grid_size, 2)

    return X_noisy, X_clean

# ---------- Fix Shared Storage Issue ----------
def ensure_no_shared_storage(model):
    for name, param in model.named_parameters():
        param.data = param.data.clone().contiguous()

# ---------- Training Function ----------
def train_model(model, dataloader, epochs=500, lr=1e-3, save_path="trained_models/wcrr_swiss_roll.pth"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ensure_no_shared_storage(model)

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

            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f"{name}: grad mean = {param.grad.abs().mean().item():.6e}")
            #     else:
            #         print(f"{name}: grad is None")

            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.6f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

# ---------- Main ----------
if __name__ == "__main__":
    n_samples = 5000
    noise_level = 0.2
    size = 63

    X_noisy, X_clean = generate_swiss_roll(n_samples=n_samples, noise_level=noise_level, grid_size=size)

    # Use x-component as scalar image input/output
    X_noisy_img = X_noisy[..., 0].unsqueeze(0).unsqueeze(0)  # shape: [1, 1, H, W]
    X_clean_img = X_clean[..., 0].unsqueeze(0).unsqueeze(0)

    dataset = TensorDataset(X_noisy_img, X_clean_img)
    dataloader = DataLoader(dataset, batch_size=1)

    # Model config
    param_multi_conv = {
        "size_kernels": [5, 5, 5],
        "num_channels": [1, 4, 8, 60]
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

    model.use_wx_for_scaling = True  # 关键：使用 wx 作为 scaling 输入

    print("Multi convolutionnal layer: ", param_multi_conv)
    train_model(model, dataloader, epochs=100, lr=1e-3, save_path="trained_models/wcrr_swiss_roll.pth")

