# train_wcrr_swiss_roll_full.py
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# ================================================================
# Utils
# ================================================================
def plot_loss_curve(loss_history, title="Training Loss Curve", save_path="loss_curve.png"):
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history, label="train loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"[Viz] Saved loss curve to {save_path}")


# ================================================================
# Data: Double Swiss Roll
# ================================================================
def generate_swiss_roll(n_samples=1000, noise_level=0.1, grid_size=25):
    from sklearn.datasets import make_swiss_roll

    n_samples = grid_size ** 2
    n_samples_x1 = (n_samples + 1) // 2
    n_samples_x2 = n_samples - n_samples_x1

    X1, _ = make_swiss_roll(n_samples_x1, noise=0)
    X2, _ = make_swiss_roll(n_samples_x2, noise=0)

    X1_2d = X1[:, [0, 2]]
    X2_2d = X2[:, [0, 2]]

    # flip + shift for separation
    X2_2d[:, 0] = -X2_2d[:, 0] + 35
    # rotate by pi
    angle = np.pi
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    X2_2d = X2_2d @ R.T

    X = np.vstack((X1_2d, X2_2d)).astype(np.float32)
    X = torch.tensor(X)

    # center x-axis
    x_center = (X[:, 0].min() + X[:, 0].max()) / 2
    X[:, 0] -= x_center

    # add noise
    noise = torch.normal(0, noise_level, X.shape)
    X_noisy = X + noise

    # reshape grids [H,W,2]
    X_clean = X[:grid_size ** 2].reshape(grid_size, grid_size, 2)
    X_noisy = X_noisy[:grid_size ** 2].reshape(grid_size, grid_size, 2)

    return X_noisy, X_clean


# --- Add below existing generators ---
def generate_infinity_symbol_data(n_samples=1000, noise_level=0.1):
    """Generate infinity symbol (lemniscate of Bernoulli) with noisy and clean coordinates"""
    t = np.random.uniform(0, 2 * np.pi, n_samples)

    a = 3.0  # scale
    denom = 1 + np.sin(t) ** 2
    x = a * np.cos(t) / denom
    y = a * np.sin(t) * np.cos(t) / denom

    X_clean = np.column_stack([x, y]).astype(np.float32)
    X_clean -= X_clean.mean(axis=0)

    noise = np.random.normal(0, noise_level, X_clean.shape).astype(np.float32)
    X_noisy = X_clean + noise

    return (torch.tensor(X_noisy, dtype=torch.float32),
            torch.tensor(X_clean, dtype=torch.float32))



def make_distance_dataset(X_clean, grid_size=61):
    """
    Reduced grid (61x61 ~ 3721 pts) for faster ICNN/NN training.
    Returns: Xg [N,2], d_true [N,1], xx, yy for plotting.
    """
    x = torch.linspace(-35, 35, grid_size)
    y = torch.linspace(-20, 20, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    Xg = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)

    GT = X_clean.reshape(-1, 2).numpy()
    nbrs = NearestNeighbors(n_neighbors=1).fit(GT)
    d_true, _ = nbrs.kneighbors(Xg.numpy())
    d_true = torch.tensor(d_true, dtype=torch.float32).view(-1, 1)
    return Xg, d_true, xx, yy


# ================================================================
# Models: ICNN / NN
# ================================================================
class ICNN(nn.Module):
    """
    Basic ICNN (Amos et al. 2017 style).
    z_{k+1} = softplus( softplus(Wz_k) @ z_k + A_x x + b )
    f(x) = <softplus(w_L), z_L> + c  (convex in x)
    """
    def __init__(self, in_dim=2, hidden_dims=(128, 128, 128)):
        super().__init__()
        self.A0 = nn.Linear(in_dim, hidden_dims[0], bias=True)

        self.Axs = nn.ModuleList()
        self.Wzs_unc = nn.ParameterList()
        self.bs = nn.ParameterList()
        last = hidden_dims[0]
        for h in hidden_dims[1:]:
            self.Axs.append(nn.Linear(in_dim, h, bias=False))
            self.Wzs_unc.append(nn.Parameter(torch.randn(h, last) * 0.1))
            self.bs.append(nn.Parameter(torch.zeros(h)))
            last = h

        self.wL_unc = nn.Parameter(torch.randn(last) * 0.1)
        self.c = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        z = F.softplus(self.A0(x))
        for A_x, Wz_unc, b in zip(self.Axs, self.Wzs_unc, self.bs):
            Wz = F.softplus(Wz_unc)
            z = F.softplus(z @ Wz.t() + A_x(x) + b)
        wL = F.softplus(self.wL_unc)
        out = (z * wL).sum(dim=1, keepdim=True) + self.c
        return out


class MLP(nn.Module):
    """Plain NN (non-convex) for comparison."""
    def __init__(self, in_dim=2, hidden_dims=(256, 256, 256)):
        super().__init__()
        layers = []
        dims = [in_dim] + list(hidden_dims) + [1]
        for i in range(len(dims)-2):
            layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(inplace=True)]
        layers += [nn.Linear(dims[-2], dims[-1])]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train_distance_net(net, Xg, d_true, epochs=1500, lr=1e-3, wd=1e-6, batch_size=512):
    dataset = TensorDataset(Xg, d_true)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=500, gamma=0.5)
    loss_fn = nn.MSELoss()

    loss_history = []
    for ep in range(1, epochs+1):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            pred = net(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        scheduler.step()
        avg = total / len(dataset)
        loss_history.append(avg)
        if ep % 100 == 0 or ep == 1:
            print(f"[ICNN/NN] epoch {ep}/{epochs}  loss={avg:.6f}")
    return net, loss_history


def eval_to_contour(net, xx, yy):
    net.eval()
    with torch.no_grad():
        grid = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        dd = net(grid).view_as(xx).cpu().numpy()
    return dd


# ================================================================
# WCRR (IWCNN) – EPFL model
# ================================================================
from models.wc_conv_net import WCvxConvNet

def ensure_no_shared_storage(model):
    for _, param in model.named_parameters():
        param.data = param.data.clone().contiguous()


def train_model_wcrr(model, dataloader, epochs=1500, lr=1e-3, save_path="trained_models/wcrr_swiss_roll.pth"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    ensure_no_shared_storage(model)
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0.0
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

        avg = total_loss / len(dataloader)
        loss_history.append(avg)

        if (epoch + 1) % 100 == 0 or epoch == epochs - 1 or epoch == 0:
            print(f"[WCRR] epoch {epoch + 1}, loss: {avg:.6f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"[WCRR] Model saved to {save_path}")
    return model, loss_history


def visualize_wcrr(model, X_noisy, X_clean, grid_size=101):
    # denser grid for prettier contours
    x = torch.linspace(-35, 35, grid_size)
    y = torch.linspace(-20, 20, grid_size)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    grid = torch.stack([xx, yy], dim=-1)  # [H,W,2]
    grid_tensor = grid.permute(2, 0, 1).unsqueeze(0)  # [1,2,H,W]
    sigma_val = torch.full((1, 1, grid_size, grid_size), 0.2)

    model.eval()
    with torch.no_grad():
        prox = model.grad_denoising(grid_tensor, grid_tensor, sigma=sigma_val)

    prox_np = prox.squeeze().permute(1, 2, 0).reshape(-1, 2).numpy()
    # distance to WCRR fitted manifold (nearest neighbor to prox outputs)
    nbrs_model = NearestNeighbors(n_neighbors=1).fit(prox_np)
    dists_model, _ = nbrs_model.kneighbors(grid.reshape(-1, 2).numpy())
    dists_model = dists_model.reshape(grid_size, grid_size)

    # true distance (to clean manifold)
    gt_np = X_clean.reshape(-1, 2).numpy()
    nbrs_true = NearestNeighbors(n_neighbors=1).fit(gt_np)
    dists_true, _ = nbrs_true.kneighbors(grid.reshape(-1, 2).numpy())
    dists_true = dists_true.reshape(grid_size, grid_size)

    return xx, yy, dists_true, dists_model


def visualize_all(*, X_noisy, X_clean, dd_true, dd_nn, dd_icnn, dd_wcrr, xx, yy, save_path="EPFL_swiss_roll_result.png"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.flatten()  # 展平为一维数组

    titles = ["True distance", "NN approximation", "ICNN approximation", "IWCNN/WCRR approximation"]
    fields = [dd_true, dd_nn, dd_icnn, dd_wcrr]

    # 子图
    for ax, field, title in zip(axes, fields, titles):
        im = ax.contourf(xx, yy, field, levels=12, cmap='RdBu_r')
        ax.scatter(X_noisy[..., 0], X_noisy[..., 1], s=4, color='r')
        ax.set_title(title)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())

    # 共用一个 colorbar
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, location="right")

    plt.savefig(save_path, dpi=200)
    print(f"[Viz] Saved figure to {save_path}")




# ================================================================
# Main
# ================================================================
if __name__ == "__main__":
    os.makedirs("trained_models", exist_ok=True)

    # ----- data -----
    size = 63  # grid for generating swiss roll samples
    X_noisy, X_clean = generate_swiss_roll(grid_size=size)

    # tensors for WCRR training
    X_noisy_img = X_noisy.permute(2, 0, 1).unsqueeze(0)  # [1,2,H,W]
    X_clean_img = X_clean.permute(2, 0, 1).unsqueeze(0)
    dataset = TensorDataset(X_noisy_img, X_clean_img)
    dataloader = DataLoader(dataset, batch_size=1)

    # ----- WCRR model (IWCNN) -----
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
        "num_knots": 200,
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

    print("Multi convolutional layer: ", param_multi_conv)
    model, wcrr_loss = train_model_wcrr(model, dataloader, epochs=1000, lr=1e-3,
                                        save_path="trained_models/wcrr_swiss_roll.pth")
    plot_loss_curve(wcrr_loss, title="IWCNN/WCRR Loss Curve", save_path="EPFL_wcrr_loss.png")

    # ----- Visualize WCRR & True distance on a denser grid -----
    xx, yy, dd_true, dd_wcrr = visualize_wcrr(model, X_noisy, X_clean, grid_size=101)

    # ----- Prepare distance supervision dataset (reduced grid for speed) -----
    Xg, d_true, xx_d, yy_d = make_distance_dataset(X_clean, grid_size=61)

    # ----- Train NN (baseline) -----
    nn_reg = MLP(in_dim=2, hidden_dims=(256, 256, 256))
    nn_reg, nn_loss = train_distance_net(nn_reg, Xg, d_true, epochs=300, lr=1e-3, wd=1e-6, batch_size=512)
    torch.save(nn_reg.state_dict(), "trained_models/nn_swiss_roll.pt")
    plot_loss_curve(nn_loss, title="NN Loss Curve", save_path="EPFL_nn_loss.png")
    dd_nn = eval_to_contour(nn_reg, xx, yy)

    # ----- Train ICNN -----
    icnn = ICNN(in_dim=2, hidden_dims=(128, 128, 128))
    icnn, icnn_loss = train_distance_net(icnn, Xg, d_true, epochs=300, lr=1e-3, wd=1e-6, batch_size=512)
    torch.save(icnn.state_dict(), "trained_models/icnn_swiss_roll.pt")
    plot_loss_curve(icnn_loss, title="ICNN Loss Curve", save_path="EPFL_icnn_loss.png")
    dd_icnn = eval_to_contour(icnn, xx, yy)

    print("[Debug] dd_true shape:", dd_true.shape, " dd_nn shape:", dd_nn.shape)
    print("[Debug] sum(dd_true)=", float(np.sum(dd_true)), " sum(dd_nn)=", float(np.sum(dd_nn)))

    mse = float(np.mean((dd_nn - dd_true) ** 2))
    mae = float(np.mean(np.abs(dd_nn - dd_true)))
    print(f"[Debug] NN vs True: MSE={mse:.6e}, MAE={mae:.6e}")

    # 可视化误差热力图
    plt.figure(figsize=(6, 4))
    plt.contourf(xx, yy, np.abs(dd_nn - dd_true), levels=12)
    plt.title("|NN - True|")
    plt.colorbar();
    plt.tight_layout()
    plt.savefig("nn_vs_true_abs_diff.png", dpi=150)
    print("[Viz] Saved nn_vs_true_abs_diff.png")

    # ----- Figure-1 style panel (True, NN, ICNN, IWCNN) -----
    visualize_all(
        X_noisy=X_noisy, X_clean=X_clean,
        dd_true=dd_true, dd_nn=dd_nn, dd_icnn=dd_icnn, dd_wcrr=dd_wcrr,
        xx=xx, yy=yy, save_path="EPFL_swiss_roll_result.png"
    )

