"""
Linear Regression - Synthetic Polynomial Features Dataset
Implements ML Task: Regression with degree-2 feature expansion, AdamW + StepLR.

Dataset: Synthetic  y = 3*x1^2 + 2*x2 - x3 + noise  (2000 samples)
Features: Raw 3 inputs expanded to 9 via PolynomialFeatures(degree=2)
Model:   nn.Linear(9, 1) with Xavier init
Loss:    MSELoss
Optimizer: AdamW with StepLR scheduler (halves LR every 50 epochs)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

torch.manual_seed(99)
np.random.seed(99)


def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name":  "task_synthetic_polynomial_linear",
        "task_type":  "regression",
        "algorithm":  "Linear Regression on Polynomial Features",
        "optimizer":  "AdamW + StepLR",
        "dataset":    "Synthetic y=3x1^2+2x2-x3+noise",
        "input_dim":  9,
        "output_dim": 1,
        "protocol":   "pytorch_task_v1",
    }


def set_seed(seed: int = 99) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    """Get computation device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def make_dataloaders(n: int = 2000, batch_size: int = 64,
                     val_ratio: float = 0.2):
    """
    Generate synthetic polynomial dataset, expand features, standardize,
    split, and return DataLoaders.

    True relationship: y = 3*x1^2 + 2*x2 - x3 + noise

    Returns:
        train_loader, val_loader, input_dim
    """
    rng    = np.random.default_rng(99)
    X_raw  = rng.uniform(-3, 3, size=(n, 3)).astype(np.float32)
    noise  = rng.normal(0, 0.5, size=n).astype(np.float32)
    y_raw  = (3 * X_raw[:, 0] ** 2
              + 2 * X_raw[:, 1]
              -     X_raw[:, 2]
              + noise)

    poly   = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X_raw)           # (n, 9)
    print(f"Polynomial features: {poly.get_feature_names_out(['x1','x2','x3'])}")

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_poly   = scaler_X.fit_transform(X_poly).astype(np.float32)
    y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel().astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        X_poly, y_scaled, test_size=val_ratio, random_state=99
    )

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).unsqueeze(1)
    X_val_t   = torch.tensor(X_val)
    y_val_t   = torch.tensor(y_val).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),
                              batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(X_train_t)}  |  Val samples: {len(X_val_t)}  "
          f"|  Features: {X_poly.shape[1]}")
    return train_loader, val_loader, X_poly.shape[1]


class PolynomialLinearRegression(nn.Module):
    """Linear model operating on polynomial-expanded features."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build_model(input_dim: int, device: torch.device) -> nn.Module:
    """Instantiate and return the polynomial regression model."""
    model = PolynomialLinearRegression(input_dim).to(device)
    print(f"Model: {model}")
    return model


def train(model, train_loader, val_loader, device,
          num_epochs=200, lr=1e-2):
    """Train with AdamW + StepLR; return best model by val loss."""
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    best_val_loss = float("inf")
    best_state    = None

    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(X_b)
        train_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                val_loss += criterion(model(X_b), y_b).item() * len(X_b)
        val_loss /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 40 == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch [{epoch:>3}/{num_epochs}]  "
                  f"Train MSE: {train_loss:.4f}  Val MSE: {val_loss:.4f}  "
                  f"LR: {cur_lr:.5f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model, data_loader, device):
    """Evaluate model; return MSE, RMSE, R2, loss."""
    criterion = nn.MSELoss()
    model.eval()

    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for X_b, y_b in data_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            preds = model(X_b)
            total_loss += criterion(preds, y_b).item() * len(X_b)
            all_preds.append(preds.cpu())
            all_labels.append(y_b.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    mse  = float(mean_squared_error(all_labels, all_preds))
    rmse = float(np.sqrt(mse))
    r2   = float(r2_score(all_labels, all_preds))

    return {
        "loss": total_loss / len(data_loader.dataset),
        "mse":  mse,
        "rmse": rmse,
        "r2":   r2,
    }


def predict(model, X, device):
    """Return regression predictions for numpy array X."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        preds = model(X_t).cpu().numpy()
    return preds.flatten()


def save_artifacts(model, metrics, output_dir="output"):
    """Save model weights, metrics JSON, and learned coefficients."""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(output_dir, "synthetic_polynomial_linear.pth"))
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save learned coefficients for interpretability
    weights = model.linear.weight.detach().cpu().numpy().flatten().tolist()
    bias    = float(model.linear.bias.detach().cpu().numpy()[0])
    coef    = {"weights": weights, "bias": bias}
    with open(os.path.join(output_dir, "coefficients.json"), "w") as f:
        json.dump(coef, f, indent=2)

    print(f"Artifacts saved to {output_dir}/")


def main():
    print("=" * 60)
    print("Linear Regression — Synthetic Polynomial Features")
    print("=" * 60)

    set_seed(99)
    device = get_device()

    print("\nGenerating data...")
    train_loader, val_loader, input_dim = make_dataloaders()

    print("\nBuilding model...")
    model = build_model(input_dim, device)

    print("\nTraining...")
    model = train(model, train_loader, val_loader, device,
                  num_epochs=200, lr=1e-2)

    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print(f"  MSE: {train_metrics['mse']:.4f} | "
          f"RMSE: {train_metrics['rmse']:.4f} | R2: {train_metrics['r2']:.4f}")

    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print(f"  MSE: {val_metrics['mse']:.4f} | "
          f"RMSE: {val_metrics['rmse']:.4f} | R2: {val_metrics['r2']:.4f}")

    print("\nSaving artifacts...")
    save_artifacts(model, {"train": train_metrics, "val": val_metrics})

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train MSE  : {train_metrics['mse']:.4f}")
    print(f"Val   MSE  : {val_metrics['mse']:.4f}")
    print(f"Train RMSE : {train_metrics['rmse']:.4f}")
    print(f"Val   RMSE : {val_metrics['rmse']:.4f}")
    print(f"Train R2   : {train_metrics['r2']:.4f}")
    print(f"Val   R2   : {val_metrics['r2']:.4f}")

    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    checks = []

    c1 = train_metrics["r2"] > 0.95
    checks.append(c1)
    print(f"{'✓' if c1 else '✗'} Train R2   > 0.95 : {train_metrics['r2']:.4f}")

    c2 = val_metrics["r2"] > 0.95
    checks.append(c2)
    print(f"{'✓' if c2 else '✗'} Val   R2   > 0.95 : {val_metrics['r2']:.4f}")

    c3 = val_metrics["mse"] < 0.05
    checks.append(c3)
    print(f"{'✓' if c3 else '✗'} Val   MSE  < 0.05 : {val_metrics['mse']:.4f}")

    c4 = val_metrics["rmse"] < 0.25
    checks.append(c4)
    print(f"{'✓' if c4 else '✗'} Val   RMSE < 0.25 : {val_metrics['rmse']:.4f}")

    gap = abs(train_metrics["r2"] - val_metrics["r2"])
    c5  = gap < 0.05
    checks.append(c5)
    print(f"{'✓' if c5 else '✗'} R2 gap     < 0.05 : {gap:.4f}")

    print("\n" + "=" * 60)
    if all(checks):
        print("PASS: All quality checks passed!")
        print("=" * 60)
        return 0
    else:
        print("FAIL: Some quality checks failed!")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
