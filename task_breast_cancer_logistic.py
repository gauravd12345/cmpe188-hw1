"""
Binary Logistic Regression - Breast Cancer Dataset
Implements ML Task: Binary classification with SGD + momentum optimizer.

Dataset: sklearn Breast Cancer Wisconsin (569 samples, 30 features)
Model:   nn.Linear(30, 1) + Sigmoid
Loss:    BCEWithLogitsLoss
Optimizer: SGD with momentum=0.9
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

torch.manual_seed(42)
np.random.seed(42)


def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name":  "task_breast_cancer_logistic",
        "task_type":  "binary_classification",
        "algorithm":  "Logistic Regression",
        "optimizer":  "SGD + momentum",
        "dataset":    "sklearn Breast Cancer Wisconsin",
        "input_dim":  30,
        "output_dim": 1,
        "protocol":   "pytorch_task_v1",
    }


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    """Get computation device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def make_dataloaders(batch_size: int = 32, val_ratio: float = 0.2):
    """
    Load Breast Cancer dataset, standardize, split, and return DataLoaders.

    Returns:
        train_loader, val_loader, input_dim
    """
    data = load_breast_cancer()
    X, y = data.data.astype(np.float32), data.target.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=42, stratify=y
    )

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train).unsqueeze(1)
    X_val_t   = torch.tensor(X_val)
    y_val_t   = torch.tensor(y_val).unsqueeze(1)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),
                              batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(X_train_t)}  |  Val samples: {len(X_val_t)}")
    return train_loader, val_loader, X_train.shape[1]


class LogisticRegressionModel(nn.Module):
    """Single-layer logistic regression."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)   # raw logits; BCEWithLogitsLoss handles sigmoid


def build_model(input_dim: int, device: torch.device) -> nn.Module:
    """Instantiate and return the logistic regression model."""
    model = LogisticRegressionModel(input_dim).to(device)
    print(f"Model: {model}")
    return model


def train(model, train_loader, val_loader, device,
          num_epochs=100, lr=0.01):
    """Train with SGD + momentum; return best model by val loss."""
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

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

        if epoch % 20 == 0:
            print(f"Epoch [{epoch:>3}/{num_epochs}]  "
                  f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model, data_loader, device):
    """Evaluate model; return dict with accuracy, MSE, R2, loss."""
    criterion = nn.BCEWithLogitsLoss()
    model.eval()

    all_logits, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for X_b, y_b in data_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            total_loss += criterion(logits, y_b).item() * len(X_b)
            all_logits.append(logits.cpu())
            all_labels.append(y_b.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs  = 1 / (1 + np.exp(-all_logits))
    all_preds  = (all_probs >= 0.5).astype(np.float32)

    accuracy = float((all_preds == all_labels).mean())
    mse      = float(mean_squared_error(all_labels, all_probs))
    r2       = float(r2_score(all_labels, all_probs))

    return {
        "loss":     total_loss / len(data_loader.dataset),
        "accuracy": accuracy,
        "mse":      mse,
        "r2":       r2,
    }


def predict(model, X, device):
    """Return predicted binary labels for numpy array X."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_t).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    return (probs >= 0.5).astype(np.float32).flatten()


def save_artifacts(model, metrics, output_dir="output"):
    """Save model weights and metrics JSON."""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(output_dir, "breast_cancer_logistic.pth"))
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Artifacts saved to {output_dir}/")


def main():
    print("=" * 60)
    print("Binary Logistic Regression — Breast Cancer Dataset")
    print("=" * 60)

    set_seed(42)
    device = get_device()

    print("\nLoading data...")
    train_loader, val_loader, input_dim = make_dataloaders()

    print("\nBuilding model...")
    model = build_model(input_dim, device)

    print("\nTraining...")
    model = train(model, train_loader, val_loader, device,
                  num_epochs=100, lr=0.01)

    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print(f"  Accuracy: {train_metrics['accuracy']:.4f} | "
          f"MSE: {train_metrics['mse']:.4f} | R2: {train_metrics['r2']:.4f}")

    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print(f"  Accuracy: {val_metrics['accuracy']:.4f} | "
          f"MSE: {val_metrics['mse']:.4f} | R2: {val_metrics['r2']:.4f}")

    print("\nSaving artifacts...")
    save_artifacts(model, {"train": train_metrics, "val": val_metrics})

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train Accuracy : {train_metrics['accuracy']:.4f}")
    print(f"Val   Accuracy : {val_metrics['accuracy']:.4f}")
    print(f"Train MSE      : {train_metrics['mse']:.4f}")
    print(f"Val   MSE      : {val_metrics['mse']:.4f}")
    print(f"Train R2       : {train_metrics['r2']:.4f}")
    print(f"Val   R2       : {val_metrics['r2']:.4f}")

    print("\n" + "=" * 60)
    print("QUALITY CHECKS")
    print("=" * 60)

    checks = []

    c1 = train_metrics["accuracy"] > 0.90
    checks.append(c1)
    print(f"{'✓' if c1 else '✗'} Train Accuracy > 0.90 : {train_metrics['accuracy']:.4f}")

    c2 = val_metrics["accuracy"] > 0.90
    checks.append(c2)
    print(f"{'✓' if c2 else '✗'} Val   Accuracy > 0.90 : {val_metrics['accuracy']:.4f}")

    c3 = val_metrics["mse"] < 0.10
    checks.append(c3)
    print(f"{'✓' if c3 else '✗'} Val   MSE      < 0.10 : {val_metrics['mse']:.4f}")

    c4 = val_metrics["r2"] > 0.60
    checks.append(c4)
    print(f"{'✓' if c4 else '✗'} Val   R2       > 0.60 : {val_metrics['r2']:.4f}")

    gap = abs(train_metrics["accuracy"] - val_metrics["accuracy"])
    c5  = gap < 0.10
    checks.append(c5)
    print(f"{'✓' if c5 else '✗'} Accuracy gap   < 0.10 : {gap:.4f}")

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
