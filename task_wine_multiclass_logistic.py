"""
Multiclass Logistic Regression - Wine Recognition Dataset
Implements ML Task: 3-class classification with RMSprop + L2 regularization.

Dataset: sklearn Wine Recognition (178 samples, 13 features, 3 classes)
Model:   nn.Linear(13, 3)  — softmax via CrossEntropyLoss
Loss:    CrossEntropyLoss
Optimizer: RMSprop with weight_decay (L2 regularization)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, mean_squared_error, r2_score

torch.manual_seed(7)
np.random.seed(7)


def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name":  "task_wine_multiclass_logistic",
        "task_type":  "multiclass_classification",
        "algorithm":  "Multiclass Logistic Regression (Softmax)",
        "optimizer":  "RMSprop + L2 weight_decay",
        "dataset":    "sklearn Wine Recognition",
        "input_dim":  13,
        "output_dim": 3,
        "protocol":   "pytorch_task_v1",
    }


def set_seed(seed: int = 7) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    """Get computation device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device


def make_dataloaders(batch_size: int = 16, val_ratio: float = 0.2):
    """
    Load Wine dataset, standardize, stratified split, return DataLoaders.

    Returns:
        train_loader, val_loader, input_dim, num_classes
    """
    data   = load_wine()
    X      = data.data.astype(np.float32)
    y      = data.target.astype(np.int64)

    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=7, stratify=y
    )

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_val_t   = torch.tensor(X_val)
    y_val_t   = torch.tensor(y_val)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t),
                              batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(TensorDataset(X_val_t, y_val_t),
                              batch_size=batch_size, shuffle=False)

    num_classes = len(np.unique(y))
    print(f"Train samples: {len(X_train_t)}  |  Val samples: {len(X_val_t)}  "
          f"|  Classes: {num_classes}")
    return train_loader, val_loader, X_train.shape[1], num_classes


class MulticlassLogisticRegression(nn.Module):
    """Single linear layer; CrossEntropyLoss applies softmax internally."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)   # raw logits


def build_model(input_dim: int, num_classes: int,
                device: torch.device) -> nn.Module:
    """Instantiate and return the multiclass logistic regression model."""
    model = MulticlassLogisticRegression(input_dim, num_classes).to(device)
    print(f"Model: {model}")
    return model


def train(model, train_loader, val_loader, device,
          num_epochs=150, lr=1e-3):
    """Train with RMSprop + L2; return best model by val loss."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.RMSprop(model.parameters(), lr=lr,
                               alpha=0.99, weight_decay=1e-4)

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

        if epoch % 30 == 0:
            print(f"Epoch [{epoch:>3}/{num_epochs}]  "
                  f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}")

    if best_state:
        model.load_state_dict(best_state)
    return model


def evaluate(model, data_loader, device):
    """Evaluate model; return accuracy, macro-F1, MSE, R2, loss."""
    criterion = nn.CrossEntropyLoss()
    model.eval()

    all_preds, all_labels = [], []
    total_loss = 0.0

    with torch.no_grad():
        for X_b, y_b in data_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            logits = model(X_b)
            total_loss += criterion(logits, y_b).item() * len(X_b)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(y_b.cpu())

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    accuracy = float((all_preds == all_labels).mean())
    f1       = float(f1_score(all_labels, all_preds, average="macro"))
    mse      = float(mean_squared_error(all_labels, all_preds))
    r2       = float(r2_score(all_labels, all_preds))

    return {
        "loss":     total_loss / len(data_loader.dataset),
        "accuracy": accuracy,
        "f1_macro": f1,
        "mse":      mse,
        "r2":       r2,
    }


def predict(model, X, device):
    """Return predicted class indices for numpy array X."""
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(X_t)
        preds  = logits.argmax(dim=1).cpu().numpy()
    return preds


def save_artifacts(model, metrics, output_dir="output"):
    """Save model weights and metrics JSON."""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(),
               os.path.join(output_dir, "wine_multiclass_logistic.pth"))
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Artifacts saved to {output_dir}/")


def main():
    print("=" * 60)
    print("Multiclass Logistic Regression — Wine Recognition Dataset")
    print("=" * 60)

    set_seed(7)
    device = get_device()

    print("\nLoading data...")
    train_loader, val_loader, input_dim, num_classes = make_dataloaders()

    print("\nBuilding model...")
    model = build_model(input_dim, num_classes, device)

    print("\nTraining...")
    model = train(model, train_loader, val_loader, device,
                  num_epochs=150, lr=1e-3)

    print("\nEvaluating on training set...")
    train_metrics = evaluate(model, train_loader, device)
    print(f"  Accuracy: {train_metrics['accuracy']:.4f} | "
          f"F1: {train_metrics['f1_macro']:.4f} | R2: {train_metrics['r2']:.4f}")

    print("\nEvaluating on validation set...")
    val_metrics = evaluate(model, val_loader, device)
    print(f"  Accuracy: {val_metrics['accuracy']:.4f} | "
          f"F1: {val_metrics['f1_macro']:.4f} | R2: {val_metrics['r2']:.4f}")

    print("\nSaving artifacts...")
    save_artifacts(model, {"train": train_metrics, "val": val_metrics})

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Train Accuracy : {train_metrics['accuracy']:.4f}")
    print(f"Val   Accuracy : {val_metrics['accuracy']:.4f}")
    print(f"Train F1 Macro : {train_metrics['f1_macro']:.4f}")
    print(f"Val   F1 Macro : {val_metrics['f1_macro']:.4f}")
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

    c3 = val_metrics["f1_macro"] > 0.85
    checks.append(c3)
    print(f"{'✓' if c3 else '✗'} Val   F1 Macro > 0.85 : {val_metrics['f1_macro']:.4f}")

    c4 = val_metrics["r2"] > 0.80
    checks.append(c4)
    print(f"{'✓' if c4 else '✗'} Val   R2       > 0.80 : {val_metrics['r2']:.4f}")

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
