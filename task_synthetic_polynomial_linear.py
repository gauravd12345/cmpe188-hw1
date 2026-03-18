"""
pytorch_task_v1
Task: Linear Regression on Synthetic Polynomial-Feature Dataset
      (AdamW Optimiser + StepLR Scheduler)
Dataset: Synthetically generated  y = 3x₁² + 2x₂ - x₃ + noise
Features: Raw inputs augmented with degree-2 polynomial features via
          sklearn.preprocessing.PolynomialFeatures (manual feature engineering)
Model: nn.Linear (operates on expanded feature set)
Optimizer: AdamW  (Adam with decoupled weight decay)
Scheduler: StepLR  — halves LR every 50 epochs
Exit: sys.exit(0) on success (R² ≥ 0.95), sys.exit(1) on failure
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(99)
rng = np.random.default_rng(99)

# ── 1. Synthetic dataset ──────────────────────────────────────────────────────
N = 2000
X_raw = rng.uniform(-3, 3, size=(N, 3)).astype(np.float32)
noise = rng.normal(0, 0.5, size=(N,)).astype(np.float32)
y_raw = (3 * X_raw[:, 0] ** 2
       + 2 * X_raw[:, 1]
       -     X_raw[:, 2]
       + noise)

# ── 2. Polynomial feature expansion (degree=2) ────────────────────────────────
poly    = PolynomialFeatures(degree=2, include_bias=False)
X_poly  = poly.fit_transform(X_raw)                 # (2000, 9)
print(f"Feature names: {poly.get_feature_names_out(['x1','x2','x3'])}")
print(f"Expanded feature matrix shape: {X_poly.shape}")

scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_poly   = scaler_X.fit_transform(X_poly)
y_scaled = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X_poly, y_scaled, test_size=0.2, random_state=99
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True
)

# ── 3. Model ──────────────────────────────────────────────────────────────────
class PolynomialLinearRegression(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        # Xavier initialisation for faster convergence
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

model = PolynomialLinearRegression(in_features=X_train.shape[1])

# ── 4. Loss, optimiser & scheduler ───────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

# ── 5. Training loop ──────────────────────────────────────────────────────────
NUM_EPOCHS = 200
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(X_batch)

    scheduler.step()                               # adjust LR after each epoch

    if epoch % 40 == 0:
        avg_loss  = epoch_loss / len(X_train_t)
        cur_lr    = scheduler.get_last_lr()[0]
        print(f"Epoch [{epoch:>3}/{NUM_EPOCHS}]  MSE: {avg_loss:.4f}  "
              f"LR: {cur_lr:.5f}")

# ── 6. Evaluation (R²) ───────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    preds  = model(X_test_t)
    ss_res = ((y_test_t - preds) ** 2).sum().item()
    ss_tot = ((y_test_t - y_test_t.mean()) ** 2).sum().item()
    r2     = 1.0 - ss_res / ss_tot

print(f"\nTest R² Score: {r2:.4f}")

# Learnt coefficients for interpretability
weights = model.linear.weight.detach().squeeze().numpy()
feature_names = poly.get_feature_names_out(['x1', 'x2', 'x3'])
print("\nLearnt coefficients:")
for name, w in zip(feature_names, weights):
    print(f"  {name:>10s}: {w:+.4f}")

# ── 7. Exit status ────────────────────────────────────────────────────────────
R2_THRESHOLD = 0.95
if r2 >= R2_THRESHOLD:
    print(f"\nPASS — R² {r2:.4f} >= threshold {R2_THRESHOLD}")
    sys.exit(0)
else:
    print(f"\nFAIL — R² {r2:.4f} < threshold {R2_THRESHOLD}")
    sys.exit(1)
