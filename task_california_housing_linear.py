"""
pytorch_task_v1
Task: Linear Regression on California Housing Dataset (Adam Optimizer)
Dataset: sklearn's fetch_california_housing (regression)
Model: Multi-feature linear regression (nn.Linear)
Optimizer: Adam
Exit: sys.exit(0) on success (R² ≥ 0.60), sys.exit(1) on failure
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(0)

# ── 1. Load & preprocess data ─────────────────────────────────────────────────
housing = fetch_california_housing()
X, y = housing.data, housing.target               # (20640, 8), continuous target

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=64, shuffle=True
)

# ── 2. Model ──────────────────────────────────────────────────────────────────
class LinearRegressionModel(nn.Module):
    """
    Extended linear model: adds one hidden layer (64 units, ReLU) so the
    network can capture the mild non-linearities present in the housing data
    while still being a straightforward regression model.
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

model = LinearRegressionModel(in_features=X_train.shape[1])

# ── 3. Loss, optimiser & scheduler ───────────────────────────────────────────
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=10
)

# ── 4. Training loop ──────────────────────────────────────────────────────────
NUM_EPOCHS = 300
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

    avg_loss = epoch_loss / len(X_train_t)
    scheduler.step(avg_loss)          # reduce LR when loss plateaus

    if epoch % 60 == 0:
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch:>3}/{NUM_EPOCHS}]  MSE: {avg_loss:.4f}  LR: {cur_lr:.5f}")

# ── 5. Evaluation (R² score) ──────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    preds   = model(X_test_t)
    ss_res  = ((y_test_t - preds) ** 2).sum().item()
    ss_tot  = ((y_test_t - y_test_t.mean()) ** 2).sum().item()
    r2      = 1.0 - ss_res / ss_tot

print(f"\nTest R² Score: {r2:.4f}")

# ── 6. Exit status ────────────────────────────────────────────────────────────
R2_THRESHOLD = 0.60
if r2 >= R2_THRESHOLD:
    print(f"PASS — R² {r2:.4f} >= threshold {R2_THRESHOLD}")
    sys.exit(0)
else:
    print(f"FAIL — R² {r2:.4f} < threshold {R2_THRESHOLD}")
    sys.exit(1)
