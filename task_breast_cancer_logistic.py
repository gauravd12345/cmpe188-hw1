"""
pytorch_task_v1
Task: Logistic Regression on Breast Cancer Dataset (SGD Optimizer)
Dataset: sklearn's load_breast_cancer (binary classification)
Model: Single-layer logistic regression (nn.Linear + Sigmoid)
Optimizer: SGD with momentum
Exit: sys.exit(0) on success, sys.exit(1) on failure
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(42)

# ── 1. Load & preprocess data ─────────────────────────────────────────────────
data = load_breast_cancer()
X, y = data.data, data.target                      # (569, 30), binary labels

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.float32).unsqueeze(1)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True
)

# ── 2. Model ──────────────────────────────────────────────────────────────────
class LogisticRegressionModel(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))

model = LogisticRegressionModel(in_features=X_train.shape[1])

# ── 3. Loss & optimiser ───────────────────────────────────────────────────────
criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ── 4. Training loop ──────────────────────────────────────────────────────────
NUM_EPOCHS = 100
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

    if epoch % 20 == 0:
        avg_loss = epoch_loss / len(X_train_t)
        print(f"Epoch [{epoch:>3}/{NUM_EPOCHS}]  Loss: {avg_loss:.4f}")

# ── 5. Evaluation ─────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    probs    = model(X_test_t)
    predicted = (probs >= 0.5).float()
    accuracy  = (predicted == y_test_t).float().mean().item()

print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# ── 6. Exit status ────────────────────────────────────────────────────────────
ACCURACY_THRESHOLD = 0.90          # expect ≥ 90 % on this clean dataset
if accuracy >= ACCURACY_THRESHOLD:
    print(f"PASS — accuracy {accuracy:.4f} >= threshold {ACCURACY_THRESHOLD}")
    sys.exit(0)
else:
    print(f"FAIL — accuracy {accuracy:.4f} < threshold {ACCURACY_THRESHOLD}")
    sys.exit(1)
