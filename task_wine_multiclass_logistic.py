"""
pytorch_task_v1
Task: Multiclass Logistic Regression on Wine Dataset (RMSprop + L2 Regularisation)
Dataset: sklearn's load_wine  (3-class classification, 13 features)
Model: nn.Linear → softmax  (multi-class logistic regression)
Optimizer: RMSprop  +  weight_decay for L2 regularisation
Exit: sys.exit(0) on success (accuracy ≥ 0.90), sys.exit(1) on failure
"""

import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── Reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(7)

# ── 1. Load & preprocess data ─────────────────────────────────────────────────
data   = load_wine()
X, y   = data.data, data.target                   # (178, 13),  3 classes

scaler = StandardScaler()
X      = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7, stratify=y
)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t  = torch.tensor(X_test,  dtype=torch.float32)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True
)

# ── 2. Model ──────────────────────────────────────────────────────────────────
NUM_CLASSES = len(data.target_names)               # 3

class MulticlassLogisticRegression(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)               # raw logits; CrossEntropyLoss handles softmax

model = MulticlassLogisticRegression(in_features=X_train.shape[1],
                                     num_classes=NUM_CLASSES)

# ── 3. Loss & optimiser (RMSprop + L2 via weight_decay) ──────────────────────
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.RMSprop(
    model.parameters(), lr=1e-3, alpha=0.99, weight_decay=1e-4
)

# ── 4. Training loop ──────────────────────────────────────────────────────────
NUM_EPOCHS = 150
for epoch in range(1, NUM_EPOCHS + 1):
    model.train()
    epoch_loss    = 0.0
    correct_train = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss    += loss.item() * len(X_batch)
        correct_train += (logits.argmax(dim=1) == y_batch).sum().item()

    if epoch % 30 == 0:
        avg_loss    = epoch_loss / len(X_train_t)
        train_acc   = correct_train / len(X_train_t)
        print(f"Epoch [{epoch:>3}/{NUM_EPOCHS}]  "
              f"Loss: {avg_loss:.4f}  Train Acc: {train_acc:.4f}")

# ── 5. Evaluation ─────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    logits    = model(X_test_t)
    predicted = logits.argmax(dim=1)
    accuracy  = (predicted == y_test_t).float().mean().item()

# Per-class accuracy for interpretability
for class_idx, class_name in enumerate(data.target_names):
    mask    = y_test_t == class_idx
    if mask.sum() > 0:
        cls_acc = (predicted[mask] == y_test_t[mask]).float().mean().item()
        print(f"  Class '{class_name}': acc = {cls_acc:.4f}  "
              f"(n={mask.sum().item()})")

print(f"\nOverall Test Accuracy: {accuracy * 100:.2f}%")

# ── 6. Exit status ────────────────────────────────────────────────────────────
ACCURACY_THRESHOLD = 0.90
if accuracy >= ACCURACY_THRESHOLD:
    print(f"PASS — accuracy {accuracy:.4f} >= threshold {ACCURACY_THRESHOLD}")
    sys.exit(0)
else:
    print(f"FAIL — accuracy {accuracy:.4f} < threshold {ACCURACY_THRESHOLD}")
    sys.exit(1)
