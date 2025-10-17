import torch
import polars as pl
from torch import nn
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt



class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(32, 4)
        )

    def forward(self, x):
        return self.net(x)

def plot_losses(train_losses, val_losses, save_path=None, title="Training vs Validation Loss"):
    """
    train_losses, val_losses: list[float] aligned by epoch
    save_path: optional path to save the PNG
    """
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (BCEWithLogits)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


root_dir = Path(__file__).resolve().parent.parent
csv_path = root_dir / "records_merged.csv"

df = pl.read_csv(csv_path)
X = df.select(['speed'] + [f"ray_{i}" for i in range(15)]).to_numpy()
y = df.select('Forward', 'Backward', 'Left', 'Right').to_numpy()

SHIFT = 1
if len(X) <= SHIFT:
    raise RuntimeError(f"Pas assez d'échantillons pour appliquer SHIFT={SHIFT}.")
X = X[:-SHIFT]
y = y[SHIFT:]

n = len(X)
cut = int(n * 0.8)
X_train, X_val = X[:cut], X[cut:]
y_train, y_val = y[:cut], y[cut:]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

train_ds = TensorDataset(X_train, y_train)
val_ds = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)

with torch.no_grad():
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    pos = y_tr.sum(0)
    neg = y_tr.shape[0] - pos
    pos = pos.clamp_min(1.0)
    pos_weight = (neg / pos).to(torch.float32)

loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

model = NeuralNetwork()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-3)
epochs = 80

# for loss curve plotting
train_hist = []
val_hist = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    train_epoch_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            val_loss += loss_fn(pred, yb).item()

    val_epoch_loss = val_loss / len(val_loader)

    train_hist.append(train_epoch_loss)
    val_hist.append(val_epoch_loss)

    print(
        f"Epoch {epoch + 1:02d}: "
        f"train_loss={total_loss / len(train_loader):.4f}, "
        f"val_loss={val_loss / len(val_loader):.4f}, "
    )

#plotting loss curve
plot_losses(train_hist, val_hist)

torch.save({
    "model_state_dict": model.state_dict(),
    "scaler": scaler
}, "rally_model.pth")