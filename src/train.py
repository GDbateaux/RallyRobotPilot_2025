import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim

from torch.utils.data import DataLoader, random_split
from dataset import DrivingDataset
from model import DrivingCNN
from pathlib import Path
from tqdm import tqdm


data_dir = Path(__file__).parent.parent / "data/simple_track"
full_dataset = DrivingDataset(data_dir)

train_ratio = 0.8
n_total = len(full_dataset)
n_train = int(n_total * train_ratio)
n_val = n_total - n_train

train_dataset, val_dataset = random_split(
    full_dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42),
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample_img, _ = full_dataset[0]
C, H, W = sample_img.shape

model = DrivingCNN((C, H, W)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_losses, val_losses = [], []
train_maes, val_maes = [], []

def train_one_epoch(model: DrivingCNN, loader: DataLoader, optimizer: optim, device: torch.device):
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_n = 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        preds = model(imgs)

        loss = F.mse_loss(preds, targets)
        mae = torch.mean(torch.abs(preds - targets))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_mae  += mae.item() * bs
        total_n += bs

    return total_loss / total_n, total_mae / total_n


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_n = 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        preds = model(imgs)

        loss = F.mse_loss(preds, targets)
        mae = torch.mean(torch.abs(preds - targets))

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_mae  += mae.item() * bs
        total_n += bs

    return total_loss / total_n, total_mae / total_n


num_epochs = 40
for epoch in tqdm(range(num_epochs)):
    train_loss, train_mae = train_one_epoch(model, train_loader, optimizer, device)
    val_loss, val_mae = eval_one_epoch(model, val_loader, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_maes.append(train_mae)
    val_maes.append(val_mae)

    print(
        f"[Epoch {epoch+1:02d}] "
        f"train_mse={train_loss:.6f} train_mae={train_mae:.6f} | "
        f"val_mse={val_loss:.6f} val_mae={val_mae:.6f}"
    )

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Train MSE")
plt.plot(epochs, val_losses, label="Val MSE")
plt.xlabel("Epoch")
plt.ylabel("Loss (MSE)")
plt.title("Learning curve - MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

OUTPUT_PATH = Path(__file__).parent.parent / "data/models/driving_cnn.pt"
torch.save(model.state_dict(), OUTPUT_PATH)
print(f"Model saved to {OUTPUT_PATH}")
