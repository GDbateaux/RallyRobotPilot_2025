import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from config import N_FRAMES, CONTROL_DELAY
from dataset import DrivingDataset
from model import DrivingCNN
from torch.utils.data import DataLoader, random_split


data_dir = Path(__file__).parent.parent / "data/simple_track_2"
full_dataset = DrivingDataset(data_dir, n_frames=N_FRAMES, control_delay=CONTROL_DELAY)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ratio = 0.8
n_total = len(full_dataset)
n_train = int(n_total * train_ratio)
n_val = n_total - n_train

train_dataset, val_dataset = random_split(
    full_dataset,
    [n_train, n_val],
    generator=torch.Generator(),
)

num_workers = 1
pin_memory = False

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
)

sample_img, _ = full_dataset[0]
C, H, W = sample_img.shape

model = DrivingCNN((C, H, W)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

train_losses, val_losses = [], []
pos_weight = torch.tensor([1.0, 3.5, 1.5, 1.5], device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)


def train_one_epoch(model: DrivingCNN, loader: DataLoader, optimizer: optim, device: torch.device):
    model.train()
    total_loss = 0.0
    total_n = 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        preds = model(imgs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / total_n


@torch.no_grad()
def eval_one_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_n = 0

    for imgs, targets in loader:
        imgs = imgs.to(device)
        targets = targets.to(device)

        preds = model(imgs)
        loss = criterion(preds, targets)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_n += bs

    return total_loss / total_n


num_epochs = 30

best_val_loss = float("inf")
best_epoch = -1
best_state = None

for epoch in tqdm(range(num_epochs)):
    train_loss = train_one_epoch(model, train_loader, optimizer, device)
    val_loss = eval_one_epoch(model, val_loader, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"[Epoch {epoch+1:02d}] train_bce={train_loss:.6f} | val_bce={val_loss:.6f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        best_state = model.state_dict()

models_dir = Path(__file__).parent.parent / "data/models"
models_dir.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = models_dir / "driving_cnn.pt"

model.load_state_dict(best_state)
torch.save(best_state, OUTPUT_PATH)

model.eval()
all_targets = []
all_preds = []

with torch.no_grad():
    for imgs, targets in val_loader:
        imgs = imgs.to(device)
        targets = targets.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int().cpu().numpy()
        t = targets.int().cpu().numpy()
        all_targets.append(t)
        all_preds.append(preds)

all_targets = np.concatenate(all_targets, axis=0)
all_preds = np.concatenate(all_preds, axis=0)

action_names = ["forward", "back", "left", "right"]

fig, axes = plt.subplots(2, 2, figsize=(8, 8))
axes = axes.ravel()

for i, name in enumerate(action_names):
    cm = confusion_matrix(all_targets[:, i], all_preds[:, i], labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(ax=axes[i], values_format="d", colorbar=False)
    axes[i].set_title(name)

plt.tight_layout()
CM_PATH = models_dir / "confusion_matrix.png"
plt.savefig(CM_PATH)
plt.close(fig)

epochs = range(1, num_epochs + 1)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, label="Train BCE")
plt.plot(epochs, val_losses, label="Val BCE")
plt.xlabel("Epoch")
plt.ylabel("Loss (BCE)")
plt.title("Learning curve - BCE")
plt.legend()
plt.grid(True)
plt.tight_layout()

PLOT_PATH = Path(__file__).parent.parent / "data/models/training_curve.png"
plt.savefig(PLOT_PATH)
print(f"Training curve saved to {PLOT_PATH}")
plt.show()
