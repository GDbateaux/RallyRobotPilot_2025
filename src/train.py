import torch
import matplotlib.pyplot as plt
import torch.optim as optim
from config import N_FRAMES, CONTROL_DELAY
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from dataset import DrivingDataset
from model import DrivingCNN
from pathlib import Path
from tqdm import tqdm


data_dir = Path(__file__).parent.parent / "data/simple_track"
full_dataset = DrivingDataset(data_dir, n_frames=N_FRAMES, control_delay=CONTROL_DELAY)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


train_ratio = 0.8
n_total = len(full_dataset)
n_train = int(n_total * train_ratio)
n_val = n_total - n_train

train_dataset, val_dataset = random_split(
    full_dataset,
    [n_train, n_val],
    generator=torch.Generator().manual_seed(42),
)

num_workers = 8
pin_memory = (device.type == "cuda")
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=True
)

sample_img, _ = full_dataset[0]
C, H, W = sample_img.shape

model = DrivingCNN((C, H, W)).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

train_losses, val_losses = [], []
pos_weight = torch.tensor([1.0, 5.0, 2.0, 2.0], device=device)
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


num_epochs = 60

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


OUTPUT_PATH = Path(__file__).parent.parent / "data/models/driving_cnn.pt"

if best_state is not None:
    torch.save(best_state, OUTPUT_PATH)
    print(f"\n Best model saved from epoch {best_epoch} with val_loss={best_val_loss:.6f}")
else:
    print("No model was saved (unexpected).")

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
