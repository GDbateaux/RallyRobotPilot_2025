import numpy as np
import pickle
import torch
import lzma
import cv2

from pathlib import Path
from torch.utils.data import Dataset


class DrivingDataset(Dataset):
    def __init__(self, dir_path: str):
        dir_path = Path(dir_path)
        all_images = []
        all_controls = []

        for filepath in sorted(dir_path.glob("*.npz")):
            with lzma.open(filepath, "rb") as file:
                data = pickle.load(file)
            
            for e in data:
                img = e.image.astype(np.float32) / 255.0
                img = cv2.resize(img, (200, 150))
                img = np.transpose(img, (2, 0, 1))
                all_images.append(img)
                all_controls.append(e.current_controls)
        self.images = np.array(all_images)
        self.controls = np.array(all_controls, dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        ctrl = self.controls[idx]

        img = torch.from_numpy(img)
        ctrl = torch.from_numpy(ctrl)

        return img, ctrl
    