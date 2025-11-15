import numpy as np
import pickle
import torch
import lzma

from pathlib import Path
from torch.utils.data import Dataset
from src.utils import preprocess_image


class DrivingDataset(Dataset):
    def __init__(self, dir_path: str, n_frames: int = 1):
        dir_path = Path(dir_path)
        all_images = []
        all_controls = []

        for filepath in sorted(dir_path.glob("*.npz")):
            with lzma.open(filepath, "rb") as file:
                data = pickle.load(file)
            
            imgs = []
            ctrls = []
            for e in data:
                imgs.append(preprocess_image(e.image, to_tensor=False))
                ctrls.append(e.current_controls)
        
            imgs = np.array(imgs)
            ctrls = np.array(ctrls, dtype=np.float32)
            L = len(imgs)

            for t in range(n_frames - 1, L):
                window = imgs[t - n_frames + 1 : t + 1]

                stacked = window.reshape(-1, *window.shape[2:])
                target = ctrls[t]
                all_images.append(stacked)
                all_controls.append(target)

        self.images = np.array(all_images, dtype=np.float32)
        self.controls = np.array(all_controls, dtype=np.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx])
        ctrl = torch.from_numpy(self.controls[idx])
        return img, ctrl
