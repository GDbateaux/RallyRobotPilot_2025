import numpy as np
import pickle
import torch
import lzma

from pathlib import Path
from torch.utils.data import Dataset
from src.utils import preprocess_image


class DrivingDataset(Dataset):
    def __init__(self, dir_path: str, frame_offset: int = 0):
        dir_path = Path(dir_path)
        all_images = []
        all_controls = []

        for filepath in sorted(dir_path.glob("*.npz")):
            with lzma.open(filepath, "rb") as file:
                data = pickle.load(file)
            
            for e in data:
                img = preprocess_image(e.image, to_tensor=False)
                all_images.append(img)
                all_controls.append(e.current_controls)
        
        images = np.array(all_images)
        controls = np.array(all_controls, dtype=np.float32)
        
        if frame_offset > 0:
            self.images = images[:-frame_offset]
            self.controls = controls[frame_offset:]
        else:
            self.images = images
            self.controls = controls

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = torch.from_numpy(self.images[idx])
        ctrl = torch.from_numpy(self.controls[idx])
        return img, ctrl
    