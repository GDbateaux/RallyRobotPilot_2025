import numpy as np
import torch
import cv2


def preprocess_image(img: np.ndarray,
                         size=(200, 150),
                         to_tensor: bool = True) -> torch.Tensor | np.ndarray:
    w, h = size
    img = cv2.resize(img, (w, h)).astype(np.float32)
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))

    if to_tensor:
        return torch.from_numpy(img)
    return img
