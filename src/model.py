import torch
import torch.nn as nn
import torch.nn.functional as F


class DrivingCNN(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        C, H, W = input_shape

        self.features = nn.Sequential(
            nn.Conv2d(C, 24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.BatchNorm2d(36),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 50, kernel_size=3, stride=1),
            nn.BatchNorm2d(50),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out_feat = self.features(dummy)
            n_flat = out_feat.view(1, -1).shape[1]

        self.mlp = nn.Sequential(
            nn.Linear(n_flat, 100),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Dropout(p=0.5),

            nn.Linear(50, 10),
            nn.ReLU(),

            nn.Linear(10, 4)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.mlp(x)
        return x
