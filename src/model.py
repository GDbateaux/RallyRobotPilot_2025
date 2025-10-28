import torch
import torch.nn as nn
import torch.nn.functional as F


class DrivingCNN(nn.Module):
    def __init__(self, input_shape):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            out_feat = self.features(dummy)
            n_flat = out_feat.view(1, -1).shape[1]

        self.fc1 = nn.Linear(n_flat, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.out = nn.Linear(10, 4)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.out(x))
        return x
