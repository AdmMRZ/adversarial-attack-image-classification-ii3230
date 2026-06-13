from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """CNN sederhana untuk klasifikasi MNIST.

    Input MNIST berukuran 1 x 28 x 28. Model ini cukup kecil supaya bisa dilatih
    cepat di laptop, tetapi tetap cukup representatif untuk eksperimen FGSM.
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # 1x28x28 -> 32x14x14
        x = self.pool(F.relu(self.conv2(x)))   # 32x14x14 -> 64x7x7
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)
