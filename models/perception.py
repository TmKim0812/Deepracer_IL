"""
Perception module.

Takes a raw camera image and produces a fixed-size feature vector z_t
that is passed to the policy network.

Architecture: lightweight CNN (3 conv blocks + adaptive avg pool + FC).
"""

import torch
import torch.nn as nn


class PerceptionModule(nn.Module):
    def __init__(self, img_channels: int = 3, feature_dim: int = 128):
        super().__init__()
        self.feature_dim = feature_dim

        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(img_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # Collapse spatial dims to 4x4 regardless of input resolution
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)  or  (B, T, C, H, W) for sequential input
        Returns:
            z: (B, feature_dim)  or  (B, T, feature_dim)
        """
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            z = self.fc(self.encoder(x))
            return z.view(B, T, self.feature_dim)

        return self.fc(self.encoder(x))
