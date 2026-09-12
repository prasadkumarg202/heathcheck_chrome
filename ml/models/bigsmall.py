"""
BigSmall: Dual-Branch Spatial Scale Model for Fast and Accurate rPPG.
Reference: McDuff et al., BigSmall: Dual-branch Architecture for On-Device Physiological and Behavioral Sensing, NeurIPS 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigSmall(nn.Module):
    def __init__(self, in_channels: int = 3, frame_depth: int = 3):
        super(BigSmall, self).__init__()
        self.frame_depth = frame_depth

        # 1. Big Branch: Low-resolution global motion (e.g. 144x144 or 72x72)
        self.big_stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        # 2. Small Branch: High-resolution local skin patches (e.g. 9x9 or 18x18)
        self.small_stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # 3. Fusion Block
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

    def forward(self, big_frame: torch.Tensor, small_frame: torch.Tensor) -> torch.Tensor:
        """
        Args:
            big_frame: (B*T, 3, H_big, W_big)
            small_frame: (B*T, 3, H_small, W_small)
        Returns:
            pulse_derivative: (B*T, 1)
        """
        f_big = self.big_stem(big_frame)
        f_small = self.small_stem(small_frame)

        # Align spatial dimensions for fusion
        f_big_resized = F.interpolate(f_big, size=f_small.shape[2:], mode="bilinear", align_corners=False)
        fused = torch.cat([f_big_resized, f_small], dim=1)

        fused_feat = self.fusion_conv(fused)
        pooled = self.global_pool(fused_feat).flatten(1)
        out = self.head(pooled)
        return out
