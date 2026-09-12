"""
EfficientPhys: Lightweight Spatio-Temporal Self-Attention Network for Edge rPPG.
Reference: Liu et al., EfficientPhys: Enabling Simple, Fast and Accurate Camera-Based Vitals Measurement, IEEE WACV 2023.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatioTemporalAttention(nn.Module):
    def __init__(self, in_channels: int):
        super(SpatioTemporalAttention, self).__init__()
        self.conv_query = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.conv_key = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.conv_value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B*T, C, H, W)
        b_t, c, h, w = x.size()
        proj_query = self.conv_query(x).view(b_t, -1, h * w).permute(0, 2, 1)  # (B*T, HW, C/4)
        proj_key = self.conv_key(x).view(b_t, -1, h * w)  # (B*T, C/4, HW)
        energy = torch.bmm(proj_query, proj_key)  # (B*T, HW, HW)
        attention = F.softmax(energy, dim=-1)

        proj_value = self.conv_value(x).view(b_t, -1, h * w)  # (B*T, C, HW)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b_t, c, h, w)
        return self.gamma * out + x


class EfficientPhys(nn.Module):
    def __init__(self, in_channels: int = 3, frame_depth: int = 10):
        super(EfficientPhys, self).__init__()
        self.frame_depth = frame_depth

        # Stage 1
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(2)
        )

        # Stage 2 with Attention
        self.stage2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(2)
        )
        self.attn2 = SpatioTemporalAttention(32)

        # Stage 3
        self.stage3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(2)
        )
        self.attn3 = SpatioTemporalAttention(64)

        # Temporal Aggregation & Regression Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Difference normalized video frames tensor of shape (B*T, 3, H, W)
        Returns:
            pulse_derivative: (B*T, 1)
        """
        feat = self.stem(x)
        feat = self.stage2(feat)
        feat = self.attn2(feat)
        feat = self.stage3(feat)
        feat = self.attn3(feat)

        pooled = self.global_pool(feat).flatten(1)
        bvp_out = self.regressor(pooled)
        return bvp_out
