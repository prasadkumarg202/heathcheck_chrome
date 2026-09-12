"""
DeepPhys: Dual-Stream Appearance and Motion Network for On-Device rPPG.
Reference: Chen & McDuff, DeepPhys: Video-Based Measurement of Heart Rate and Respiration Rate via Deep CNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepPhys(nn.Module):
    def __init__(self, in_channels: int = 3):
        super(DeepPhys, self).__init__()
        
        self.app_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2)
        )
        
        self.app_mask1 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.app_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2)
        )
        
        self.app_mask2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.mot_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.AvgPool2d(2)
        )
        
        self.mot_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.AvgPool2d(2)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

    def forward(self, appearance_frame: torch.Tensor, motion_diff: torch.Tensor) -> torch.Tensor:
        a1 = self.app_conv1(appearance_frame)
        m1 = self.mot_conv1(motion_diff)
        mask1 = self.app_mask1(a1)
        m1_gated = m1 * mask1
        
        a2 = self.app_conv2(a1)
        m2 = self.mot_conv2(m1_gated)
        mask2 = self.app_mask2(a2)
        m2_gated = m2 * mask2
        
        feat = self.global_pool(m2_gated).flatten(1)
        out = self.fc(feat)
        return out
