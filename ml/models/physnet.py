"""
PhysNet: 3D Spatio-Temporal Convolutional Network for Contactless rPPG Estimation.
Reference: Yu et al., Remote Photoplethysmograph Signal Measurement from Facial Videos Using Spatio-Temporal Networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysNet(nn.Module):
    def __init__(self, in_channels: int = 3, num_frames: int = 64):
        super(PhysNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(1, 5, 5), stride=1, padding=(0, 2, 2)),
            nn.BatchNorm3d(16),
            nn.ELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        self.block1 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.ELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        self.block2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        )
        
        self.block3 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        )
        
        self.temporal_conv = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ELU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(4, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ELU()
        )
        
        self.global_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc = nn.Conv1d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input facial video tensor of shape (B, C, T, H, W) e.g., (B, 3, 64, 72, 72)
        Returns:
            bvp: Estimated 1D rPPG pulse signal of shape (B, T)
        """
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        x = self.temporal_conv(x)
        x = self.global_pool(x)
        x = x.squeeze(-1).squeeze(-1)
        bvp = self.fc(x).squeeze(1)
        return bvp
