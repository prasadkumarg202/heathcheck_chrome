"""
TS-CAN: Temporal Shift Convolutional Attention Network for Remote Physiological Sensing.
Reference: Liu et al., Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalShift(nn.Module):
    def __init__(self, n_segment: int = 10, n_div: int = 8):
        super(TemporalShift, self).__init__()
        self.n_segment = n_segment
        self.n_div = n_div

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bt, c, h, w = x.size()
        t = self.n_segment
        b = bt // t
        x = x.view(b, t, c, h, w)
        
        fold = c // self.n_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]
        out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
        out[:, :, 2*fold:] = x[:, :, 2*fold:]
        
        return out.view(bt, c, h, w)

class AttentionMask(nn.Module):
    def __init__(self, in_channels: int):
        super(AttentionMask, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.conv(x)
        return x * mask

class TSCAN(nn.Module):
    def __init__(self, in_channels: int = 3, n_segment: int = 10):
        super(TSCAN, self).__init__()
        self.n_segment = n_segment
        
        self.ts1 = TemporalShift(n_segment=n_segment)
        self.m_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.AvgPool2d(2)
        )
        
        self.ts2 = TemporalShift(n_segment=n_segment)
        self.m_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.AvgPool2d(2)
        )
        
        self.attn1 = AttentionMask(32)
        self.attn2 = AttentionMask(64)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ts1(x)
        x = self.m_conv1(x)
        x = self.attn1(x)
        
        x = self.ts2(x)
        x = self.m_conv2(x)
        x = self.attn2(x)
        
        x = self.global_pool(x).flatten(1)
        out = self.head(x)
        return out
