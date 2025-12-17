"""
Polar U-Net implementation for medical image segmentation.
Combines polar coordinate transformation with U-Net architecture.

The key idea is to transform the image to polar coordinates, apply U-Net,
and then transform back to Cartesian coordinates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class PolarTransform(nn.Module):
    """
    Polar coordinate transformation layer
    """
    def __init__(self, input_size=256, output_size=256):
        super(PolarTransform, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Create polar coordinate grid
        self.register_buffer('polar_grid', self._create_polar_grid())
    
    def _create_polar_grid(self):
        """Create polar coordinate sampling grid"""
        # Create polar coordinates
        r_max = self.input_size // 2
        theta_steps = self.output_size
        r_steps = self.output_size
        
        # Create theta and r arrays
        theta = torch.linspace(0, 2 * math.pi, theta_steps)
        r = torch.linspace(0, r_max, r_steps)
        
        # Create meshgrid
        theta_grid, r_grid = torch.meshgrid(theta, r, indexing='ij')
        
        # Convert to Cartesian coordinates for sampling
        center = self.input_size // 2
        x = r_grid * torch.cos(theta_grid) + center
        y = r_grid * torch.sin(theta_grid) + center
        
        # Normalize to [-1, 1] for grid_sample
        x = 2.0 * x / (self.input_size - 1) - 1.0
        y = 2.0 * y / (self.input_size - 1) - 1.0
        
        # Stack to create grid [H, W, 2]
        grid = torch.stack([x, y], dim=-1)
        
        return grid
    
    def forward(self, x):
        """Transform from Cartesian to polar coordinates"""
        batch_size = x.size(0)
        
        # Expand grid for batch
        grid = self.polar_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Apply grid sampling
        polar_x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return polar_x

class InversePolarTransform(nn.Module):
    """
    Inverse polar coordinate transformation layer
    """
    def __init__(self, input_size=256, output_size=256):
        super(InversePolarTransform, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        # Create inverse polar coordinate grid
        self.register_buffer('inverse_grid', self._create_inverse_grid())
    
    def _create_inverse_grid(self):
        """Create inverse polar coordinate sampling grid"""
        # Create Cartesian coordinate grid
        x = torch.linspace(-1, 1, self.output_size)
        y = torch.linspace(-1, 1, self.output_size)
        
        x_grid, y_grid = torch.meshgrid(x, y, indexing='ij')
        
        # Convert to polar coordinates
        center_x, center_y = 0.0, 0.0
        dx = x_grid - center_x
        dy = y_grid - center_y
        
        r = torch.sqrt(dx**2 + dy**2)
        theta = torch.atan2(dy, dx)
        
        # Normalize theta to [0, 2π]
        theta = (theta + 2 * math.pi) % (2 * math.pi)
        
        # Normalize for grid sampling
        r_max = math.sqrt(2)  # Maximum radius in normalized coordinates
        r_norm = 2.0 * r / r_max - 1.0
        theta_norm = 2.0 * theta / (2 * math.pi) - 1.0
        
        # Clamp to valid range
        r_norm = torch.clamp(r_norm, -1.0, 1.0)
        theta_norm = torch.clamp(theta_norm, -1.0, 1.0)
        
        # Stack to create grid [H, W, 2]
        grid = torch.stack([theta_norm, r_norm], dim=-1)
        
        return grid
    
    def forward(self, x):
        """Transform from polar to Cartesian coordinates"""
        batch_size = x.size(0)
        
        # Expand grid for batch
        grid = self.inverse_grid.unsqueeze(0).expand(batch_size, -1, -1, -1)
        
        # Apply grid sampling
        cartesian_x = F.grid_sample(x, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        
        return cartesian_x

class DoubleConv(nn.Module):
    """Double convolution block"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Output convolution"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class PolarUNet(nn.Module):
    """
    Polar U-Net: U-Net with polar coordinate transformation
    
    The network transforms input to polar coordinates, applies U-Net processing,
    and transforms back to Cartesian coordinates.
    """
    def __init__(self, n_channels, n_classes, bilinear=False, polar_transform=True):
        super(PolarUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.polar_transform = polar_transform
        
        # Polar transformation layers
        if self.polar_transform:
            self.to_polar = PolarTransform(input_size=256, output_size=256)
            self.to_cartesian = InversePolarTransform(input_size=256, output_size=256)
        
        # U-Net architecture
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        # Transform to polar coordinates if enabled
        if self.polar_transform:
            x = self.to_polar(x)
        
        # U-Net forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        # Transform back to Cartesian coordinates if enabled
        if self.polar_transform:
            logits = self.to_cartesian(logits)
        
        return logits

    def get_model_info(self):
        """Return model information"""
        return {
            'name': 'PolarUNet',
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'bilinear': self.bilinear,
            'polar_transform': self.polar_transform,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        } 