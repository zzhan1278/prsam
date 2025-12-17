"""
VNet implementation adapted for 2D medical image segmentation.
Based on "V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class VNetBlock(nn.Module):
    """VNet convolution block with residual connection"""
    def __init__(self, in_channels, out_channels, num_convs=2, use_residual=True):
        super(VNetBlock, self).__init__()
        self.use_residual = use_residual
        
        layers = []
        for i in range(num_convs):
            if i == 0:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2))
            else:
                layers.append(nn.Conv2d(out_channels, out_channels, kernel_size=5, padding=2))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.PReLU(out_channels))
        
        self.conv_block = nn.Sequential(*layers)
        
        # Residual connection
        if use_residual and in_channels != out_channels:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        elif use_residual:
            self.residual_conv = nn.Identity()
    
    def forward(self, x):
        out = self.conv_block(x)
        
        if self.use_residual:
            residual = self.residual_conv(x)
            out = out + residual
            
        return out

class DownTransition(nn.Module):
    """Downsampling transition"""
    def __init__(self, in_channels, out_channels, num_convs=2, use_residual=True):
        super(DownTransition, self).__init__()
        
        self.down_conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        self.conv_block = VNetBlock(out_channels, out_channels, num_convs, use_residual)
        
    def forward(self, x):
        x = self.down_conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        x = self.conv_block(x)
        return x

class UpTransition(nn.Module):
    """Upsampling transition with skip connection"""
    def __init__(self, in_channels, skip_channels, out_channels, num_convs=2, use_residual=True):
        super(UpTransition, self).__init__()
        
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)
        
        # Combine upsampled features with skip connection
        combined_channels = out_channels + skip_channels
        self.conv_block = VNetBlock(combined_channels, out_channels, num_convs, use_residual)
        
    def forward(self, x, skip):
        x = self.up_conv(x)
        x = self.bn(x)
        x = self.prelu(x)
        
        # Handle size mismatch
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        
        # Concatenate with skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        
        return x

class VNet2D(nn.Module):
    """
    2D VNet implementation for medical image segmentation.
    
    Args:
        n_channels: Number of input channels
        n_classes: Number of output classes
        use_residual: Use residual connections in VNet blocks
    """
    def __init__(self, n_channels, n_classes, use_residual=True):
        super(VNet2D, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_residual = use_residual
        
        # Initial convolution
        self.input_conv = VNetBlock(n_channels, 16, num_convs=1, use_residual=False)
        
        # Encoder (downsampling path)
        self.down1 = DownTransition(16, 32, num_convs=2, use_residual=use_residual)
        self.down2 = DownTransition(32, 64, num_convs=3, use_residual=use_residual)
        self.down3 = DownTransition(64, 128, num_convs=3, use_residual=use_residual)
        self.down4 = DownTransition(128, 256, num_convs=3, use_residual=use_residual)
        
        # Decoder (upsampling path)
        self.up4 = UpTransition(256, 128, 128, num_convs=3, use_residual=use_residual)
        self.up3 = UpTransition(128, 64, 64, num_convs=3, use_residual=use_residual)
        self.up2 = UpTransition(64, 32, 32, num_convs=2, use_residual=use_residual)
        self.up1 = UpTransition(32, 16, 16, num_convs=1, use_residual=use_residual)
        
        # Output layer
        self.output_conv = nn.Conv2d(16, n_classes, kernel_size=1)
        
    def forward(self, x):
        # Input
        x0 = self.input_conv(x)
        
        # Encoder
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        # Decoder
        y3 = self.up4(x4, x3)
        y2 = self.up3(y3, x2)
        y1 = self.up2(y2, x1)
        y0 = self.up1(y1, x0)
        
        # Output
        output = self.output_conv(y0)
        
        return output
    
    def get_model_info(self):
        """Return model information"""
        return {
            'name': 'VNet2D',
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'use_residual': self.use_residual,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

def test_vnet2d():
    """Test function for VNet2D"""
    model = VNet2D(n_channels=1, n_classes=1, use_residual=True)
    x = torch.randn(2, 1, 256, 256)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model info: {model.get_model_info()}")
    
    return output.shape == (2, 1, 256, 256)

if __name__ == "__main__":
    success = test_vnet2d()
    print(f"Test passed: {success}")
