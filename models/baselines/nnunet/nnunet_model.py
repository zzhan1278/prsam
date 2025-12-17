"""
nnU-Net: Self-configuring method for deep learning-based biomedical image segmentation
Simplified implementation for 2D liver segmentation

Reference: Isensee et al. "nnU-Net: a self-configuring method for deep learning-based 
biomedical image segmentation." Nature methods 18.2 (2021): 203-211.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDropoutNormReLU(nn.Module):
    """
    nnU-Net basic building block: Conv -> Dropout -> InstanceNorm -> LeakyReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, 
                 dropout_p=0.0, norm_type='instance'):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        if dropout_p > 0:
            self.dropout = nn.Dropout2d(dropout_p)
        else:
            self.dropout = None
            
        if norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None
            
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.norm is not None:
            x = self.norm(x)
        x = self.activation(x)
        return x


class StackedConvLayers(nn.Module):
    """
    Stacked convolution layers used in nnU-Net encoder/decoder
    """
    def __init__(self, in_channels, out_channels, num_convs=2, dropout_p=0.0):
        super().__init__()
        
        layers = []
        for i in range(num_convs):
            if i == 0:
                layers.append(ConvDropoutNormReLU(in_channels, out_channels, dropout_p=dropout_p))
            else:
                layers.append(ConvDropoutNormReLU(out_channels, out_channels, dropout_p=dropout_p))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)


class Downsample(nn.Module):
    """
    Downsampling block with strided convolution
    """
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class Upsample(nn.Module):
    """
    Upsampling block with transposed convolution
    """
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=stride, bias=False)
        self.norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.activation = nn.LeakyReLU(negative_slope=0.01, inplace=True)
    
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))


class nnUNet(nn.Module):
    """
    Simplified nnU-Net implementation for 2D medical image segmentation
    
    Args:
        n_channels: Number of input channels
        n_classes: Number of output classes
        deep_supervision: Use deep supervision (multiple outputs at different scales)
        base_channels: Base number of channels (will be scaled up in deeper layers)
        num_pool: Number of pooling operations (depth of the network)
    """
    def __init__(self, n_channels=1, n_classes=1, deep_supervision=True, 
                 base_channels=32, num_pool=5):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.deep_supervision = deep_supervision
        self.num_pool = num_pool
        
        # Calculate channel numbers for each level
        self.encoder_channels = [base_channels * (2**i) for i in range(num_pool + 1)]
        
        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()
        
        # First encoder block (input -> first feature level)
        self.encoder_blocks.append(
            StackedConvLayers(n_channels, self.encoder_channels[0], num_convs=2)
        )
        
        # Subsequent encoder blocks with downsampling
        for i in range(num_pool):
            # Downsampling
            self.downsample_blocks.append(
                Downsample(self.encoder_channels[i], self.encoder_channels[i+1])
            )
            # Encoder block
            self.encoder_blocks.append(
                StackedConvLayers(self.encoder_channels[i+1], self.encoder_channels[i+1], 
                                num_convs=2, dropout_p=0.0 if i < 3 else 0.3)
            )
        
        # Decoder path
        self.upsample_blocks = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        
        for i in range(num_pool):
            # Upsampling
            self.upsample_blocks.append(
                Upsample(self.encoder_channels[num_pool-i], self.encoder_channels[num_pool-i-1])
            )
            # Decoder block (concatenated features)
            self.decoder_blocks.append(
                StackedConvLayers(self.encoder_channels[num_pool-i-1] * 2, 
                                self.encoder_channels[num_pool-i-1], num_convs=2)
            )
        
        # Output layers
        self.final_conv = nn.Conv2d(self.encoder_channels[0], n_classes, kernel_size=1)
        
        # Deep supervision outputs (if enabled)
        if deep_supervision:
            self.deep_supervision_outputs = nn.ModuleList()
            for i in range(1, num_pool):
                # Use decoder channels instead of encoder channels
                self.deep_supervision_outputs.append(
                    nn.Conv2d(self.encoder_channels[num_pool-i], n_classes, kernel_size=1)
                )
    
    def forward(self, x):
        # Store encoder features for skip connections
        encoder_features = []
        
        # Encoder path
        current = x
        for i in range(self.num_pool + 1):
            current = self.encoder_blocks[i](current)
            encoder_features.append(current)
            
            if i < self.num_pool:
                current = self.downsample_blocks[i](current)
        
        # Decoder path
        for i in range(self.num_pool):
            # Upsample
            current = self.upsample_blocks[i](current)
            
            # Skip connection
            skip_feature = encoder_features[self.num_pool - i - 1]
            
            # Resize if necessary (handle odd dimensions)
            if current.shape[-2:] != skip_feature.shape[-2:]:
                current = F.interpolate(current, size=skip_feature.shape[-2:], 
                                      mode='bilinear', align_corners=False)
            
            # Concatenate and process
            current = torch.cat([current, skip_feature], dim=1)
            current = self.decoder_blocks[i](current)
        
        # Final output
        output = self.final_conv(current)
        
        # Deep supervision (return multiple outputs if enabled)
        if self.deep_supervision and self.training:
            # For simplicity, just return the main output for now
            # Deep supervision can be implemented later if needed
            return output
        else:
            return output
    
    def get_model_info(self):
        """Return model information"""
        return {
            'name': 'nnUNet',
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'deep_supervision': self.deep_supervision,
            'num_pool': self.num_pool,
            'encoder_channels': self.encoder_channels,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


def test_nnunet():
    """Test function for nnUNet"""
    model = nnUNet(n_channels=1, n_classes=1, deep_supervision=True)
    x = torch.randn(2, 1, 256, 256)
    
    # Test training mode (with deep supervision)
    model.train()
    with torch.no_grad():
        outputs = model(x)
    
    print(f"Input shape: {x.shape}")
    if isinstance(outputs, list):
        print(f"Number of outputs (deep supervision): {len(outputs)}")
        for i, output in enumerate(outputs):
            print(f"Output {i} shape: {output.shape}")
    else:
        print(f"Output shape: {outputs.shape}")
    
    # Test evaluation mode (single output)
    model.eval()
    with torch.no_grad():
        output = model(x)
    
    print(f"Eval mode output shape: {output.shape}")
    print(f"Model info: {model.get_model_info()}")
    
    return True


if __name__ == "__main__":
    success = test_nnunet()
    print(f"Test passed: {success}")
