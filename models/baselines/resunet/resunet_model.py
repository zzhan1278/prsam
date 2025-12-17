"""
ResUNet implementation using ResNet as encoder backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResBlock(nn.Module):
    """Residual block"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out

class DecoderBlock(nn.Module):
    """Decoder block with residual connections"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv_block = nn.Sequential(
            ResBlock(in_channels // 2 + skip_channels, out_channels),
            ResBlock(out_channels, out_channels)
        )
        
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # Handle size mismatch
        diffY = skip.size()[2] - x.size()[2]
        diffX = skip.size()[3] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv_block(x)
        
        return x

class ResUNet(nn.Module):
    """
    ResUNet with ResNet34 encoder backbone.
    
    Args:
        n_channels: Number of input channels
        n_classes: Number of output classes
        backbone: ResNet backbone ('resnet18', 'resnet34', 'resnet50')
        pretrained: Use pretrained weights
    """
    def __init__(self, n_channels, n_classes, backbone='resnet34', pretrained=False):
        super(ResUNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.backbone_name = backbone
        
        # Load ResNet backbone
        if backbone == 'resnet18':
            backbone_model = models.resnet18(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet34':
            backbone_model = models.resnet34(pretrained=pretrained)
            encoder_channels = [64, 64, 128, 256, 512]
        elif backbone == 'resnet50':
            backbone_model = models.resnet50(pretrained=pretrained)
            encoder_channels = [64, 256, 512, 1024, 2048]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Modify first conv layer for single channel input
        if n_channels != 3:
            backbone_model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Extract encoder layers
        self.encoder1 = nn.Sequential(
            backbone_model.conv1,
            backbone_model.bn1,
            backbone_model.relu
        )  # 64 channels
        
        self.encoder2 = nn.Sequential(
            backbone_model.maxpool,
            backbone_model.layer1
        )  # 64 channels (ResNet18/34) or 256 (ResNet50)
        
        self.encoder3 = backbone_model.layer2  # 128 channels (ResNet18/34) or 512 (ResNet50)
        self.encoder4 = backbone_model.layer3  # 256 channels (ResNet18/34) or 1024 (ResNet50)
        self.encoder5 = backbone_model.layer4  # 512 channels (ResNet18/34) or 2048 (ResNet50)
        
        # Decoder
        self.decoder4 = DecoderBlock(encoder_channels[4], encoder_channels[3], 256)
        self.decoder3 = DecoderBlock(256, encoder_channels[2], 128)
        self.decoder2 = DecoderBlock(128, encoder_channels[1], 64)
        self.decoder1 = DecoderBlock(64, encoder_channels[0], 64)
        
        # Final classifier
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)      # 1/2
        enc2 = self.encoder2(enc1)   # 1/4
        enc3 = self.encoder3(enc2)   # 1/8
        enc4 = self.encoder4(enc3)   # 1/16
        enc5 = self.encoder5(enc4)   # 1/32
        
        # Decoder with skip connections
        dec4 = self.decoder4(enc5, enc4)  # 1/16
        dec3 = self.decoder3(dec4, enc3)  # 1/8
        dec2 = self.decoder2(dec3, enc2)  # 1/4
        dec1 = self.decoder1(dec2, enc1)  # 1/2
        
        # Final upsampling to original size
        dec1 = F.interpolate(dec1, scale_factor=2, mode='bilinear', align_corners=True)
        
        # Final classification
        out = self.final_conv(dec1)
        
        return out
    
    def get_model_info(self):
        """Return model information"""
        return {
            'name': 'ResUNet',
            'backbone': self.backbone_name,
            'n_channels': self.n_channels,
            'n_classes': self.n_classes,
            'total_params': sum(p.numel() for p in self.parameters()),
            'trainable_params': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }

def test_resunet():
    """Test function for ResUNet"""
    model = ResUNet(n_channels=1, n_classes=1, backbone='resnet34', pretrained=False)
    x = torch.randn(2, 1, 256, 256)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model info: {model.get_model_info()}")
    
    return output.shape == (2, 1, 256, 256)

if __name__ == "__main__":
    success = test_resunet()
    print(f"Test passed: {success}")
