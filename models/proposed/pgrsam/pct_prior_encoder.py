import torch
import torch.nn as nn
import torch.nn.functional as F


class PCTPriorEncoder(nn.Module):
    """
    Plan-CT prior encoder (N_prior)
    - Input: plan CT slice tensor [B, 1 or 3, 256, 256] (we'll use 1 channel)
    - Output: prior feature map [B, 256, 64, 64], aligned with SAM image encoder output
    """

    def __init__(self, input_atlas_channels: int = 1, output_prior_channels: int = 256,
                 target_spatial_size=(64, 64)):
        super().__init__()

        self.target_h, self.target_w = target_spatial_size

        # Simple CNN downsampling from 256x256 -> 128x128 -> 64x64, then project to 256 channels
        self.encoder = nn.Sequential(
            nn.Conv2d(input_atlas_channels, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, output_prior_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(output_prior_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, atlas_image: torch.Tensor) -> torch.Tensor:
        """
        atlas_image: [B, C, 256, 256] in [0,1]
        returns: [B, 256, 64, 64]
        """
        x = self.encoder(atlas_image)
        if x.shape[-2:] != (self.target_h, self.target_w):
            x = F.interpolate(x, size=(self.target_h, self.target_w), mode='bilinear', align_corners=False)
        return x


