import torch
import torch.nn as nn
import torch.nn.functional as F


class CorrectionNetwork(nn.Module):
    def __init__(self,
                 ecbct_channels: int = 256,
                 eprior_channels: int = 256,
                 output_delta_e_channels: int = 256,
                 hidden_channels_factor: int = 1):
        super().__init__()

        input_channels = ecbct_channels + eprior_channels
        mid_channels = input_channels // (2 * hidden_channels_factor)
        if mid_channels == 0:
            mid_channels = output_delta_e_channels // 2

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.conv_out = nn.Conv2d(mid_channels, output_delta_e_channels, kernel_size=1)

    def forward(self, E_cbct: torch.Tensor, E_prior: torch.Tensor) -> torch.Tensor:
        if E_cbct.shape[2:] != E_prior.shape[2:]:
            E_prior = F.interpolate(E_prior, size=E_cbct.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([E_cbct, E_prior], dim=1)
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        delta_E = self.conv_out(x)
        return delta_E


