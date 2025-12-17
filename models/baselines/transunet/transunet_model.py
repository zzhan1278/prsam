import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""
    
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention"""
    
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """MLP block"""
    
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Block"""
    
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer Encoder"""
    
    def __init__(self, img_size=256, patch_size=16, in_chans=1, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Initialize position embeddings
        torch.nn.init.normal_(self.pos_embed, std=.02)
        
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        for blk in self.blocks:
            x = blk(x)
            
        x = self.norm(x)
        return x


class ConvBlock(nn.Module):
    """Basic convolutional block"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class TransUNet(nn.Module):
    """TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation"""
    
    def __init__(self, img_size=256, patch_size=16, n_channels=1, n_classes=1, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Vision Transformer Encoder
        self.vit = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=n_channels,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 512, 2, stride=2),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(512, 256, 2, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, n_classes, 1)
        )
        
    def forward(self, x):
        # Vision Transformer Encoder
        vit_out = self.vit(x)  # [B, num_patches, embed_dim]
        
        # Reshape transformer output back to feature map
        B, N, C = vit_out.shape
        h = w = int(math.sqrt(N))
        vit_out = vit_out.transpose(1, 2).reshape(B, C, h, w)
        
        # Decoder
        x = self.decoder(vit_out)
        
        return x  # Return raw logits (sigmoid will be applied in loss function)


def test_transunet():
    """Test function for TransUNet"""
    model = TransUNet(img_size=256, patch_size=16, n_channels=1, n_classes=1, embed_dim=768, depth=12, num_heads=12)
    x = torch.randn(1, 1, 256, 256)
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    return output.shape == (1, 1, 256, 256)


if __name__ == "__main__":
    test_transunet() 