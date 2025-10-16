import torch
import torch.nn as nn

class ImageTokenizer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, d_model):
        super().__init__()
        self.img_size = img_size
        self.num_patches = (img_size // patch_size)**2
        self.proj=nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        self.pos_enc = nn.Parameter(torch.randn(1, self.num_patches, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x):
        out=self.proj(x).flatten(-2).transpose(-1, -2)
        B = x.shape[0]
        out=out+self.pos_enc
        out=torch.cat([self.cls_token.expand(B, -1, -1), out], dim=1)
        return out