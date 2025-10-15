import torch
import torch.nn as nn
from norms import RMSNorm, LayerNorm

class ViTHead(nn.Module):
    def __init__(self, d_model, n_classes, norm='rms'):
        super().__init__()
        self.head=nn.Linear(d_model, n_classes)
        if(norm=="rms"):
            self.norm=RMSNorm(d_model)
        elif(norm=="layer"):
            self.norm=LayerNorm(d_model)
        else:
            raise ValueError(f"Invalid normalization function: {norm}")

    def forward(self, x):
        return self.head(self.norm(x))[:, 0]
