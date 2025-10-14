import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma=nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        
    
class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma=nn.Parameter(torch.ones(d_model))
        self.beta=nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        