import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma=nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x*torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True)+1e-6)*self.gamma
    
class LayerNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.gamma=nn.Parameter(torch.ones(d_model))
        self.beta=nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        return (x-x.mean(dim=-1, keepdim=True))*torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True)+1e-6)*self.gamma+self.beta