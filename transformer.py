import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from norms import RMSNorm, LayerNorm

class Attention(nn.Module):
    def __init__(self, d_model, n_head, d_head, d_v):
        super().__init__()
        self.W_Q=nn.Linear(d_model, n_head*d_head, bias=False)
        self.W_K=nn.Linear(d_model, n_head*d_head, bias=False)
        self.W_V=nn.Linear(d_model, n_head*d_v, bias=False)
        self.W_O=nn.Linear(n_head*d_v, d_model)
        self.n_head=n_head
        self.d_head=d_head
        self.d_v=d_v
        
    def forward(self, x, mask=None):
        B, L, _ = x.shape
        Q=self.W_Q(x).reshape(B, L, self.n_head, self.d_head).transpose(1, 2)
        K=self.W_K(x).reshape(B, L, self.n_head, self.d_head).transpose(1, 2)
        V=self.W_V(x).reshape(B, L, self.n_head, self.d_v).transpose(1, 2)
        
        attn=Q@K.transpose(-1, -2)/(self.d_head**0.5)
        
        if mask is not None:
            attn=attn.masked_fill(mask==0, float('-inf'))
        
        return self.W_O((attn.softmax(dim=-1)@V).transpose(1,2).flatten(-2))
        
                
class FeedForward(nn.Module):
    def __init__(self, n_linear, d_linear, d_model, act):
        super().__init__()
        if(act=="gelu"):
            self.act=nn.GELU()
        elif(act=="relu"):
            self.act=nn.ReLU()
        elif(act=="none"):
            self.act=nn.Identity()
        else:
            raise ValueError(f"Invalid activation function: {act}")
        
        layers=[nn.Linear(d_model, d_linear)]
        layers.append(self.act)
        for _ in range(n_linear-2):
            layers.append(nn.Linear(d_linear, d_linear))
            layers.append(self.act)
        layers.append(nn.Linear(d_linear, d_model))
        self.layers=nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_head, d_head, d_v, n_linear, d_linear, act="gelu", norm='rms', normpos='pre'):
        super().__init__()
        
        if(norm=="rms"):
            self.norm1=RMSNorm(d_model)
            self.norm2=RMSNorm(d_model)
        elif(norm=="layer"):
            self.norm1=LayerNorm(d_model)
            self.norm2=LayerNorm(d_model)
        else:
            raise ValueError(f"Invalid normalization function: {norm}")
        
        self.normpos=normpos
        self.attn=Attention(d_model, n_head, d_head, d_v)
        self.ffn=FeedForward(n_linear, d_linear, d_model, act)
        
    def forward(self, x, mask=None):
        if(self.normpos=="pre"):
            x=self.attn(self.norm1(x), mask=mask)+x
            x=self.ffn(self.norm2(x))+x
        elif(self.normpos=="post"):
            x=self.norm1(self.attn(x, mask=mask)+x)
            x=self.norm2(self.ffn(x)+x)
        else:
            raise ValueError(f"Invalid normalization position: {self.normpos}")
        
        return x
        
class Transformer(nn.Module):
    def __init__(self, tokenizer, head, n_layers, d_model, n_head, d_head, d_v, n_linear, d_linear, act="gelu", norm='rms', normpos='pre'):
        super().__init__()
        layers=[]
        self.tokenizer=tokenizer
        self.layers=nn.ModuleList([TransformerBlock(d_model, n_head, d_head, d_v, n_linear, d_linear, act=act, norm=norm, normpos=normpos) for _ in range(n_layers)])
        self.head=head

    def forward(self, x, mask=None):
        x=self.tokenizer(x)
        for layer in self.layers:
            x=layer(x, mask=mask)
        return self.head(x)
    