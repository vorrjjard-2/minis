import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn as nn
import pytorch_lightning as pl 
import einops
from einops import rearrange
import math 

import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, sizes=[256, 256], S=16, d_model=1024):
        super().__init__()
        self.sizes = sizes
        self.S = S 
        self.P = self.sizes[0] // self.S
        self.d_model = d_model
        self.proj = nn.Linear(768, self.d_model)

    def forward(self, x):
        """
        x.shape = [B, C, H, W]
        Return B, N, D
        """

        P = self.P

        B, C, H, W = x.shape

        x = rearrange(x, 'b c (hp p1) (wp p2) -> b (hp wp) (c p1 p2)', p1=P, p2=P)
        x = self.proj(x)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads=8, d_model=1024):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.scaler = math.sqrt(self.d_model // self.n_heads)
        self.QKV = nn.Linear(self.d_model, 3 * self.d_model)
        self.proj = nn.Linear(self.d_model, self.d_model)

    def forward(self, x):
        qkv = self.QKV(x) # B, P, d_model * 3
        Q, K, V = torch.chunk(qkv, 3, dim=-1) # B, P, d_model (per tensor)

        Q = rearrange(Q, 'b p (h x) -> b h p x', h=self.n_heads) # B, H, p, x ()
        K = rearrange(K, 'b p (h x) -> b h p x', h=self.n_heads) # B, H, p, x ()
        V = rearrange(V, 'b p (h x) -> b h p x', h=self.n_heads) # B, H, p, x ()

        head_attns = torch.matmul(F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / self.scaler, dim=-1), V)
        attn_score = rearrange(head_attns, 'b h t d -> b t (h d)')

        y = self.proj(attn_score)

        return attn_score
    
class MLP(nn.Module):
    def __init__(self, sizes=[1024, 1024 * 4, 1024]):
        super().__init__()
        self.sizes = sizes
        self.layers = nn.ModuleList([
            nn.Linear(self.sizes[0], self.sizes[1]),
            nn.GELU(),
            nn.Linear(self.sizes[1], self.sizes[2])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, d_model=1024):
        super().__init__()
        self.d_model = d_model

        self.attn = MultiHeadAttention()
        self.mlp = MLP()
        self.ln1 = nn.LayerNorm(self.d_model)
        self.ln2 = nn.LayerNorm(self.d_model)

    def forward(self, x):
        x = self.ln1(x)
        x = self.attn(x) + x 

        x = self.ln2(x) 
        x = self.mlp(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, N, d_model=1024):
        super().__init__()
        self.N = N
        self.d_model = d_model
        self.layers = nn.ModuleList([
            EncoderBlock(self.d_model) for i in range(self.N)
            ])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    enc = TransformerEncoder(4)
    x = torch.ones(3, 257, 1024)

    y = enc.forward(x)
    print(y.shape)