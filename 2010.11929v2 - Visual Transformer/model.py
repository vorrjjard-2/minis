from blocks import PatchEmbedding

import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn as nn
import pytorch_lightning as pl 

class ViT(nn.Module):
    def __init__(self, sizes=[256, 256], S=16, d_model=1024):
        super().__init__()
        self.sizes = sizes
        self.S = S
        self.d_model = d_model
        self.n_tokens = self.S ** 2

        self.patcher = PatchEmbedding(self.sizes, self.S, self.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens + 1, self.d_model)) # + 1 to accomodate cls token

    def forward(self, x):
        B = x.shape[0]
        # 1. Patch images and return token 
        x = self.patcher(x)

        # 2. add cls
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # 3. add pos embed
        pos_embeds = self.pos_embed.expand(B, -1 , -1) # has shape B, N+1, d_model
        x = x + pos_embeds
        return x

class VitWrapper(pl.LightningModule):
    def __init__(self, model):
        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, x):
        x = self.forward(x)



if __name__ == "__main__":
    model = ViT()
    test_input = torch.zeros(3, 3, 256, 256)
    out = model(test_input)
