from .blocks import PatchEmbedding, EncoderBlock

import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn as nn
import pytorch_lightning as pl 
import torch.optim as optim

class ViT(nn.Module):
    def __init__(self, sizes=[256, 256], S=16, d_model=1024, N=12, num_classes=80):
        super().__init__()
        self.sizes = sizes
        self.S = S
        self.N = N
        self.d_model = d_model
        self.n_tokens = self.S ** 2
        self.num_classes = num_classes

        self.patcher = PatchEmbedding(self.sizes, self.S, self.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_tokens + 1, self.d_model)) # + 1 to accomodate cls token
        self.ln = nn.LayerNorm(self.d_model)

        self.blocks = nn.ModuleList([
            EncoderBlock(self.d_model) for i in range(self.N)
        ])

        self.head = nn.Linear(d_model, num_classes)

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

        #4. transformer
        for block in self.blocks:
            x = block(x)

        x = self.ln(x)

        #5. extract the CLS token
        x = x[:, 0]

        #6. make classification
        x = self.head(x)
        return x

class VitWrapper(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()

        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train/acc", acc, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

if __name__ == "__main__":
    model = ViT()
    test_input = torch.ones(8, 3, 256, 256)
    out = model(test_input)

    print(out.shape)