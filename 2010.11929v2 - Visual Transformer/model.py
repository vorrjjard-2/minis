import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn as nn
import pytorch_lightning as pl 

class PatchEmbedding():
    def __init__(self, sizes=[256, 256], S=16):
        self.sizes = sizes
        self.S = S 
        self.P = self.sizes[0] // self.S
        self.proj = nn.Linear(768, 1024)

    def forward(self, x):
        """
        x.shape = [B, C, H, W]
        Return B, N, D
        """

        P = self.P

        B, C, H, W = x.shape

        x = x.reshape(B, C, H // P, P, W // P, P)
        x = x.permute(0, 2, 4, 1, 3, 5) 
        x = x.reshape(B, (H // P) * (W // P), C * P * P)
        x = self.proj(x)
        return x

if __name__ == "__main__":
    patcher = PatchEmbedding()
    img = torch.zeros((1, 3, 256, 256))

    res = patcher.forward(img)
    print(res.shape)






        
