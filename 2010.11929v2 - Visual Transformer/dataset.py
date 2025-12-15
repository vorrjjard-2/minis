import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn as nn


class MiniDataset(Dataset):
    def __init__(self):
        self.config = config
        self.split = split


