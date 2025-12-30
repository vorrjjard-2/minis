import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn as nn
import pandas as pd 


from utils import pre_index

import albumentations as A 
from albumentations.pytorch import ToTensorV2

class MiniDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.transform = A.Compose([
            A.Resize(400, 400),
            ToTensorV2()
        ])

        id_annotations, id_images, id_categories = pre_index(
            os.path.join('datasets', 'coco_dataset', self.split, '_annotations.coco.json')
        )
        targets = {img_id: v[0][0] for img_id, v in id_annotations.items()}

        self.id_annotations = {i:v for i, v in enumerate(id_annotations.values())}
        self.id_images = {i:v for i, v in enumerate(id_images.values())}
        self.targets = {i:v for i, v in enumerate(targets.values())}

        self.len = len(self.id_annotations.keys())
       

    def __len__(self):
        return self.len
    
    
    def __getitem__(self, index):
        """
        Docstring for __getitem__
        
        Returns an image and target in a tuple.
        """
        raw_path = self.id_images[index][0]
        image_bgr = cv2.imread(os.path.join('datasets', 'coco_dataset', self.split, raw_path))
        image = np.array(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)) # type: ignore

        image = self.transform(image=image)['image'] if self.transform else None
            
        target = self.targets[index]
        return (image, target)


if __name__ == "__main__":
    dataset = MiniDataset('train')
    sample_img = dataset[0]
    assert sample_img[0].ndim == 3 and type(sample_img[1]) == int
    


