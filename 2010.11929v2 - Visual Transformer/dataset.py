import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn as nn

from utils import pre_index

import albumentations as A 
from albumentations.pytorch import ToTensorV2


train_transform = A.Compose([
    A.Resize(400, 400),
    ToTensorV2()

])



class MiniDataset(Dataset):
    def __init__(self, split, transform):
        self.split = split
        self.id_annotations, self.id_images, self.id_categories = pre_index(
            os.path.join('datasets', 'coco_dataset', self.split, '_annotations.coco.json')
        )
        self.targets = {img_id: v[0][0] for img_id, v in self.id_annotations.items()}
        self.transform = transform

    def __getitem__(self, index):
        """
        Docstring for __getitem__
        
        Returns an image and target in a tuple.
        """
        raw_path, _ , _ = self.id_images[index]
        image_bgr = cv2.imread(os.path.join('datasets', 'coco_dataset', self.split, raw_path))
        image = np.array(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)) # type: ignore

        if self.transform :
            image = self.transform(image=image)['image']
            
        target = self.targets[index]
        return (image, target)


if __name__ == "__main__":
    dataset = MiniDataset('train', train_transform)
    assert dataset[0][0].ndim == 3 and type(dataset[0][1]) == int
    print('All Tests Passed!')

