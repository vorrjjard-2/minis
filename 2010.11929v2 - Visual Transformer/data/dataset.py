import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn as nn
import pandas as pd 

import albumentations as A 
from albumentations.pytorch import ToTensorV2

import collections 
from collections import defaultdict
import json 

def pre_index(annotation_path):
    with open(annotation_path, 'r') as file:
        annotations = json.load(file)

    id_annotations = defaultdict(list)
    id_images = {}
    id_categories = {} 

    for item in annotations["annotations"]:
        img_id = item['image_id']
        id_annotations[img_id].append([item['category_id']] + item['bbox'])

    for item in annotations["images"]:
        id = item['id']
        id_images[id] = (item['file_name'], item['width'], item['height'])
    
    for item in annotations['categories']:
        category_id = item['id']
        id_categories[category_id] = item['name']
        
    return (id_annotations, id_images, id_categories)


class MiniDataset(Dataset):
    def __init__(self, split, path):
        self.split = split
        self.path = path
        self.transform = A.Compose([
            A.Resize(256, 256),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0,
                p=1),
            ToTensorV2()
        ])

        id_annotations, id_images, id_categories = pre_index(
            os.path.join(self.path, self.split, '_annotations.coco.json')
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
        image_bgr = cv2.imread(os.path.join(self.path, self.split, raw_path))
        image = np.array(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)) # type: ignore

        image = self.transform(image=image)['image'] if self.transform else image

        image = image.to(torch.float32)
            
        target = self.targets[index]
        return (image, target)


if __name__ == "__main__":
    dataset = MiniDataset('train')
    sample_img = dataset[0]
    assert sample_img[0].ndim == 3 and type(sample_img[1]) == int
    


