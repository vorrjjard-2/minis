import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import torch.nn as nn

from utils import pre_index

class MiniDataset(Dataset):
    def __init__(self, split):
        self.split = split
        self.id_annotations, self.id_images, self.id_categories = pre_index(
            os.path.join('datasets', 'coco_dataset', self.split, '_annotations.coco.json')
        )
        self.targets = {img_id: v[0][0] for img_id, v in self.id_annotations.items()}

    def __getitem__(self, index):
        """
        Docstring for __getitem__
        
        Returns an image and target in a tuple.
        """
        raw_path, _ , _ = self.id_images[index]
        image_bgr = cv2.imread(os.path.join('datasets', 'coco_dataset', self.split, raw_path))
        image = np.array(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)) # type: ignore

        target = self.targets[index]
        return (image, target)


if __name__ == "__main__":
    dataset = MiniDataset('train')
    assert dataset[0][0].ndim == 3 and type(dataset[0][1]) == int
    print('All Tests Passed!')

