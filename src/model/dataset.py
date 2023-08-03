import torch
import os
import pandas as pd
import PIL
from mask2bbox import BBoxes

class CINDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, images_folder, transform = None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        filename = self.df.loc[index, "image"]
        label = self.df.loc[index, "count"]
        image = PIL.Image.open(os.path.join(self.images_folder, filename))
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class CINPrediction(torch.utils.data.Dataset):
    def __init__(self, image, mask, resizing_factor=0.7, size=(256, 256), transform=None):
        self.size = size
        self.transform = transform
        self.boxes = BBoxes.from_mask(mask)
        self.boxes.image = image
        self.rf = self.boxes.calculate_resizing_factor(resizing_factor, self.size)

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, index):
        image = self.boxes.extract_single(index, resize_factor=self.rf[index], size=self.size)
        if self.transform is not None:
            image = self.transform(image)
        return image
