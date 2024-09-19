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


class micronuclAI_inference(torch.utils.data.Dataset):
    def __init__(self, image, mask, resizing_factor=0.6, expansion=30, size=(256, 256), transform=None):
        self.size = size
        self.transform = transform
        self.boxes = BBoxes.from_mask(mask)
        self.boxes.image = image
        self.rf = self.boxes.calculate_resizing_factor(resizing_factor, self.size)
        self.boxes = self.boxes.expand(expansion)

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, index):
        image = self.boxes.grab_pixels_from(index, source="image", resize_factor=self.rf[index], size=self.size, rescale_intensity=True)
        if self.transform is not None:
            image = self.transform(image)
        return image
