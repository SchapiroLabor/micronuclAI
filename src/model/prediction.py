import argparse
import time

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from augmentations import preprocess_test as preprocess
from dataset import CINDataset
from models import (EfficientNetClassifier, BinaryClassifierModel)


def get_args():
    # Script description
    description = """Predits the probability of occurence of micronuclei in the cropped images in the input folder."""

    # Add parser
    parser = argparse.ArgumentParser(description)

    # Tool Input
    input = parser.add_argument_group(title="Input")
    input.add_argument("-i", "--images", dest="images", action="store", required=True,
                       help="Pathway to image folder.")
    input.add_argument("-m", "--model", dest="model", action="store", required=True,
                       help="Pathway to prediction model.")
    input.add_argument("-d", "--device", dest="device", action="store", required=False, default="cpu",
                       help="Device to be used for training [default='cpu']")

    # Tool output
    output = parser.add_argument_group(title="Output")
    output.add_argument("-o", "--out", dest="out", action="store", required=True,
                        help="Path to the output data folder")

    # Parse arguments
    args = parser.parse_args()

    # Standardize paths
    args.images = Path(args.images).resolve()
    args.labels = Path(args.model).resolve()
    args.out = Path(args.out).resolve()

    return args


def main(args):
    # Get list of file names
    list_cropped_files = [path.name for path in args.images.iterdir()]

    # Load model and set to evaluation
    device = args.device
    print(f"Using device = {device}")
    net = torch.load(args.model, map_location=device)#.to(device)
    net.eval()

    # Iterate over files
    list_predictions = []
    for image in tqdm(args.images.iterdir()):
        img_pil = Image.open(image)
        img_tensor = preprocess(img_pil).unsqueeze(0).to(device)
        y = net(img_tensor).cpu().detach().numpy()
        list_predictions.append(y[0][0])

    # Create dictionary with results
    dict_tmp = {
        "image": list_cropped_files,
        "prediction": list_predictions
    }

    # Save output file
    df_predictions = pd.DataFrame.from_dict(dict_tmp)
    args.out.mkdir(parents=True, exist_ok=True)
    df_predictions.to_csv(args.out.joinpath(f"{args.images.name}_predictions.csv"), index=False)


if __name__ == "__main__":
    # Read arguments from command line
    args = get_args()

    # Run script and calculate run time
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finish in {rt//60:.0f}m {rt%60:.0f}s")