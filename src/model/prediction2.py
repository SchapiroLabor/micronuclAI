import argparse
import time

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from augmentations import preprocess_test as preprocess
from dataset import CINPrediction
from torch.utils.data import DataLoader
from models import (EfficientNetClassifier, MulticlassRegression)
from augmentations import get_transforms


def get_args():
    # Script description
    description = """Predits the probability of occurence of micronuclei in the cropped images in the input folder."""

    # Add parser
    parser = argparse.ArgumentParser(description)

    # Tool Input
    input = parser.add_argument_group(title="Input")
    input.add_argument("-i", "--image", dest="image", action="store", required=True,
                       help="Pathway to image.")
    input.add_argument("-m", "--mask", dest="mask", action="store", required=True,
                       help="Pathway to mask.")
    input.add_argument("-mod", "--model", dest="model", action="store", required=True,
                       help="Pathway to prediction model.")

    # Optional input
    options = parser.add_argument_group(title="Non-required arguments")
    options.add_argument("-s", "--size", dest="size", action="store", required=False, default=(256, 256),
                         type=int, nargs="+", help="Size of images for training. [Default = (256, 256)]")
    options.add_argument("-rf", "--resizing_factor", dest="resizing_factor", action="store", required=False,
                         default=0.7, type=float, help="Resizing factor for images. [Default = 0.7]")
    options.add_argument("-p", "--precision", dest="precision", action="store", default="32",
                        choices=["16-mixed", "bf16-mixed", "16-true", "bf16-true", "32", "64"],
                        help="Precision for training. [Default = bf16-mixed]")
    options.add_argument("-d", "--device", dest="device", action="store", required=False, default="cpu",
                         help="Device to be used for training [default='cpu']")

    # Tool output
    output = parser.add_argument_group(title="Output")
    output.add_argument("-o", "--out", dest="out", action="store", required=True,
                        help="Path to the output data folder")

    # Parse arguments
    args = parser.parse_args()

    # Standardize paths
    args.image = Path(args.image).resolve()
    args.mask = Path(args.mask).resolve()
    args.out = Path(args.out).resolve()

    return args


def summarize(df_predictions):
    # Get micronuclei counts
    df_predictions["micronuclei"] = df_predictions["score"].apply(lambda x: round(x) if x > 0.5 else 0)

    # Get dataset summary
    print("Calculating summary.")
    df_summary = df_predictions["micronuclei"].value_counts()

    total = df_summary.sum()
    total_micronuclei = sum(df_summary.index * df_summary.values)
    cells_with_micronuclei = df_summary[df_summary.index > 0].sum()
    cells_with_micronuclei_ratio = cells_with_micronuclei / total
    micronuclei_ratio = total_micronuclei / total

    # Add summary to dataframe
    df_summary["total_cells"] = total
    df_summary["total_micronuclei"] = total_micronuclei
    df_summary["cells_with_micronuclei"] = cells_with_micronuclei
    df_summary["cells_with_micronuclei_ratio"] = cells_with_micronuclei_ratio
    df_summary["micronuclei_ratio"] = micronuclei_ratio

    return df_summary


def main(args):
    torch.set_float32_matmul_precision('high')
    # Load model
    model = torch.load(args.model, map_location=args.device)

    # Predicting
    trainer = pl.Trainer(precision=args.precision,
                         accelerator=args.device)

    # Load data transformations
    transform = get_transforms(resize=args.size, training=False, prediction=True)

    # Dataset
    dataset = CINPrediction(args.image,
                            args.mask,
                            resizing_factor=args.resizing_factor,
                            size=args.size,
                            transform=transform)

    # Dataloader
    dataloader = DataLoader(dataset, num_workers=8, pin_memory=True, batch_size=1)

    #  Getting predictions
    predictions = np.concatenate(trainer.predict(model, dataloader), axis=0)
    ids = np.arange(1, len(predictions)+1)

    # Create dictionary with results
    dict_tmp = {
        "image": ids,
        "score": predictions
    }

    # Create dataframes for predictions and for the summary
    df_predictions = pd.DataFrame.from_dict(dict_tmp)
    df_summary = summarize(df_predictions)

    # Save output file
    print("Finished prediction. Saving output file.")
    args.out.mkdir(parents=True, exist_ok=True)
    df_predictions.to_csv(args.out.joinpath(f"{args.mask.stem}_predictions.csv"), index=False)
    df_summary.to_csv(args.out.joinpath(f"{args.mask.stem}_summary.csv"), index=True)



if __name__ == "__main__":
    # Read arguments from command line
    args = get_args()

    # Run script and calculate run time
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finish in {rt//60:.0f}m {rt%60:.0f}s")