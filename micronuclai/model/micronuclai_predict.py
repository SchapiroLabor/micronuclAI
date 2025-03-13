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
from dataset import micronuclAI_inference
from torch.utils.data import DataLoader
from augmentations import get_transforms
from pytorch_lightning import LightningModule
from typing import Union, Tuple, Any
from logger import set_logger

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
                         default=0.6, type=float, help="Resizing factor for images. [Default = 0.6]")
    options.add_argument("-e", "--expansion", dest="expansion", action="store", required=False, default=25,
                         type=int, help="Expansion factor for images. [Default = 25]")
    options.add_argument("-d", "--device", dest="device", action="store", required=False, default="cpu",
                         help="Device to be used for training [default='cpu']")
    options.add_argument("-bs", "--batch_size", dest="batch_size", action="store", required=False, default=1,
                         type=int, help="Batch size for training. [Default = 0]")

    # Tool output
    output = parser.add_argument_group(title="Output")
    output.add_argument("-o", "--out", dest="out", action="store", required=True,
                        help="Path to the output data folder")

    # Script options
    options = parser.add_argument_group(title="Script options")
    options.add_argument("-log", "--log-level", dest="log_level", action="store", default="info",
                         choices=["debug", "info"],
                         help="Set the logging level. [Default = info]")
    options.add_argument("--version", action="version", version="micronuclAI 1.0.0")

    # Parse arguments
    args = parser.parse_args()

    # Standardize paths
    args.image = Path(args.image).resolve()
    args.mask = Path(args.mask).resolve()
    args.out = Path(args.out).resolve()

    return args


def inference(model, dataloader, device, log=None):
    # Set model to evaluation mode and disable gradient computation for inference
    model.to(device)
    model.eval()

    # Iterate over batches and make predictions
    with torch.no_grad():
        scores = []
        for batch in tqdm(dataloader):
            x = batch.to(device)
            y_pred = model(x)
            scores.append(y_pred.cpu().numpy().flatten())

    # Convert scores to dataframe with cellID names
    scores = np.hstack(scores)
    ids = np.arange(1, len(scores)+1)

    # Create dictionary with results
    dict_tmp = {
        "cellID": ids,
        "score": scores
    }

    # Create dataframes for predictions and for the summary
    df_predictions = pd.DataFrame.from_dict(dict_tmp)

    # Get micronuclei counts
    df_predictions["micronuclei"] = df_predictions["score"].apply(lambda x: round(x) if x > 0.5 else 0)

    # Return list of predictions
    return df_predictions


def summarize(df_mn_counts, log=None):
    # Get dataset summary
    total = df_mn_counts.sum()
    total_micronuclei = sum(df_mn_counts.index * df_mn_counts.values)
    cells_with_micronuclei = df_mn_counts[df_mn_counts.index > 0].sum()
    cells_with_micronuclei_ratio = cells_with_micronuclei / total
    micronuclei_ratio = total_micronuclei / total

    # Add summary to dataframe
    summary = {
        "total_cells": int(total),
        "total_micronuclei": int(total_micronuclei),
        "cells_with_micronuclei": int(cells_with_micronuclei),
        "cells_with_micronuclei_ratio": cells_with_micronuclei_ratio,
        "micronuclei_ratio": micronuclei_ratio,}

    df_summary = pd.DataFrame.from_dict(summary, orient="index")

    return df_summary


def main():
    # Read arguments from CLI
    args = get_args()

    # Set logger and output info
    log = set_logger(log_level=args.log_level)
    log.info(f"Starting micronuclAI prediction")
    log.info(f"Input image      = {args.image}")
    log.info(f"Input mask       = {args.mask}")
    log.info(f"Prediction model = {args.model}")
    log.info(f"Output folder    = {args.out}")

    # Set device
    device = torch.device(args.device)
    log.info(f"Using device     = {device}")

    # Load model
    log.info("Loading model")
    model = torch.jit.load(args.model)

    # Load data transformations
    transform = get_transforms(resize=args.size, training=False, prediction=True)

    # Load dataset
    dataset = micronuclAI_inference(args.image,
                                    args.mask,
                                    resizing_factor=args.resizing_factor,
                                    expansion=args.expansion,
                                    size=args.size,
                                    transform=transform)

    # Create a data loader
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)

    # Inference step
    log.info("Predicting micronuclei")
    df_predictions = inference(model, loader, device, log=log)

    # Get micronuclei group by counts
    log.info("Calculating summary")
    df_mn_counts = df_predictions["micronuclei"].value_counts()

    # Get summary
    log.info("Summarizing predictions")
    df_summary = summarize(df_mn_counts, log=log)

    # Save predictions
    log.info("Saving predictions")
    args.out.mkdir(parents=True, exist_ok=True)
    df_predictions.to_csv(args.out.joinpath(f"{args.mask.stem}_predictions.csv"), index=False)
    df_mn_counts.to_csv(args.out.joinpath(f"{args.mask.stem}_counts.csv"), index=True)
    df_summary.to_csv(args.out.joinpath(f"{args.mask.stem}_summary.csv"), index=True, header=False)


if __name__ == "__main__":
    main()