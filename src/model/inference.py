import argparse
import time
import torch
from pandas import DataFrame
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from src.model.dataset import micronuclAI_inference
from src.model.augmentations import get_transforms
from src.model.logger import set_logger
from typing import Any


def get_args():
    # Script description
    description = """micronuclAI prediction script - Predicts the amount of micro-nuclei in the input image."""

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

    # Tool output
    output = parser.add_argument_group(title="Output")
    output.add_argument("-o", "--out", dest="out", action="store", required=True,
                        help="Path to the output folder")
    output.add_argument("-f", "--file-name", dest="file_name", action="store", required=False, default=None,
                        help="Name of the output file, if not filename provided it will take the mask file name.")

    # Optional input
    options = parser.add_argument_group(title="Tool options")
    options.add_argument("-s", "--size", dest="size", action="store", required=False, default=(256, 256),
                         type=int, nargs="+", help="Size of images for training. [Default = (256, 256)]")
    options.add_argument("-rf", "--resizing_factor", dest="resizing_factor", action="store", required=False,
                         default=0.6, type=float, help="Resizing factor for images. [Default = 0.6]")
    options.add_argument("-e", "--expansion", dest="expansion", action="store", required=False, default=25,
                         type=int, help="Expansion factor for images. [Default = 25]")
    options.add_argument("-d", "--device", dest="device", action="store", required=False, default="cpu",
                         help="Device to be used for training [default='cpu']")
    options.add_argument("-log", "--log_level", dest="log_level", action="store", default="info",
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


def predict(model, dataloader, device) -> list:
    """
    Predicts the amount of micro-nuclei in the input image.
    :param model: Model to be used for prediction.
    :param dataloader: Dataloader to be used for prediction.
    :param device: Device to be used for prediction.
    :return: List with the amount of micro-nuclei in the input image.
    """
    # Set the model to evaluation mode
    model.eval()
    model.to(device)

    # Predict
    predictions = []
    for batch in tqdm(dataloader, desc="Predicting"):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch).squeeze().cpu().numpy()
            predictions.extend(pred)

    return predictions


def summarize(predictions) -> tuple[DataFrame, Any]:
    """
    Summarizes the amount of micro-nuclei in the input image.
    :param predictions: list with the amount of micro-nuclei in the input image.
    :return: Dataframe with the amount of micro-nuclei in the input image.
    """
    # Generate a list of IDs
    ids = np.arange(1, len(predictions)+1)

    # Create dictionary with results
    df_predictions = pd.DataFrame({"CellID": ids, "score": predictions})

    # Get micronuclei counts
    df_predictions["micronuclei"] = df_predictions["score"].apply(lambda x: round(x) if x > 0.5 else 0)

    # Get dataset summary
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

    return df_predictions, df_summary


def main():
    # Get arguments
    args = get_args()

    # Set logger
    lg = set_logger(log_level=args.log_level)
    lg.info("micronuclAI")

    # Load model
    lg.info("Loading model")
    model = torch.jit.load(args.model)

    # Load data transformations
    transform = get_transforms(resize=args.size, training=False, prediction=True)

    # Dataset
    lg.info(f"Creating dataset")
    lg.debug(f"Loading image from     = {args.image}")
    lg.debug(f"Loading mask from      = {args.mask}")
    lg.debug(f"Resizing factor        = {args.resizing_factor}")
    lg.debug(f"Expansion              = {args.expansion}")
    lg.debug(f"Size                   = {args.size}")
    dataset = micronuclAI_inference(args.image,
                                    args.mask,
                                    resizing_factor=args.resizing_factor,
                                    expansion=args.expansion,
                                    size=args.size,
                                    transform=transform)

    # Create dataloader, set to 0 workers to avoid overhead with multiprocessing.
    # a batch size of 32 should be easy to handle in most systems.
    lg.info(f"Creating dataloader")
    dataloader = DataLoader(dataset,
                            num_workers=0,
                            batch_size=64)

    # Predict
    lg.info("Predicting")
    lg.debug(f"Predicting with device = {args.device}")
    lg.debug(f"Number of cell patches = {len(dataset)}")
    predictions = predict(model, dataloader, args.device)

    # Summarize
    lg.info("Summarizing predictions")
    df_predictions, df_summary = summarize(predictions)

    # Save predictions and summary
    lg.info("Saving predictions and summary")

    # Take the mask file name and use it
    if args.file_name is not None:
        file_name = args.file_name
    else:
        file_name = args.mask.stem

    lg.debug(f"Saving predictions to {args.out / file_name}_predictions.csv")
    lg.debug(f"Saving summary to {args.out / file_name}_summary.csv")
    args.out.mkdir(parents=True, exist_ok=True)  # Creates output directory if it doesn't exist
    df_predictions.to_csv(args.out / f"{file_name}_predictions.csv", index=False)
    df_summary.to_csv(args.out / f"{file_name}_summary.csv", index=True)
    lg.debug(f"{df_summary}")
    lg.info("Done!")


if __name__ == "__main__":
    main()
