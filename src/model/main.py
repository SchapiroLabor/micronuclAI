import pickle
import torch
import time
from pathlib import Path
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import PIL
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import numpy as np
import pandas as pd
from dataset import CINDataset
from models import (EfficientNetClassifier, BinaryClassifierModel)
from augmentations import get_transforms
from augmentations import preprocess_test as pt
from augmentations import preprocess_train as ptr
from utils import evaluate_binary_model, evaluate_multiclass_model, plot_confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse


def get_args():
    # Script description
    description = """Use EfficientNet to train a classifier"""

    # Add parser
    parser = argparse.ArgumentParser(description)

    # Tool Input
    input = parser.add_argument_group(title="Inputs")
    input.add_argument("-i", "--images", dest="images", action="store", required=True,
                       help="Pathway to input image folder.")
    input.add_argument("-l", "--labels", dest="labels", action="store", required=True,
                       help="Pathway to label file.")

    # Training options
    training = parser.add_argument_group(title="Training")
    training.add_argument("-s", "--size", dest="size", action="store", default=(256, 256), type=int, nargs="+",
                            help="Size of images for training. [Default = (256, 256)]")
    training.add_argument("-b", "--batch_size", dest="batch_size", action="store", default=32, type=int,
                            help="Batch size for training. [Default = 32]")


    # Tool output
    output = parser.add_argument_group(title="Output")
    output.add_argument("-o", "--out", dest="out", action="store", required=True,
                        help="Pathway to results folder.")

    # Parse arguments
    args = parser.parse_args()

    # Standardize paths
    args.images = Path(args.images).resolve()
    args.labels = Path(args.labels).resolve()
    args.out = Path(args.out).resolve()
    args.size = tuple(args.size)

    return args


def main(args):
    torch.set_float32_matmul_precision('high')

    # Set transformations
    transform = {
        "train": get_transforms(training=True),
        "val": get_transforms(training=False)
    }

    # Set pathways
    RESULTS_FOLDER = args.out

    # Create two dataset objects with different transformations (to not have augmentations in validation and test set later)
    data_train = CINDataset(csv_path=args.labels, images_folder=args.images, transform=transform["train"])
    data_valid = CINDataset(csv_path=args.labels, images_folder=args.images, transform=transform["val"])
    print(f"Dataset contains =  {len(data_train)} images.")

    # Train test split
    train_val_indices, test_indices = train_test_split(data_train.df.index, test_size=0.2,
                                                       stratify=data_train.df["label"], random_state=42)

    # Cross validation k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (train_indices, val_indices) in enumerate(skf.split(data_train.df["image"].loc[train_val_indices],
                                                data_train.df["label"].loc[train_val_indices])):

        # Oversample/undersample training set due to class imbalance
        train_idx = data_train.df.loc[train_indices].groupby("label").sample(n=len(train_indices), replace=True).index

        # Get subsets of data
        train_set = torch.utils.data.Subset(data_train, train_idx)
        val_set = torch.utils.data.Subset(data_valid, val_indices)
        test_set = torch.utils.data.Subset(data_valid, test_indices)
        datasets = (train_set, val_set, test_set)
        print(len(train_set), len(val_set), len(test_set))

        # set hyper parameters
        hparams = {
            "batch_size": args.batch_size,
            "learning_rate": 3e-4,
        }

        # Set model
        model = BinaryClassifierModel(hparams, datasets, EfficientNetClassifier(out_features=1))

        # Training model
        trainer = pl.Trainer(
            precision=32,
            accelerator="auto",
            max_epochs=300,
            log_every_n_steps=5,
            callbacks=[EarlyStopping(monitor="val_loss", mode="min")]
        )
        trainer.fit(model)

        # Save model
        MODEL_FILE = RESULTS_FOLDER.joinpath(f"models/model_{str(k)}.pt")
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, MODEL_FILE)

        ########################################
        # FROM HERE ON TEST
        # Create test folder if it does not exist
        RESULTS_FOLDER.joinpath("test").mkdir(parents=True, exist_ok=True)

        # Save test results and metrics
        TEST_METRICS = RESULTS_FOLDER.joinpath(f"test/test_scores_{str(k)}.csv")
        TEST_CONFMTRX = RESULTS_FOLDER.joinpath(f"test/test_confusion_matrix_{str(k)}.pdf")
        TEST_PREDICTIONS = RESULTS_FOLDER.joinpath(f"test/test_predictions_{str(k)}.csv")

        # Get test scores
        df_test = model.get_test_pred_scores()

        # Get test metrics
        df_test_metrics = evaluate_multiclass_model(df_test["prediction"], df_test["target"])
        df_test_metrics.to_csv(TEST_METRICS, index=False)

        # Get plot for test data
        fig = plot_confusion_matrix(df_test["prediction"], df_test["target"])
        fig.savefig(TEST_CONFMTRX, dpi=300)

        # Save test predictions
        df_test.to_csv(TEST_PREDICTIONS, index=False)

        ########################################
        # FROM HERE ON VALIDATION
        # Create validation folder if it does not exist
        RESULTS_FOLDER.joinpath("validation").mkdir(parents=True, exist_ok=True)

        # Save validation results and metrics
        VAL_METRICS = RESULTS_FOLDER.joinpath(f"validation/val_scores_{str(k)}.csv")
        VAL_CONFMTRX = RESULTS_FOLDER.joinpath(f"validation/val_confusion_matrix_{str(k)}.pdf")
        VAL_PREDICTIONS = RESULTS_FOLDER.joinpath(f"validation/val_predictions_{str(k)}.csv")

        # Get validation scores
        df_val = model.get_val_pred_scores()

        # Get validation metrics
        df_val_metrics = evaluate_multiclass_model(df_val["prediction"], df_val["target"])
        df_val_metrics.to_csv(VAL_METRICS, index=False)

        # Get plot for validation data
        fig = plot_confusion_matrix(df_val["prediction"], df_val["target"])
        fig.savefig(VAL_CONFMTRX, dpi=300)

        # Save validation predictions
        df_val.to_csv(VAL_PREDICTIONS, index=False)




if __name__ == "__main__":
    # Read arguments from command line
    args = get_args()

    # Log arguments
    print(f"Batch size = {args.batch_size}")
    print(f"Image size = {args.size}")

    # Run script and calculate run time
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finish in {rt//60:.0f}m {rt%60:.0f}s")
