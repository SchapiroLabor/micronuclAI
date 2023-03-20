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
from augmentations import preprocess_test as p_t
from augmentations import preprocess_train as p_tr
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

    return args


def main(args):
    transform = {
        "train": p_tr,
        "val": p_t
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

        # set hyper parameters
        hparams = {
            "batch_size": 64,
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

        # Evaluate the model
        # validation_scores, validation_labels = model.get_val_pred_scores()
        # dict_val_evaluation = evaluate_binary_model(validation_scores, validation_labels)
        # df_val_evaluation = pd.DataFrame.from_dict(dict_val_evaluation)
        # EVALUATION_FILE = RESULTS_FOLDER.joinpath(f"evaluation/validation_evaluation_scores_{str(k)}.csv")
        # EVALUATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        # df_val_evaluation.to_csv(EVALUATION_FILE, index=False)

        # Get test score

        # tuple_scores = (test_scores, test_labels)
        # SCORE_FILE = RESULTS_FOLDER.joinpath(f"test_scores_{str(k)}.p")
        # SCORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        # pickle.dump(tuple_scores, open(SCORE_FILE, "wb"))

        # Evaluate test
        # dict_test_evaluation = evaluate_binary_model(test_scores, test_labels)
        # df_test_evaluation = pd.DataFrame.from_dict(dict_test_evaluation)

        # df_test_evaluation = pd.DataFrame.from_dict([test_scores, test_labels])
        # RESULTS_FILE = RESULTS_FOLDER.joinpath(f"test_evaluation_scores_{str(k)}.csv")
        #
        # df_test_evaluation.to_csv(RESULTS_FILE, index=False)

        # Save test results and metrics
        TEST_METRICS = RESULTS_FOLDER.joinpath(f"test/test_scores_{str(k)}.csv")
        TEST_CONFMTRX = RESULTS_FOLDER.joinpath(f"test/test_confusion_matrix_{str(k)}.pdf")
        TEST_PREDICTIONS = RESULTS_FOLDER.joinpath(f"test/test_predictions_{str(k)}.csv")

        TEST_METRICS.parent.mkdir(parents=True, exist_ok=True)
        test_scores, test_labels = model.get_test_pred_scores()
        df_test_metrics = evaluate_multiclass_model(test_scores, test_labels)
        df_test_metrics.to_csv(TEST_METRICS, index=False)

        # Get plot for test data
        fig = plot_confusion_matrix(test_scores, test_labels)
        fig.savefig(TEST_CONFMTRX, dpi=300)

        # Save predictions
        df_test_predictions = pd.DataFrame([test_scores, test_labels], columns=["scores", "labels"])
        df_test_predictions.to_csv(TEST_PREDICTIONS, index=False)



        # Save model
        MODEL_FILE = RESULTS_FOLDER.joinpath(f"models/model_{str(k)}.pt")
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, MODEL_FILE)


if __name__ == "__main__":
    # Read arguments from command line
    args = get_args()

    # Run script and calculate run time
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finish in {rt//60:.0f}m {rt%60:.0f}s")