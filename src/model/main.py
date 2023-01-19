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
from utils import evaluate_binary_model
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
    image_size = 256

    transform = {
        "train": torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.RandomRotation(degrees=30),
            torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size=(3, 3))], p=0.3),
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ]),
        "val": torchvision.transforms.Compose([
            torchvision.transforms.Resize((image_size, image_size)),
            torchvision.transforms.Grayscale(num_output_channels=3),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        ])
    }

    # Set pathways
    IMAGES_ROOT = args.images
    LABELS_FILE = args.labels
    RESULTS_FOLDER = args.out

    # Create two dataset objects with different transformations (to not have augmentations in validation and test set later)
    data_train = CINDataset(csv_path=LABELS_FILE, images_folder=IMAGES_ROOT, transform=transform["train"])
    data_val_test = CINDataset(csv_path=LABELS_FILE, images_folder=IMAGES_ROOT, transform=transform["val"])
    print("Dataset contains %d images." % len(data_train))

    train_val_indices, test_indices = train_test_split(data_train.df.index, test_size=0.2,
                                                       stratify=data_train.df["label"], random_state=42)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    #k = 1

    for k, (train_indices, val_indices) in enumerate(skf.split(data_train.df["image"].loc[train_val_indices],
                                                data_train.df["label"].loc[train_val_indices])):
        # Oversample/ undersample training set due to class imbalance
        sampling_number = len(train_indices)
        df_tmp = data_train.df.loc[train_indices].groupby("label").sample(n=sampling_number, replace=True)
        train_indices_sampled = df_tmp.index

        train_set = torch.utils.data.Subset(data_train, train_indices_sampled)
        val_set = torch.utils.data.Subset(data_val_test, val_indices)
        test_set = torch.utils.data.Subset(data_val_test, test_indices)

        datasets = (train_set, val_set, test_set)

        hparams = {
            "batch_size": 64,
            "learning_rate": 3e-4,
        }

        model = BinaryClassifierModel(hparams, datasets, EfficientNetClassifier(out_features=1))
        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=300,
            log_every_n_steps=5,
            callbacks=[EarlyStopping(monitor="val_loss", mode="min")]
        )
        trainer.fit(model)

        validation_scores, validation_labels = model.get_val_pred_scores()
        dict_val_evaluation = evaluate_binary_model(validation_scores, validation_labels)
        df_val_evaluation = pd.DataFrame.from_dict(dict_val_evaluation)
        EVALUATION_FILE = RESULTS_FOLDER.joinpath(f"evaluation/validation_evaluation_scores_{str(k)}.csv")
        EVALUATION_FILE.parent.mkdir(parents=True, exist_ok=True)
        df_val_evaluation.to_csv(EVALUATION_FILE, index=False)

        test_scores, test_labels = model.get_test_pred_scores()
        tuple_scores = (test_scores, test_labels)
        SCORE_FILE = RESULTS_FOLDER.joinpath(f"test_scores_{str(k)}.p")
        SCORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        pickle.dump(tuple_scores, open(SCORE_FILE, "wb"))

        dict_test_evaluation = evaluate_binary_model(test_scores, test_labels)
        df_test_evaluation = pd.DataFrame.from_dict(dict_test_evaluation)
        RESULTS_FILE = RESULTS_FOLDER.joinpath(f"test_evaluation_scores_{str(k)}.csv")
        RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        df_test_evaluation.to_csv(RESULTS_FILE, index=False)

        MODEL_FILE = RESULTS_FOLDER.joinpath(f"models/model_{str(k)}.pt")
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, MODEL_FILE)

        #k += 1


if __name__ == "__main__":
    # Read arguments from command line
    args = get_args()

    # Run script and calculate run time
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finish in {rt//60:.0f}m {rt%60:.0f}s")