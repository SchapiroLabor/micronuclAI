import torch
import time
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import CINDataset
from models import (EfficientNetClassifier, MulticlassRegression)
from augmentations import get_transforms
from pytorch_lightning.loggers import CSVLogger
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
    input.add_argument("-t", "--test", dest="test", action="store", required=True,
                       help="Pathway to test label file.")

    # Training options
    training = parser.add_argument_group(title="Training")
    training.add_argument("-m", "--model", dest="model", action="store", default="efficientnet_b0",
                            help="Model to use for training. [Default = efficientnet_b0]")
    training.add_argument("-s", "--size", dest="size", action="store", default=(256, 256), type=int, nargs="+",
                            help="Size of images for training. [Default = (256, 256)]")
    training.add_argument("-b", "--batch_size", dest="batch_size", action="store", default=32, type=int,
                            help="Batch size for training. [Default = 32]")
    training.add_argument("-p" "--precision", dest="precision", action="store", default="32",
                          choices=["16-mixed", "bf16-mixed", "16-true", "bf16-true", "32", "64"],
                          help="Precision for training. [Default = bf16-mixed]")
    training.add_argument("-sc", "--single_channel", dest="single_channel", action="store_true",
                            help="Use single channel images. [Default = False]")


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

    # Set transformations for the data
    print("Loading transformations")
    transform = {
        "train": get_transforms(resize=args.size, single_channel=args.single_channel, training=True),
        "val":   get_transforms(resize=args.size, single_channel=args.single_channel, training=False)
    }

    # Create two dataset objects with different transformations (to not have augmentations in validation and test set later)
    data_train = CINDataset(csv_path=args.labels, images_folder=args.images, transform=transform["train"])
    data_test  = CINDataset(csv_path=args.test, images_folder=args.images, transform=transform["val"])
    print(f"Dataset contains  = {len(data_train)} images.")
    print(f"Data distribution = \n{data_train.df['label'].value_counts()} images.")

    # Cross validation k-fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for k, (train_indices, val_indices) in enumerate(skf.split(data_train.df["image"],
                                                               data_train.df["label"])):

        # Set loggers
        csv_logger = CSVLogger(args.out,
                               name=f"{args.model}_s{args.size[0]}_bs{args.batch_size}_p{args.precision}",
                               version=f"log")

        # Get subsets of data for train/validation and test
        train_set = torch.utils.data.Subset(data_train, train_indices)
        val_set = torch.utils.data.Subset(data_train, val_indices)
        #test_set = torch.utils.data.Subset(data_test, data_test.df.index) # Use all indexes for the test dataset
        datasets = (train_set, val_set, data_test)

        # Print final ammount of training, validation and test data
        print(f"Training data   = {len(train_set)}")
        print(f"Validation data = {len(val_set)}")
        print(f"Test data       = {len(data_test)}")

        # set hyper parameters
        hparams = {
            "batch_size": args.batch_size,
            "learning_rate": 3e-4,
        }

        # Set model
        model = MulticlassRegression(hparams, datasets, EfficientNetClassifier(model=args.model))

        # Training model
        trainer = pl.Trainer(
            precision= args.precision,
            accelerator="auto",
            max_epochs=300,
            #log_every_n_steps=1,
            callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
            logger=csv_logger
        )
        trainer.fit(model)

        # Save model
        # MODEL_FILE = RESULTS_FOLDER.joinpath(f"models/model_{str(k)}.pt")
        MODEL_FOLDER = args.out / f"{args.model}_s{args.size[0]}_bs{args.batch_size}_p{args.precision}" / "trained_models"
        MODEL_FILE = MODEL_FOLDER / f"model_{k}.pt"
        MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model, MODEL_FILE)

        ########################################
        # FROM HERE ON VALIDATION
        # Create validation folder if it does not exist
        VALIDATION_FOLDER = args.out / f"{args.model}_s{args.size[0]}_bs{args.batch_size}_p{args.precision}" / "validation"
        VALIDATION_FOLDER.mkdir(parents=True, exist_ok=True)

        # Save validation results and metrics
        VAL_METRICS = VALIDATION_FOLDER / f"val_scores_{k}.csv"
        VAL_CONFMTRX = VALIDATION_FOLDER / f"confusion_matrix_{k}.pdf"
        VAL_PREDICTIONS = VALIDATION_FOLDER / f"predictions_{k}.csv"

        # Get validation scores
        df_val = model.get_val_pred_scores()

        # Get validation metrics
        df_val_metrics = evaluate_multiclass_model(df_val["prediction"], df_val["target"])
        df_val_metrics.T.to_csv(VAL_METRICS, index=True)

        # Get plot for validation data
        fig = plot_confusion_matrix(df_val["prediction"], df_val["target"])
        fig.savefig(VAL_CONFMTRX, dpi=300)

        # Save validation predictions
        df_val.to_csv(VAL_PREDICTIONS, index=False)

        ########################################
        # FROM HERE ON TEST
        # Create validation folder if it does not exist
        TEST_FOLDER = args.out / f"{args.model}_s{args.size[0]}_bs{args.batch_size}_p{args.precision}" / "test"
        TEST_FOLDER.mkdir(parents=True, exist_ok=True)

        # Save test results and metrics
        TEST_METRICS = TEST_FOLDER / f"test_scores{k}.csv"
        TEST_CONFMTRX = TEST_FOLDER / f"test_confusion_matrix{k}.pdf"
        TEST_PREDICTIONS = TEST_FOLDER / f"test_predictions{k}.csv"

        # Get test scores
        df_test = model.get_test_pred_scores()

        # Get test metrics
        df_test_metrics = evaluate_multiclass_model(df_test["prediction"], df_test["target"])
        df_test_metrics.T.to_csv(TEST_METRICS, index=True, header=False)

        # Get plot for test data
        fig = plot_confusion_matrix(df_test["prediction"], df_test["target"])
        fig.savefig(TEST_CONFMTRX, dpi=300)

        # Save test predictions
        df_test.to_csv(TEST_PREDICTIONS, index=False)


if __name__ == "__main__":
    # Read arguments from command line
    args = get_args()

    # Log arguments
    print(f"Model      = {args.model}")
    print(f"Batch size = {args.batch_size}")
    print(f"Image size = {args.size}")

    # Run script and calculate run time
    st = time.time()
    main(args)
    rt = time.time() - st
    print(f"Script finish in {rt//60:.0f}m {rt%60:.0f}s")
