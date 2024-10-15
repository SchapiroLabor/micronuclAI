import torch
import time
from pathlib import Path
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import CINDataset
from models import micronuclAI
from augmentations import get_transforms
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split, StratifiedKFold
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
from logger import set_logger


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

    # Tool output
    output = parser.add_argument_group(title="Output")
    output.add_argument("-o", "--out", dest="out", action="store", required=True,
                        help="Pathway to results folder.")

    # Training options
    training = parser.add_argument_group(title="Training")
    training.add_argument("-m", "--model", dest="model", action="store", default="efficientnet_b0",
                          help="Model to use for training. [Default = efficientnet_b0]")
    training.add_argument("-s", "--size", dest="size", action="store", default=(256, 256), type=int, nargs="+",
                          help="Size of images for training. [Default = (256, 256)]")
    training.add_argument("-b", "--batch_size", dest="batch_size", action="store", default=32, type=int,
                          help="Batch size for training. [Default = 32]")
    training.add_argument("-w", "--num_workers", dest="num_workers", action="store", default=4, type=int,
                          help="Number of workers for data loading. [Default = 4]")
    training.add_argument("-p" "--precision", dest="precision", action="store", default="32",
                          choices=["16-mixed", "bf16-mixed", "16-true", "bf16-true", "32", "64"],
                          help="Precision for training. [Default = 32]")
    training.add_argument("-sc", "--single_channel", dest="single_channel", action="store_true",
                          help="Use single channel images. [Default = False]")
    training.add_argument("-k", "--kfold", dest="kfold", action="store", default=5, type=int,
                          help="Number of folds for cross validation. [Default = 5]")

    # Script options
    options = parser.add_argument_group(title="Script options")
    options.add_argument("-log", "--log_level", dest="log_level", action="store", default="info",
                         choices=["debug", "info"],
                         help="Set the logging level. [Default = info]")
    options.add_argument("--version", action="version", version="micronuclAI 1.0.0")

    # Parse arguments
    args = parser.parse_args()

    # Standardize paths
    args.images = Path(args.images).resolve()
    args.labels = Path(args.labels).resolve()
    args.out = Path(args.out).resolve()
    args.size = tuple(args.size)

    return args


def save_model(model, output_path, suffix="") -> None:
    """
    Save model to TorchScript format

    :param model: Model to save
    :param output_path: Path to save model
    :param suffix: Suffix to add to model name
    :return: None
    """
    # Save model
    MODEL_FOLDER = output_path / "trained_models"
    MODEL_FILE = MODEL_FOLDER / f"model{suffix}.pt"
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving model to = {MODEL_FILE}")

    # Convert model to TorchScript and save
    script = model.to_torchscript()
    torch.jit.save(script, MODEL_FILE)


def cv_training(args, data_train, data_test, hparams) -> None:
    """
    Perform k-fold cross-validation and train the model on each fold.

    :param args: Arguments containing model, size, batch_size precision, k-fold and output path.
    :param data_train: Training data
    :param data_test: Test data
    :param hparams: Hyperparameters for the model.
    :return: None
    """

    # Stratified K-Fold split
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)

    # Iterate through each fold
    for k, (train_indices, val_indices) in enumerate(skf.split(data_train.df["image"], data_train.df["label"])):
        # Set logger
        csv_logger = CSVLogger(
            args.out,
            name=f"{args.model}_s{args.size[0]}_bs{args.batch_size}_p{args.precision}",
            version=f"log"
        )

        # Get subsets for training, validation, and test
        train_set = torch.utils.data.Subset(data_train, train_indices)
        val_set = torch.utils.data.Subset(data_train, val_indices)
        datasets = (train_set, val_set, data_test)

        # Print dataset sizes
        print(f"Training data   = {len(train_set)}")
        print(f"Validation data = {len(val_set)}")
        print(f"Test data       = {len(data_test)}")

        # Initialize model with hyperparameters, datasets, and model name
        model = micronuclAI(hparams, datasets, args.model)

        # Trainer setup
        trainer = pl.Trainer(
            precision=args.precision,
            accelerator="auto",
            max_epochs=300,
            callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
            logger=csv_logger
        )

        # Train the model
        trainer.fit(model)

        # Save model after training for the current fold
        save_model(model, args.out, suffix=f"_{k}")

        # Validation: get validation scores
        model.get_val_pred_scores(output_path=args.out, suffix=f"_{k}")

        # Test: get test scores
        model.get_test_pred_scores(output_path=args.out, suffix=f"_{k}")


def no_cv_training(args, data_train, data_test, hparams) -> None:
    """
    Train a model with no cross-validation on all the data.
    
    :param args: Arguments containing model, size, batch_size precision, k-fold and output path.
    :param data_train: Training data
    :param data_test: Test data
    :param hparams: Hyperparameters for the model.
    :return: None
    """
    # Set logger
    csv_logger = CSVLogger(
        args.out,
        name=f"{args.model}_s{args.size[0]}_bs{args.batch_size}_p{args.precision}",
        version=f"log"
    )

    # Split data into train and test (uses 1/10 of data for testing)
    train_indices, val_indices = train_test_split(data_train.df.index,
                                                  test_size=0.05,
                                                  stratify=data_train.df["label"],
                                                  random_state=42)

    train_set = torch.utils.data.Subset(data_train, train_indices)
    val_set = torch.utils.data.Subset(data_train, val_indices)

    # Get subsets for training, validation, and test
    datasets = (train_set, val_set, data_test)

    # Print final ammount of training, validation and test data
    print(f"Training data   = {len(train_set)}")
    print(f"Validation data = {len(val_set)}")
    print(f"Test data       = {len(data_test)}")

    # Initialize model with hyperparameters, datasets, and model name
    model = micronuclAI(hparams, datasets, args.model)

    # Trainer setup
    trainer = pl.Trainer(
        precision=args.precision,
        accelerator="auto",
        max_epochs=300,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min")],
        logger=csv_logger
    )

    # Train the model
    trainer.fit(model)

    # Save model after training for the current fold
    save_model(model, args.out)

    # Validation: get validation scores
    model.get_val_pred_scores(output_path=args.out)

    # Test: get test scores
    model.get_test_pred_scores(output_path=args.out)


def main():
    torch.set_float32_matmul_precision('high')
    # Read arguments from command line
    args = get_args()

    # Set logger
    lg = set_logger(log_level=args.log_level)
    lg.info("micronuclAI - Training")
    lg.info(f"Model      = {args.model}")
    lg.info(f"Batch size = {args.batch_size}")
    lg.info(f"Image size = {args.size}")

    # Load transformations for train and validation
    lg.info("Loading transformations")
    transform = {
        "train": get_transforms(resize=args.size, single_channel=args.single_channel, training=True),
        "test": get_transforms(resize=args.size, single_channel=args.single_channel, training=False)
    }

    # Create two dataset objects with different transformations
    # (to not have augmentations in validation and test set later)
    data_train = CINDataset(csv_path=args.labels, images_folder=args.images, transform=transform["train"])
    data_test = CINDataset(csv_path=args.test, images_folder=args.images, transform=transform["test"])
    lg.info(f"Dataset contains  = {len(data_train)} images.")
    lg.info(f"Data distribution = \n{data_train.df['label'].value_counts()} images.")

    # set hyper parameters
    hparams = {
        "batch_size": args.batch_size,
        "learning_rate": 3e-4,
        "workers": args.num_workers,
    }

    # Call training with k-folds
    lg.info(f"Training model ")
    if args.kfold != 0:
        cv_training(args, data_train, data_test, hparams)
    else:
        no_cv_training(args, data_train, data_test, hparams)
    lg.info("Done!")

if __name__ == "__main__":
    main()
