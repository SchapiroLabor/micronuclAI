import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import torchvision.models as models
from src.model.utils import evaluate_multiclass_model, plot_confusion_matrix

class EfficientNetClassifier(nn.Module):
    def __init__(self, model="efficientnet_b0", weights="IMAGENET1K_V1", out_features=1):
        super().__init__()

        # Load model
        self.model = models.get_model(model, weights=weights)

        # Get the correct last layer depending on the selected model
        if model == "efficientnet_b0":
            self.model.classifier[1] = nn.Linear(in_features=1280, out_features=out_features)
        elif model == "efficientnet_b1":
            self.model.classifier[1] = nn.Linear(in_features=1280, out_features=out_features)
        elif model == "efficientnet_b2":
            self.model.classifier[1] = nn.Linear(in_features=1408, out_features=out_features)
        elif model == "efficientnet_b3":
            self.model.classifier[1] = nn.Linear(in_features=1536, out_features=out_features)
        elif model == "efficientnet_b4":
            self.model.classifier[1] = nn.Linear(in_features=1792, out_features=out_features)
        elif model == "efficientnet_b5":
            self.model.classifier[1] = nn.Linear(in_features=2048, out_features=out_features)
        elif model == "efficientnet_b6":
            self.model.classifier[1] = nn.Linear(in_features=2304, out_features=out_features)
        elif model == "efficientnet_b7":
            self.model.classifier[1] = nn.Linear(in_features=2560, out_features=out_features)
        elif model == "efficientnet_v2_s":
            self.model.classifier[1] = nn.Linear(in_features=1280, out_features=out_features)
        elif model == "efficientnet_v2_m":
            self.model.classifier[1] = nn.Linear(in_features=1280, out_features=out_features)
        elif model == "efficientnet_v2_l":
            self.model.classifier[1] = nn.Linear(in_features=1280, out_features=out_features)

        for param in self.model.parameters():
            param.require_grad = True

    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)
        return x


class micronuclAI(pl.LightningModule):
    def __init__(self, hparams, dataset, model):
        super().__init__()
        self.model = EfficientNetClassifier(model)
        self.save_hyperparameters(hparams)
        self.dataset = {"train": dataset[0], "val": dataset[1], "test": dataset[2]}

    def forward(self, x):
        x = self.model(x)
        # x = torch.sigmoid(x)
        return x

    def general_step(self, batch, batch_idx, mode):
        images, targets = batch

        # forward pass
        out = self.forward(images)

        # loss
        targets = targets.float().view(-1, 1)
        loss = F.mse_loss(out, targets)
        
        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum().detach()
        return loss, n_correct

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.dataset[mode])
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'train_n_correct': n_correct}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_n_correct': n_correct}

    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch).squeeze().cpu().numpy()

    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        # print("Val-Acc={}".format(acc))
        return {'val_loss': avg_loss, 'val_acc': acc}

    def train_dataloader(self):
        return DataLoader(self.dataset["train"],
                          num_workers=self.hparams["workers"],
                          persistent_workers=True,
                          pin_memory=True,
                          batch_size=self.hparams["batch_size"],
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset["val"],
                          num_workers=0,
                          pin_memory=True,
                          batch_size=self.hparams["batch_size"])

    def test_dataloader(self):
        return DataLoader(self.dataset["test"],
                          num_workers=0,
                          pin_memory=True,
                          batch_size=self.hparams["batch_size"])

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), self.hparams["learning_rate"])
        return optim

    def get_val_pred_scores(self, loader=None, output_path="", suffix="") -> None:
        self.model.eval()
        self.model = self.model.to(self.device)

        if not loader: loader = self.val_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        self.model.train()

        # Convert to dataframe and round scores
        df_val = pd.DataFrame([scores.squeeze(), labels]).T.rename(columns={0: "scores", 1: "target"})
        df_val["target"] = df_val["target"].astype(int)
        df_val["prediction"] = df_val["scores"].apply(lambda x: round(x))

        # FROM HERE ON VALIDATION
        # Create validation folder if it does not exist
        VALIDATION_FOLDER = output_path / "validation"
        VALIDATION_FOLDER.mkdir(parents=True, exist_ok=True)

        # Save validation results and metrics
        VAL_METRICS = VALIDATION_FOLDER / f"val_scores{suffix}.csv"
        VAL_CONFMTRX = VALIDATION_FOLDER / f"confusion_matrix{suffix}.pdf"
        VAL_PREDICTIONS = VALIDATION_FOLDER / f"predictions{suffix}.csv"

        # Get validation metrics
        df_val_metrics = evaluate_multiclass_model(df_val["prediction"], df_val["target"])
        print(f"Saving validation metrics to = {VAL_METRICS}")
        df_val_metrics.T.to_csv(VAL_METRICS, index=True, header=False)

        # Get plot for validation data
        fig = plot_confusion_matrix(df_val["prediction"], df_val["target"])
        print(f"Saving validation confusion matrix to = {VAL_CONFMTRX}")
        fig.savefig(VAL_CONFMTRX, dpi=300)

        # Save validation predictions
        print(f"Saving validation predictions to = {VAL_PREDICTIONS}")
        df_val.to_csv(VAL_PREDICTIONS, index=False)

    def get_test_pred_scores(self, loader=None, output_path="", suffix="") -> None:
        self.model.eval()
        self.model = self.model.to(self.device)

        if not loader: loader = self.test_dataloader()

        scores = []
        labels = []

        for batch in loader:
            X, y = batch
            X = X.to(self.device)
            score = self.forward(X)
            scores.append(score.detach().cpu().numpy())
            labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(scores, axis=0)
        labels = np.concatenate(labels, axis=0)

        self.model.train()

        # Convert to dataframe and round scores
        df_test = pd.DataFrame([scores.squeeze(), labels]).T.rename(columns={0: "scores", 1: "target"})
        df_test["target"] = df_test["target"].astype(int)
        df_test["prediction"] = df_test["scores"].apply(lambda x: round(x))

        # Convert to dataframe and round scores
        df_test = pd.DataFrame([scores.squeeze(), labels]).T.rename(columns={0: "scores", 1: "target"})
        df_test["target"] = df_test["target"].astype(int)
        df_test["prediction"] = df_test["scores"].apply(lambda x: round(x))

        # FROM HERE ON TEST
        # Create test folder if it does not exist
        TEST_FOLDER = output_path / "test"
        TEST_FOLDER.mkdir(parents=True, exist_ok=True)

        # Save test results and metrics
        TEST_METRICS = TEST_FOLDER / f"test_scores{suffix}.csv"
        TEST_CONFMTRX = TEST_FOLDER / f"test_confusion_matrix{suffix}.pdf"
        TEST_PREDICTIONS = TEST_FOLDER / f"test_predictions{suffix}.csv"

        # Get test metrics
        df_test_metrics = evaluate_multiclass_model(df_test["prediction"], df_test["target"])
        print(f"Saving test metrics to = {TEST_METRICS}")
        df_test_metrics.T.to_csv(TEST_METRICS, index=True, header=False)

        # Get plot for test data
        fig = plot_confusion_matrix(df_test["prediction"], df_test["target"])
        print(f"Saving test confusion matrix to = {TEST_CONFMTRX}")
        fig.savefig(TEST_CONFMTRX, dpi=300)

        # Save test predictions
        print(f"Saving test predictions to = {TEST_PREDICTIONS}")
        df_test.to_csv(TEST_PREDICTIONS, index=False)

    def reset_weights(self):
        self.model.reset_parameters()