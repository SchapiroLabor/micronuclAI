import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import torchvision.models as models


class EfficientNetClassifier(nn.Module):
    def __init__(self, out_features):
        super().__init__()

        self.model = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=out_features)

        for param in self.model.parameters():
            param.require_grad = True

    def forward(self, x):
        x = self.model.features(x)

        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.model.classifier(x)

        return x


class BinaryClassifierModel(pl.LightningModule):

    def __init__(self, hparams, dataset, model):
        super().__init__()

        self.model = model
        self.save_hyperparameters(hparams)
        self.dataset = {"train": dataset[0], "val": dataset[1], "test": dataset[2]}

    def forward(self, x):
        x = self.model(x)
        x = torch.sigmoid(x)
        return x

    def general_step(self, batch, batch_idx, mode):
        images, targets = batch

        # forward pass
        out = self.forward(images)

        # loss
        targets = targets.float().view(-1, 1)
        loss = F.binary_cross_entropy(out, targets)
        #loss = F.cross_entropy(out, targets)
        
        preds = out.argmax(axis=1)
        n_correct = (targets == preds).sum()
        return loss, n_correct

    def general_end(self, outputs, mode):
        # average over all batches aggregated during one epoch
        avg_loss = torch.stack([x[mode + '_loss'] for x in outputs]).mean()
        total_correct = torch.stack([x[mode + '_n_correct'] for x in outputs]).sum().cpu().numpy()
        acc = total_correct / len(self.dataset[mode])
        return avg_loss, acc

    def training_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "train")
        self.log("train_loss", loss)
        return {'loss': loss, 'train_n_correct': n_correct}

    def validation_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "val")
        self.log("val_loss", loss)
        return {'val_loss': loss, 'val_n_correct': n_correct}

    def test_step(self, batch, batch_idx):
        loss, n_correct = self.general_step(batch, batch_idx, "test")
        return {'test_loss': loss, 'test_n_correct': n_correct}

    def validation_end(self, outputs):
        avg_loss, acc = self.general_end(outputs, "val")
        # print("Val-Acc={}".format(acc))
        return {'val_loss': avg_loss, 'val_acc': acc}

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], num_workers=8, batch_size=self.hparams["batch_size"], shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset["val"], num_workers=8, batch_size=self.hparams["batch_size"])

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], num_workers=8, batch_size=self.hparams["batch_size"])

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(), self.hparams["learning_rate"])
        return optim

    def get_val_pred_scores(self, loader=None):
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
        return scores, labels

    def get_test_pred_scores(self, loader=None):
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
        return scores, labels

    def reset_weights(self):
        self.model.reset_parameters()
