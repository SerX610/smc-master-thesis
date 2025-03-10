"""
This script defines the `CLAPTransferLearning` class, a PyTorch Lightning
module designed for multi-label classification using precomputed audio
embeddings and a Multi-Layer Perceptron (MLP). The model is intended for
transfer learning tasks, where audio samples are classified into multiple
categories based on their embeddings.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from pytorch_lightning import LightningModule
from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelAUROC,
    MultilabelConfusionMatrix,
)


class CLAPTransferLearning(LightningModule):
    """
    A PyTorch Lightning module for transfer learning using precomputed audio embeddings.

    This module implements a multi-label classification model using a Multi-Layer Perceptron (MLP)
    to classify audio samples based on their embeddings. It supports training, validation, and testing
    with metrics such as AUROC, Average Precision, and Confusion Matrix.

    Attributes:
        audio_embeddings (torch.Tensor): Precomputed audio embeddings loaded from a file.
        num_labels (int): Number of output labels for classification.
        lr (float): Learning rate for the optimizer.
        criterion (nn.BCEWithLogitsLoss): Loss function combining Sigmoid and BCELoss.
        threshold (float): Threshold for binary classification.
        top_n (int): Number of top labels to consider for evaluation.
        top_n_indices (list): Indices of the top N labels.
        tag_index_mapping (dict): Mapping from tag indices to tag names.
        plot_dir (str): Directory to save the confusion matrix plot.
        classifier (nn.Sequential): MLP classifier for multimodal input.
        val_metrics (nn.ModuleDict): Dictionary of validation metrics.
        test_metrics (nn.ModuleDict): Dictionary of test metrics.
        test_confusion_matrix (MultilabelConfusionMatrix): Confusion matrix for test data.
        best_val_metric (dict): Dictionary to store the best validation metric values.
    """
    def __init__(
            self,
            audio_embeddings_file,
            hidden_size=512,
            num_labels=188,
            dropout=0.2,
            lr=0.0001,
            threshold=0.5,
            top_n=50,
            top_n_indices=None,
            tag_index_mapping=None,
            plot_dir="confusion_matrix.png",
            ):
        super().__init__()

        # Load precomputed embeddings from file
        self.audio_embeddings = torch.from_numpy(np.load(audio_embeddings_file)).float()

        # Initialize parameters for model definition
        input_size = self.audio_embeddings.shape[1]
        self.num_labels = num_labels
        self.lr = lr
        self.criterion = nn.BCEWithLogitsLoss()

        # Initialize parameters for model evaluation
        self.threshold = threshold
        self.top_n = top_n
        if top_n_indices is not None:
            self.top_n_indices = top_n_indices.copy()
        if tag_index_mapping is not None:
            self.tag_index_mapping = tag_index_mapping
        self.plot_dir = plot_dir

        # Initialize MLP classifier for multimodal input
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels),
        )

        # Initialize metrics
        self.init_metrics()
        self.best_val_metric = {metric: 0.0 for metric in self.val_metrics.keys()}
    

    def setup(self, stage=None):
        """
        Moves the audio embeddings to the appropriate device (CPU or GPU).
        """
        self.audio_embeddings = self.audio_embeddings.to(self.device)


    def init_metrics(self):
        """
        Initializes the metrics for validation and testing.
        """
        self.val_metrics = nn.ModuleDict(
            {
                "val-AUROC-macro": MultilabelAUROC(
                    num_labels=self.num_labels, average="macro"
                ),
                "val-MAP-macro": MultilabelAveragePrecision(
                    num_labels=self.num_labels, average="macro"
                ),
            }
        )
        self.test_metrics = nn.ModuleDict(
            {
                "test-AUROC-macro": MultilabelAUROC(
                    num_labels=self.top_n, average="macro"
                ),
                "test-MAP-macro": MultilabelAveragePrecision(
                    num_labels=self.top_n, average="macro"
                ),
            }
        )
        self.test_confusion_matrix = MultilabelConfusionMatrix(
            num_labels=self.top_n
        )


    def forward(self, audio_idx):
        """
        Forward pass of the model.
        """
        embeddings = self.audio_embeddings[audio_idx]
        predictions = self.classifier(embeddings)
        return predictions


    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        """
        audio_idx, labels = batch
        logits = self.forward(audio_idx)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def predict(self, batch, return_predicted_class=False):
        """
        Predicts the output for a given batch.
        """
        audio_idx, labels = batch
        logits = self.forward(audio_idx)
        loss = self.criterion(logits, labels)
        if return_predicted_class:
            predicted_class = (torch.sigmoid(logits) > self.threshold).int()
            return logits, loss, predicted_class
        return logits, loss


    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        """
        logits, loss = self.predict(batch)
        self.log("val_loss", loss, prog_bar=True)
        labels = batch[1].int()
        for _, metric in self.val_metrics.items():
            metric.update(logits, labels)


    def on_validation_epoch_end(self):
        """
        Called at the end of the validation epoch to compute and log metrics.
        """
        for name, metric in self.val_metrics.items():
            self.log(name, metric, on_epoch=True, prog_bar=True)
            metric_value = metric.compute().cpu().numpy()
            if metric_value > self.best_val_metric[name]:
                self.best_val_metric[name] = metric_value


    def test_step(self, batch, batch_idx):
        """
        Test step for the model.
        """
        labels = batch[1].int()
        logits, _ = self.predict(batch)
        labels = labels[:, self.top_n_indices]
        logits = logits[:, self.top_n_indices]
        for _, metric in self.test_metrics.items():
            metric.update(logits, labels)
        self.test_confusion_matrix.update(logits, labels)


    def on_test_epoch_end(self):
        """
        Called at the end of the test epoch to compute and log metrics, and save the confusion matrix plot.
        """
        for name, metric in self.test_metrics.items():
            self.log(name, metric, on_epoch=True)
        conf_matrix = self.test_confusion_matrix.compute()
        fig = self.plot_confusion_matrix(conf_matrix)
        fig.savefig(self.plot_dir, bbox_inches='tight')


    def configure_optimizers(self):
        """
        Configures the optimizer for the model.
        """
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


    def plot_confusion_matrix(self, conf_matrix):
        """
        Plots the confusion matrix for the top N labels.
        """
        conf_matrix = conf_matrix.cpu().numpy()
        fig, axes = plt.subplots(nrows=10, ncols=5, figsize=(25, 50), constrained_layout=True)
        axes = axes.flatten()
        fig.suptitle(f'Confusion Matrix for Top {self.top_n} Labels', fontsize=24)
        labels = [self.tag_index_mapping[str(i)] for i in self.top_n_indices]
        for ax, cm, label in zip(axes, conf_matrix, labels):
            im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
            ax.set_title(label, fontsize=15)
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center", color="red")
            ax.set_xticks(np.arange(cm.shape[1]))
            ax.set_yticks(np.arange(cm.shape[0]))
            ax.set_xticklabels(["False", "True"])
            ax.set_yticklabels(["False", "True"])
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
        return fig
