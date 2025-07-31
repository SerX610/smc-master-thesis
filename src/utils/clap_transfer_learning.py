"""
CLAPTransferLearning: PyTorch Lightning module for multi-label classification using precomputed audio embeddings.

This script provides a transfer learning framework for audio classification tasks. It uses a Multi-Layer Perceptron (MLP)
to classify audio samples into multiple categories based on their embeddings. The module supports training, validation,
and testing with metrics such as AUROC, Average Precision, and Confusion Matrix.
"""

import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from torchmetrics.classification import (
    MultilabelAveragePrecision,
    MultilabelAUROC,
    MultilabelConfusionMatrix,
)

seed = 42
seed_everything(seed, workers=True)


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
            num_labels=50,
            dropout=0.2,
            lr=0.0001,
            threshold=0.5,
            tag_index_mapping=None,
            plot_dir="results/mtt_clap_multilabel_classification_confusion_matrix.png",
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
                    num_labels=self.num_labels, average="macro"
                ),
                "test-MAP-macro": MultilabelAveragePrecision(
                    num_labels=self.num_labels, average="macro"
                ),
            }
        )
        self.test_confusion_matrix = MultilabelConfusionMatrix(
            num_labels=self.num_labels
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
        fig.suptitle("Confusion Matrix for Top 50 Labels", fontsize=24)
        labels = [self.tag_index_mapping[i] for i in range(self.num_labels)]
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


class AudioEmbeddingDataset(Dataset):
    """
    A custom PyTorch Dataset for loading audio embeddings and corresponding labels.

    Args:
        audio_embeddings_file (str): Path to the .npy file containing audio embeddings.
        labels_file (str): Path to the .npy file containing labels corresponding to the audio embeddings.
    """
    def __init__(self, audio_embeddings_file, labels_file):
        self.audio_embeddings = torch.from_numpy(np.load(audio_embeddings_file)).float()
        self.labels = torch.from_numpy(np.load(labels_file)).float()

    def __len__(self):
        """Returns the number of samples in the dataset."""
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the index and the corresponding label.
        """
        return idx, self.labels[idx]


@dataclass
class CLAPTrainerConfig:
    """
    Configuration class for CLAPTrainer that holds various hyperparameters and file paths.
    """

    embeddings_folder: str = "mtt_embeddings"  # Path to the folder containing the embeddings
    model_dir: str = "models/trained/"  # Directory to save the model checkpoint
    model_ckpt: str = "mtt-clap-multilabel-classifier-checkpoint"  # Filename for saving the model checkpoint
    batch_size: int = 64  # Batch size for training
    max_epochs: int = 10  # Maximum number of training epochs


class CLAPTrainer:
    """
    Trainer class to handle data preparation, model initialization, and training.

    Args:
        config (CLAPTrainerConfig): A configuration object that contains training parameters.
    """
    def __init__(self, config: CLAPTrainerConfig):
        self.config = config

        # Ensure the model directory exists
        os.makedirs(self.config.model_dir, exist_ok=True)

        # Define dataset paths for training and validation
        self.train_audio_embeddings_file = f"{self.config.embeddings_folder}/train_audio_embeddings.npy"
        self.train_labels_file = f"{self.config.embeddings_folder}/train_labels.npy"
        self.valid_audio_embeddings_file = f"{self.config.embeddings_folder}/valid_audio_embeddings.npy"
        self.valid_labels_file = f"{self.config.embeddings_folder}/valid_labels.npy"

        # Initialize the DataLoader for both training and validation datasets
        self.train_dataloader = DataLoader(
            AudioEmbeddingDataset(self.train_audio_embeddings_file, self.train_labels_file),
            batch_size=self.config.batch_size, shuffle=True, num_workers=4
        )
        self.valid_dataloader = DataLoader(
            AudioEmbeddingDataset(self.valid_audio_embeddings_file, self.valid_labels_file),
            batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )

        # Initialize the CLAP model for transfer learning
        self.model = CLAPTransferLearning(self.train_audio_embeddings_file)

    def train_model(self):
        """
        Trains the CLAP model using PyTorch Lightning with checkpointing.
        """

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            dirpath=self.config.model_dir,
            filename=self.config.model_ckpt,
        )

        # Initialize the PyTorch Lightning trainer with GPU acceleration if available
        trainer = Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            callbacks=[checkpoint_callback],
        )

        # Train the model
        trainer.fit(self.model, self.train_dataloader, self.valid_dataloader)


@dataclass
class CLAPTestConfig:
    """
    Configuration class for CLAPTester that holds various file paths and parameters for testing.
    """
    
    embeddings_folder: str = "mtt_embeddings"  # Path to the folder containing the embeddings
    model_ckpt: str = "models/trained/mtt-clap-multilabel-classifier-checkpoint.ckpt"  # Path to the model checkpoint
    batch_size: int = 64  # Batch size for testing
    tag_index_mapping_path: str = "data/mtt/MTAT_split/top50_tags.txt"  # Path to the tag list `.txt` file
    plot_dir: str = "results/mtt_clap_multilabel_classification_confusion_matrix.png"  # Directory to save the plot


class CLAPTester:
    """
    Tester class to handle data preparation, model loading, and testing.

    Args:
        config (CLAPTestConfig): A configuration object that contains testing parameters.
    """
    def __init__(self, config: CLAPTestConfig):
        self.config = config
        self.test_results = None  # Will hold test metrics after test_model()

        # Define dataset paths for testing
        self.test_audio_embeddings_file = f"{self.config.embeddings_folder}/test_audio_embeddings.npy"
        self.test_labels_file = f"{self.config.embeddings_folder}/test_labels.npy"

        # Initialize the DataLoader for testing dataset
        self.test_dataloader = DataLoader(
            AudioEmbeddingDataset(self.test_audio_embeddings_file, self.test_labels_file),
            batch_size=self.config.batch_size, shuffle=False, num_workers=4
        )

        # Load the tag-to-index mapping from the .txt file
        self.tag_index_mapping = self._load_tag_index_mapping_from_txt()

        # Load the model from the checkpoint
        self.model = CLAPTransferLearning.load_from_checkpoint(
            self.config.model_ckpt,
            audio_embeddings_file=self.test_audio_embeddings_file,
            tag_index_mapping=self.tag_index_mapping,
            plot_dir=self.config.plot_dir,
        )

    def test_model(self):
        """
        Tests the CLAP model using PyTorch Lightning and saves the test results.
        """
        trainer = Trainer(accelerator="gpu" if torch.cuda.is_available() else "cpu")
        results = trainer.test(self.model, self.test_dataloader)
        if results:
            self.test_results = results[0]  # Save the first dict of test metrics

    def _load_tag_index_mapping_from_txt(self):
        """
        Loads the tag-to-index mapping from a .txt file.
        
        Returns:
            dict: A dictionary mapping indices to tags.
        """
        with open(self.config.tag_index_mapping_path, 'r') as f:
            tags = [line.strip() for line in f.readlines()]
        
        return {index: tag for index, tag in enumerate(tags)}