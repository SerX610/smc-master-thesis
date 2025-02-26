import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import numpy as np


class CLAPTransferLearning(LightningModule):
    def __init__(self, audio_embeddings_file, hidden_size=512, num_labels=188, dropout=0.2, lr=0.0001):
        super().__init__()

        # Load precomputed embeddings from file
        audio_embeddings = torch.from_numpy(np.load(audio_embeddings_file)).float()

        # Use `register_buffer` to ensure embeddings move to the correct device
        self.register_buffer("audio_embeddings", audio_embeddings)

        # Define the input size for the linear layer
        input_size = self.audio_embeddings.shape[1]

        # MLP classifier for multimodal input
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_size, num_labels),
        )

        self.criterion = nn.BCEWithLogitsLoss()  # Handles a Sigmoid layer and the BCELoss in one single class 
        self.lr = lr


    def forward(self, audio_idx):
        """
        Forward pass through the model using precomputed embeddings.
        
        Args:
            audio_idx (Tensor): A tensor of indices referring to rows in `self.audio_embeddings`.
            
        Returns:
            tensor: Predictions (logits) from the classifier for the given sample
        """
        embeddings = self.audio_embeddings[audio_idx].to(self.device)
        predictions = self.classifier(embeddings)
        return predictions


    def training_step(self, batch, batch_idx):
        """ Single training step """
        audio_idx, labels = batch
        logits = self.forward(audio_idx)
        loss = self.criterion(logits, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss


    def configure_optimizers(self):
        """ Optimizer and Learning Rate Scheduler """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
