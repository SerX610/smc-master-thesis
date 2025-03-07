"""
This module performs zero-shot classification on the GTZAN dataset using the
Microsoft Contrastive Language-Audio Pretraining (CLAP) model implementation. 
It loads the GTZAN dataset, creates text prompts for each genre, computes text
and audio embeddings, and evaluates the model's performance using accuracy and
a confusion matrix. The results are saved as a confusion matrix plot.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch.nn.functional as F

from gtzan_dataset import GTZANDataset
from msclap import CLAP
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm


def load_dataset(root_path):
    """
    Load the GTZAN dataset from the specified root path.

    Args:
        root_path (str): Path to the root directory of the GTZAN dataset.

    Returns:
        GTZANDataset: Initialized GTZAN dataset object.
    """
    print("Loading GTZAN dataset...")
    return GTZANDataset(root=root_path)


def create_arbitrary_prompts(classes):
    """
    Create text prompts for zero-shot classification.

    Args:
        classes (list): List of genre names.

    Returns:
        list: List of text prompts for each genre.
    """
    print("Creating text prompts...")
    return [f"This is a {genre} song." for genre in classes]


def compute_text_embeddings(clap_model, prompts):
    """
    Compute text embeddings for the given prompts using the CLAP model.

    Args:
        clap_model (CLAP): Initialized CLAP model.
        prompts (list): List of text prompts.

    Returns:
        np.ndarray: Text embeddings for the prompts.
    """
    print("Computing text embeddings...")
    return clap_model.get_text_embeddings(prompts)


def perform_zero_shot_classification(clap_model, dataset, text_embeddings):
    """
    Perform zero-shot classification on the dataset using the CLAP model.

    Args:
        clap_model (CLAP): Initialized CLAP model.
        dataset (GTZANDataset): GTZAN dataset object.
        text_embeddings (np.ndarray): Text embeddings for the prompts.

    Returns:
        tuple: Predicted probabilities and true labels.
    """
    y_preds, y_labels = [], []
    print("Performing zero-shot classification...")
    for i in tqdm(range(len(dataset))):
        x, _, one_hot_target = dataset[i]  # Get audio file path, target, and one-hot target
        audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)  # Compute audio embeddings
        similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)  # Compute similarity
        y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()  # Apply softmax to get probabilities
        y_preds.append(y_pred)
        y_labels.append(one_hot_target.detach().cpu().numpy())

    # Concatenate predictions and labels
    y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
    return y_preds, y_labels


def calculate_accuracy(true_labels, pred_labels):
    """
    Calculate the accuracy of the predictions.

    Args:
        true_labels (np.ndarray): Ground truth labels.
        pred_labels (np.ndarray): Predicted labels.

    Returns:
        float: Accuracy score.
    """
    acc = accuracy_score(true_labels, pred_labels)
    return acc * 100


def plot_confusion_matrix(true_labels, pred_labels, classes, save_path):
    """
    Plot and save the confusion matrix.

    Args:
        true_labels (np.ndarray): Ground truth labels.
        pred_labels (np.ndarray): Predicted labels.
        classes (list): List of genre names.
        save_path (str): Path to save the confusion matrix plot.
    """
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix for GTZAN Zero-Shot Classification')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")


def main():
    # Define path to the GTZAN dataset
    root_path = "../../data/gtzan"

    # Define path to save the confusion matrix plot
    plot_path = "../../results/gtzan_zero_shot_class_confusion_matrix.png"

    # Load dataset
    dataset = load_dataset(root_path)

    # Create prompts
    prompts = create_arbitrary_prompts(dataset.classes)

    # Load and initialize CLAP
    clap_model = CLAP(version='2023', use_cuda=False)  # Set use_cuda=True if you have a GPU

    # Compute text embeddings
    text_embeddings = compute_text_embeddings(clap_model, prompts)

    # Perform zero-shot classification
    y_preds, y_labels = perform_zero_shot_classification(clap_model, dataset, text_embeddings)

    # Calculate accuracy
    true_labels = np.argmax(y_labels, axis=1)
    pred_labels = np.argmax(y_preds, axis=1)
    accuracy = calculate_accuracy(true_labels, pred_labels)
    print(f'GTZAN Zero-Shot Classification Accuracy: {accuracy:.2f}%')

    # Plot and save confusion matrix
    plot_confusion_matrix(true_labels, pred_labels, dataset.classes, plot_path)


if __name__ == "__main__":
    main()
