"""
This module performs zero-shot classification on the GTZAN dataset using the
Contrastive Language-Audio Pretraining (CLAP) model. It loads the GTZAN
dataset, creates text prompts for each genre, computes text and audio
embeddings, and evaluates the model's performance using accuracy and a
confusion matrix. The results are saved as a confusion matrix plot.
"""

import laion_clap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix

from gtzan_dataset import GTZANDataset


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
    return [f"This audio is a {genre} song." for genre in classes]


def load_laion_clap(model_checkpoint_path):
    """
    Load and initialize the LAION-CLAP model.

    Args:
        model_checkpoint_path (str): Path to the model checkpoint.

    Returns:
        CLAP_Module: Initialized LAION-CLAP model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Initializing LAION-CLAP model...")
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base', device=device)

    print("Loading model checkpoint...")
    model.load_ckpt(model_checkpoint_path)

    return model


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
    return clap_model.get_text_embedding(prompts)


def compute_audio_embeddings(clap_model, dataset):
    """
    Compute audio embeddings for the given dataset using the CLAP model.

    Args:
        clap_model (CLAP): Initialized CLAP model.
        dataset (GTZANDataset): GTZAN dataset object.

    Returns:
        np.ndarray: Audio embeddings for the dataset.
    """
    print("Computing audio embeddings...")
    audio_embed = []
    audio_files = dataset.audio_paths
    for i in tqdm(range(0, len(audio_files), 32)):
        batch_paths = audio_files[i:i + 32]
        batch_embeddings = clap_model.get_audio_embedding_from_filelist(batch_paths)
        audio_embed.append(batch_embeddings)
    audio_embed = np.concatenate(audio_embed)
    return audio_embed
    # return clap_model.get_audio_embedding_from_filelist(dataset.audio_paths)


def perform_zero_shot_classification(dataset, audio_embeddings, text_embeddings):
    """
    Perform zero-shot classification using the computed audio and text embeddings.

    Args:
        dataset (GTZANDataset): GTZAN dataset object.
        audio_embeddings (np.ndarray): Audio embeddings for the dataset.
        text_embeddings (np.ndarray): Text embeddings for the prompts.

    Returns:
        tuple: A tuple containing true labels and predicted labels.
    """
    print("Performing zero-shot classification...")
    true_labels = torch.tensor([dataset.class_to_idx[genre] for genre in dataset.targets]).view(-1, 1)
    ranking = torch.argsort(torch.tensor(audio_embeddings) @ torch.tensor(text_embeddings).t(), descending=True)
    pred_labels = ranking[:, 0].unsqueeze(1)
    return true_labels, pred_labels


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
    plt.title('Confusion Matrix for GTZAN Zero-Shot Classification using LAION-CLAP')
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

    # Define the path to the LAION-CLAP model checkpoint
    model_checkpoint_path = "../../models/laion-clap/music_audioset_epoch_15_esc_90.14.pt"

    # Define path to save the confusion matrix plot
    plot_path = "../../results/gtzan_laion_clap_zero_shot_class_confusion_matrix.png"

    # Load dataset
    dataset = load_dataset(root_path)

    # Create prompts
    prompts = create_arbitrary_prompts(dataset.classes)

    # Load and initialize LAION-CLAP
    clap_model = load_laion_clap(model_checkpoint_path)
    
    with torch.no_grad():

        # Compute text embeddings
        text_embeddings = compute_text_embeddings(clap_model, prompts)

        # Compute audio embeddings
        audio_embeddings = compute_audio_embeddings(clap_model, dataset)

        # Perform zero-shot classification
        true_labels, pred_labels = perform_zero_shot_classification(dataset, audio_embeddings, text_embeddings)

    # Calculate accuracy    
    accuracy = calculate_accuracy(true_labels, pred_labels)
    print(f'GTZAN Zero-Shot Classification Accuracy: {accuracy:.2f}%')

    # Plot and save confusion matrix
    plot_confusion_matrix(true_labels, pred_labels, dataset.classes, plot_path)


if __name__ == "__main__":
    main()
