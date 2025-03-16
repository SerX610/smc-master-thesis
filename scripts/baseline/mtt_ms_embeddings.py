"""
This script extracts audio and text embeddings from the MagnaTagATune (MTT)
dataset using the Microsoft CLAP model implementation. The extracted embeddings
are saved as .npy files for further use in machine learning models.
"""

import numpy as np
import os
import torch

from msclap import CLAP
from mtt_dataset import MagnaTagATuneDataset
from tqdm import tqdm


def extract_embeddings(dataset, clap_model):
    """
    Extracts audio and text embeddings from the MagnaTagATune dataset using a CLAP model.

    Args:
        dataset (MagnaTagATuneDataset): Instance of the MagnaTagATune dataset.
        clap_model (CLAP): Initialized CLAP model.

    Returns:
        tuple: A tuple containing:
            - all_audio_embeddings (numpy.ndarray): Audio embeddings with shape (num_samples, embedding_dim).
            - all_text_embeddings (numpy.ndarray): Text embeddings with shape (num_samples, embedding_dim).
            - all_labels (numpy.ndarray): One-hot encoded labels with shape (num_samples, num_classes).
    """

    # Initialize lists to store all embeddings and labels
    all_audio_embeddings = []
    all_text_embeddings = []
    all_labels = []

    # Iterate through the dataset to extract embeddings
    for idx in tqdm(range(len(dataset)), desc="Extracting embeddings..."):
        
        # Retrieve audio file path, tags, and one-hot encoded labels from the dataset
        file_path, target, one_hot_target = dataset[idx]

        # Generate audio embeddings from the file path
        audio_embeddings = clap_model.get_audio_embeddings([file_path], resample=True)

        # Combine tags into a text description and generate text embeddings
        text_description = ", ".join(target)
        text_embeddings = clap_model.get_text_embeddings([text_description])

        # Append embeddings and labels to their respective lists
        all_audio_embeddings.append(audio_embeddings.cpu().numpy())
        all_text_embeddings.append(text_embeddings.cpu().numpy())
        all_labels.append(one_hot_target.numpy())

    # Convert lists to numpy arrays
    all_audio_embeddings = np.stack(all_audio_embeddings).squeeze(1)
    all_text_embeddings = np.stack(all_text_embeddings).squeeze(1)
    all_labels = np.stack(all_labels)

    return all_audio_embeddings, all_text_embeddings, all_labels


def save_embeddings(audio_embeddings, text_embeddings, labels, embeddings_folder, split):
    """
    Saves extracted embeddings and labels as .npy files.

    Args:
        audio_embeddings (numpy.ndarray): Array containing extracted audio embeddings.
        text_embeddings (numpy.ndarray): Array containing extracted text embeddings.
        labels (numpy.ndarray): One-hot encoded label array.
        embeddings_folder (str): Path to the directory where embeddings will be saved.
        split (str): Dataset split name (e.g., "train", "valid", "test").
    """

    # Ensure the embeddings folder exists
    os.makedirs(embeddings_folder, exist_ok=True)

    # Define file paths for storing the embeddings
    audio_embeddings_path = f"{embeddings_folder}/{split}_audio_embeddings.npy"
    text_embeddings_path = f"{embeddings_folder}/{split}_text_embeddings.npy"
    labels_path = f"{embeddings_folder}/{split}_labels.npy"

    # Save embeddings and labels as .npy files
    np.save(audio_embeddings_path, audio_embeddings)
    print(f"Audio embeddings saved to: {audio_embeddings_path}")

    np.save(text_embeddings_path, text_embeddings)
    print(f"Text embeddings saved to: {text_embeddings_path}")

    np.save(labels_path, labels)
    print(f"Labels saved to: {labels_path}")


def main():

    # Define the folder where embeddings will be stored
    embeddings_folder = "mtt_ms_embeddings"

    # Initialize CLAP model with GPU support if available
    print("Initializing CLAP Model...")
    use_cuda = True if torch.cuda.is_available() else False
    clap_model = CLAP(version='2023', use_cuda=use_cuda)

    # Process train, validation, and test splits
    for split in ["train", "valid", "test"]:
        print(f"Processing {split} dataset...")

        # Load dataset split
        dataset = MagnaTagATuneDataset(split)

        # Extract embeddings
        audio_embeddings, text_embeddings, labels = extract_embeddings(dataset, clap_model)

        # Save embeddings
        save_embeddings(audio_embeddings, text_embeddings, labels, embeddings_folder, split)

        # Print embeddings and labels shapes
        print(f"{split} audio embeddings shape: {audio_embeddings.shape}")
        print(f"{split} text embeddings shape: {text_embeddings.shape}")
        print(f"{split} labels shape: {labels.shape}")


if __name__ == "__main__":
    main()
