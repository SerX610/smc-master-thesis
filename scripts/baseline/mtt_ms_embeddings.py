"""
This script extracts audio and text embeddings from the MagnaTagATune (MTT) dataset using the Microsft CLAP model implementation.
The extracted embeddings are saved as .npy files for further use in machine learning models.
"""

import numpy as np

from msclap import CLAP
from mtt_dataset import MagnaTagATuneDataset, split_dataset
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


def save_embeddings(audio_embeddings, text_embeddings, labels, audio_embeddings_path, text_embeddings_path, labels_path):
    """
    Saves extracted embeddings and labels as .npy files.

    Args:
        audio_embeddings (numpy.ndarray): Array of audio embeddings.
        text_embeddings (numpy.ndarray): Array of text embeddings.
        labels (numpy.ndarray): Array of one-hot encoded labels.
        audio_embeddings_path (str): File path for saving audio embeddings.
        text_embeddings_path (str): File path for saving text embeddings.
        labels_path (str): File path for saving labels.
    """
    # Save embeddings as .npy files
    np.save(audio_embeddings_path, audio_embeddings)
    print(f"Audio embeddings saved to {audio_embeddings_path}")
    np.save(text_embeddings_path, text_embeddings)
    print(f"Text embeddings saved to {text_embeddings_path}")
    np.save(labels_path, labels)
    print(f"Labels saved to {labels_path}")


if __name__ == "__main__":
    mtt_data_path = "../../data/mtt"
    annotations_file = "annotations.csv"
    
    embeddings_folder = "mtt_ms_embeddings"
    train_audio_embeddings_path = f"{embeddings_folder}/train_audio_embeddings.npy"
    train_text_embeddings_path = f"{embeddings_folder}/train_text_embeddings.npy"
    train_labels_path = f"{embeddings_folder}/train_labels.npy"
    test_audio_embeddings_path = f"{embeddings_folder}/test_audio_embeddings.npy"
    test_text_embeddings_path = f"{embeddings_folder}/test_text_embeddings.npy"
    test_labels_path = f"{embeddings_folder}/test_labels.npy"

    print("Initializing CLAP Model...")
    clap_model = CLAP(version='2023', use_cuda=True)  # Set use_cuda=True if you have a GPU

    print("Loading MagnaTagATune Dataset...")
    dataset = MagnaTagATuneDataset(root=mtt_data_path, annotations_file=annotations_file)

    print("Splitting dataset...")
    train_dataset, test_dataset = split_dataset(dataset, train_ratio=0.75, seed=42)

    print("Extracting embeddings for training set...")
    train_audio_embeddings, train_text_embeddings, train_labels = extract_embeddings(train_dataset, clap_model)
    save_embeddings(train_audio_embeddings, train_text_embeddings, train_labels, train_audio_embeddings_path, train_text_embeddings_path, train_labels_path)

    print("Extracting embeddings for test set...")
    test_audio_embeddings, test_text_embeddings, test_labels = extract_embeddings(test_dataset, clap_model)
    save_embeddings(test_audio_embeddings, test_text_embeddings, test_labels, test_audio_embeddings_path, test_text_embeddings_path, test_labels_path)

    print(f"Train Audio embeddings shape: {train_audio_embeddings.shape}")
    print(f"Train Text embeddings shape: {train_text_embeddings.shape}")
    print(f"Train Labels shape: {train_labels.shape}")
    print(f"Test Audio embeddings shape: {test_audio_embeddings.shape}")
    print(f"Test Text embeddings shape: {test_text_embeddings.shape}")
    print(f"Test Labels shape: {test_labels.shape}")
