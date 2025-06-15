"""
This script extracts audio and text embeddings from the MagnaTagATune (MTT) dataset using the LAION-AI CLAP model implementation.
The extracted embeddings are saved as .npy files for further use in machine learning models.
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))  # get the current directory of this script
scripts_dir = os.path.abspath(os.path.join(current_dir, ".."))  # go one level up to the scripts directory
sys.path.insert(0, scripts_dir)  # add the scripts directory to the Python path
import laion_clap  # this should now work as laion_clap is in the scripts directory

import gc
import glob
import numpy as np
import os
import torch

from mtt_dataset import MagnaTagATuneDataset
from tqdm import tqdm


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


def extract_embeddings(dataset, clap_model, save_every=1000, save_path="./embeddings"):
    """
    Extracts audio and text embeddings from the MagnaTagATune dataset using a CLAP model.

    Args:
        dataset (MagnaTagATuneDataset): Instance of the MagnaTagATune dataset.
        clap_model (CLAP): Initialized CLAP model.
        save_every (int, optional): Number of samples after which embeddings are saved to disk. Defaults to 1000.
        save_path (str, optional): Base path for saving embeddings. Defaults to './embeddings'.

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

    # Extract embeddings for each sample in the dataset
    for idx, (file_path, target, one_hot_target) in enumerate(tqdm(dataset, desc="Extracting embeddings...")):

        # Extract audio embeddings
        audio_embeddings = clap_model.get_audio_embedding_from_filelist(x=[file_path], use_tensor=False)

        # Combine tags into a single text description and extract text embeddings
        text_description = ", ".join(target)
        text_embeddings = clap_model.get_text_embedding([text_description])

        # Append embeddings and labels to their respective lists
        all_audio_embeddings.append(audio_embeddings)
        all_text_embeddings.append(text_embeddings)
        all_labels.append(one_hot_target.numpy())

        # Save embeddings periodically to avoid memory issues
        if (idx + 1) % save_every == 0:
            save_embeddings(
                np.stack(all_audio_embeddings).squeeze(1),
                np.stack(all_text_embeddings).squeeze(1),
                np.stack(all_labels),
                f"{save_path}_audio_embeddings_{idx}.npy",
                f"{save_path}_text_embeddings_{idx}.npy",
                f"{save_path}_labels_{idx}.npy"
            )
            # Clear lists and free up memory
            all_audio_embeddings.clear()
            all_text_embeddings.clear()
            all_labels.clear()
            gc.collect()
            torch.cuda.empty_cache()

    # Return any remaining embeddings that were not saved
    if all_audio_embeddings:
        return (np.stack(all_audio_embeddings).squeeze(1), np.stack(all_text_embeddings).squeeze(1), np.stack(all_labels))
    else:
        return np.array([]), np.array([]), np.array([])
    

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


def normalize_embeddings(embeddings):
    """
    Normalizes the given embeddings by subtracting the mean and dividing by the standard deviation.

    Args:
        embeddings (numpy.ndarray): A NumPy array containing embeddings.

    Returns:
        numpy.ndarray: The normalized embeddings, where the mean is 0 and the standard deviation is 1.
    """
    return (embeddings - np.mean(embeddings)) / np.std(embeddings)


def concatenate_and_save_all_embeddings(folder_path, split_name):
    """
    Concatenates all partial .npy embedding files into a single file.

    Args:
        folder_path (str): Folder containing saved .npy files.
        split_name (str): Dataset split name (e.g., 'train' or 'test').
    """
    # Find all partial embedding and label files
    audio_files = sorted(glob.glob(f'{folder_path}_audio_embeddings_*.npy'))
    text_files = sorted(glob.glob(f'{folder_path}_text_embeddings_*.npy'))
    label_files = sorted(glob.glob(f'{folder_path}_labels_*.npy'))

    # Concatenate all partial files
    all_audio_embeddings = np.concatenate([np.load(f) for f in audio_files], axis=0)
    all_text_embeddings = np.concatenate([np.load(f) for f in text_files], axis=0)
    all_labels = np.concatenate([np.load(f) for f in label_files], axis=0)

    # Standardize embeddings
    all_audio_embeddings = normalize_embeddings(all_audio_embeddings)
    all_text_embeddings = normalize_embeddings(all_text_embeddings)
    
    # Save concatenated embeddings and labels
    np.save(f'{folder_path}_audio_embeddings.npy', all_audio_embeddings)
    np.save(f'{folder_path}_text_embeddings.npy', all_text_embeddings)
    np.save(f'{folder_path}_labels.npy', all_labels)

    # Remove partial files to save disk space
    for f in audio_files + text_files + label_files:
        os.remove(f)

    print(f"Concatenated and saved all {split_name} embeddings.")


if __name__ == "__main__":
    # Define the folder where embeddings will be stored
    embeddings_folder = "mtt_laion_embeddings"
    os.makedirs(embeddings_folder, exist_ok=True)

    # Initialize CLAP model from checkpoint with GPU support if available
    model_checkpoint_path = "../../models/laion-clap/music_audioset_epoch_15_esc_90.14.pt"
    clap_model = load_laion_clap(model_checkpoint_path)

    with torch.no_grad():
        for split in ["valid", "train", "test"]:
            print(f"Processing {split} dataset...")

            # Load dataset split
            dataset = MagnaTagATuneDataset(split)

            # Extract embeddings
            audio_embeddings, text_embeddings, labels = extract_embeddings(dataset, clap_model, save_path=f"{embeddings_folder}/{split}")
            save_embeddings(audio_embeddings,
                            text_embeddings,
                            labels,
                            f"{embeddings_folder}/{split}_audio_embeddings_last.npy",
                            f"{embeddings_folder}/{split}_text_embeddings_last.npy",
                            f"{embeddings_folder}/{split}_labels_last.npy"
                        )
            concatenate_and_save_all_embeddings(f"{embeddings_folder}/{split}", split)
