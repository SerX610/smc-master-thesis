import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))  # get the current directory of this script
scripts_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))  # go two levels up to the scripts directory
sys.path.insert(0, scripts_dir)  # add the scripts directory to the Python path

import gc
import glob
import laion_clap  # this should now work as laion_clap is in the scripts directory
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm


def load_laion_clap(model_checkpoint_path, amodel='HTSAT-base'):
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
    model = laion_clap.CLAP_Module(enable_fusion=False, amodel=amodel, device=device)

    print("Loading model checkpoint...")
    model.load_ckpt(model_checkpoint_path)

    return model


def compute_gtzan_text_embeddings(clap_model, prompts):
    """
    Compute text embeddings for the GTZAN given prompts using the CLAP model.

    Args:
        clap_model (CLAP): Initialized CLAP model.
        prompts (list): List of text prompts.

    Returns:
        np.ndarray: Text embeddings for the prompts.
    """
    print("Computing text embeddings...")
    return clap_model.get_text_embedding(prompts)


def compute_gtzan_audio_embeddings(clap_model, dataset):
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


def extract_mtt_embeddings(dataset, clap_model, save_every=1000, save_path="./embeddings"):
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
            save_mtt_embeddings(
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
    

def save_mtt_embeddings(audio_embeddings, text_embeddings, labels, audio_embeddings_path, text_embeddings_path, labels_path):
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


def normalize_mtt_embeddings(embeddings):
    """
    Normalizes the given embeddings by subtracting the mean and dividing by the standard deviation.

    Args:
        embeddings (numpy.ndarray): A NumPy array containing embeddings.

    Returns:
        numpy.ndarray: The normalized embeddings, where the mean is 0 and the standard deviation is 1.
    """
    return (embeddings - np.mean(embeddings)) / np.std(embeddings)


def concatenate_and_save_all_mtt_embeddings(folder_path, split_name):
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
    all_audio_embeddings = normalize_mtt_embeddings(all_audio_embeddings)
    all_text_embeddings = normalize_mtt_embeddings(all_text_embeddings)

    # Save concatenated embeddings and labels
    np.save(f'{folder_path}_audio_embeddings.npy', all_audio_embeddings)
    np.save(f'{folder_path}_text_embeddings.npy', all_text_embeddings)
    np.save(f'{folder_path}_labels.npy', all_labels)

    # Remove partial files to save disk space
    for f in audio_files + text_files + label_files:
        os.remove(f)

    print(f"Concatenated and saved all {split_name} embeddings.")


def compute_sdd_text_embeddings(clap_model, dataset):
    """
    Compute text embeddings for the SDD dataset using the CLAP model.

    Args:
        clap_model (CLAP): Initialized CLAP model.
        dataset (SongDescriberDataset): The dataset containing text descriptions.

    Returns:
        np.ndarray: Array of text embeddings for the dataset.
    """
    print("Computing text embeddings...")
    captions = [dataset[idx][1] for idx in range(len(dataset))]
    return clap_model.get_text_embedding(captions)


def compute_sdd_audio_embeddings(clap_model, dataset, batch_size=32):
    """
    Compute audio embeddings for the SDD dataset using the CLAP model.

    Args:
        clap_model (CLAP): Initialized CLAP model.
        dataset (SongDescriberDataset): The dataset containing audio files.
        batch_size (int, optional): Number of audio files to process at each iteration. Defaults to 32.

    Returns:
        np.ndarray: Array of audio embeddings for the dataset.
    """
    print("Computing audio embeddings...")
    audio_embed = []
    audio_files = [dataset[idx][0] for idx in range(len(dataset))]
    
    # Process the audio files in batches for efficiency
    for i in tqdm(range(0, len(audio_files), batch_size)):
        batch_paths = audio_files[i:i + batch_size]
        batch_embeddings = clap_model.get_audio_embedding_from_filelist(batch_paths)
        audio_embed.append(batch_embeddings)
        
    # Concatenate all batch embeddings into a single array
    audio_embed = np.concatenate(audio_embed)
    return audio_embed


def get_idx_to_audio_paths(dataset):
    """
    Extracts a mapping from dataset indices to corresponding audio paths.

    Args:
        dataset (SongDescriberDataset): Dataset containing audio-text pairs.

    Returns:
        list[str]: A list where each index corresponds to a text description 
                   and contains the associated audio file path.
    """
    return [dataset[idx][0] for idx in tqdm(range(len(dataset)), desc="Extracting audio paths")]


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
    plt.title('Confusion Matrix for Zero-Shot Classification on the GTZAN dataset')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")

def compute_recall_at_k(similarity_matrix, k, dataset, idx_to_audio_paths):
    """
    Compute recall at k for text-to-audio retrieval.

    Args:
        similarity_matrix (np.ndarray): Similarity scores between text and audio embeddings.
        k (int): Number of top-ranked results to consider.
        dataset (SongDescriberDataset): Dataset containing audio-text pairs.
        idx_to_audio_paths (list[str]): Mapping of dataset indices to audio paths.

    Returns:
        float: Recall at k (proportion of queries where at least one relevant audio file is in top k results).
    """
    num_queries = similarity_matrix.shape[0]
    ranks = np.argsort(-similarity_matrix, axis=1)  # Sort in descending order (higher similarity first)
    correct_at_k = []

    for idx in range(num_queries):
        valid_audio_paths = dataset[idx][0]  # Valid audio paths for the current text
        top_k_audio_paths = []
        seen_audio_path = set()

        # Get the top-k unique audio paths
        for i in ranks[idx]:
            if idx_to_audio_paths[i] not in seen_audio_path:
                top_k_audio_paths.append(idx_to_audio_paths[i])
                seen_audio_path.add(idx_to_audio_paths[i])
            if len(top_k_audio_paths) == k:  # Stop when we have exactly k unique elements
                break

        # Check if any of the top-k audio paths match the valid audio paths for the current text
        if valid_audio_paths in top_k_audio_paths:
            correct_at_k.append(1)
        else:
            correct_at_k.append(0)

    return np.mean(correct_at_k)


def compute_median_rank(similarity_matrix, dataset, idx_to_audio_paths):
    """
    Compute the median rank of the first relevant audio sample for each query.

    Args:
        similarity_matrix (np.ndarray): Similarity scores between text and audio embeddings.
        dataset (SongDescriberDataset): Dataset containing audio-text pairs.
        idx_to_audio_paths (list[str]): Mapping of dataset indices to audio paths.

    Returns:
        float: Median rank of correct audio samples.
    """
    num_queries = similarity_matrix.shape[0]
    ranks = np.argsort(-similarity_matrix, axis=1)  # Sort in descending order (higher similarity first)
    median_ranks = []

    for idx in range(num_queries):
        valid_audio_paths = dataset[idx][0]  # Valid audio paths for the current text

        unique_ranks = []
        seen_ranks = set()

        # Ensure unique ranks while maintaining order
        for rank in ranks[idx]:
            if rank not in seen_ranks:
                unique_ranks.append(rank)
                seen_ranks.add(rank)

        # Find the rank of the first valid audio match
        for i, rank in enumerate(unique_ranks):
            if idx_to_audio_paths[rank] in valid_audio_paths:
                correct_rank = i + 1  # Rank is 1-based
                break

        median_ranks.append(correct_rank)

    return np.median(median_ranks)
