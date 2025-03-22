"""
This module performs text-to-music retrieval on the SDD dataset using the
LAION-AI Contrastive Language-Audio Pretraining (CLAP) model implementation.
It loads the SDD dataset, computes text and audio embeddings, and evaluates
the model's performance using different metrics.
"""

import laion_clap
import numpy as np
import torch

from collections import defaultdict
from sdd_dataset import SongDescriberDataset
from sklearn.metrics import pairwise_distances
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


def compute_text_embeddings(clap_model, dataset):
    """
    Compute text embeddings for the dataset using the CLAP model.

    Args:
        clap_model (CLAP): Initialized CLAP model.
        dataset (SongDescriberDataset): The dataset containing text descriptions.

    Returns:
        np.ndarray: Array of text embeddings for the dataset.
    """
    print("Computing text embeddings...")
    captions = [dataset[idx][1] for idx in range(len(dataset))]
    return clap_model.get_text_embedding(captions)


def compute_audio_embeddings(clap_model, dataset, batch_size=32):
    """
    Compute audio embeddings for the dataset using the CLAP model.

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
    
    # Process the audio files in batches for efficiency.
    for i in tqdm(range(0, len(audio_files), batch_size)):
        batch_paths = audio_files[i:i + batch_size]
        batch_embeddings = clap_model.get_audio_embedding_from_filelist(batch_paths)
        audio_embed.append(batch_embeddings)
        
    # Concatenate all batch embeddings into a single array.
    audio_embed = np.concatenate(audio_embed)
    return audio_embed


def get_audio_to_text_indices(dataset):
    """
    Extracts the mapping between audio files and corresponding text descriptions.

    Args:
        dataset (SongDescriberDataset): Instance of the SongDescriberDataset containing audio-text pairs.

    Returns:
        dict: A dictionary mapping each audio file path to a list of corresponding text indices.
    """
    audio_to_text_indices = defaultdict(list)

    # Iterate over the dataset to create the audio-to-text mapping.
    for idx in tqdm(range(len(dataset)), desc="Extracting audio-to-text indices..."):
        # Retrieve audio file path and text captions.
        audio_path, _ = dataset[idx]

        # Store the mapping from audio index to text index.
        audio_to_text_indices[audio_path].append(idx)

    return audio_to_text_indices


def compute_recall_at_k(similarity_matrix, k, dataset, audio_to_text_indices):
    """
    Compute the recall at k for the retrieval system.

    Args:
        similarity_matrix (np.ndarray): Matrix of similarity scores between text and audio embeddings.
        k (int): The number of top matches to consider for recall calculation.
        dataset (SongDescriberDataset): The dataset to access audio-text relationships.
        audio_to_text_indices (dict): Dictionary mapping audio index to valid text indices.

    Returns:
        float: The recall at k, which is the proportion of queries for which at least one relevant audio 
               is found in the top k matches.
    """
    num_queries = similarity_matrix.shape[0]
    ranks = np.argsort(-similarity_matrix, axis=1)  # Sort in descending order (higher similarity first)
    correct_at_k = []

    for idx in range(num_queries):
        valid_audio_indices = set(audio_to_text_indices[dataset[idx][0]])  # Valid audio indices for the current text
        top_k_audio_indices = set(ranks[idx, :k])  # Top k audio indices based on similarity scores

        # Check if any of the top-k audio indices match the valid audio indices for the current text.
        if not valid_audio_indices.isdisjoint(top_k_audio_indices):
            correct_at_k.append(1)
        else:
            correct_at_k.append(0)

    return np.mean(correct_at_k)


def compute_median_rank(similarity_matrix, dataset, audio_to_text_indices):
    """
    Compute the median rank of the first relevant audio sample for each query.

    Args:
        similarity_matrix (np.ndarray): Matrix of similarity scores between text and audio embeddings.
        dataset (SongDescriberDataset): The dataset to access audio-text relationships.
        audio_to_text_indices (dict): Dictionary mapping audio index to valid text indices.

    Returns:
        float: The median rank of correct audio samples, where lower ranks are better.
    """
    num_queries = similarity_matrix.shape[0]
    ranks = np.argsort(-similarity_matrix, axis=1)  # Sort in descending order (higher similarity first)
    median_ranks = []

    for idx in range(num_queries):
        valid_audio_indices = set(audio_to_text_indices[dataset[idx][0]])  # Valid audio indices for the current text
        correct_rank = None

        # Find the rank of the first valid audio match
        for rank in ranks[idx]:
            if rank in valid_audio_indices:
                correct_rank = np.where(ranks[idx] == rank)[0][0] + 1  # Rank is 1-based
                break

        if correct_rank:
            median_ranks.append(correct_rank)

    return np.median(median_ranks)


def main():
    # Initialize the Song Describer Dataset
    dataset = SongDescriberDataset()

    # Define the path to the LAION-CLAP model checkpoint
    model_checkpoint_path = "../../models/laion-clap/music_audioset_epoch_15_esc_90.14.pt"

    # Load and initialize LAION-CLAP
    clap_model = load_laion_clap(model_checkpoint_path)

    with torch.no_grad():
        # Compute text embeddings
        text_embeddings = compute_text_embeddings(clap_model, dataset)

        # Compute audio embeddings
        audio_embeddings = compute_audio_embeddings(clap_model, dataset)

        # Get audio to text indices
        audio_to_text_indices = get_audio_to_text_indices(dataset)

    # Compute similarity matrix between text and audio embeddings
    print("Computing similarity scores...")
    similarity_matrix = 1 - pairwise_distances(text_embeddings, audio_embeddings, metric="cosine")

    # Evaluate retrieval performance
    recall_at_1 = compute_recall_at_k(similarity_matrix, 1, dataset, audio_to_text_indices)
    recall_at_5 = compute_recall_at_k(similarity_matrix, 5, dataset, audio_to_text_indices)
    recall_at_10 = compute_recall_at_k(similarity_matrix, 10, dataset, audio_to_text_indices)
    median_rank = compute_median_rank(similarity_matrix, dataset, audio_to_text_indices)

    # Print results
    print("\nEvaluation Results:")
    print(f"Recall at 1 (R@1): {recall_at_1:.4f}")
    print(f"Recall at 5 (R@5): {recall_at_5:.4f}")
    print(f"Recall at 10 (R@10): {recall_at_10:.4f}")
    print(f"Median Rank (MedR): {int(median_rank)}")


if __name__ == "__main__":
    main()
