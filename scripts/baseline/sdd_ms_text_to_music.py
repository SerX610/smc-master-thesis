"""
This module performs text-to-music retrieval on the SDD dataset using the
Microsoft Contrastive Language-Audio Pretraining (CLAP) model implementation.
It loads the SDD dataset, computes text and audio embeddings, and evaluates
the model's performance using different metrics.
"""

import numpy as np
import torch

from collections import defaultdict
from msclap import CLAP
from sdd_dataset import SongDescriberDataset
from sklearn.metrics import pairwise_distances
from tqdm import tqdm


def extract_embeddings(dataset, clap_model):
    """
    Extracts audio and text embeddings from the SongDescriberDataset using CLAP.

    Args:
        dataset (SongDescriberDataset): Instance of the dataset.
        clap_model (CLAP): Initialized CLAP model.

    Returns:
        tuple: A tuple containing:
            - all_audio_embeddings (numpy.ndarray): Audio embeddings (num_samples, embedding_dim).
            - all_text_embeddings (numpy.ndarray): Text embeddings (num_samples, embedding_dim).
            - audio_to_text_indices (dict): Mapping from audio index to text indices.
    """

    all_audio_embeddings = []
    all_text_embeddings = []
    audio_to_text_indices = defaultdict(list)

    for idx in tqdm(range(len(dataset)), desc="Extracting embeddings..."):
        # Retrieve audio file path and text captions from the dataset
        audio_path, text_caption = dataset[idx]

        # Compute embeddings
        audio_embedding = clap_model.get_audio_embeddings([audio_path], resample=True)
        text_embedding = clap_model.get_text_embeddings([text_caption])

        # Append to lists
        all_audio_embeddings.append(audio_embedding.cpu().numpy())
        all_text_embeddings.append(text_embedding.cpu().numpy())

        # Store the mapping from audio index to text index
        audio_to_text_indices[audio_path].append(idx)

    # Convert lists to numpy arrays
    all_audio_embeddings = np.stack(all_audio_embeddings).squeeze(1)
    all_text_embeddings = np.stack(all_text_embeddings).squeeze(1)

    return all_audio_embeddings, all_text_embeddings, audio_to_text_indices


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

    # Initialize CLAP model with GPU support if available
    print("Initializing CLAP Model...")
    use_cuda = True if torch.cuda.is_available() else False
    clap_model = CLAP(version='2023', use_cuda=use_cuda)

    # Extract embeddings
    audio_embeddings, text_embeddings, audio_to_text_indices = extract_embeddings(dataset, clap_model)

    # Compute similarity
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
