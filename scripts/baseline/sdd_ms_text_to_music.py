"""
This module performs text-to-music retrieval on the SDD dataset using the
Microsoft Contrastive Language-Audio Pretraining (CLAP) model implementation.
It loads the SDD dataset, computes text and audio embeddings, and evaluates
the model's performance using different metrics.
"""

import numpy as np
import torch

from msclap import CLAP
from sdd_dataset import SongDescriberDataset
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
    """

    all_audio_embeddings = []
    all_text_embeddings = []

    for idx in tqdm(range(len(dataset)), desc="Extracting embeddings..."):
        # Retrieve audio file path and text captions from the dataset
        audio_path, text_caption = dataset[idx]

        # Compute embeddings
        audio_embedding = clap_model.get_audio_embeddings([audio_path], resample=True)
        text_embedding = clap_model.get_text_embeddings([text_caption])

        # Append to lists
        all_audio_embeddings.append(audio_embedding.cpu().numpy())
        all_text_embeddings.append(text_embedding.cpu().numpy())

    # Convert lists to numpy arrays
    all_audio_embeddings = np.stack(all_audio_embeddings).squeeze(1)
    all_text_embeddings = np.stack(all_text_embeddings).squeeze(1)

    return all_audio_embeddings, all_text_embeddings


def get_idx_to_audio_paths(dataset):
    '''
    Extracts a mapping from dataset indices to corresponding audio paths.

    Args:
        dataset (SongDescriberDataset): Dataset containing audio-text pairs.

    Returns:
        list[str]: A list where each index corresponds to a text description 
                   and contains the associated audio file path.
    '''
    return [dataset[idx][0] for idx in tqdm(range(len(dataset)), desc="Extracting audio paths")]


def compute_recall_at_k(similarity_matrix, k, dataset, idx_to_audio_paths):
    '''
    Compute recall at k for text-to-audio retrieval.

    Args:
        similarity_matrix (np.ndarray): Similarity scores between text and audio embeddings.
        k (int): Number of top-ranked results to consider.
        dataset (SongDescriberDataset): Dataset containing audio-text pairs.
        idx_to_audio_paths (list[str]): Mapping of dataset indices to audio paths.

    Returns:
        float: Recall at k (proportion of queries where at least one relevant audio file is in top k results).
    '''
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
    '''
    Compute the median rank of the first relevant audio sample for each query.

    Args:
        similarity_matrix (np.ndarray): Similarity scores between text and audio embeddings.
        dataset (SongDescriberDataset): Dataset containing audio-text pairs.
        idx_to_audio_paths (list[str]): Mapping of dataset indices to audio paths.

    Returns:
        float: Median rank of correct audio samples.
    '''
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


def main():
    # Initialize the Song Describer Dataset
    dataset = SongDescriberDataset()

    # Initialize CLAP model with GPU support if available
    print("Initializing CLAP Model...")
    use_cuda = True if torch.cuda.is_available() else False
    clap_model = CLAP(version='2023', use_cuda=use_cuda)

    # Extract embeddings
    audio_embeddings, text_embeddings = extract_embeddings(dataset, clap_model)

    # Get index to audio paths mapping
    idx_to_audio_paths = get_idx_to_audio_paths(dataset)

    # Compute similarity matrix between text and audio embeddings
    print("Computing similarity scores...")
    similarity_matrix = np.dot(text_embeddings, audio_embeddings.T)

    # Evaluate retrieval performance
    recall_at_1 = compute_recall_at_k(similarity_matrix, 1, dataset, idx_to_audio_paths)
    recall_at_5 = compute_recall_at_k(similarity_matrix, 5, dataset, idx_to_audio_paths)
    recall_at_10 = compute_recall_at_k(similarity_matrix, 10, dataset, idx_to_audio_paths)
    median_rank = compute_median_rank(similarity_matrix, dataset, idx_to_audio_paths)

    # Print results
    print("\nEvaluation Results:")
    print(f"Recall at 1 (R@1): {100*recall_at_1:.2f}")
    print(f"Recall at 5 (R@5): {100*recall_at_5:.2f}")
    print(f"Recall at 10 (R@10): {100*recall_at_10:.2f}")
    print(f"Median Rank (MedR): {int(median_rank)}")


if __name__ == "__main__":
    main()
