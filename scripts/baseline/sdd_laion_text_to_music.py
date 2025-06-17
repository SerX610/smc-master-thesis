"""
This module performs text-to-music retrieval on the SDD dataset using the
LAION-AI Contrastive Language-Audio Pretraining (CLAP) model implementation.
It loads the SDD dataset, computes text and audio embeddings, and evaluates
the model's performance using different metrics.
"""

import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))  # get the current directory of this script
scripts_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))  # go two levels up to the scripts directory
sys.path.insert(0, scripts_dir)  # add the scripts directory to the Python path
import laion_clap  # this should now work as laion_clap is in the scripts directory

import numpy as np
import torch

from sdd_dataset import SongDescriberDataset
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

    # Get index to audio paths mapping
    idx_to_audio_paths = get_idx_to_audio_paths(dataset)

    # Uncomment to store computed embeddings
    # np.save("sdd_laion_embeddings/text_embeddings.npy", text_embeddings)
    # np.save("sdd_laion_embeddings/audio_embeddings.npy", audio_embeddings)

    # Uncomment to load precomputed embeddings
    # text_embeddings = np.load("sdd_laion_embeddings/text_embeddings.npy")
    # audio_embeddings = np.load("sdd_laion_embeddings/audio_embeddings.npy")

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
