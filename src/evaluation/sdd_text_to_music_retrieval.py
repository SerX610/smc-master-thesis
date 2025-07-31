"""
This module performs text-to-music retrieval on the SDD dataset using the
LAION-AI Contrastive Language-Audio Pretraining (CLAP) model implementation.
It loads the SDD dataset, computes text and audio embeddings, and evaluates
the model's performance using different metrics.
"""

import numpy as np
import torch

from src.utils.sdd_dataset import SongDescriberDataset
from src.utils.utils import load_laion_clap, compute_sdd_text_embeddings, compute_sdd_audio_embeddings, get_idx_to_audio_paths, compute_recall_at_k, compute_median_rank

def main():
    # Initialize the Song Describer Dataset
    dataset = SongDescriberDataset()

    # Define the path to the LAION-CLAP model checkpoint
    model_checkpoint_path = "models/laion-clap/music_audioset_epoch_15_esc_90.14.pt"

    # Define the path to save results
    metrics_file = "results/metrics.txt"

    # Load and initialize LAION-CLAP
    clap_model = load_laion_clap(model_checkpoint_path)

    with torch.no_grad():
        # Compute text embeddings
        text_embeddings = compute_sdd_text_embeddings(clap_model, dataset)

        # Compute audio embeddings
        audio_embeddings = compute_sdd_audio_embeddings(clap_model, dataset)

    # Get index to audio paths mapping
    idx_to_audio_paths = get_idx_to_audio_paths(dataset)

    # Uncomment to store computed embeddings
    # np.save("results/sdd_laion_embeddings/text_embeddings.npy", text_embeddings)
    # np.save("results/sdd_laion_embeddings/audio_embeddings.npy", audio_embeddings)

    # Uncomment to load precomputed embeddings
    # text_embeddings = np.load("results/sdd_laion_embeddings/text_embeddings.npy")
    # audio_embeddings = np.load("results/sdd_laion_embeddings/audio_embeddings.npy")

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

    # Append results to metrics file
    with open(metrics_file, "a") as f:
        f.write(f"SDD Text-to-Music Retrieval - R@1: {100*recall_at_1:.2f}\n")
        f.write(f"SDD Text-to-Music Retrieval - R@5: {100*recall_at_5:.2f}\n")
        f.write(f"SDD Text-to-Music Retrieval - R@10: {100*recall_at_10:.2f}\n")
        f.write(f"SDD Text-to-Music Retrieval - MedR: {int(median_rank)}\n\n")


if __name__ == "__main__":
    main()
