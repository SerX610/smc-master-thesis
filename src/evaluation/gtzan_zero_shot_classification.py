"""
This module performs zero-shot classification on the GTZAN dataset using the
LAION-AI Contrastive Language-Audio Pretraining (CLAP) model implementation.
It loads the GTZAN dataset, creates text prompts for each genre, computes text
and audio embeddings, and evaluates the model's performance using accuracy and
a confusion matrix. The results are saved as a confusion matrix plot.
"""

import torch

from src.utils.gtzan_dataset import GTZANDataset
from src.utils.utils import load_laion_clap, compute_gtzan_text_embeddings, compute_gtzan_audio_embeddings, perform_zero_shot_classification, calculate_accuracy, plot_confusion_matrix


def main():
    # Define path to the GTZAN dataset
    root_path = "data/gtzan"

    # Define the path to the LAION-CLAP model checkpoint and model type
    model_checkpoint_path = "models/laion-clap/music_audioset_epoch_15_esc_90.14.pt"
    amodel = 'HTSAT-base' 

    # Define the path to save results
    metrics_file = "results/metrics.txt"

    # Define path to save the confusion matrix plot
    plot_path = "results/gtzan_laion_clap_zero_shot_class_confusion_matrix.png"

    # Load dataset
    dataset = GTZANDataset(root_path)

    # Create prompts
    prompts = [f"This audio is a {genre} song." for genre in dataset.classes]

    # Load and initialize LAION-CLAP
    clap_model = load_laion_clap(model_checkpoint_path, amodel)
    
    with torch.no_grad():

        # Compute text embeddings
        text_embeddings = compute_gtzan_text_embeddings(clap_model, prompts)

        # Compute audio embeddings
        audio_embeddings = compute_gtzan_audio_embeddings(clap_model, dataset)

        # Perform zero-shot classification
        true_labels, pred_labels = perform_zero_shot_classification(dataset, audio_embeddings, text_embeddings)

    # Calculate accuracy    
    accuracy = calculate_accuracy(true_labels, pred_labels)
    print(f'GTZAN Zero-Shot Classification Accuracy: {accuracy:.2f}%')
    # Save accuracy to a text file in the results folder
    with open(metrics_file, "a") as f:
        f.write(f"GTZAN Zero-Shot Classification - Accuracy: {accuracy:.2f}%\n")

    # Plot and save confusion matrix
    plot_confusion_matrix(true_labels, pred_labels, dataset.classes, plot_path)


if __name__ == "__main__":
    main()
