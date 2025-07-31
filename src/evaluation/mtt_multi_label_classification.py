"""
This script extracts audio and text embeddings from the MagnaTagATune (MTT) dataset using the LAION-AI CLAP model implementation.
The extracted embeddings are saved as .npy files for further use in machine learning models.
"""

import os
import torch

from src.utils.utils import (
    load_laion_clap,
    extract_mtt_embeddings,
    save_mtt_embeddings,
    concatenate_and_save_all_mtt_embeddings,
)
from src.utils.mtt_dataset import MagnaTagATuneDataset
from src.utils.clap_transfer_learning import (
    CLAPTrainer,
    CLAPTrainerConfig,
    CLAPTester,
    CLAPTestConfig,
)


def main():
    # Define the folder where embeddings will be stored
    embeddings_folder = "results/mtt_laion_embeddings"

    # Define the path to the LAION-CLAP model checkpoint
    model_checkpoint_path = "models/laion-clap/music_audioset_epoch_15_esc_90.14.pt"

    # Define the trained model directory and checkpoint
    trained_model_dir="models/trained/"
    trained_model_ckpt="mtt-laion-clap-multilabel-classifier-checkpoint"

    # Define the tag index mapping path
    tag_index_mapping_path="data/mtt/MTAT_split/top50_tags.txt"
    
    # Define the path to save results
    metrics_file = "results/metrics.txt"

    # Define the path to save confusion matrix plot
    plot_dir="results/mtt_laion_clap_multilabel_classification_confusion_matrix.png"

    compute_embeddings = True  # Set to False to skip embedding extraction and use precomputed embeddings

    if compute_embeddings:
        # Ensure the embeddings folder exists    
        os.makedirs(embeddings_folder, exist_ok=True)

        # Initialize CLAP model from checkpoint with GPU support if available
        clap_model = load_laion_clap(model_checkpoint_path)

        with torch.no_grad():
            for split in ["valid", "train", "test"]:
                print(f"Processing {split} dataset...")

                # Load dataset split
                dataset = MagnaTagATuneDataset(split)

                # Extract embeddings
                audio_embeddings, text_embeddings, labels = extract_mtt_embeddings(dataset, clap_model, save_path=f"{embeddings_folder}/{split}")
                save_mtt_embeddings(audio_embeddings,
                                text_embeddings,
                                labels,
                                f"{embeddings_folder}/{split}_audio_embeddings_last.npy",
                                f"{embeddings_folder}/{split}_text_embeddings_last.npy",
                                f"{embeddings_folder}/{split}_labels_last.npy"
                            )
                concatenate_and_save_all_mtt_embeddings(f"{embeddings_folder}/{split}", split)
    
    # Initialize configuration with specified parameters
    train_config = CLAPTrainerConfig(
        embeddings_folder=embeddings_folder,
        model_dir=trained_model_dir,
        model_ckpt=trained_model_ckpt,
        batch_size=64,
        max_epochs=10,
    )

    # Initialize and train the model using the configuration.
    trainer = CLAPTrainer(train_config)
    trainer.train_model()

    # Initialize configuration with specified parameters
    test_config = CLAPTestConfig(
        embeddings_folder=embeddings_folder, 
        model_ckpt=f"{trained_model_dir}/{trained_model_ckpt}.ckpt",
        batch_size=64,
        tag_index_mapping_path=tag_index_mapping_path,
        plot_dir=plot_dir
    )

    # Initialize the tester and run the test
    tester = CLAPTester(test_config)
    tester.test_model()

    # Append results to metrics file
    with open(metrics_file, "a") as f:
        for metric, value in tester.test_results.items():
            # Strip 'test-' prefix and '-macro' suffix if present
            clean_metric = metric.removeprefix("test-").removesuffix("-macro")
            f.write(f"MTT Multi-Label Classification - {clean_metric}: {value:.3f}\n")


if __name__ == "__main__":
    main()
