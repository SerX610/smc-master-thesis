import numpy as np
from tqdm import tqdm
from msclap import CLAP
from mtt_dataset import MagnaTagATuneDataset, split_dataset


def extract_embeddings(dataset, clap_model):
    """
    Extracts audio and text embeddings from the MagnaTagATune dataset using a CLAP model.

    Args:
        dataset (MagnaTagATuneDataset): Instance of the MagnaTagATune dataset.
        clap_model (CLAP): Initialized CLAP model.

    Returns:
        tuple: 
            - all_audio_embeddings (numpy.ndarray): Array of shape (num_samples, embedding_dim) containing audio embeddings.
            - all_text_embeddings (numpy.ndarray): Array of shape (num_samples, embedding_dim) containing text embeddings.
    """

    # Initialize lists to store all embeddings
    all_audio_embeddings = []
    all_text_embeddings = []
    all_labels = []

    # Extract embeddings
    for idx in tqdm(range(len(dataset)), desc="Extracting embeddings..."):

        # Get file path and tags
        file_path, target, one_hot_target = dataset[idx]

        # Extract audio and text embeddings
        audio_embeddings = clap_model.get_audio_embeddings([file_path], resample=True)
        text_description = ", ".join(target)  # Combine tags into a text description
        text_embeddings = clap_model.get_text_embeddings([text_description])

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
    Saves extracted embeddings as .npy files.

    Args:
        audio_embeddings (numpy.ndarray): Numpy array containing audio embeddings.
        text_embeddings (numpy.ndarray): Numpy array containing text embeddings.
        labels (numpy.ndarray): Numpy array containing labels.
        audio_embeddings_path (str): Path where the audio embeddings will be saved.
        text_embeddings_path (str): Path where the text embeddings will be saved.
        labels_path (str): Path where the labels will be saved.
    """
    # Save embeddings as .npy files
    np.save(audio_embeddings_path, audio_embeddings)
    np.save(text_embeddings_path, text_embeddings)
    np.save(labels_path, labels)

    print(f"Audio embeddings saved to {audio_embeddings_path}")
    print(f"Text embeddings saved to {text_embeddings_path}")
    print(f"Labels saved to {labels_path}")


if __name__ == "__main__":
    mtt_data_path = "../../data/mtt"
    annotations_file = "annotations.csv"
    train_audio_embeddings_path = "train_audio_embeddings.npy"
    train_text_embeddings_path = "train_text_embeddings.npy"
    train_labels_path = "train_labels.npy"
    test_audio_embeddings_path = "test_audio_embeddings.npy"
    test_text_embeddings_path = "test_text_embeddings.npy"
    test_labels_path = "test_labels.npy"

    print("Initializing CLAP Model...")
    clap_model = CLAP(version='2023', use_cuda=True)  # Set use_cuda=True if you have a GPU

    print("Loading MagnaTagATune Dataset...")
    dataset = MagnaTagATuneDataset(root=mtt_data_path, annotations_file=annotations_file)

    print("Splitting dataset...")
    train_dataset, test_dataset = split_dataset(dataset, train_ratio=0.75)

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
