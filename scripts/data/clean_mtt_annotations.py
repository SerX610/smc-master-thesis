import pandas as pd
import sys


def clean_annotations(annotations_file, corrupted_files, output_file):
    """
    Cleans the annotations CSV by removing rows where 'mp3_path' is in the corrupted files list.
    
    Args:
        annotations_file (str): Path to the input annotations CSV file.
        corrupted_files (list): List of corrupted file paths to be removed from the CSV.
        output_file (str): Path where the cleaned CSV should be saved.
    """
    # Load the annotations file
    annotations = pd.read_csv(annotations_file, sep="\t")

    # Convert corrupted files list to a set for faster lookup
    corrupted_set = set(corrupted_files)

    # Filter the annotations to exclude corrupted files
    cleaned_annotations = annotations[~annotations["mp3_path"].isin(corrupted_set)]

    # Save the cleaned annotations to a new CSV file
    cleaned_annotations.to_csv(output_file, sep="\t", index=False)

    print(f"Cleaned annotations saved to: {output_file}")


if __name__ == "__main__":
    # Parse the arguments
    annotations_file = sys.argv[1]
    corrupted_files = sys.argv[2:-1]
    output_file = sys.argv[-1]

    # Clean the annotations and save to output file
    clean_annotations(annotations_file, corrupted_files, output_file)
