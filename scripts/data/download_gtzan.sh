#!/bin/bash

# Define variables
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification"
DEST_DIR="../../data/gtzan"
ZIP_FILE="$DEST_DIR/gtzan-dataset-music-genre-classification.zip"
EXTRACT_DIR="$DEST_DIR/Data"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Download the dataset
echo "Downloading GTZAN dataset..."
curl -L -o "$ZIP_FILE" "$DATASET_URL"

# Extract the dataset
echo "Extracting files..."
unzip "$ZIP_FILE" -d "$DEST_DIR"

# Move extracted files and clean up directory structure
mv "$EXTRACT_DIR"/* "$DEST_DIR/"
rm -rf "$EXTRACT_DIR"

# Remove unnecessary files
echo "Removing unnecessary files..."
rm -rf "$ZIP_FILE" \
       "$DEST_DIR/images_original" \
       "$DEST_DIR/features_30_sec.csv" \
       "$DEST_DIR/features_3_sec.csv" \
       "$DEST_DIR/genres_original/jazz/jazz.00054.wav"

# Confirm completion
echo "Dataset setup complete."
