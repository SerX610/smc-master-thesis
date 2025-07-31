#!/bin/bash

# Define variables
DEST_DIR="data/sdd/"
RECORD_ID="10072001"
SONG_DESCRIBER="https://zenodo.org/record/$RECORD_ID/files/song_describer.csv"
AUDIO_DATA="https://zenodo.org/record/$RECORD_ID/files/audio.zip"
ZIP_PATH="$DEST_DIR/audio.zip"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Download the dataset csv file
echo "Downloading Song Describer..."
curl -L -o "$DEST_DIR/song_describer.csv" "$SONG_DESCRIBER"

# Download the audio dataset
echo "Downloading Audio Data..."
curl -L -o "$ZIP_PATH" "$AUDIO_DATA"

# Extract the audio dataset
echo "Extracting files..."
unzip "$ZIP_PATH" -d "$DEST_DIR"

# Remove unnecessary zip files
rm "$ZIP_PATH"
