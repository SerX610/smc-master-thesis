#!/bin/bash
set -e

# Extract the file ID from the Google Drive URL
FILE_ID="1ZrW_SvOoc8ZKJ-1Ci5dLlk2GMbqSPGsA"
FILE_NAME="HTSAT-pretrained.ckpt"

# Define the destination directory and create it if it doesn't exist
DEST_DIR="models/laion-clap"
mkdir -p "$DEST_DIR"

# Use gdown with the file ID
gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "$DEST_DIR/$FILE_NAME"

echo "Checkpoint downloaded to $DEST_DIR/$FILE_NAME"
