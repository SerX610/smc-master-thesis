#!/bin/bash

# Define the model checkpoint to download and its URL
MODEL_CKPT="music_audioset_epoch_15_esc_90.14.pt"
URL="https://huggingface.co/lukewys/laion_clap/resolve/main/$MODEL_CKPT"

# Define the destination directory and create it if it doesn't exist
DEST_DIR="models/laion-clap"
mkdir -p "$DEST_DIR"

# Download the file using wget
wget -O "$DEST_DIR/$MODEL_CKPT" "$URL"

# Check if the download was successful
if [ $? -eq 0 ]; then
    echo "Download completed successfully."
else
    echo "Download failed. Please check the URL and try again."
    exit 1
fi
