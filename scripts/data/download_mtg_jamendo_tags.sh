#!/bin/bash

# Define variables
TAGS_URL="https://raw.githubusercontent.com/MTG/mtg-jamendo-dataset/refs/heads/master/data/splits/split-0"
TRAIN_TAGS_FILE="autotagging-train.tsv"
VALID_TAGS_FILE="autotagging-validation.tsv"
DEST_DIR="data/mtg_jamendo/split-0"

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Download MTG Jamendo tags file
echo "Downloading MTG Jamendo tags..."
curl -L -o "$DEST_DIR/$TRAIN_TAGS_FILE" "$TAGS_URL/$TRAIN_TAGS_FILE"
curl -L -o "$DEST_DIR/$VALID_TAGS_FILE" "$TAGS_URL/$VALID_TAGS_FILE"

# Confirm completion
echo "Tags file downloaded."