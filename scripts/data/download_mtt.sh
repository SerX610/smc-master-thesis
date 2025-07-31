#!/bin/bash

# Define variables
BASE_URL="https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip."
DEST_DIR="data/mtt/"
MERGED_ZIP="$DEST_DIR/mp3.zip"
TAGS_URL="https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv"
SPLIT_URL="https://github.com/jongpillee/music_dataset_split/archive/refs/heads/master.zip"
SPLIT_DIR="$DEST_DIR/MTAT_split"

# List of corrupted files to remove
corrupted_files=(
    "6/norine_braun-now_and_zen-08-gently-117-146.mp3"
    "8/jacob_heringman-josquin_des_prez_lute_settings-19-gintzler__pater_noster-204-233.mp3"
    "9/american_baroque-dances_and_suites_of_rameau_and_couperin-25-le_petit_rien_xiveme_ordre_couperin-88-117.mp3"
)

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Download all parts
for i in {001..003}; do
    FILE="$DEST_DIR/mp3.zip.$i"
    URL="$BASE_URL$i"
    echo "Downloading $URL..."
    curl -L -o "$FILE" "$URL"
done

# Merge the zip parts
echo "Merging zip parts..."
cat "$DEST_DIR"/mp3.zip.* > "$MERGED_ZIP"

# Extract the dataset
echo "Extracting files..."
unzip "$MERGED_ZIP" -d "$DEST_DIR"

# Remove unnecessary zip files
rm "$DEST_DIR"/mp3.zip*

# Remove corrupted files
echo "Removing corrupted files..."
for file in "$DEST_DIR/${corrupted_files[@]}"; do
    if [ -f "$file" ]; then
        rm "$file"
        echo "Removed: $file"
    else
        echo "File not found: $file"
    fi
done

# Download and extract MTAT_split folder
echo "Downloading MTAT_split..."
wget -O "$DEST_DIR/master.zip" "$SPLIT_URL"

echo "Extracting MTAT_split..."
unzip "$DEST_DIR/master.zip" -d "$DEST_DIR"

# Move MTAT_split to correct location and clean up
mv "$DEST_DIR/music_dataset_split-master/MTAT_split" "$SPLIT_DIR"
rm -rf "$DEST_DIR/music_dataset_split-master" "$DEST_DIR/master.zip"

# Remove README.md from MTAT_split
rm "$SPLIT_DIR/README.md"
echo "Removed README.md from $SPLIT_DIR"

# Confirm completion
echo "Dataset setup complete."
