#!/bin/bash

# Define variables
BASE_URL="https://mirg.city.ac.uk/datasets/magnatagatune/mp3.zip."
DEST_DIR="../../data/mtt/"
MERGED_ZIP="$DEST_DIR/mp3.zip"
TAGS_URL="https://mirg.city.ac.uk/datasets/magnatagatune/annotations_final.csv"
ANNOTATIONS_FILE="$DEST_DIR/annotations_final.csv"
CLEANED_ANNOTATIONS_FILE="$DEST_DIR/annotations.csv"

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

# Download the annotations file
echo "Downloading annotations file..."
curl -L -o "$ANNOTATIONS_FILE" "$TAGS_URL"

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

# Clean annotations from corrupted files
python clean_mtt_annotations.py "$ANNOTATIONS_FILE" "${corrupted_files[@]}" "$CLEANED_ANNOTATIONS_FILE"
rm -rf $ANNOTATIONS_FILE

# Confirm completion
echo "Dataset setup complete."
