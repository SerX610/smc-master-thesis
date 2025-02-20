echo "Downloading GTZAN dataset..."
curl -L -o ../../data/gtzan/gtzan-dataset-music-genre-classification.zip https://www.kaggle.com/api/v1/datasets/download/andradaolteanu/gtzan-dataset-music-genre-classification

echo "Extracting files..."
unzip ../../data/gtzan/gtzan-dataset-music-genre-classification.zip -d ../../data/gtzan
mv ../../data/gtzan/Data/* ../../data/gtzan/
rmdir ../../data/gtzan/Data

echo "Removing unnecessary files..."
rm -rf ../../data/gtzan/gtzan-dataset-music-genre-classification.zip ../../data/gtzan/images_original ../../data/gtzan/features_30_sec.csv ../../data/gtzan/features_3_sec.csv ../../data/gtzan/genres_original/jazz/jazz.00054.wav
