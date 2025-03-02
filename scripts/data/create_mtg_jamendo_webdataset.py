import os
import csv
import json
import webdataset as wds


from collections import defaultdict

def clean_tag(tag):
    """
    Remove prefixes like 'genre---', 'instrument---', etc., from tags.
    """
    return tag.split('---', 1)[1] if '---' in tag else tag


def get_writer(writers, output_dir, xx):
    """
    Return (or create if needed) a TarWriter for a specific 'xx' folder (00-99).
    Each shard is named like '00.tar', '01.tar', etc.
    """
    if writers[xx] is None:
        tar_path = os.path.join(output_dir, f"{xx}.tar")
        writers[xx] = wds.TarWriter(tar_path)
    return writers[xx]


def create_webdataset(tsv_file, audio_root, output_dir):
    writers = defaultdict(lambda: None)  # Dictionary to hold open shard writers, keyed by folder prefix (from '00' to '99')
    shard_counts = defaultdict(int)  # File counts per shard
    count = 0  # Track number of samples written

    # Open TSV file
    with open(tsv_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)  # Skip header

        for row in reader:
            # Unpack the first five fixed columns
            track_id, artist_id, album_id, rel_path, duration = row[:5]

            # Process and clean tags
            tag_tokens = row[5:]
            tags_list = sorted(set(clean_tag(tag.strip()) for tag in tag_tokens if tag.strip()))
            tags = ", ".join(tags_list)  # Join tags as a comma-separated string

            # Adjust audio file path to insert ".low" before ".mp3"
            base, ext = os.path.splitext(rel_path)
            audio_filename = base + ".low" + ext
            audio_path = os.path.join(audio_root, audio_filename)

            # Extract the folder prefix (e.g., '14' from '14/214.mp3')
            xx = base.split("/")[0]

            # Load the audio file
            try:
                with open(audio_path, "rb") as af:
                    audio_data = af.read()
            except FileNotFoundError:
                print(f"Missing: {audio_path}")
                continue

            # Create metadata dictionary
            metadata = {
                "track_id": track_id,
                "artist_id": artist_id,
                "album_id": album_id,
                "duration": float(duration),
                "tags": tags,
            }

            # Build WebDataset sample
            sample = {
                "__key__": track_id,
                "mp3": audio_data,
                "json": json.dumps(metadata),
            }

            # Write sample to appropriate shard
            writer = get_writer(writers, output_dir, xx)
            writer.write(sample)

            # Track counts
            count += 1
            shard_counts[f"{xx}.tar"] += 1
            
            if count % 1000 == 0:
                print(f"[Progress] {count} samples written...")

    # Close all writers
    for writer in writers.values():
        if writer:
            writer.close()

    # Save shard sizes
    sizes_path = os.path.join(output_dir, "sizes.json")
    with open(sizes_path, "w", encoding="utf-8") as f:
        json.dump(shard_counts, f, indent=4)

    print(f"[Done] Total samples written: {count}")


def main():
    audio_root = "../../../../../../gpfs/projects/mtg/audio/stable/mtg-jamendo/raw_30s/audio-low/"  # Root folder of audio files
    data_dir = "../../data/mtg_jamendo/split-0"  # Directory for the dataset
    splits = ["train", "validation"]  # Splits for the dataset

    for split in splits:
        tsv_file = f"{data_dir}/autotagging-{split}.tsv"  # Dataset tsv file
        output_dir = f"{data_dir}/{split}"  # Output directory for the webdataset

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Create the webdataset for the given split
        create_webdataset(tsv_file, audio_root, output_dir)


if __name__=="__main__":
    main()
