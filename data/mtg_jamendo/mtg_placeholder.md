# MTG-Jamendo Dataset Directory Structure

This directory should contain the processsed MTG-Jamendo dataset, organized as follows:

```
mtg_jamendo/
└── split-0/
    ├── train/
    │   ├── 00.tar
    │   ├── 01.tar
    │   ├── ...
    │   ├── ...
    │   ├── 99.tar
    │   └── sizes.json
    ├── validation/
    │   ├── 00.tar
    │   ├── 01.tar
    │   ├── ...
    │   ├── ...
    │   ├── 99.tar
    │   └── sizes.json
    ├── autotagging-train.tsv
    └── autotagging-validation.tsv
```

The `train/` and `validation/` folders contain `.tar` archives with preprocessed audio samples and a `sizes.json` file describing the contents of the dataset shards.

The `autotagging-train.tsv` and `autotagging-validation.tsv` files containing audio metadatat can be downloaded using the [`scripts/data/download_mtg_jamendo_tags.sh`](../../scripts/data/download_mtg_jamendo_tags.sh) script.

Once the tag files are downloaded, generate the dataset in WebDataset format using the [`create_mtg_jamendo_webdataset`](../../scripts/data/create_mtg_jamendo_webdataset.sh) script.

Note that the audio files for MTG-Jamendo split-0 are already available in the MTG project environment. If you are setting this up in a different environment, you must manually download the original audio files corresponding to split-0 before processing the dataset.
