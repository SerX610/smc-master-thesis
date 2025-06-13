# Song Describer Dataset Directory Structure

This directory should contain the Song Describer Dataset (run the [`scripts/data/download_sdd.sh`](../../scripts/data/download_sdd.sh) script to download it), organized as follows:

```
sdd/
├── audio/
│   ├── 00/
│   |   ├── 18500.2min.mp3
│   |   ├── 133600.2min.mp3
│   |   └── ...
│   ├── 01/
│   |   ├── 12301.2min.mp3
│   |   ├── 246601.2min.mp3
│   |   └── ...
...
│   ├── 98/
│   |   ├── 973498.2min.mp3
│   |   ├── 1051198.2min.mp3
│   |   └── ...
│   └── 99/
│       ├── 238099.2min.mp3
│       ├── 945199.2min.mp3
│       └── ...
└── song_describer.csv
```

The `song_describer.csv` file contains information about the audio tracks and their captions.