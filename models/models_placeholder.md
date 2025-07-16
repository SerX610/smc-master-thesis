# Models Directory Structure

This directory should contain the models checkpoints for this project:

```
laion-clap/
├── laion-clap/
│   ├── HTSAT-pretrained.ckpt
│   ├── music_audioset_epoch_15_esc_90.14.pt
│   ├── ...
│   └── ...
└── trained/
    ├── ...
    └── ...
```

The `laion-clap/` folder contains an HTSAT pretrained checkpoint with initialized weights, which can be downloaded using the [`download_htsat_pretrained_ckpt.sh`](../scripts/models/download_htsat_pretrained_ckpt.sh) script, and the LAION-AI CLAP model for music, which can be downloaded using the [`download_laion_clap_ckpt.sh`](../scripts/models/download_laion_clap_ckpt.sh) script. The CLAP models that you train will be stored in this folder.

The `trained/` folder will be used to store models that you train for downstream tasks, such as music classifiers.
