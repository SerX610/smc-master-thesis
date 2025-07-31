# Master's Thesis in Sound and Music Computing - Sergio Cárdenas Gracia

This repository contains the code and resources for the Master's Thesis in Sound and Music Computing by Sergio Cárdenas Gracia, focusing on the **Comparison of Audio Encoders for Audio-Text Contrastive Learning Representations**.

The goal of this thesis is to systematically compare state-of-the-art audio encoders (e.g., MAEST, HTSAT) for audio-text contrastive learning scenarios (CLAP). The project explores how different architectures and training strategies affect the quality of learned representations for downstream tasks such as zero-shot classification, multi-label classification, and text-to-music retrieval. For more information, check the project report (TODO: add link to PDF)

## Repository Structure

```
smc-master-thesis/
├── laion_clap/                # CLAP model implementation, training, and evaluation utilities
├── models/                    # Pretrained and trained model checkpoints
├── data/                      # Datasets (GTZAN, MagnaTagATune, SDD, MTG-Jamendo)
├── results/                   # Evaluation results, plots, and metrics
├── scripts/                   # Data processing, baseline, and training scripts
│   ├── baseline/              # Baseline scripts for embedding extraction and classifier training/testing
│   ├── data/                  # Data conversion and webdataset creation scripts
│   └── models/                # Training scripts for different encoder architectures
├── src/                       # Utility modules and evaluation scripts
│   ├── evaluation/            # Evaluation scripts for downstream tasks
│   └── utils/                 # General utilities and dataset loaders
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── LICENSE                    # License information
```

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SerX610/smc-master-thesis.git
   cd smc-master-thesis
   ```

2. **Install dependencies:**
   ```bash
   conda create -n env python=3.11
   conda activate env
   pip install -r requirements.txt
   ```
   

## Usage


- **Datasets:**
  - Use scripts in `scripts/data/` to download the datasets.
  - Example:
    ```bash
    chmod +x scripts/data/download_gtzan.sh
    ./scripts/data/download_gtzan.sh
    ```

- **Training:**
  - Use scripts in `scripts/models/` to train models with different encoders and configurations.
  - Example:
    ```bash
    chmod +x scripts/models/train_maest.sh
    ./scripts/models/train_maest.sh
    ```
  - Alternatively, you can download a pretrained checkpoint.
  - Example:
    ```bash
    chmod +x scripts/models/download_laion_clap_ckpt.sh
    ./scripts/models/download_laion_clap_ckpt.sh
    ```

- **Evaluation:**
  - Run evaluation scripts in `src/evaluation/` for zero-shot classification, multi-label classification, and text-to-music retrieval.
  - Example:
    ```bash
    chmod +x scripts/evaluation/evaluate_model.sh
    bash scripts/evaluation/evaluate_model.sh
    ```


## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.
