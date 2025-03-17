# MagnaTagATune Dataset Directory Structure

This directory should contain the MagnaTagATune dataset (run the [`scripts/data/download_mtt.sh`](../../scripts/data/download_mtt.sh) script to download it), organized as follows:

```
mtt/
├── 0/
│   ├── file_00000.mp3
│   ├── file_00001.mp3
│   └── ...
├── 1/
│   ├── file_00002.mp3
│   ├── file_00003.mp3
│   └── ...
├── 2/
│   ├── file_00004.mp3
│   ├── file_00005.mp3
│   └── ...
...
├── a/
│   ├── file_00010.mp3
│   ├── file_00011.mp3
│   └── ...
├── f/
│   ├── file_00100.mp3
│   ├── file_00101.mp3
│   └── ...
└── MTAT_split/
    ├── top50_tags.txt
    ├── train_list_pub.cP
    ├── valid_list_pub.cP
    ├── test_list_pub.cP
    ├── y_train_pub.npy
    ├── y_valid_pub.npy
    └── y_test_pub.npy
```

The files `/6/norine_braun-now_and_zen-08-gently-117-146.mp3`, `/8/jacob_heringman-josquin_des_prez_lute_settings-19-gintzler__pater_noster-204-233.mp3`, and `/9/american_baroque-dances_and_suites_of_rameau_and_couperin-25-le_petit_rien_xiveme_ordre_couperin-88-117.mp3` have been removed from the dataset as they are corrupted.

The `MTAT_split` folder contains predefined splits (from https://github.com/jongpillee/music_dataset_split/tree/master/MTAT_split) for the MagnaTagATune dataset, including only tracks with the 50 most frequently occurring tags.
