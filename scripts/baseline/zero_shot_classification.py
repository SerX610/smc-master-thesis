"""
This script uses CLAP to perform zero-shot classification on the MTG Jamendo dataset.

The MTG Jamendo dataset can be found at:
    https://github.com/MTG/mtg-jamendo-dataset/tree/master
"""

import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from msclap import CLAP
from mtg_jamendo_dataset import MTGJamendo
from sklearn.metrics import accuracy_score


# Load the MTG Jamendo dataset
dataset = MTGJamendo(...)
prompt = 'this is the sound of '
y = [prompt + x for x in dataset.classes]

# Load and initialize CLAP
clap_model = CLAP(version = '2023', use_cuda=False)

# Computing text embeddings
text_embeddings = clap_model.get_text_embeddings(y)

# Computing audio embeddings
y_preds, y_labels = [], []
for i in tqdm(range(len(dataset))):
    x, _, one_hot_target = dataset.__getitem__(i)
    audio_embeddings = clap_model.get_audio_embeddings([x], resample=True)
    similarity = clap_model.compute_similarity(audio_embeddings, text_embeddings)
    y_pred = F.softmax(similarity.detach().cpu(), dim=1).numpy()
    y_preds.append(y_pred)
    y_labels.append(one_hot_target.detach().cpu().numpy())

# Computing zero-shot classification
y_labels, y_preds = np.concatenate(y_labels, axis=0), np.concatenate(y_preds, axis=0)
acc = accuracy_score(np.argmax(y_labels, axis=1), np.argmax(y_preds, axis=1))
print('ESC50 Accuracy {}'.format(acc))