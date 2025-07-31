cd "$(dirname "$0")/../.."

python -m src.evaluation.gtzan_zero_shot_classification
python -m src.evaluation.mtt_multi_label_classification
python -m src.evaluation.sdd_text_to_music_retrieval