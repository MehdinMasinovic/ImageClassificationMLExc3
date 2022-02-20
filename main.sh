#!/bin/sh
# Traditional methods:
# All models: Fashion MNIST
python main_ml_exc3.py --path_to_config configs/AllModels.json --data "clothes" --extract_feature_with "SIFT" --random_state 42 --n_jobs 6 --n_words 100 --pixels 28
python main_ml_exc3.py --path_to_config configs/AllModels.json --data "clothes" --extract_feature_with "CH" --random_state 42 --n_jobs 6 --pixels 28

# All models: Labelled Faces in the Wild
python main_ml_exc3.py --path_to_config configs/AllModels.json --data "faces" --extract_feature_with "SIFT" --random_state 42 --n_jobs 6 --n_words 500 --pixels 154
python main_ml_exc3.py --path_to_config configs/AllModels.json --data "faces" --extract_feature_with "CH" --random_state 42 --n_jobs 6 --pixels 154




