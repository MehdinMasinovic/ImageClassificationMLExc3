#!/bin/sh

# Random Forrest
python main_ml_exc3.py --path_to_config configs/RandomForrestClassifier.json --data "faces" --extract_feature_with "SIFT" --random_state 42 --n_jobs 6 --n_words 100 --pixels 48
python main_ml_exc3.py --path_to_config configs/RandomForrestClassifier.json --data "faces" --extract_feature_with "CH" --random_state 42 --n_jobs 6 --n_words 100 --pixels 48

# Decision Tree
python main_ml_exc3.py --path_to_config configs/DecisionTreeClassifier.json --data "faces" --extract_feature_with "SIFT" --random_state 42 --n_jobs 6 --n_words 100 --pixels 48
python main_ml_exc3.py --path_to_config configs/DecisionTreeClassifier.json --data "faces" --extract_feature_with "CH" --random_state 42 --n_jobs 6 --n_words 100 --pixels 48

# MLP
python main_ml_exc3.py --path_to_config configs/MLPClassifier.json --data "faces" --extract_feature_with "SIFT" --random_state 42 --n_jobs 6 --n_words 100 --pixels 48
python main_ml_exc3.py --path_to_config configs/MLPClassifier.json --data "faces" --extract_feature_with "CH" --random_state 42 --n_jobs 6 --n_words 100 --pixels 48

# SVC
python main_ml_exc3.py --path_to_config configs/SVC.json --data "faces" --extract_feature_with "SIFT" --random_state 42 --n_jobs 6 --n_words 100 --pixels 48
python main_ml_exc3.py --path_to_config configs/SVC.json --data "faces" --extract_feature_with "CH" --random_state 42 --n_jobs 6 --n_words 100 --pixels 48

# Other Models
python main_ml_exc3.py --path_to_config configs/OtherModels.json --data "faces" --extract_feature_with "SIFT" --random_state 42 --n_jobs 6 --n_words 100 --pixels 48
python main_ml_exc3.py --path_to_config configs/OtherModels.json --data "faces" --extract_feature_with "CH" --random_state 42 --n_jobs 6 --n_words 100 --pixels 48

# All models
#python main_ml_exc3.py --path_to_config configs/AllModels.json --data "faces" --extract_feature_with "SIFT" --random_state 42 --n_jobs 6 --n_words 100 --pixels 48
#python main_ml_exc3.py --path_to_config configs/AllModels.json --data "faces" --extract_feature_with "CH" --random_state 42 --n_jobs 6 --n_words 100 --pixels 48



