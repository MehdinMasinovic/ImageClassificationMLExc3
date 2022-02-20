#!/bin/sh

# Random Forrest
python main_ml_exc3.py --path_to_config configs/RandomForrestClassifier.json --data "clothes" --extract_feature_with "SIFT" --random_state 42 --n_jobs 6 --n_words 100 --pixels 28
python main_ml_exc3.py --path_to_config configs/RandomForrestClassifier.json --data "clothes" --extract_feature_with "CH" --random_state 42 --n_jobs 6 --n_words 100 --pixels 28

# Decision Trees
python main_ml_exc3.py --path_to_config configs/DecisionTreeClassifier.json --data "clothes" --extract_feature_with "SIFT" --random_state 42 --n_jobs 6 --n_words 100 --pixels 28
python main_ml_exc3.py --path_to_config configs/DecisionTreeClassifier.json --data "clothes" --extract_feature_with "CH" --random_state 42 --n_jobs 6 --n_words 100 --pixels 28

# MLP
python main_ml_exc3.py --path_to_config configs/MLPClassifier.json --data "clothes" --extract_feature_with "SIFT" --random_state 42 --n_jobs 6 --n_words 100 --pixels 28
python main_ml_exc3.py --path_to_config configs/MLPClassifier.json --data "clothes" --extract_feature_with "CH" --random_state 42 --n_jobs 6 --n_words 100 --pixels 28

# SVC
python main_ml_exc3.py --path_to_config configs/SVC.json --data "clothes" --extract_feature_with "SIFT" --random_state 42 --n_jobs 6 --n_words 100 --pixels 28
python main_ml_exc3.py --path_to_config configs/SVC.json --data "clothes" --extract_feature_with "CH" --random_state 42 --n_jobs 6 --n_words 100 --pixels 28

# Other Models
python main_ml_exc3.py --path_to_config configs/OtherModels.json --data "clothes" --extract_feature_with "SIFT" --random_state 42 --n_jobs 6 --n_words 100 --pixels 28
python main_ml_exc3.py --path_to_config configs/OtherModels.json --data "clothes" --extract_feature_with "CH" --random_state 42 --n_jobs 6 --n_words 100 --pixels 28

# All models
#python main_ml_exc3.py --path_to_config configs/AllModels.json --data "clothes" --extract_feature_with "SIFT" --random_state 42 --n_jobs 6 --n_words 100 --pixels 28
#python main_ml_exc3.py --path_to_config configs/AllModels.json --data "clothes" --extract_feature_with "CH" --random_state 42 --n_jobs 6 --n_words 100 --pixels 28



