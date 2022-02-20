# Import libraries
import argparse
import json
from timeit import default_timer as dt

from utils.mnist_utils import *
from utils.faces_utils import *
from utils.feature_extraction_utils import *
from utils.classification_utils import *


def change_jsontypes(configuration):
    for key, value in configuration.items():
        if value == "True":
            configuration[key] = True
            continue
        elif value == "False":
            configuration[key] = False
            continue
        elif value == "None":
            configuration[key] = None
            continue
        else:
            continue

    return configuration

def load_data(data):
    if data == "clothes":
        X_train_data, y_train_data, X_test_data, y_test_data = load_mnist_data()
        return X_train_data, y_train_data, X_test_data, y_test_data
    elif data == "faces":
        X_train_data, y_train_data, X_test_data, y_test_data, labelNames = fetch_lfw()
        return X_train_data, y_train_data, X_test_data, y_test_data
    else:
        print("Preprocessing for dataset " + data + " is not implemented. Please consider changing your command line option to 'clothes' or 'faces'.")
        print("Exiting program")
        exit()

def pipeline(config_file=None, data="clothes", extract_feature_with="SIFT", random_state=42, n_jobs=1, n_words=100, pixels=28):
    print("Loading data for the " + data + " dataset")
    X_train_data, y_train_data, X_test_data, y_test_data = load_data(data)
    print("Extracting features using method: " + extract_feature_with)
    start = dt()
    X_train, X_test, y_train, y_test = extract_features(extract_feature_with=extract_feature_with, n_words=n_words,
                                                        random_state=random_state, n_jobs=n_jobs, pixels=pixels,
                                                        X_train_data=X_train_data, y_train_data=y_train_data,
                                                        X_test_data=X_test_data, y_test_data=y_test_data)
    end = dt()
    extractFeatures = end-start

    for experiment_setup in config_file:
        print("Loading configurations for: ", experiment_setup)
        configuration = change_jsontypes(config_file[experiment_setup])
        print("Retrieving classifier: " + configuration['name'])
        classifier, name = retrieve_classifier(configuration)
        try:
            print("Training the classifier")
            start = dt()
            classifier = train_classifier(classifier, X_train, y_train, X_test)
            end = dt()
            trainClassifier = end-start
            print("Making predictions")
            start = dt()
            predicted = make_prediction(classifier, X_test)
            end = dt()
            makePrediction = end-start
            print("Storing metrics")
            store_metrics(predicted, y_test, data, extract_feature_with, experiment_setup, name, extractFeatures, trainClassifier, makePrediction)
        except Exception as e:
            print("Classification failed using: " + configuration['name'])
            print("Reason: ")
            print(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_config", default="configs/AllModels.json")
    parser.add_argument("--data", default="clothes")
    parser.add_argument("--extract_feature_with", default="SIFT")
    parser.add_argument("--random_state", default=42)
    parser.add_argument("--n_jobs", default=1)
    parser.add_argument("--n_words", default=100)
    parser.add_argument("--pixels", default=28)
    args = parser.parse_args()

    with open(args.path_to_config) as config_file:
        config_file = json.load(config_file)

    for experiment_setup in config_file:

        pipeline(
            config_file=config_file,
            data=args.data,
            extract_feature_with=args.extract_feature_with,
            random_state=int(args.random_state),
            n_jobs=int(args.n_jobs),
            n_words=int(args.n_words),
            pixels=int(args.pixels)
        )
