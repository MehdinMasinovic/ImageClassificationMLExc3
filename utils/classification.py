import os
import time
import numpy as np
from matplotlib import pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import ConfusionMatrixDisplay

def retrieve_classifier(configuration):
    model_name = configuration['name']

    if model_name == "RandomForrestClassifier":
        classifier = RandomForestClassifier(
            max_depth=configuration['max_depth'],
            n_estimators=configuration['n_estimators'],
            n_jobs=configuration['n_jobs']
        )
        return classifier, model_name
    elif model_name == "MLPClassifier":
        classifier = MLPClassifier(
            solver=configuration["solver"],
            alpha=configuration["alpha"],
            activation=configuration["activation"],
            learning_rate=configuration["learning_rate"],
            hidden_layer_sizes=configuration["hidden_layer_sizes"],
            random_state=configuration["random_state"],
            max_iter=configuration["max_iter"]
        )
        return classifier, model_name
    elif model_name == "DecisionTreeClassifier":
        classifier = DecisionTreeClassifier(
            max_depth=configuration["max_depth"]
        )
        return classifier, model_name
    elif model_name == "GaussianNB":
        classifier = GaussianNB()
        return classifier, model_name
    elif model_name == "QuadraticDiscriminantAnalysis":
        classifier = QuadraticDiscriminantAnalysis()
        return classifier, model_name
    elif model_name == "SVC":
        classifier = SVC(
            kernel=configuration["kernel"],
            gamma=configuration["gamma"],
            C=configuration["C"]
        )
        return classifier, model_name
    else:
        print("Model " + model_name + " not implemented. Please consider editing your configuration.")
        print("Exiting program")
        exit()


def train_classifier(classifier, X_train, y_train, X_test):
    classifier.fit(X_train, y_train)
    return classifier

def make_prediction(classifier, X_test):
    predicted = classifier.predict(X_test)
    return predicted

def store_metrics(predicted, y_test, data, extract_feature_with, experiment_setup, name, extractFeatures=None, trainClassifier=None, makePrediction=None):
    t = time.localtime()
    current_time = time.strftime("T%H_%M_", t)

    if not os.path.exists("results/" + current_time + data + "_" + extract_feature_with + "_" + name):
        os.mkdir("results/" + current_time + data + "_" + extract_feature_with + "_" + name)

    # Save confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix: " + data + "_" + name)
    np.savetxt("results/" + current_time + data + "_" + extract_feature_with + "_" + name + "/CM_" + data + "_" + extract_feature_with + "_" + experiment_setup + "_predicted.txt",
               disp.confusion_matrix.astype(int), fmt="%i")
    # Plot confusion matrix
    plt.savefig("results/" + current_time + data + "_" + extract_feature_with + "_" + name + "/CM_" + data + "_" + extract_feature_with + "_" + experiment_setup + "_predicted.png")
    plt.close()
    # Save predictions
    np.savetxt("results/" + current_time + data + "_" + extract_feature_with + "_" + name + "/" + data + "_" + extract_feature_with + "_" + experiment_setup + "_predicted.txt",
               predicted.astype(int), fmt="%i")
    # Save timing
    np.savetxt("results/" + current_time + data + "_" + extract_feature_with + "_" + name + "/" + data + "_" + extract_feature_with + "_" + experiment_setup + "_timing.txt",
               np.array([extractFeatures, trainClassifier, makePrediction]).astype(int), fmt="%i")