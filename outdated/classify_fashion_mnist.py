import os
import gzip
import numpy as np
import time
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.neural_network import MLPClassifier
from outdated.handle_sift import *

def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def load_mnist_data():
    X_train_data, y_train_data = load_mnist('data/fashion-mnist', kind='train')
    X_test_data, y_test_data = load_mnist('data/fashion-mnist', kind='t10k')

    return X_train_data, y_train_data, X_test_data, y_test_data


def load_classifiers():
    names = [
        "Random_Forest_md5_ne10",
        "Random_Forest_md10_ne100",
        "Random_Forest_md20_ne50",
        "Random_Forest_md20_ne100",
        "MLPClassifier_relu_constant_5_2",
        "MLPClassifier_relu_constant_50_50_50",
        "MLPClassifier_relu_constant_50_100_50",
        "MLPClassifier_relu_constant_100",
        "MLPClassifier_tanh_adaptive_5_2",
        "MLPClassifier_tanh_adaptive_50_50_50",
        "MLPClassifier_tanh_adaptive_50_100_50",
        "MLPClassifier_tanh_adaptive_100",
        "Decision_tree_md3",
        "Decision_tree_md15",
        "Decision_tree_md35",
        "Decision_tree_md80",
        "Naive_Bayes",
        "QDA",
        "Linear_SVM_lin_0_025",
        "Linear_SVM_lin_0_5",
        "Linear_SVM_lin_1",
        "Linear_SVM_gamma_2_C_0_025",
        "Linear_SVM_gamma_2_C_0_5",
        "Linear_SVM_gamma_2_C_1"
    ]

    classifier = [
        RandomForestClassifier(max_depth=5, n_estimators=10, n_jobs=4),
        RandomForestClassifier(max_depth=10, n_estimators=100, n_jobs=4),
        RandomForestClassifier(max_depth=20, n_estimators=50, n_jobs=4),
        RandomForestClassifier(max_depth=20, n_estimators=100, n_jobs=4),
        MLPClassifier(solver='adam', alpha=1e-5, activation = "relu", learning_rate= "constant", hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000),
        MLPClassifier(solver='adam', alpha=1e-5, activation = "relu", learning_rate= "constant", hidden_layer_sizes=(50,50,50), random_state=1, max_iter=1000),
        MLPClassifier(solver='adam', alpha=1e-5, activation = "relu", learning_rate= "constant", hidden_layer_sizes=(50,100,50), random_state=1, max_iter=1000),
        MLPClassifier(solver='adam', alpha=1e-5, activation = "relu", learning_rate= "constant", hidden_layer_sizes=(100,), random_state=1, max_iter=1000),
        MLPClassifier(solver='adam', alpha=1e-5, activation = "tanh", learning_rate= "adaptive", hidden_layer_sizes=(5, 2), random_state=1, max_iter=1000),
        MLPClassifier(solver='adam', alpha=1e-5, activation = "tanh", learning_rate= "adaptive", hidden_layer_sizes=(50, 50, 50), random_state=1, max_iter=1000),
        MLPClassifier(solver='adam', alpha=1e-5, activation = "tanh", learning_rate= "adaptive", hidden_layer_sizes=(50, 100, 50), random_state=1, max_iter=1000),
        MLPClassifier(solver='adam', alpha=1e-5, activation = "tanh", learning_rate= "adaptive", hidden_layer_sizes=(100,), random_state=1, max_iter=1000),
        DecisionTreeClassifier(max_depth=3),
        DecisionTreeClassifier(max_depth=15),
        DecisionTreeClassifier(max_depth=35),
        DecisionTreeClassifier(max_depth=80),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        SVC(kernel="linear", C=0.025),
        SVC(kernel="linear", C=0.5),
        SVC(kernel="linear", C=1),
        SVC(gamma=2, C=0.025),
        SVC(gamma=2, C=0.5),
        SVC(gamma=2, C=1)

    ]

    return names, classifier

def compute_histogram(data):
    #return pd.DataFrame([np.histogram(row, bins=256, range=(0, 255))[0] for row in data])
    return np.array([np.histogram(row, bins=256, range=(0, 255))[0] for row in data])

def train_classifier(classifier, X_train, y_train, X_test):
    classifier.fit(X_train, y_train)
    return classifier

def make_prediction(classifier, X_test):
    predicted = classifier.predict(X_test)
    return predicted

def store_metrics(predicted, y_test, name):
    if not os.path.exists("results/" + name):
        os.mkdir("results/" + name)

    t = time.localtime()
    current_time = time.strftime("_%H_%M_", t)


    disp = ConfusionMatrixDisplay.from_predictions(y_test, predicted)
    disp.figure_.suptitle("Confusion Matrix: " + name)
    np.savetxt("results/" + name + "/CM_" + name + "predicted.txt", disp.confusion_matrix.astype(int), fmt="%i")
    plt.savefig("results/" + name + "/CM_" + name + "predicted.png")
    plt.close()
    np.savetxt("results/" + name + "/" + name + "predicted.txt", predicted.astype(int), fmt="%i")


if __name__ == "__main__":
    method = "SIFT"
    n_words = 100
    random_state = 42
    pixels = 28
    n_jobs = 6

    print("Loading Classifiers.\n")
    names, classifiers = load_classifiers()
    print("Loading mnist-data.\n")
    X_train_data, y_train_data, X_test_data, y_test_data = load_mnist_data()
    print("Transform data using histograms.\n")

    if method == "SIFT":
        X_train, X_test, y_train, y_test = transform_w_sift(n_words=n_words, random_state=random_state,
                                                            n_jobs=n_jobs, pixels=pixels, X_train_data=X_train_data,
                                                            y_train_data=y_train_data, X_test_data=X_test_data,
                                                            y_test_data=y_test_data)
    else:
        X_train, X_test, y_train, y_test = transform_w_histograms(X_train_data, y_train_data, X_test_data, y_test_data)

    print("Training Classifiers.\n")
    for name, classifier in zip(names, classifiers):
        print("Training " + name + ".\n")
        classifier = train_classifier(classifier, X_train, y_train, X_test)
        print("Making predictions.\n")
        predicted = make_prediction(classifier, X_test)
        print("Storing metrics.\n")
        store_metrics(predicted, y_test, name)









#
# print(
#     f"Classification report for classifier {classifier}:\n"
#     f"{metrics.classification_report(y_test, predicted)}\n"
# )
#
# np.loadtxt("results/"+"Random_Forest"+"_predicted.txt").astype(int)
