from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split

def fetch_lfw():
    #Following part is based on the import to be found in
    #https://github.com/emanuelfakh/Face-Recognition/blob/master/FR_Final.ipynb
    lfw_people = fetch_lfw_people(resize=0.315, color=False, min_faces_per_person=20, slice_=(slice(48, 202), slice(48, 202)))
    X = lfw_people.images
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    #splitting X and y into train and test
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.33, random_state=42)
    #turning labelNames to a list equivalent to Fashion MNIST for consistency
    labelNames = list(target_names)
    return trainX, trainY, testX, testY, labelNames
