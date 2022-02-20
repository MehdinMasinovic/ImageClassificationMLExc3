print("CNN MiniGoogLeNet Script")

# import the necessary packages
from minigooglenet import MiniGoogLeNet
from minivggnet import MiniVGGNet
from sklearn.metrics import classification_report
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import fashion_mnist
from keras.preprocessing.image import ImageDataGenerator
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from imutils import build_montages
import matplotlib.pyplot as plt
import numpy as np
import cv2
from datetime import datetime
import os

#where the images will be saved. If path doesn't exist, will fail, so creating it if doesn't exist here
path = "cnn_images"
isExist = os.path.exists(path)
if not isExist:
  os.makedirs(path)

np.random.seed(1) # we initialize a random seed here to make the experiments repeatable with same results

#initialize GPU functionality if available

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

#MAIN FUNCTIONS

# Fetch Fashion MNIST Dataset
def fetch_fashion_mnist():
    #loading dataset
    ((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
    
    # initialize the label names
    labelNames = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    return trainX, trainY, testX, testY, labelNames

def fetch_and_resize_fashion_mnist():
    #loading dataset
    ((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
    
    # initialize the label names
    labelNames = ["top", "trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    trainX = np.array([list(cv2.resize(i, dsize=(36, 36), interpolation=cv2.INTER_CUBIC)) for i in trainX])
    testX = np.array([list(cv2.resize(i, dsize=(36, 36), interpolation=cv2.INTER_CUBIC)) for i in testX])
    return trainX, trainY, testX, testY, labelNames

# Fetch LFW People Dataset
def fetch_lfw():
    #Following part is based on the import to be found in 
    #https://github.com/emanuelfakh/Face-Recognition/blob/master/FR_Final.ipynb
    lfw_people = fetch_lfw_people(resize=0.315, color=True, min_faces_per_person=20,
                              slice_=(slice(48, 202), slice(48, 202)))
    X = lfw_people.images
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]
    #splitting X and y into train and test
    trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.33, random_state=42)
    #turning labelNames to a list equivalent to Fashion MNIST for consistency
    labelNames = list(target_names)
    return trainX, trainY, testX, testY, labelNames

# reshape data
def reshape_data(trainX, testX):
    shape = trainX.shape
    if len(shape) == 3:
        _, h, w = shape
        d = 1
    elif len(shape) == 4:
        _, h, w, d = shape
    # if we are using "channels first" ordering, then reshape the design
    # matrix such that the matrix is:
    # 	num_samples x depth x rows x columns
    if K.image_data_format() == "channels_first":
        trainX = trainX.reshape((trainX.shape[0], d, h, w))
        testX = testX.reshape((testX.shape[0], d, h, w))
    # otherwise, we are using "channels last" ordering, so the design
    # matrix shape should be: num_samples x rows x columns x depth
    else:
        trainX = trainX.reshape((trainX.shape[0], h, w, d))
        testX = testX.reshape((testX.shape[0], h, w, d))
    return trainX, testX

# scale data to the range of [0, 1]
def scale_data(trainX, testX):
    trainX = trainX.astype("float32") / 255.0
    testX = testX.astype("float32") / 255.0
    return trainX, testX

# one-hot encode the training and testing labels
def onehot_data(trainY, testY, labelNames):
    len_lN = len(labelNames)
    trainY = to_categorical(trainY, len_lN)
    testY = to_categorical(testY, len_lN)
    return trainY, testY

def init_model(trainX, epochs, learning_rate, labelNames, architecture):
    print("[INFO] training model...")
    shape = trainX.shape
    if len(shape) == 3:
        _, h, w = shape
        d = 1
    elif len(shape) == 4:
        _, h, w, d = shape
    # initialize model and normalizer
    opt = SGD(learning_rate=learning_rate, momentum=0.9, decay=learning_rate / epochs)
    model = architecture.build(width=w, height=h, depth=d, classes=len(labelNames))
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model

def train_model(trainX, testX, trainY, testY, batch_size, epochs, model):
    # training the network
    start = datetime.now()
    H = model.fit(x=trainX, y=trainY, validation_data=(testX, testY), batch_size=batch_size, epochs=epochs, verbose=0)
    end = datetime.now()
    print("total seconds for training: " + str((end-start).total_seconds()))
    return model, H

# make predictions on the test set
def pred_test(model, testX):
    print("[INFO] testing model...")
    start = datetime.now()
    preds = model.predict(testX)
    end = datetime.now()
    print("total seconds for testing: " + str((end-start).total_seconds()))
    return preds

# show a nicely formatted classification report
def evaluate_model(preds, testY, labelNames):
    print("[INFO] evaluating network...")
    print(classification_report(testY.argmax(axis=1), preds.argmax(axis=1), target_names=labelNames))

# plot the training accuracy
def plot_acc(epochs, H, naming):
    N = epochs
    plt.style.use("ggplot")
    plt.figure()
    #plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    #plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Accuracy on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("cnn_images/"+naming+"_accuracycurve.png")

# plot the training loss
def plot_loss(epochs, H, naming):
    N = epochs
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    #plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
    #plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
    plt.title("Training Loss on Dataset")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig("cnn_images/"+naming+"_losscurve.png")
    
def augment_data_all():
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    return datagen

def augment_data_reduced():
    datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True)
    return datagen

def train_augmented_model(trainX, testX, trainY, testY, batch_size, epochs, model, datagen):
    # training the network
    start = datetime.now()
    H = model.fit(datagen.flow(trainX, trainY, batch_size=batch_size), validation_data=(testX, testY), batch_size=batch_size, epochs=epochs, verbose=0)
    end = datetime.now()
    print("total seconds for training: " + str((end-start).total_seconds()))
    return model, H

def create_montage(testX, testY, model, labelNames, naming):
    _, _, _, d = testX.shape
    # initialize our list of output images
    images = []
    # randomly select a few testing fashion items
    for i in np.random.choice(np.arange(0, len(testY)), size=(16,)):
        # classify the clothing
        probs = model.predict(testX[np.newaxis, i])
        prediction = probs.argmax(axis=1)
        label = labelNames[prediction[0]]

        # extract the image from the testData if using "channels_first"
        # ordering
        if K.image_data_format() == "channels_first":
            image = (testX[i][0] * 255).astype("uint8")

        # otherwise we are using "channels_last" ordering
        else:
            image = (testX[i] * 255).astype("uint8")
            # initialize the text label color as green (correct)
            color = (0, 255, 0)

            # otherwise, the class label prediction is incorrect
            if prediction[0] != np.argmax(testY[i]):
                color = (0, 0, 255)

            # merge the channels into one image and resize the image from
            # 28x28 to 96x96 so we can better see it and then draw the
            # predicted label on the image
            if d == 1:
                image = cv2.merge([image] * 3)
            image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
            cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                        color, 2)

            # add the image to our list of output images
            images.append(image)
    # construct the montage for the images
    montage = build_montages(images, (96, 96), (4, 4))[0]
    # show the output montage
    fig, ax = plt.subplots()
    plt.imshow(montage)
    ax.grid(False)
    _ = plt.axis('off')
    plt.savefig("cnn_images/"+naming+"_montage.png")

def plot_confusion_matrix(testY, preds, labelNames, naming):
    fig, ax = plt.subplots(figsize = (20,20))
    cm=confusion_matrix(tf.argmax(testY, axis = 1),tf.argmax(preds, axis = 1))
    ax.set_title("Confusion Matrix")
    _ = sns.heatmap(cm, annot=True, fmt='d', cbar=False, xticklabels = labelNames, yticklabels= labelNames)
    _ = plt.xlabel("Predicted")
    _ = plt.ylabel("Actual")
    plt.savefig("cnn_images/"+naming+"_confusion.png")

#WRAPPER FUNCTIONS

def wrapper_f(epochs, learning_rate, batch_size, fetcher, augment, architecture, naming):
    epochs = epochs
    learning_rate = learning_rate
    batch_size = batch_size
    trainX, trainY, testX, testY, labelNames = fetcher()
    trainX, testX = reshape_data(trainX, testX)
    trainX, testX = scale_data(trainX, testX)
    trainY, testY = onehot_data(trainY, testY, labelNames)
    model = init_model(trainX, epochs, learning_rate, labelNames, architecture)
    if augment == "full":
        datagen = augment_data_all()
        model, H = train_augmented_model(trainX, testX, trainY, testY, batch_size, epochs, model, datagen)
    elif augment == "reduced":
        
        datagen = augment_data_reduced()
        model, H = train_augmented_model(trainX, testX, trainY, testY, batch_size, epochs, model, datagen)
    else:
        model, H = train_model(trainX, testX, trainY, testY, batch_size, epochs, model)
    preds = pred_test(model, testX)
    evaluate_model(preds, testY, labelNames)
    plot_acc(epochs, H, naming)
    plot_loss(epochs, H, naming)
    create_montage(testX, testY, model, labelNames,naming)
    plot_confusion_matrix(testY, preds, labelNames,naming)

print("\nMiniGoogLeNet")
print("Fashion MNIST")
print("LR: 1e-4")
print("No augmentation")

naming = "goog_fmnist_noaug_lr_1e-4"

wrapper_f(fetcher       = fetch_and_resize_fashion_mnist,
          epochs        = 5,
          learning_rate = 1e-4,
          batch_size    = 32,
          augment       = "none",
          architecture  = MiniGoogLeNet,
          naming        = naming)

print("Accuracy curve, loss curve, montage and confusion matrix images saved in cnn_images/" + naming + "*")

print("\nMiniGoogLeNet")
print("Fashion MNIST")
print("LR: 1e-3")
print("No augmentation")

naming = "goog_fmnist_noaug_lr_1e-3"

wrapper_f(fetcher       = fetch_and_resize_fashion_mnist,
          epochs        = 5,
          learning_rate = 1e-3,
          batch_size    = 32,
          augment       = "none",
          architecture  = MiniGoogLeNet,
          naming        = naming)

print("Accuracy curve, loss curve, montage and confusion matrix images saved in cnn_images/" + naming + "*")

print("\nMiniGoogLeNet")
print("Fashion MNIST")
print("LR: 1e-2")
print("No augmentation")

naming = "goog_fmnist_noaug_lr_1e-2"

wrapper_f(fetcher       = fetch_and_resize_fashion_mnist,
          epochs        = 5,
          learning_rate = 1e-2,
          batch_size    = 32,
          augment       = "none",
          architecture  = MiniGoogLeNet,
          naming        = naming)

print("Accuracy curve, loss curve, montage and confusion matrix images saved in cnn_images/" + naming + "*")

print("\nMiniGoogLeNet")
print("Fashion MNIST")
print("LR: 1e-1")
print("No augmentation")

naming = "goog_fmnist_noaug_lr_1e-1"

wrapper_f(fetcher       = fetch_and_resize_fashion_mnist,
          epochs        = 5,
          learning_rate = 1e-1,
          batch_size    = 32,
          augment       = "none",
          architecture  = MiniGoogLeNet,
          naming        = naming)

print("Accuracy curve, loss curve, montage and confusion matrix images saved in cnn_images/" + naming + "*")


print("\nMiniGoogLeNet")
print("Fashion MNIST")
print("LR: 1e-4")
print("With augmentation")

naming = "goog_fmnist_aug_lr_1e-4"

wrapper_f(fetcher       = fetch_and_resize_fashion_mnist,
          epochs        = 5,
          learning_rate = 1e-4,
          batch_size    = 32,
          augment       = "reduced",
          architecture  = MiniGoogLeNet,
          naming        = naming)

print("Accuracy curve, loss curve, montage and confusion matrix images saved in cnn_images/" + naming + "*")

print("\nMiniGoogLeNet")
print("Fashion MNIST")
print("LR: 1e-3")
print("With augmentation")

naming = "goog_fmnist_aug_lr_1e-3"

wrapper_f(fetcher       = fetch_and_resize_fashion_mnist,
          epochs        = 5,
          learning_rate = 1e-3,
          batch_size    = 32,
          augment       = "reduced",
          architecture  = MiniGoogLeNet,
          naming        = naming)

print("Accuracy curve, loss curve, montage and confusion matrix images saved in cnn_images/" + naming + "*")

print("\nMiniGoogLeNet")
print("Fashion MNIST")
print("LR: 1e-2")
print("With augmentation")

naming = "goog_fmnist_aug_lr_1e-2"

wrapper_f(fetcher       = fetch_and_resize_fashion_mnist,
          epochs        = 5,
          learning_rate = 1e-2,
          batch_size    = 32,
          augment       = "reduced",
          architecture  = MiniGoogLeNet,
          naming        = naming)

print("Accuracy curve, loss curve, montage and confusion matrix images saved in cnn_images/" + naming + "*")

print("\nMiniGoogLeNet")
print("Fashion MNIST")
print("LR: 1e-1")
print("With augmentation")

naming = "goog_fmnist_aug_lr_1e-1"

wrapper_f(fetcher       = fetch_and_resize_fashion_mnist,
          epochs        = 5,
          learning_rate = 1e-1,
          batch_size    = 32,
          augment       = "reduced",
          architecture  = MiniGoogLeNet,
          naming        = naming)

print("Accuracy curve, loss curve, montage and confusion matrix images saved in cnn_images/" + naming + "*")


print("\nMiniGoogLeNet")
print("LFW People")
print("LR: 1e-4")
print("No augmentation")


naming = "goog_lfw_noaug_lr_1e-4"

wrapper_f(fetcher       = fetch_lfw,
          epochs        = 75,
          learning_rate = 1e-4,
          batch_size    = 32,
          augment       = "none",
          architecture  = MiniGoogLeNet,
          naming        = naming)

print("Accuracy curve, loss curve, montage and confusion matrix images saved in cnn_images/" + naming + "*")

print("\nMiniGoogLeNet")
print("LFW People")
print("LR: 1e-3")
print("No augmentation")

naming = "goog_lfw_noaug_lr_1e-3"

wrapper_f(fetcher       = fetch_lfw,
          epochs        = 75,
          learning_rate = 1e-3,
          batch_size    = 32,
          augment       = "none",
          architecture  = MiniGoogLeNet,
          naming        = naming)

print("Accuracy curve, loss curve, montage and confusion matrix images saved in cnn_images/" + naming + "*")

print("\nMiniGoogLeNet")
print("LFW People")
print("LR: 1e-2")
print("No augmentation")

naming = "goog_lfw_noaug_lr_1e-2"

wrapper_f(fetcher       = fetch_lfw,
          epochs        = 75,
          learning_rate = 1e-2,
          batch_size    = 32,
          augment       = "none",
          architecture  = MiniGoogLeNet,
          naming        = naming)

print("Accuracy curve, loss curve, montage and confusion matrix images saved in cnn_images/" + naming + "*")

print("\nMiniGoogLeNet")
print("LFW People")
print("LR: 1e-1")
print("No augmentation")

naming = "goog_lfw_noaug_lr_1e-1"

wrapper_f(fetcher       = fetch_lfw,
          epochs        = 75,
          learning_rate = 1e-1,
          batch_size    = 32,
          augment       = "none",
          architecture  = MiniGoogLeNet,
          naming        = naming)

print("Accuracy curve, loss curve, montage and confusion matrix images saved in cnn_images/" + naming + "*")


print("\nMiniGoogLeNet")
print("LFW People")
print("LR: 1e-4")
print("With augmentation")

naming = "goog_lfw_aug_lr_1e-4"

wrapper_f(fetcher       = fetch_lfw,
          epochs        = 75,
          learning_rate = 1e-4,
          batch_size    = 32,
          augment       = "full",
          architecture  = MiniGoogLeNet,
          naming        = naming)

print("Accuracy curve, loss curve, montage and confusion matrix images saved in cnn_images/" + naming + "*")

print("\nMiniGoogLeNet")
print("LFW People")
print("LR: 1e-3")
print("With augmentation")

naming = "goog_lfw_aug_lr_1e-3"

wrapper_f(fetcher       = fetch_lfw,
          epochs        = 75,
          learning_rate = 1e-3,
          batch_size    = 32,
          augment       = "full",
          architecture  = MiniGoogLeNet,
          naming        = naming)

print("Accuracy curve, loss curve, montage and confusion matrix images saved in cnn_images/" + naming + "*")

print("\nMiniGoogLeNet")
print("LFW People")
print("LR: 1e-2")
print("With augmentation")

naming = "goog_lfw_aug_lr_1e-2"

wrapper_f(fetcher       = fetch_lfw,
          epochs        = 75,
          learning_rate = 1e-2,
          batch_size    = 32,
          augment       = "full",
          architecture  = MiniGoogLeNet,
          naming        = naming)

print("Accuracy curve, loss curve, montage and confusion matrix images saved in cnn_images/" + naming + "*")

print("\nMiniGoogLeNet")
print("LFW People")
print("LR: 1e-1")
print("With augmentation")

naming = "goog_lfw_aug_lr_1e-1"

wrapper_f(fetcher       = fetch_lfw,
          epochs        = 75,
          learning_rate = 1e-1,
          batch_size    = 32,
          augment       = "full",
          architecture  = MiniGoogLeNet,
          naming        = naming)

print("Accuracy curve, loss curve, montage and confusion matrix images saved in cnn_images/" + naming + "*")
