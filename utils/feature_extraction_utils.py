import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from skimage.feature import SIFT

# Color histogram
def compute_histogram(data):
    #return pd.DataFrame([np.histogram(row, bins=256, range=(0, 255))[0] for row in data])
    return np.array([np.histogram(row, bins=256, range=(0, 255))[0] for row in data])

def transform_w_histograms(X_train_data, y_train_data, X_test_data, y_test_data):
    X_train_freqs = compute_histogram(X_train_data)
    X_test_freqs = compute_histogram(X_test_data)
    X_train = X_train_freqs
    y_train = y_train_data
    X_test = X_test_freqs
    y_test = y_test_data

    return X_train, X_test, y_train, y_test

# SIFT
def handle_sift_exceptions(array, sift):
    try:
        sift.detect_and_extract(array)
        descriptor = sift.descriptors
        no_features_extracted = False
        return descriptor, no_features_extracted
    except Exception as e:
        print(e)
        descriptor = None
        no_features_extracted = True
        return descriptor, no_features_extracted

def extract_descriptor(array, sift, index):
    try:
        sift.detect_and_extract(array)
        descriptor = sift.descriptors
        return descriptor
    except Exception as e:
        orig_c_dog = sift.c_dog
        orig_c_edge = sift.c_edge
        orig_n_octaves = sift.n_octaves
        orig_upsampling = sift.upsampling
        no_features_extracted = True
        while(no_features_extracted):
            print("No features could be extracted for image", str(index), ", loosening threshold to discard low contrast and edge extremas.")
            print("Current: ")
            print("c_dog: "+ str(sift.c_dog))
            print("c_edge: "+ str(sift.c_edge))
            print("n_octaves: "+ str(sift.n_octaves))
            print("upsampling: "+ str(sift.upsampling))
            sift.c_dog = sift.c_dog * 0.01
            sift.c_edge = int(sift.c_edge * 2)
            sift.n_octaves = sift.n_octaves * 2
            sift.upsampling = 4
            print("New: ")
            print("c_dog: " + str(sift.c_dog))
            print("c_edge: " + str(sift.c_edge))
            print("n_octaves: " + str(sift.n_octaves))
            print("upsampling: " + str(sift.upsampling))
            descriptor, no_features_extracted = handle_sift_exceptions(array, sift)

        sift.c_dog = orig_c_dog
        sift.c_edge = orig_c_edge
        sift.n_octaves = orig_n_octaves
        sift.upsampling = orig_upsampling

        return descriptor

def extract_descriptors(n_jobs=3, sift=None, images=None):
    descriptors = Parallel(n_jobs=n_jobs)(
            delayed(extract_descriptor)(image, sift, index) for index, image in enumerate(images)
        )
    # Remove images for which SIFT could not be extracted: non-informative image
    return [x for x in descriptors if x is not None]

def cluster_descriptors(dictionary, descriptors):

    return dictionary.fit(np.concatenate(descriptors))


def descriptors_to_histogram(dictionary, descriptors):
    return np.histogram(
        dictionary.predict(descriptors), bins=range(dictionary.n_clusters), density=True
    )[0]

def extract_histograms(n_jobs=3, descriptors=None, dictionary=None):
    X_trans = Parallel(n_jobs=n_jobs)(
                delayed(descriptors_to_histogram)(dictionary=dictionary, descriptors=descriptor)
                for descriptor in descriptors
            )
    return np.array(X_trans)

def reshape_pixels(data=None, pixels=None, nr_of_images=None):
    return np.array(np.split(np.reshape(data, (-1, pixels)), nr_of_images))

def compute_sift_train(n_words=3, random_state=42, n_jobs=3, pixels=28, nr_of_images=60000, data=None):
    data_reshaped = reshape_pixels(data=data, pixels=pixels, nr_of_images=nr_of_images)
    sift = SIFT(n_scales=5, n_octaves=8, c_dog=0.001, c_edge=15)
    descriptors = extract_descriptors(n_jobs=n_jobs, sift=sift, images=data_reshaped)
    dictionary = KMeans(n_clusters=n_words, random_state=random_state)
    dictionary = cluster_descriptors(dictionary=dictionary, descriptors=descriptors)
    data_transformed = extract_histograms(n_jobs=n_jobs, descriptors=descriptors, dictionary=dictionary)
    return data_transformed, dictionary

def compute_sift_test(n_words=3, random_state=42, n_jobs=3, pixels=28, nr_of_images=60000, data=None, dictionary=None):
    data_reshaped = reshape_pixels(data=data, pixels=pixels, nr_of_images=nr_of_images)
    sift = SIFT(n_scales=5, n_octaves=8, c_dog=0.001, c_edge=15)
    descriptors = extract_descriptors(n_jobs=n_jobs, sift=sift, images=data_reshaped)
    data_transformed = extract_histograms(n_jobs=n_jobs, descriptors=descriptors, dictionary=dictionary)
    return data_transformed

def transform_w_sift(n_words=3, random_state=42, n_jobs=3, pixels=28, X_train_data=None, y_train_data=None, X_test_data=None, y_test_data=None):
    data = X_train_data
    nr_of_images= data.shape[0]
    X_train, dictionary = compute_sift_train(n_words=n_words, random_state=random_state,
                               n_jobs=n_jobs, pixels=pixels,
                               nr_of_images=nr_of_images, data=data)
    data = X_test_data
    nr_of_images = data.shape[0]
    X_test = compute_sift_test(n_words=n_words, random_state=random_state,
                              n_jobs=n_jobs, pixels=pixels,
                              nr_of_images=nr_of_images, data=data, dictionary=dictionary)
    y_train = y_train_data
    y_test = y_test_data

    return X_train, X_test, y_train, y_test

# Feature extraction methods

def extract_features(extract_feature_with="SIFT", n_words=100, random_state=42, n_jobs=1, pixels=28, X_train_data=None, y_train_data=None, X_test_data=None, y_test_data=None):
    if extract_feature_with == "SIFT":
        X_train, X_test, y_train, y_test = transform_w_sift(n_words=n_words, random_state=random_state,
                                                            n_jobs=n_jobs, pixels=pixels, X_train_data=X_train_data,
                                                            y_train_data=y_train_data, X_test_data=X_test_data,
                                                            y_test_data=y_test_data)
        return X_train, X_test, y_train, y_test
    elif extract_feature_with == "CH":
        X_train, X_test, y_train, y_test = transform_w_histograms(X_train_data, y_train_data,
                                                                  X_test_data, y_test_data)
        return X_train, X_test, y_train, y_test
    else:
        print("Feature extraction method " + extract_feature_with + " not implemented. Please choose between 'SIFT' and 'CH'.")
        print("Exiting program")
        exit()
