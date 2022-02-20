import cv2
from glob import glob
from os.path import join,basename
import numpy as np

def get_images(path):
    all_images = []
    for fname in glob(path + "/*.png"):
        all_images.extend([join(path, basename(fname))])
    return all_images

def extractSift(img_files):
	img_Files_Sift_dict = {}
	for file in img_files:
		img = cv2.imread (file)
		gray = cv2.cvtColor (img, cv2.COLOR_BGR2GRAY)
		sift = cv2.xfeatures2d.SIFT_create()
		kp, des = sift.detectAndCompute (gray, None)
		img_Files_Sift_dict[file] = des
	return img_Files_Sift_dict


images = get_images("data/")
img = cv2.imread(images[1])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute (gray, None)
img_kp=cv2.drawKeypoints(gray,kp,img)
cv2.imwrite('sift_keypoints.jpg',img_kp)
#img_Files_Sift_dict[file] = des

cv2.imshow("image window", img)
cv2.waitKey(0)
cv2.imshow("image window", gray)
cv2.waitKey(0)



def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

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


from skimage.io import imread
images = get_images("data/")
img = imread(images[0], as_gray=True)
img2 = imread(images[1], as_gray=True)


X_train_data, y_train_data = load_mnist('data/fashion-mnist', kind='train')
X_test_data, y_test_data = load_mnist('data/fashion-mnist', kind='t10k')

from sklearn.cluster import MiniBatchKMeans
from skimage.feature import SIFT
sift = SIFT()
sift.detect_and_extract(img)
descriptor = sift.descriptors

sift = SIFT()
sift.detect_and_extract(img2)
descriptor2 = sift.descriptors

descriptors = np.concatenate([descriptor, descriptor2])
descriptors = Parallel(n_jobs=self.n_jobs)(
            delayed(_load_and_extract_sift)(filename, self.sift) for filename in X
        )

import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans
from skimage.feature import SIFT


def handle_sift_exceptions(array, sift):
    try:
        sift.detect_and_extract(array)
        descriptor = sift.descriptors
        no_features_extracted = False
        return descriptor, no_features_extracted
    except Exception as e:
        descriptor = None
        no_features_extracted = True
        return descriptor, no_features_extracted

def extract_descriptor(array, sift):
    try:
        sift.detect_and_extract(array)
        descriptor = sift.descriptors
        return descriptor
    except Exception as e:
        orig_c_dog = sift.c_dog
        orig_c_edge = sift.c_edge
        no_features_extracted = True
        while(no_features_extracted):
            print("No features could be extracted, loosening threshold to discard low contrast and edge extremas.")
            print("Current ")
            sift.c_dog = sift.c_dog * 0.1
            sift.c_edge = int(sift.c_edge * 1.5)
            descriptor, no_features_extracted = handle_sift_exceptions(array, sift)

        sift.c_dog = orig_c_dog
        sift.c_edge = orig_c_edge
        return descriptor

def extract_descriptors(n_jobs=3, sift=None, images=None):
    descriptors = Parallel(n_jobs=n_jobs)(
            delayed(extract_descriptor)(image, sift) for image in images
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

def compute_sift(n_words=3, random_state=42, n_jobs=3, pixels=28, nr_of_images=60000, data=None):
    data_reshaped = reshape_pixels(data=data, pixels=pixels, nr_of_images=nr_of_images)
    sift = SIFT(n_scales = 5, n_octaves=3, c_dog=0.001, c_edge=15)
    descriptors = extract_descriptors(n_jobs=n_jobs, sift=sift, images=data_reshaped)
    dictionary = MiniBatchKMeans(n_clusters=n_words, random_state=random_state)
    dictionary = cluster_descriptors(dictionary=dictionary, descriptors=descriptors)
    data_transformed = extract_histograms(n_jobs=n_jobs, descriptors=descriptors, dictionary=dictionary)
    return data_transformed

def transform_w_sift(n_words=3, random_state=42, n_jobs=3, pixels=28, X_train_data=None, y_train_data=None, X_test_data=None, y_test_data=None):
    data = X_train_data[:200]
    nr_of_images= data.shape[0]
    X_train = transform_w_sift(n_words=n_words, random_state=random_state,
                               n_jobs=n_jobs, pixels=pixels,
                               nr_of_images=nr_of_images, data=data)
    data = X_test_data[:200]
    nr_of_images = data.shape[0]
    X_test = transform_w_sift(n_words=n_words, random_state=random_state,
                              n_jobs=n_jobs, pixels=pixels,
                              nr_of_images=nr_of_images, data=data)
    y_train = y_train_data[:200]
    y_test = y_test_data[:200]

    return X_train, X_test, y_train, y_test

n_words = 20
random_state = 42
pixels = 28
n_jobs=3

X_train_data, y_train_data, X_test_data, y_test_data = load_mnist_data()

data = X_train_data[:200]
nr_of_images = data.shape[0]
X_train = transform_w_sift(n_words=n_words, random_state=random_state, n_jobs=3, pixels=pixels, nr_of_images=nr_of_images, data=data)

[i for i,v in enumerate(descriptors) if v == None]


data = X_test_data
nr_of_images = data.shape[0]
X_test = transform_w_sift(n_words=n_words, random_state=random_state, n_jobs=3, pixels=pixels, nr_of_images=nr_of_images, data=X_test_data)

descriptors = extract_descriptors(n_jobs=n_jobs, sift=sift, images=data_reshaped)
def extract_descriptor(array, sift):
    sift.detect_and_extract(array)
    descriptor = sift.descriptors
    return descriptor

def extract_descriptors(n_jobs=3, sift=None, images=None):
    descriptors = Parallel(n_jobs=n_jobs)(
            delayed(extract_descriptor)(image, sift) for image in images
        )
    return descriptors

count = 1
for image in data_reshaped:
    print(count)
    sift.detect_and_extract(image)
    count = count + 1



sift.detect_and_extract(data_reshaped[18])















sift = SIFT()
sift.detect_and_extract(mnist)
descriptor_mnist = sift.descriptors

dictionary.fit(np.array([np.concatenate(descriptor), np.concatenate(descriptor), np.concatenate(descriptor)]))

np.histogram(dictionary.predict(
    np.array([np.concatenate(descriptor), np.concatenate(descriptor), np.concatenate(descriptor)])
), bins=range(dictionary.n_clusters), density=True)[0]
