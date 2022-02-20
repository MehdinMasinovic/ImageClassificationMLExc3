import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans
from sklearn.utils import check_random_state
from skimage.feature import SIFT
from skimage.io import imread


def _load_and_extract_sift(filename, sift):
    img = imread(filename, as_gray=True)
    sift.detect_and_extract(img)
    return sift.descriptors

def _descriptors_to_histogram(descriptors, dictionary):
    return np.histogram(
        dictionary.predict(descriptors), bins=range(dictionary.n_clusters), density=True
    )[0]


class BagOfVisualWords(TransformerMixin, BaseEstimator):
    def __init__(self, n_words, batch_size=1024, n_jobs=None, random_state=None):
        self.n_words = n_words
        self.batch_size = batch_size
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        random_state = check_random_state(self.random_state)

        self.dictionary = MiniBatchKMeans(
            n_clusters=self.n_words, random_state=random_state
        )
        self.sift = SIFT()

        descriptors = Parallel(n_jobs=self.n_jobs)(
            delayed(_load_and_extract_sift)(filename, self.sift) for filename in X
        )

        self.dictionary.fit(np.concatenate(descriptors))

        X_trans = Parallel(n_jobs=self.n_jobs)(
            delayed(_descriptors_to_histogram)(descr_img, self.dictionary)
            for descr_img in descriptors
        )

        return np.array(X_trans)

    def transform(self, X, y=None):
        descriptors = Parallel(n_jobs=self.n_jobs)(
            delayed(_load_and_extract_sift)(filename, self.sift) for filename in X
        )

        X_trans = Parallel(n_jobs=self.n_jobs)(
            delayed(_descriptors_to_histogram)(descr_img, self.dictionary)
            for descr_img in descriptors
        )

        return np.array(X_trans)


# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

bovw = BagOfVisualWords(n_words=1000, n_jobs=-1)
classifier = RandomForestClassifier(n_estimators=100, n_jobs=-1)
model = make_pipeline(bovw, classifier)

# %%
from sklearn.model_selection import cross_validate

cv_results = cross_validate(
    model, image_filenames, labels, cv=5, scoring='accuracy', return_train_score=True
)

# %%
import pandas as pd

cv_results = pd.DataFrame(cv_results)
cv_results


